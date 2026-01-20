import os
import streamlit as st
import pandas as pd
import requests
import time
from functools import lru_cache
from io import BytesIO
import folium
from pathlib import Path
from streamlit_folium import st_folium
import re
import json
import numpy as np
import matplotlib.colors as mcolors
from folium.plugins import BeautifyIcon
from dataclasses import dataclass

# =========================================================
# SIMPLE ORDER STRUCT (for OSRM & traffic matrices)
# =========================================================
@dataclass
class SimpleOrder:
    id: int
    enlem: float
    boylam: float
    desi: float = 0.0
    servis: float = 0.0


def df_to_orders(df_orders: pd.DataFrame):
    """Convert Streamlit orders_df → list[SimpleOrder] for OSRMClient / traffic."""
    return [
        SimpleOrder(
            id=int(row["OrderID"]),
            enlem=float(row["Enlem"]),
            boylam=float(row["Boylam"]),
            desi=float(row.get("Desi", 0)),
            servis=float(row.get("Servis Süresi (dk)", 0)),
        )
        for _, row in df_orders.iterrows()
    ]


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="EVRP Optimizer", layout="wide", page_icon="🚚")

# Apply custom styling
from utils.ui_components import apply_custom_css, render_header, info_card, success_card, warning_card

apply_custom_css()

# Main header
render_header(
    "Electric Vehicle Routing Problem Optimizer",
    "Elektrikli araç filosu için akıllı rota optimizasyonu"
)

# =========================================================
# SESSION STATE
# =========================================================
if "single_results" not in st.session_state:
    st.session_state["single_results"] = []

if "orders_df" not in st.session_state:
    st.session_state["orders_df"] = None

if "osrm_D" not in st.session_state:
    st.session_state["osrm_D"] = None

if "osrm_T" not in st.session_state:
    st.session_state["osrm_T"] = None

# cached structures for optimization
for key in ["evrp_problem", "ortools_data", "tabu_result",
            "ortools_routes", "ga_best_routes", "ga_best_fitness"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================================
# CONSTANTS
# =========================================================
DEPOT_LAT = 40.900
DEPOT_LON = 29.300
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"

# Load local .env (optional) so `os.getenv(...)` works in dev runs.
# Streamlit does NOT automatically load `.env` into the process environment.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(), override=False)
except Exception:
    # If python-dotenv isn't available (or no .env exists), we fall back to OS env vars.
    pass


def _get_streamlit_secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# Get API key from Streamlit secrets (primary) or environment variables.
OPENCAGE_API_KEY = _get_streamlit_secret("OPENCAGE_API_KEY") or os.getenv("OPENCAGE_API_KEY")

if not OPENCAGE_API_KEY or not str(OPENCAGE_API_KEY).strip() or str(OPENCAGE_API_KEY).strip() in {"YOUR_KEY_HERE", "YOUR_OPENCAGE_API_KEY"}:
    st.error(
        "🔑 OPENCAGE_API_KEY bulunamadı. Şunlardan birini yapın:\n"
        "- `.streamlit/secrets.toml` içine `OPENCAGE_API_KEY = \"...\"` ekleyin\n"
        "- veya ortam değişkeni olarak `OPENCAGE_API_KEY` tanımlayın\n"
        "İpucu: Bu repo için örnek dosya: `.streamlit/secrets.toml.example`"
    )
    st.stop()

# =========================================================
# LOAD TRAFFIC DATA (CONSTANT, ALWAYS LOADED)
# =========================================================
@st.cache_data
def load_traffic_data():
    path = DATA_DIR / "traffic_density_2024_clean_with_dayofweek.csv"
    df = pd.read_csv(path)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df[["LATITUDE", "LONGITUDE", "HOUR", "DAY_OF_WEEK", "AVG_SPEED_CLEAN"]]
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    df["HOUR"] = df["HOUR"].astype(int)
    df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype(int)
    df["AVG_SPEED_CLEAN"] = df["AVG_SPEED_CLEAN"].astype(float)
    return df


if "traffic_df" not in st.session_state:
    st.session_state["traffic_df"] = load_traffic_data()

# =========================================================
# IMPORT UTILS
# =========================================================
from utils.osrm_client import OSRMClient
from utils.visualize_routes_osrm import visualize_routes_osrm
from utils.data_builder import (
    build_problem_and_data_from_globals,
    CAPACITY_DESI,
    BATTERY_CAPACITY,
    ENERGY_B,
    BASE_KWH_PER_100KM,
)
from utils.ortools_tabu_solver import solve_with_ortools_tabu
from utils.ga_optimizer import ga_optimize_sequences, total_plan_cost
from utils.traffic_osrm import osrm_route_with_traffic
from utils.traffic_time_matrices import build_time_matrices_with_traffic_optimized
from utils.energy_comparator import (
    compare_ortools_vs_ga,
    format_route_report,
    format_fleet_comparison,
)
from utils.normalization_ai import ai_normalize_address
from utils.parser import (
    smart_mahalle_detector,
    parse_cadde,
    parse_sokak
)
from utils.parser import parse_mahalle_regex, parse_cadde, parse_sokak
from utils.normalization_ai import ascii_fallback
from utils.depot_distance_filter import depot_distance_feasibility


# create OSRM client once
if "osrm_client" not in st.session_state:
    st.session_state["osrm_client"] = OSRMClient(
        host="https://router.project-osrm.org",
        profile="driving",
    )
EUROPE_DISTRICTS = {
    "avcılar", "bakırköy", "bahçelievler", "bağcılar", "başakşehir",
    "bayrampaşa", "beşiktaş", "beylikdüzü", "beyoğlu", "büyükçekmece",
    "çatalca", "esenler", "esenyurt", "eyüpsultan", "fatih",
    "gaziosmanpaşa", "güngören", "kağıthane", "küçükçekmece",
    "sarıyer", "silivri", "şişli", "zeytinburnu", "arnavutköy"
}
ALLOWED_CITY = "istanbul"
import unicodedata

def normalize_tr(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.strip()

    # Normalize Unicode (CRITICAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    return (
        s.lower()
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
    )



# =========================================================
# LOAD MAHALLE DATA
# =========================================================
@st.cache_data
def load_mahalle_data():
    df = pd.read_excel(DATA_DIR / "Istanbul_Mahalle_Listesi.xlsx")
    df.columns = [c.lower() for c in df.columns]
    return df


mahalle_df = load_mahalle_data()

# =========================================================
# CLEANERS
# =========================================================
def clean_street(street):
    if not isinstance(street, str):
        return ""
    street = street.strip()
    street = street.replace("İ", "i").replace("I", "ı").title()

    replace_map = {
        r"\bSk\b": "Sokak",
        r"\bSk.\b": "Sokak",
        r"\bCd\b": "Caddesi",
        r"\bCd.\b": "Caddesi",
        r"No:": "",
        r"No": "",
    }
    for pat, rep in replace_map.items():
        street = re.sub(pat, rep, street)

    street = re.sub(r"\s+", " ", street)
    return street.strip()


def clean_mahalle(mahalle):
    if not isinstance(mahalle, str):
        return ""
    m = mahalle.lower()
    m = re.sub(r"\bmah.*\b", "", m)
    m = m.strip().title()
    return f"{m} Mahallesi"


# =========================================================
# GEOCODERS
# =========================================================
@lru_cache(maxsize=5000)
def geocode_opencage(query):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": query, "key": OPENCAGE_API_KEY, "limit": 1}
    try:
        r = requests.get(url, params=params, timeout=6)
        return r.json(), r.url
    except Exception:
        return None, None


@lru_cache(maxsize=5000)
def geocode_nominatim(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "EVRP-Geocoder/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=6)
        return r.json(), r.url
    except Exception:
        return None, None


from utils.normalization_ai import ascii_fallback
import json
import time


def smart_geocode(street, mahalle, ilce, il):

    # ---------------------------------------------
    # Build the detailed geocode query (full)
    # ---------------------------------------------
    full_q = f"{street}, {mahalle}, {ilce}, {il}, Türkiye"
    full_q_ascii = ascii_fallback(full_q)

    # ---------------------------------------------
    # 1) TRY OPENCAGE — FULL QUERY
    # ---------------------------------------------
    oc_json, oc_url = geocode_opencage(full_q_ascii)

    if oc_json and oc_json.get("results"):
        best = oc_json["results"][0]
        comp = best.get("components", {})
        confidence = best.get("confidence", 0)

        # Extract OpenCage admin levels safely
        city_like = ascii_fallback(
            comp.get("city")
            or comp.get("town")
            or comp.get("county")
            or ""
        ).lower()

        suburb_like = ascii_fallback(
            comp.get("suburb")
            or comp.get("neighbourhood")
            or comp.get("city_district")
            or ""
        ).lower()

        ilce_ascii = ascii_fallback(ilce).lower()
        mahalle_ascii = ascii_fallback(mahalle).lower()

        # ---------------------------------------------
        # VALIDATION CHECKS
        # ---------------------------------------------
        good_confidence = confidence >= 6
        matches_ilce = (ilce_ascii in city_like) or (city_like == "")
        matches_mahalle = (mahalle_ascii in suburb_like) or (suburb_like == "")

        # ---------------------------------------------
        # Acceptable result
        # ---------------------------------------------
        if good_confidence and matches_ilce and matches_mahalle:
            lat = best["geometry"]["lat"]
            lon = best["geometry"]["lng"]
            return (
                lat,
                lon,
                "opencage",
                oc_url,
                json.dumps(oc_json, indent=2, ensure_ascii=False)
            )

    # ---------------------------------------------
    # 2) OPENCAGE FALLBACK — MAHALLE-FOCUSED QUERY
    # ---------------------------------------------
    mahalle_q = f"{mahalle}, {ilce}, {il}, Turkey"
    mahalle_q_ascii = ascii_fallback(mahalle_q)

    oc_json2, oc_url2 = geocode_opencage(mahalle_q_ascii)

    if oc_json2 and oc_json2.get("results"):
        best2 = oc_json2["results"][0]
        lat = best2["geometry"]["lat"]
        lon = best2["geometry"]["lng"]
        return (
            lat,
            lon,
            "opencage_mahalle",
            oc_url2,
            json.dumps(oc_json2, indent=2, ensure_ascii=False)
        )

    # ---------------------------------------------
    # 3) NOMINATIM FALLBACK — MAHALLE ONLY
    # ---------------------------------------------
    nom_json, nom_url = geocode_nominatim(mahalle_q_ascii)

    if nom_json:
        try:
            time.sleep(1)
            lat = float(nom_json[0]["lat"])
            lon = float(nom_json[0]["lon"])
            return (
                lat,
                lon,
                "nominatim",
                nom_url,
                json.dumps(nom_json, indent=2, ensure_ascii=False)
            )
        except:
            pass

    # ---------------------------------------------
    # 4) TOTAL FAILURE → RETURN BLANKS
    # ---------------------------------------------
    return None, None, "failed", None, None



# =========================================================
# OSRM MATRIX BUILDER (fallback, but OSRMClient is preferred)
# =========================================================
def build_osrm_matrices(
    df_orders,
    depot_lat,
    depot_lon,
    osrm_host="https://router.project-osrm.org",
    profile="driving",
):
    coords = [(depot_lon, depot_lat)] + [
        (row["Boylam"], row["Enlem"]) for _, row in df_orders.iterrows()
    ]

    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
    url = f"{osrm_host}/table/v1/{profile}/{coord_str}?annotations=distance,duration"

    r = requests.get(url, timeout=120, verify=False)
    r.raise_for_status()
    j = r.json()

    D = np.array(j["distances"], dtype=float) / 1000.0
    T = np.array(j["durations"], dtype=float) / 60.0
    return D, T


# =========================================================
# HELPER: EXTRACT ROUTES FROM OR-TOOLS SOLUTION (Option A)
# =========================================================
def extract_routes_from_solution(data, routing, manager, solution):
    """
    Returns list of routes by vehicle:
    [
      [node_idx_1, node_idx_2, ...],   # vehicle 0
      ...
    ]
    where nodes are 1..N (0 = depot)
    """
    routes = []
    n_vehicles = data["num_vehicles"]
    depot = data["depot"]

    for v in range(n_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)
    return routes

# =========================================================
# ⚡ ADVANCED EVRP FEASIBILITY ANALYZER
# =========================================================
import numpy as np

BASE_KWH_PER_KM = 0.436
ENERGY_PER_DESI_KM = 0.00136

def evrp_feasibility_detailed(data, work_start_min=9*60, work_end_min=18*60):
    """
    EVRP Feasibility Debugger
    Returns:
        (feasible: bool, report_text: str, sections: dict)
    sections = {
        "capacity": [...],
        "time": [...],
        "battery": [...],
        "summary": [...]
    }
    """

    depot = data["depot"]
    D = np.array(data["distance_km"])
    T = np.array(data["time_min"])
    demand = np.array(data["demand_desi"])
    service = np.array(data["service_min"])
    num_vehicles = data["num_vehicles"]
    cap = data["vehicle_cap_desi"]
    battery = float(data["battery_capacity"])
    n = len(D)

    horizon = work_end_min - work_start_min

    feasible = True
    sections = {"capacity": [], "time": [], "battery": [], "summary": []}

    # ============================================================
    # 1) CAPACITY
    # ============================================================
    oversized = np.where(demand > cap)[0]
    if len(oversized) > 0:
        feasible = False
        sections["capacity"].append("❌ Aşağıdaki müşteriler kapasiteyi aşıyor:")
        for i in oversized:
            sections["capacity"].append(f" - Node {i}: {demand[i]} > {cap}")
    else:
        sections["capacity"].append("✅ Hiçbir müşteri kapasite aşmıyor.")

    total_demand = demand.sum()
    total_capacity = num_vehicles * cap

    if total_capacity < total_demand:
        feasible = False
        sections["capacity"].append(
            f"❌ Toplam talep {total_demand:.1f} > toplam filo kapasitesi {total_capacity:.1f}"
        )
    else:
        sections["capacity"].append("✅ Toplam filo kapasitesi yeterli.")

    lb_cap = int(np.ceil(total_demand / cap))
    sections["capacity"].append(f"ℹ️ Minimum araç (kapasite): {lb_cap}")

    # ============================================================
    # 2) TIME
    # ============================================================
    impossible_nodes = []
    for i in range(n):
        if i == depot:
            continue
        travel_out = T[depot, i]
        travel_back = T[i, depot]
        req = travel_out + service[i] + travel_back
        if req > horizon:
            feasible = False
            impossible_nodes.append((i, req))

    if impossible_nodes:
        sections["time"].append("❌ Aşağıdaki müşteriler süreye sığmıyor:")
        for node, req in impossible_nodes:
            sections["time"].append(
                f" - Node {node}: {req:.1f} dk > {horizon} dk"
            )
    else:
        sections["time"].append("✅ Tüm müşteriler süre açısından uygun.")

    min_travel = [
        min(T[depot, i], T[i, depot]) for i in range(n) if i != depot
    ]
    approx_total_min = sum(min_travel) + sum(service)
    lb_time = int(np.ceil(approx_total_min / horizon))
    sections["time"].append(f"ℹ️ Minimum araç (zaman): {lb_time}")

    # ============================================================
    # 3) BATTERY
    # ============================================================
    def energy_cost(dist_km, load):
        return dist_km * (BASE_KWH_PER_KM + ENERGY_PER_DESI_KM * load)

    too_far_nodes = []
    for i in range(n):
        if i == depot:
            continue
        e1 = energy_cost(D[depot, i], demand[i])
        e2 = energy_cost(D[i, depot], 0)
        if e1 > battery or e2 > battery:
            feasible = False
            too_far_nodes.append((i, e1, e2))

    if too_far_nodes:
        sections["battery"].append("❌ Batarya nedeniyle ulaşılamayan müşteriler:")
        for i, e1, e2 in too_far_nodes:
            sections["battery"].append(
                f" - Node {i}: gidiş {e1:.2f} kWh, dönüş {e2:.2f} kWh (batarya={battery})"
            )
    else:
        sections["battery"].append("✅ Batarya tüm müşteriler için yeterli.")

    min_energy = sum(D[depot, i] * BASE_KWH_PER_KM for i in range(n) if i != depot)
    lb_energy = int(np.ceil(min_energy / battery))
    sections["battery"].append(f"ℹ️ Minimum araç (enerji): {lb_energy}")

    # ============================================================
    # SUMMARY
    # ============================================================
    required = max(lb_cap, lb_time, lb_energy)
    sections["summary"].append(f"➡️ Minimum araç gereksinimi: {required}")
    sections["summary"].append(f"➡️ Mevcut araç sayısı: {num_vehicles}")

    if num_vehicles < required:
        feasible = False
        sections["summary"].append("❌ Filo boyutu yetersiz.")
    else:
        sections["summary"].append("🎉 Filo boyutu yeterli!")

    report = "\n".join(
        ["\n".join(v) for v in sections.values()]
    )

    return feasible, report, sections



# =========================================================
# MAIN TABS (Adres / Orders / Map / OSRM)
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1️⃣ Adres → Koordinat",
        "2️⃣ Sipariş Oluştur",
        "3️⃣ Siparişleri Haritada Göster",
        "4️⃣ OSRM Mesafe & Süre Matrisi",
        "5️⃣ Trafikli Süre Matrisleri",
        "6️⃣ Problem Çözümü",
        "7️⃣ Çoklu Görev Optimizasyonu",
    ]
)

with tab1:
    # =========================================================
    # 1) SINGLE ADDRESS GEOCODER
    # =========================================================
    st.header("1) Adres Bileşenleri")

    il = st.selectbox("İl", sorted(mahalle_df["il"].unique()), key="il_sel")
    ilce = st.selectbox(
        "İlçe",
        sorted(mahalle_df[mahalle_df["il"] == il]["ilce"].unique()),
        key="ilce_sel",
    )
    mah = st.selectbox(
        "Mahalle",
        sorted(
            mahalle_df[
                (mahalle_df["il"] == il) & (mahalle_df["ilce"] == ilce)
            ]["mahalle"].unique()
        ),
        key="mah_sel",
    )

    street_raw = st.text_input("Sokak + Kapı No", key="street_raw")

    street_clean = clean_street(street_raw)
    mahalle_clean = clean_mahalle(mah)

    st.json(
        {
            "street": street_clean,
            "mahalle": mahalle_clean,
            "ilce": ilce,
            "il": il,
        }
    )

    if st.button("📍 Koordinatları Bul", key="btn_geocode"):
        lat, lon, kaynak, req_url, resp_json = smart_geocode(
            street_clean, mahalle_clean, ilce, il
        )

        if lat:
            st.success(f"📌 {lat}, {lon} — Kaynak: {kaynak}")
            st.session_state["single_results"].append(
                {
                    "Street": street_clean,
                    "Mahalle": mahalle_clean,
                    "Ilce": ilce,
                    "Il": il,
                    "Enlem": lat,
                    "Boylam": lon,
                    "Kaynak": kaynak,
                }
            )
        else:
            st.error("❌ Adres bulunamadı")

    st.subheader("2) Sorgulama Geçmişi")
    if st.session_state.get("single_results"):
        st.dataframe(pd.DataFrame(st.session_state["single_results"]), use_container_width=True)
    else:
        st.info("Henüz bir adres sorgulanmadı.")

    # =========================================================
    # BULK GEOCODER
    # =========================================================
    st.markdown("---")
    st.header("📤 Toplu Adres → Koordinat İşleme")

    bulk_file = st.file_uploader(
        "Excel yükle (id, il, ilçe, adres, desi, tahmini servis süresi)",
        type=["xlsx"],
        key="bulk_upload_tab1",
    )

    if bulk_file:
        df_bulk = pd.read_excel(bulk_file)

        required_cols = ["id", "il", "ilçe", "adres", "desi", "tahmini servis süresi"]
        if not all(col in df_bulk.columns for col in required_cols):
            st.error(
                f"❌ Excel sütunları eksik. Gerekli sütunlar: {', '.join(required_cols)}"
            )
            st.stop()

        st.success("✔ Dosya yüklendi.")
        st.dataframe(df_bulk.head(), use_container_width=True)

        # ---------------------------------------------------------
        # STEP 0 — NORMALIZE CITY & DISTRICT
        # ---------------------------------------------------------
        df_bulk["il_norm"] = df_bulk["il"].apply(normalize_tr)
        df_bulk["ilçe_norm"] = df_bulk["ilçe"].apply(normalize_tr)

        # ---------------------------------------------------------
        # STEP 1 — KEEP ONLY İSTANBUL ORDERS
        # ---------------------------------------------------------
        ALLOWED_CITY = "istanbul"

        removed_city_count = (df_bulk["il_norm"] != ALLOWED_CITY).sum()
        df_bulk = df_bulk[df_bulk["il_norm"] == ALLOWED_CITY]

        if removed_city_count > 0:
            st.warning(f"❗ İstanbul dışı {removed_city_count} sipariş çıkarıldı.")

        if df_bulk.empty:
            st.error("📭 İstanbul içinde işlenecek sipariş yok.")
            st.stop()

        # ---------------------------------------------------------
        # STEP 2 — REMOVE EUROPE-SIDE ORDERS
        # ---------------------------------------------------------
        df_europe = df_bulk[df_bulk["ilçe_norm"].isin(EUROPE_DISTRICTS)]
        df_bulk = df_bulk[~df_bulk["ilçe_norm"].isin(EUROPE_DISTRICTS)]

        removed_count = len(df_europe)

        if removed_count > 0:
            st.warning(f"❗ Avrupa yakasından {removed_count} sipariş çıkarıldı.")

        if df_bulk.empty:
            st.error("📭 Anadolu yakasında işlenecek sipariş yok.")
            st.stop()

        # ---------------------------------------------------------
        # STEP 3 — GROUP DUPLICATE ADDRESSES
        # ---------------------------------------------------------
        grouped = (
            df_bulk
            .groupby("adres")
            .agg({
                "id": lambda x: ",".join(x.astype(str)),
                "desi": "sum",
                "tahmini servis süresi": "mean",
                "il": "first",
                "ilçe": "first",
            })
            .reset_index()
        )

        st.info(
            f"🔄 {len(df_bulk)} sipariş → {len(grouped)} eşsiz adrese indirildi."
        )
        df_bulk = grouped

        # ---------------------------------------------------------
        # STEP 4 — NEW SEQUENTIAL IDS
        # ---------------------------------------------------------
        df_bulk["new_id"] = range(1, len(df_bulk) + 1)

        # ---------------------------------------------------------
        # GEOCODE BUTTON
        # ---------------------------------------------------------
        if st.button("🚀 Toplu Geocode Başlat", key="bulk_geocode_btn"):
            results = []
            total = len(df_bulk)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, row in df_bulk.iterrows():
                status_text.text(f"⏳ İşleniyor: {i + 1} / {total}")

                normalized = ai_normalize_address(str(row["adres"]))

                mahalle = parse_mahalle_regex(normalized)
                cadde = parse_cadde(normalized)
                sokak = parse_sokak(normalized)
                street = cadde if cadde else sokak

                lat, lon, src, req_url, raw_json = smart_geocode(
                    street,
                    mahalle,
                    row["ilçe"],
                    row["il"]
                )

                results.append({
                    "id": row["new_id"],
                    "enlem": lat,
                    "boylam": lon,
                    "desi": row["desi"],
                    "tahmini servis süresi": row["tahmini servis süresi"],
                    "il": row["il"],
                    "ilçe": row["ilçe"],
                    "mahalle": mahalle,
                    "cadde": cadde,
                    "sokak": sokak,
                    "adres": normalized,
                    "Kaynak": src,
                })

                progress_bar.progress((i + 1) / total)

            status_text.empty()
            df_result = pd.DataFrame(results)

            st.success(
                f"🎉 Toplu adres sorgulama tamamlandı! "
                f"Avrupa yakasından çıkarılan: {removed_count}, "
                f"işlenen adres sayısı: {len(df_bulk)}."
            )

            st.subheader("📄 Sonuçlar (Sipariş Oluştur Formatında)")
            st.dataframe(df_result, use_container_width=True)

            buffer = BytesIO()
            df_result.to_excel(buffer, index=False)

            st.download_button(
                label="📥 Excel Sonuçlarını İndir",
                data=buffer.getvalue(),
                file_name="siparis_olustur_bulkgis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )





# =========================================================
# TAB 2 — SİPARİŞ OLUŞTUR (Excel + Manuel)
# =========================================================
with tab2:
    st.header("3) Sipariş Tablosu Oluştur")

    # -------- Sample Excel --------
    st.subheader("📥 Örnek Excel Şablonu İndir")

    sample_df = pd.DataFrame(
        {
            "id": [1, 2],
            "enlem": [40.9000, 40.9500],
            "boylam": [29.3000, 29.3500],
            "desi": [500, 1200],
            "tahmini servis süresi": [30, 45],
        }
    )

    sample_out = BytesIO()
    sample_df.to_excel(sample_out, index=False)

    st.download_button(
        label="📄 Örnek Sipariş Excel Dosyası",
        data=sample_out.getvalue(),
        file_name="ornek_siparis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")

    # -------- Excel Upload --------
    st.subheader("📤 Excel'den Sipariş Yükle")

    uploaded_file = st.file_uploader(
        "Excel yükle (id, enlem, boylam, desi, tahmini servis süresi)",
        type=["xlsx"],
        key="orders_upload",
    )

    if uploaded_file is not None:
        try:
            df_up = pd.read_excel(uploaded_file)

            required_cols = ["id", "enlem", "boylam", "desi", "tahmini servis süresi"]
            missing = [c for c in required_cols if c not in df_up.columns]

            if missing:
                st.error(f"❌ Eksik kolonlar: {missing}")
            else:
                df_orders = df_up.rename(
                    columns={
                        "id": "OrderID",
                        "enlem": "Enlem",
                        "boylam": "Boylam",
                        "desi": "Desi",
                        "tahmini servis süresi": "Servis Süresi (dk)",
                    }
                )

                st.session_state["orders_df"] = df_orders

                st.success("📥 Excel başarıyla yüklendi!")
                st.dataframe(df_orders, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Excel okunamadı: {e}")

    st.markdown("---")

    # -------- Manual Order Creation --------
    st.subheader("📝 Manuel Sipariş Oluştur")

    df_hist = pd.DataFrame(st.session_state["single_results"])

    if df_hist.empty:
        st.info("Önce Tab 1'de adres sorgulayın veya üstten Excel yükleyin.")
    else:
        selected = []

        st.subheader("Satır Seçimi + Desi / Servis Süresi")

        h = st.columns([0.6, 2.8, 2.2, 1.4, 1.4, 1.3, 1.7])
        h[0].markdown("**Seç**")
        h[1].markdown("**Street**")
        h[2].markdown("**Mahalle**")
        h[3].markdown("**Enlem**")
        h[4].markdown("**Boylam**")
        h[5].markdown("**Desi**")
        h[6].markdown("**Servis (dk)**")

        for i, row in df_hist.iterrows():
            cols = st.columns([0.6, 2.8, 2.2, 1.4, 1.4, 1.3, 1.7])
            chk = cols[0].checkbox("", key=f"sel_{i}")
            cols[1].write(row["Street"])
            cols[2].write(row["Mahalle"])
            cols[3].write(round(row["Enlem"], 5))
            cols[4].write(round(row["Boylam"], 5))
            cols[5].text_input("", key=f"desi_{i}", placeholder="örn: 500")
            cols[6].text_input("", key=f"svc_{i}", placeholder="örn: 30")

            if chk:
                selected.append(i)

        if st.button("📦 Sipariş Tablosunu Oluştur", key="btn_orders"):
            orders = []

            for order_id, idx in enumerate(selected, start=1):
                row = df_hist.loc[idx]

                def parse_int(key):
                    v = st.session_state.get(key, "")
                    try:
                        return int(v)
                    except Exception:
                        return 0

                orders.append(
                    {
                        "OrderID": order_id,
                        "Street": row["Street"],
                        "Mahalle": row["Mahalle"],
                        "Ilce": row["Ilce"],
                        "Il": row["Il"],
                        "Enlem": row["Enlem"],
                        "Boylam": row["Boylam"],
                        "Desi": parse_int(f"desi_{idx}"),
                        "Servis Süresi (dk)": parse_int(f"svc_{idx}"),
                    }
                )

            df_orders = pd.DataFrame(orders)
            st.session_state["orders_df"] = df_orders

            st.success("📦 Sipariş tablosu oluşturuldu.")
            st.dataframe(df_orders, use_container_width=True)


# =========================================================
# TAB 3 — ORDERS MAP
# =========================================================
with tab3:
    st.header("4) Siparişleri Haritada Göster")

    df_orders = st.session_state.get("orders_df")

    if df_orders is None or df_orders.empty:
        st.info("Önce sipariş oluşturun.")
    else:
        st.dataframe(df_orders, use_container_width=True)

        all_coords = [(DEPOT_LAT, DEPOT_LON)]
        avg_lat = df_orders["Enlem"].mean()
        avg_lon = df_orders["Boylam"].mean()

        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=11,
            tiles="cartodbpositron",
        )

        # depot
        folium.Marker(
            [DEPOT_LAT, DEPOT_LON],
            tooltip="🚩 <b>Depot (Start/End)</b>",
            popup=f"<b>Depot</b><br>Lat: {DEPOT_LAT:.4f}<br>Lon: {DEPOT_LON:.4f}",
            icon=BeautifyIcon(
                icon_shape="circle",
                border_color="red",
                border_width=3,
                text_color="white",
                background_color="red",
                inner_icon_style="font-size:24px;margin-top:-10px;font-weight:bold;",
                number="H",
            ),
        ).add_to(m)

        # orders
        blue_color = list(mcolors.TABLEAU_COLORS.values())[0]

        for _, row in df_orders.iterrows():
            tooltip_html = (
                f"<b>Order ID:</b> {row['OrderID']}<br>"
                f"<b>Desi:</b> {row['Desi']}<br>"
                f"<b>Servis Süresi:</b> {row['Servis Süresi (dk)']} dk"
            )

            folium.Marker(
                [row["Enlem"], row["Boylam"]],
                tooltip=tooltip_html,
                popup=tooltip_html,
                icon=BeautifyIcon(
                    number=str(row["OrderID"]),
                    border_color="black",
                    border_weight=2,
                    text_color="white",
                    background_color=blue_color,
                    inner_icon_style="margin-top:0px;",
                    spin=False,
                ),
            ).add_to(m)

            all_coords.append((row["Enlem"], row["Boylam"]))

        m.fit_bounds(all_coords)

        _, col_map, _ = st.columns([1, 6, 1])
        with col_map:
            st_folium(m, width=1200, height=750)


# =========================================================
# TAB 4 — OSRM MATRICES (NEW CLIENT)
# =========================================================
with tab4:
    st.header("5) OSRM Mesafe & Süre Matrisi")

    df_orders = st.session_state.get("orders_df")

    if df_orders is None or df_orders.empty:
        st.info("Önce sipariş oluşturun.")
        st.stop()

    st.dataframe(df_orders, use_container_width=True)

    # ---------------- OSRM MATRIX BUILD ----------------
    if st.button("🧮 Hesapla", key="btn_osrm"):
        with st.spinner("OSRM çağrısı yapılıyor..."):
            depot_obj = SimpleOrder(
                id=0,
                enlem=DEPOT_LAT,
                boylam=DEPOT_LON,
            )

            orders = df_to_orders(df_orders)
            osrm = st.session_state["osrm_client"]

            D, T = osrm.build_matrices_from_orders(depot_obj, orders)

            st.session_state["osrm_D"] = D
            st.session_state["osrm_T"] = T

            st.success("OSRM matrisleri hazır!")

    D = st.session_state.get("osrm_D")
    T = st.session_state.get("osrm_T")

    if D is None or T is None:
        st.info("Henüz OSRM matrisi yok.")
        st.stop()

    # ---------------- SHOW MATRICES ----------------
    st.write("📏 Mesafe Matrisi (km)")
    st.dataframe(pd.DataFrame(D), use_container_width=True)

    st.write("⏱ Süre Matrisi (dk)")
    st.dataframe(pd.DataFrame(T), use_container_width=True)

    # =========================================================
    # 🚦 DEPOT DISTANCE & ENERGY FEASIBILITY (PRE-EVRP)
    # =========================================================
    st.markdown("---")
    st.subheader("🚦 Depot Distance & Energy Feasibility (Pre-EVRP)")

    MAX_KM = st.number_input(
        "Max depot → customer distance (km)",
        min_value=10,
        max_value=300,
        value=110,
        step=5,
    )

    BATTERY = BATTERY_CAPACITY  # global in your app

    if st.button("🧹 Check & Remove Infeasible Orders"):
        feasible_nodes, removed = depot_distance_feasibility(
            D=D,
            demand=df_orders["Desi"].values,
            battery_kwh=BATTERY,
            max_one_way_km=MAX_KM,
            depot=0,
        )

        if removed:
            st.error(f"❌ {len(removed)} order(s) removed")

            removed_df = pd.DataFrame(removed)

            # node_index -> OrderID (node 0 = depot)
            removed_df["OrderID"] = removed_df["node_index"].apply(
                lambda i: df_orders.iloc[i - 1]["OrderID"]
            )

            st.dataframe(removed_df, use_container_width=True)

            # ---- DOWNLOAD ----
            buffer = BytesIO()
            removed_df.to_excel(buffer, index=False)

            st.download_button(
                "📥 Download Removed Orders",
                buffer.getvalue(),
                file_name="removed_by_distance_energy.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # ---- REMOVE FROM SESSION ----
            drop_idx = [r["node_index"] - 1 for r in removed]
            df_orders_clean = (
                df_orders.drop(df_orders.index[drop_idx])
                .reset_index(drop=True)
            )

            st.session_state["orders_df"] = df_orders_clean
            st.session_state["osrm_D"] = None
            st.session_state["osrm_T"] = None

            st.warning("Remaining feasible orders:")
            st.dataframe(df_orders_clean, use_container_width=True)

            st.info("➡ OSRM matrisi sıfırlandı. Lütfen yeniden hesaplayın.")

        else:
            st.success("✅ All orders pass depot distance & energy feasibility.")


# =========================================================
# TAB 5 — TRAFİKLİ OSRM ROTA ANALİZİ (Optimized builder)
# =========================================================
with tab5:
    st.header("🚦 Trafik Bazlı Süre Matrisleri (09:00–18:00)")

    df_orders = st.session_state.get("orders_df")
    D = st.session_state.get("osrm_D")
    traffic = st.session_state.get("traffic_df")

    if df_orders is None or D is None:
        st.warning("Önce siparişleri ve OSRM matrislerini oluşturun (Tab 4).")
        st.stop()

    # Use SimpleOrder for depot & customers (matches optimized util expectations)
    depot_obj = SimpleOrder(id=0, enlem=DEPOT_LAT, boylam=DEPOT_LON)
    customers_tmp = df_to_orders(df_orders)

    WEEKDAY_FOR_EXAMPLE = 2  # Çarşamba

    # ===== Build or load T_by_hour with progress bar =====
    if st.button("⏱ Trafik Matrislerini Hesapla (09–18)"):

        progress = st.progress(0.0)
        status = st.empty()

        def cb(done, total):
            progress.progress(done / total)
            status.text(f"Saat {done}/{total} işleniyor...")

        with st.spinner("Trafikli süre matrisleri hesaplanıyor..."):
            T_by_hour = build_time_matrices_with_traffic_optimized(
                D=D,
                depot=depot_obj,
                customers=customers_tmp,
                traffic=traffic,
                weekday=WEEKDAY_FOR_EXAMPLE,
                hours=range(9, 19),
                cache_path="T_by_hour_wd2.pkl",   # or None if you don't want disk cache
                use_gpu=False,                    # set True if you install CuPy
                use_multiprocessing=False,        # ❌ keep False in Streamlit on Windows
                progress_callback=cb,
            )

        st.session_state["T_by_hour"] = T_by_hour
        st.success("T_by_hour hazır (09:00–18:00).")

    T_by_hour = st.session_state.get("T_by_hour")
    if not T_by_hour:
        st.info("Henüz trafik matrisi hesaplanmadı.")
        st.stop()

    # ===== Show only a small preview instead of full big matrices =====
    def truncate_matrix(M, k=10):
        n = min(k, M.shape[0])
        return pd.DataFrame(M[:n, :n])

    col9, col10 = st.columns(2)

    with col9:
        st.subheader("🕘 09:00 Trafik Süre Matrisi (ilk 10×10)")
        st.dataframe(truncate_matrix(T_by_hour[9]), use_container_width=True)

        with st.expander("Tam matrisi göster (yavaş olabilir)"):
            st.dataframe(pd.DataFrame(T_by_hour[9]), use_container_width=True)

    with col10:
        st.subheader("🕙 10:00 Trafik Süre Matrisi (ilk 10×10)")
        st.dataframe(truncate_matrix(T_by_hour[10]), use_container_width=True)

        with st.expander("Tam matrisi göster (yavaş olabilir)"):
            st.dataframe(pd.DataFrame(T_by_hour[10]), use_container_width=True)


# =========================================================
# TAB 6 — PROBLEM ÇÖZÜMÜ
# =========================================================
with tab6:
    st.header("📦 Problem Çözümü")

    evrp_tab1, evrp_tab2, evrp_tab3, evrp_tab4 = st.tabs(
        [
            "📦 Problem Kurulumu",
            "🧠 Tabu Search",
            "🧬 Genetik Algoritma",
            "🗺 Çözümü Haritada Göster",
        ]
    )

    # ---------- TAB 1: Problem Builder ----------
    with evrp_tab1:
        st.header("🚚 EVRP Model Oluşturma")

        df_orders = st.session_state.get("orders_df")
        D = st.session_state.get("osrm_D")
        T_osrm = st.session_state.get("osrm_T")
        T_by_hour_all = st.session_state.get("T_by_hour")

        # ---- SAFETY CHECK ----
        if df_orders is None or D is None:
            st.warning("Önce siparişleri ve OSRM matrislerini oluşturun (Tab 4).")
            st.info("➡ OSRM matrisi olmadan EVRP oluşturulamaz.")
            st.stop()

        # =============== USER INPUTS FOR EVRP ======================
        num_vehicles = st.number_input("Araç Sayısı", min_value=1, value=1)

        day_map = {
            "Pazartesi": 0,
            "Salı": 1,
            "Çarşamba": 2,
            "Perşembe": 3,
            "Cuma": 4,
            "Cumartesi": 5,
            "Pazar": 6,
        }

        selected_day = st.selectbox("Gün Seç (Trafiğe Göre)", list(day_map.keys()))
        weekday = day_map[selected_day]

        # ======================= TRAFFIC MATRIX BUTTON =======================
        if st.button("📊 Bu Gün İçin Trafik Matrisi Oluştur"):

            traffic = st.session_state["traffic_df"]

            depot_obj = SimpleOrder(id=0, enlem=DEPOT_LAT, boylam=DEPOT_LON)
            customers_tmp = df_to_orders(df_orders)

            with st.spinner("⚡ Trafik matrisleri hızlı modda hesaplanıyor..."):
                T_by_hour = build_time_matrices_with_traffic_optimized(
                    D=D,
                    depot=depot_obj,
                    customers=customers_tmp,
                    traffic=traffic,
                    weekday=weekday,
                    cache_path="traffic_matrix_cache.pkl",
                    use_gpu=False,
                    use_multiprocessing=False,   # ❌ no multiprocessing in Streamlit
                    progress_callback=None,
                )

            st.session_state["T_by_hour"] = T_by_hour
            st.session_state["selected_weekday"] = weekday
            st.success(f"{selected_day} için trafik matrisleri hazır.")

    # ================= EVRP MODEL OLUŞTUR ======================
    if st.button("🚀 EVRP Modelini Derle"):
        
        T_by_hour = st.session_state.get("T_by_hour")

        if T_by_hour is not None:
            planning_hour = 9  # always start at 09:00
            problem, data = build_problem_and_data_from_globals(
                df_orders=df_orders,
                D=D,
                T=None,  # use T_by_hour
                num_vehicles=int(num_vehicles),
                T_by_hour=T_by_hour,
                planning_hour=planning_hour,
            )
        else:
            problem, data = build_problem_and_data_from_globals(
                df_orders=df_orders,
                D=D,
                T=T_osrm,
                num_vehicles=int(num_vehicles),
            )

        # store
        st.session_state["evrp_problem"] = problem
        st.session_state["ortools_data"] = data
        st.session_state["tabu_result"] = None
        st.session_state["ortools_routes"] = None
        st.session_state["ga_best_routes"] = None
        st.session_state["ga_best_fitness"] = None

        st.success("EVRP modeli başarıyla oluşturuldu.")
        st.subheader("🧪 Detaylı Feasibility Analizi")

        ok, full_report, sections = evrp_feasibility_detailed(
            data,
            work_start_min=9*60,
            work_end_min=18*60
        )

        # ---- CAPACITY ----
        if "❌" in "".join(sections["capacity"]):
            st.error("📦 Kapasite Problemi Var")
        else:
            st.success("📦 Kapasite Uygun")
        st.code("\n".join(sections["capacity"]))

        # ---- TIME ----
        if "❌" in "".join(sections["time"]):
            st.error("⏱ Süre Problemi Var")
        else:
            st.success("⏱ Süre Uygun")
        st.code("\n".join(sections["time"]))

        # ---- BATTERY ----
        if "❌" in "".join(sections["battery"]):
            st.error("🔋 Batarya Problemi Var")
        else:
            st.success("🔋 Batarya Uygun")
        st.code("\n".join(sections["battery"]))

        # ---- SUMMARY ----
        if ok:
            st.success("🎉 Model FEASIBLE – tüm kısıtlar sağlanıyor!")
        else:
            st.error("⚠️ Model INFEASIBLE – yukarıdaki kırmızı bölümlere bakın.")

        st.code("\n".join(sections["summary"]))

        # Debug values
        st.write("Kapasite (desi):", CAPACITY_DESI)
        st.write("Batarya (kWh):", BATTERY_CAPACITY)
        st.write("Enerji (kWh/100km):", BASE_KWH_PER_100KM)
        st.write("Enerji (kWh/desi-km):", ENERGY_B)

        # === OR-Tools Debug Diagnostics (INSIDE the button block!) ===
        # === OR-Tools Debug Diagnostics (INSIDE the button block!) ===
        with st.expander("🔍 OR-Tools Debug Diagnostics"):
            import numpy as np

            st.write("### OR-Tools Data Summary")

            num_vehicles = data.get("num_vehicles")
            vehicle_cap = data.get("vehicle_cap_desi")
            battery_cap = float(data.get("battery_capacity", 100.0))
            D = data.get("distance_km")
            T = data.get("time_min")
            demand = data.get("demand_desi")

            st.write("**num_vehicles:**", num_vehicles)
            st.write("**vehicle_cap_desi:**", vehicle_cap)
            st.write("**battery_capacity:**", battery_cap)
            st.write("**distance_km shape:**", None if D is None else D.shape)
            st.write("**time_min shape:**", None if T is None else T.shape)

            if D is not None:
                st.write("**Max distance (km):**", float(np.max(D)))
                st.write("**Min distance (km):**", float(np.min(D[D > 0])))

            if T is not None:
                st.write("**Max time (min):**", float(np.max(T)))
                st.write("**Min time (min):**", float(np.min(T[T > 0])))

            if demand is not None:
                st.write("**Total demand (desi):**", float(np.sum(demand)))
                st.write("**Max single customer desi:**", float(np.max(demand)))
                st.write("**Num nodes:**", len(demand))

            # ===== ENERGY DIAGNOSTIC (MATCHING OR-TOOLS) =====
            if D is not None:
                BASE = 0.436

                # round-trip energy depot -> i -> depot for each node
                depot = data.get("depot", 0)
                n = D.shape[0]

                round_trip_energy = np.zeros(n)
                for i in range(n):
                    if i == depot:
                        continue
                    d_out = D[depot, i]
                    d_back = D[i, depot]
                    round_trip_energy[i] = (d_out + d_back) * BASE

                worst_idx = int(np.argmax(round_trip_energy))
                worst_energy = float(round_trip_energy[worst_idx])

                st.write("**Worst round-trip energy (depot → i → depot):**", worst_energy)
                st.write(f"**Worst customer index:** {worst_idx}")
                st.write(
                    f"   depot→{worst_idx}: {D[depot, worst_idx]:.2f} km, "
                    f"{worst_idx}→depot: {D[worst_idx, depot]:.2f} km"
                )

                if worst_energy > battery_cap:
                    st.error(
                        "❌ At least one customer requires more energy for a round trip "
                        "than the battery capacity → no OR-Tools solution possible.\n"
                        f"   (Node {worst_idx}, round-trip energy {worst_energy:.1f} kWh)"
                    )


    # ---------- TAB 2: OR-Tools Tabu Search ----------
    with evrp_tab2:
        st.subheader("🧠 OR-Tools Çözücü")

        data = st.session_state.get("ortools_data")

        if data is None:
            st.warning("Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        else:
            col_solver1, col_solver2, col_solver3 = st.columns(3)
            
            with col_solver1:
                time_limit = st.number_input("Zaman limiti (saniye)", min_value=1, value=10)
            with col_solver2:
                seed = st.number_input("Random Seed", min_value=0, value=42)
            with col_solver3:
                solver_mode = st.selectbox(
                    "Çözücü Modu",
                    ["Tek Tur (Tabu)", "Çoklu Tur (Multi-Trip)"],
                    help="Çoklu Tur: Araçlar yeterli enerji ve zaman varsa depoya dönüp yeni tur yapabilir"
                )
            
            # Multi-trip option
            allow_multitrip = (solver_mode == "Çoklu Tur (Multi-Trip)")
            
            # ========================================
            # MINIMUM VEHICLES CALCULATOR
            # ========================================
            st.markdown("---")
            st.subheader("🔢 Minimum Araç Hesaplayıcı")
            
            if st.button("📊 Minimum Araç Sayısını Hesapla", key="calc_min_vehicles"):
                from utils.min_vehicles_calculator import calculate_min_vehicles_multitrip, calculate_min_vehicles_single_trip
                
                D = np.array(data["distance_km"], dtype=float)
                T = np.array(data["time_min"], dtype=float)
                demands = np.array(data["demand_desi"], dtype=float)
                service_times = np.array(data["service_min"], dtype=float)
                
                with st.spinner("Hesaplanıyor..."):
                    if allow_multitrip:
                        result = calculate_min_vehicles_multitrip(
                            D=D,
                            T=T,
                            demands=demands,
                            depot=data["depot"],
                            vehicle_capacity=data["vehicle_cap_desi"],
                            battery_capacity=data["battery_capacity"],
                            work_start_min=9*60,
                            work_end_min=18*60,
                            service_times=service_times,
                        )
                    else:
                        result = calculate_min_vehicles_single_trip(
                            demands=demands,
                            vehicle_capacity=data["vehicle_cap_desi"],
                        )
                    
                    st.session_state["min_vehicles_result"] = result
            
            # Display result if available
            if "min_vehicles_result" in st.session_state:
                result = st.session_state["min_vehicles_result"]
                
                # Show recommended minimum prominently
                col_rec1, col_rec2, col_rec3 = st.columns([1, 2, 1])
                with col_rec2:
                    st.metric(
                        "✅ Önerilen Minimum Araç Sayısı",
                        f"{result['recommended_min']} araç",
                        help="Bu sayı teorik minimum. Gerçek rotalar için biraz daha fazla olabilir."
                    )
                
                # Show detailed explanation
                with st.expander("📋 Detaylı Hesaplama"):
                    st.text(result['explanation'])
                
                # Quick set button
                if st.button(f"⚡ Problem Kurulumuna {result['recommended_min']} Araç Olarak Ayarla", key="set_min_vehicles"):
                    st.info(f"Problem Kurulumu sekmesinde 'Araç Sayısı' değerini {result['recommended_min']} olarak ayarlayın.")
            
            st.markdown("---")
            
            if allow_multitrip:
                st.info("🔄 Çoklu Tur Modu: Araçlar depoya dönüp batarya doldurduktan sonra yeni rota yapabilir.")

            if st.button("🚀 Çöz"):
                import time
                start_time = time.time()
                
                if allow_multitrip:
                    # Use multi-trip solver
                    from utils.multitrip_solver import solve_multitrip_ortools
                    
                    with st.spinner("Multi-Trip Solver çalışıyor..."):
                        result = solve_multitrip_ortools(
                            data,
                            time_limit_s=int(time_limit),
                            seed=int(seed),
                            allow_multi_trip=True,
                        )
                else:
                    # Use standard tabu solver
                    with st.spinner("OR-Tools Tabu Search çalışıyor..."):
                        result = solve_with_ortools_tabu(
                            data,
                            time_limit_s=int(time_limit),
                            seed=int(seed),
                        )
                
                ortools_time = time.time() - start_time

                st.session_state["tabu_result"] = result
                st.session_state["solver_mode"] = solver_mode

                # extract routes (Option A: node indices)
                if result.get("solution") is not None:
                    routes = extract_routes_from_solution(
                        data,
                        result["routing"],
                        result["manager"],
                        result["solution"],
                    )
                    st.session_state["ortools_routes"] = routes
                    
                    # For multi-trip: extract trip details
                    if allow_multitrip:
                        from utils.multitrip_route_extractor import extract_multitrip_routes, get_trip_statistics
                        
                        trips = extract_multitrip_routes(
                            data,
                            result["routing"],
                            result["manager"],
                            result["solution"],
                        )
                        trip_stats = get_trip_statistics(trips, data)
                        
                        st.session_state["multitrip_routes"] = trips
                        st.session_state["multitrip_stats"] = trip_stats
                        
                        # Count total trips
                        n_vehicles = data["num_vehicles"]
                        total_trips = sum(len(vehicle_trips) for vehicle_trips in trips)
                        total_customers = sum(len(route) for route in routes)
                        
                        st.success(f"✅ Çözüm bulundu! {n_vehicles} araç ile {total_trips} tur yapıldı, {total_customers} müşteri servis edildi. (⏱️ {ortools_time:.1f} saniye)")
                        
                        # Display trip summary
                        st.markdown("### 🔄 Tur Özeti")
                        for v, vehicle_trips in enumerate(trips):
                            if len(vehicle_trips) > 1:
                                st.write(f"**Araç {v+1}:** {len(vehicle_trips)} tur")
                                for i, trip in enumerate(vehicle_trips, 1):
                                    stats = trip_stats[v][i-1]
                                    st.write(f"  • Tur {i}: {stats['num_customers']} müşteri, "
                                            f"{stats['distance_km']:.1f} km, "
                                            f"{stats['energy_kwh']:.1f} kWh, "
                                            f"{stats['load_desi']:.0f} desi")
                            elif len(vehicle_trips) == 1:
                                stats = trip_stats[v][0]
                                st.write(f"**Araç {v+1}:** 1 tur - {stats['num_customers']} müşteri, "
                                        f"{stats['distance_km']:.1f} km")
                    else:
                        st.success(f"✅ Çözüm bulundu! (⏱️ {ortools_time:.1f} saniye)")
                        
                    st.text("✅ Rotalar cache'lendi (GA için hazır).")
                else:
                    st.session_state["ortools_routes"] = None
                    st.error("❌ Çözüm bulunamadı.")

                st.text_area(
                    "Çözüm Detayları",
                    value=result.get("log", ""),
                    height=400,
                )


    # ---------- TAB 3: Genetic Algorithm ----------
    with evrp_tab3:
        st.subheader("🧬 Genetik Algoritma Çözücü")

        from utils.ga_optimizer import (
            ga_optimize_sequences,
            print_ga_detailed_solution,
            total_plan_cost,
        )

        data = st.session_state.get("ortools_data")
        ortools_routes = st.session_state.get("ortools_routes")
        df_orders = st.session_state.get("orders_df")

        if data is None or df_orders is None:
            st.warning("Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        else:
            st.info("💡 GA, Tabu Search'ten **tamamen bağımsız** çalışır. Aynı siparişleri kullanır ama sıfırdan optimize eder.")
            st.info("🔧 GA, OR-Tools ile **aynı kısıtlar ve amaç fonksiyonuyla** çalışır: "
                    "mesafe-based enerji (0.436 kWh/km), kapasite, batarya ve çalışma saati kısıtları.")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pop_size = st.number_input(
                    "Popülasyon boyutu",
                    min_value=20,
                    max_value=500,
                    value=150,
                    step=10,
                    help="Daha büyük popülasyon = daha iyi çözüm ama daha yavaş",
                    key="evrp_tab3_pop_size"
                )
            with col2:
                generations = st.number_input(
                    "Generasyon sayısı",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Daha fazla generasyon = daha iyi optimizasyon",
                    key="evrp_tab3_generations"
                )
            with col3:
                mutation_rate = st.slider(
                    "Mutasyon oranı",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.15,
                    step=0.05,
                    help="Yüksek oran = daha fazla keşif",
                    key="evrp_tab3_mutation_rate"
                )
            with col4:
                ga_seed = st.number_input(
                    "Random seed", 
                    min_value=0, 
                    value=123,
                    step=1,
                    help="Farklı seed = farklı sonuçlar",
                    key="evrp_tab3_ga_seed"
                )

            col_obj, col_imp = st.columns(2)
            with col_obj:
                objective = st.selectbox(
                    "Amaç fonksiyonu", 
                    ["energy", "distance"], 
                    index=0,
                    help="Energy: Mesafe-based enerji modeli (OR-Tools ile aynı)",
                    key="evrp_tab3_objective"
                )
            with col_imp:
                improvement_mode = st.selectbox(
                    "İyileştirme modu",
                    ["none", "selective", "full"],
                    format_func=lambda x: {
                        "none": "Hızlı (Sadece GA)",
                        "selective": "Dengeli (Seçici 2-opt)",
                        "full": "Maksimum Kalite (Full 2-opt)"
                    }[x],
                    index=0,
                    help="2-opt lokal arama ile çözümü iyileştir (daha yavaş ama daha iyi)",
                    key="evrp_tab3_improvement_mode"
                )

            if st.button("🧬 GA Çalıştır", key="evrp_tab3_run_ga"):
                # GA always starts from scratch - completely independent from Tabu
                st.markdown("### 📊 GA Başlangıç: Tüm Müşteriler (Bağımsız Çözüm)")
                num_customers = len(df_orders)
                all_customers = list(range(1, num_customers + 1))
                base_routes = [all_customers]
                
                # Calculate detailed energy metrics
                D_matrix = np.array(data["distance_km"], dtype=float)
                demands = np.array(data["demand_desi"], dtype=float)
                depot = data["depot"]
                
                st.write("**Başlangıç Rotaları:**")
                total_energy_distance_only = 0.0
                
                for v, route in enumerate(base_routes):
                    if route:
                        energy_dist = 0.0
                        prev = depot
                        for node in route:
                            energy_dist += D_matrix[prev, node] * 0.436
                            prev = node
                        energy_dist += D_matrix[prev, depot] * 0.436
                        
                        total_energy_distance_only += energy_dist
                        
                        st.write(f"Rota {v+1}: {len(route)} müşteri")
                        st.write(f"  → Enerji (mesafe-based): {energy_dist:.3f} kWh")
                
                st.markdown("---")
                st.write(f"**Toplam Enerji (mesafe-based):** {total_energy_distance_only:.3f} kWh")
                
                original_cost = total_plan_cost(data, base_routes, objective)
                st.write(f"**Başlangıç Maliyeti ({objective}):** {original_cost:.4f}")
                
                st.markdown("---")
                
                with st.spinner(f"Genetik Algoritma çalışıyor ({generations} generasyon)..."):
                    import time
                    start_time = time.time()
                    
                    best_routes, best_fit = ga_optimize_sequences(
                        data=data,
                        base_routes=base_routes,
                        pop_size=int(pop_size),
                        generations=int(generations),
                        objective=objective,
                        elitism=2,
                        seed=int(ga_seed),
                        improvement_mode=improvement_mode
                    )
                    
                    ga_time = time.time() - start_time

                st.session_state["ga_best_routes"] = best_routes
                st.session_state["ga_best_fitness"] = best_fit
                st.session_state["ga_original_cost"] = original_cost

                improvement = (
                    (original_cost - best_fit) / original_cost * 100
                    if original_cost > 0
                    else 0.0
                )
                
                # Check if routes actually changed
                routes_changed = False
                for v in range(min(len(base_routes), len(best_routes))):
                    if v < len(base_routes) and v < len(best_routes):
                        if base_routes[v] != best_routes[v]:
                            routes_changed = True
                            break

                st.markdown("---")
                st.markdown("### ✅ GA Sonuçları")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric(
                        "Başlangıç Maliyeti",
                        f"{original_cost:.4f}",
                        help=f"{objective.upper()} değeri"
                    )
                with col_r2:
                    st.metric(
                        "GA Sonrası",
                        f"{best_fit:.4f}",
                        delta=f"{improvement:.2f}%",
                        delta_color="normal" if improvement > 0 else "off"
                    )
                with col_r3:
                    if routes_changed:
                        st.success("🔄 Rotalar değişti!")
                    else:
                        st.warning("⚠️ Rotalar değişmedi")
                
                if improvement > 0.002:
                    st.success(f"🎉 GA ile **{improvement:.2f}%** iyileşme sağlandı! (⏱️ {ga_time:.1f} saniye)")
                elif improvement > 0:
                    st.info(f"✅ GA ile **{improvement:.2f}%** küçük iyileşme sağlandı. (⏱️ {ga_time:.1f} saniye)")
                else:
                    st.warning(
                        "⚠️ GA iyileştirme bulamadı. Şunları deneyin:\n"
                        "- Popülasyon boyutunu artırın (200+)\n"
                        "- Generasyon sayısını artırın (1000+)\n"
                        "- Farklı random seed deneyin\n"
                        "- Mutasyon oranını artırın"
                    )
                
                # Show which routes changed
                if routes_changed:
                    st.markdown("### 🔄 Değişen Rotalar")
                    for v in range(len(base_routes)):
                        if base_routes[v] != best_routes[v]:
                            st.write(f"**Araç {v+1}:**")
                            st.write(f"  Önce: {base_routes[v][:10]}{'...' if len(base_routes[v]) > 10 else ''}")
                            st.write(f"  Sonra: {best_routes[v][:10]}{'...' if len(best_routes[v]) > 10 else ''}")

                txt_ga = print_ga_detailed_solution(
                    data=data,
                    routes=best_routes,
                    df_orders=df_orders,
                )

                st.text_area("GA Detaylı Çıktı", txt_ga, height=600)


    # ---------- TAB 4: Solution Maps ----------
    with evrp_tab4:
        st.subheader("🗺 Çözümü Haritada Göster")

        tabu_result = st.session_state.get("tabu_result")
        ga_routes = st.session_state.get("ga_best_routes")
        data = st.session_state.get("ortools_data")
        df_orders = st.session_state.get("orders_df")
        osrm_client = st.session_state.get("osrm_client")

        # Check what solutions are available
        has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
        has_ga = ga_routes is not None
        
        if data is None or df_orders is None:
            st.warning("Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        elif not has_tabu and not has_ga:
            st.info("Önce Tabu Search veya GA çözümünü oluşturun.")
        else:
            # Display based on what's available
            if has_tabu and has_ga:
                # Both solutions available - show comparison
                st.markdown("### 🔄 Tabu vs GA Karşılaştırması")
                st.info("Her iki çözüm de mevcut. Haritalar ve istatistikler yan yana gösteriliyor.")
                
                # Side-by-side maps
                map_col1, map_col2 = st.columns(2)
                
                with map_col1:
                    st.markdown("#### 🧠 Tabu Search Çözümü")
                    with st.spinner("Tabu haritası oluşturuluyor..."):
                        routing = tabu_result["routing"]
                        manager = tabu_result["manager"]
                        solution = tabu_result["solution"]
                        time_dim = tabu_result["time_dim"]
                        energy_dim = tabu_result["energy_dim"]
                        
                        m_tabu = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=routing,
                            manager=manager,
                            solution=solution,
                            time_dim=time_dim,
                            energy_dim=energy_dim,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                        )
                        st_folium(m_tabu, width=550, height=500, key="comparison_map_tabu")
                
                with map_col2:
                    st.markdown("#### 🧬 GA Çözümü")
                    with st.spinner("GA haritası oluşturuluyor..."):
                        m_ga = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=None,
                            manager=None,
                            solution={"routes": ga_routes},
                            time_dim=None,
                            energy_dim=None,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                        )
                        st_folium(m_ga, width=550, height=500, key="comparison_map_ga")
                
                # Comparison statistics
                st.markdown("---")
                st.markdown("### 📊 Detaylı Karşılaştırma")
                
                # Extract tabu routes
                n_vehicles = data["num_vehicles"]
                tabu_routes = []
                for v in range(n_vehicles):
                    idx = routing.Start(v)
                    route = []
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        if node != data["depot"]:
                            route.append(node)
                        idx = solution.Value(routing.NextVar(idx))
                    tabu_routes.append(route)
                
                # Calculate statistics
                D = np.array(data["distance_km"], dtype=float)
                T = np.array(data["time_min"], dtype=float)
                loads = np.array(data["demand_desi"], dtype=float)
                depot = data["depot"]
                battery_capacity = float(data.get("battery_capacity", 100.0))
                vehicle_capacity = float(data.get("vehicle_cap_desi", 15000.0))
                
                comparison_data = []
                
                for v in range(n_vehicles):
                    # Tabu statistics
                    tabu_route = tabu_routes[v]
                    tabu_km = 0.0
                    tabu_time = 0.0
                    tabu_load = 0.0
                    tabu_energy = 0.0
                    
                    if tabu_route:
                        prev_node = depot
                        cum_load = 0.0
                        
                        for node in tabu_route:
                            if node >= len(loads):
                                continue
                            d_km = float(D[prev_node, node])
                            t_min = float(T[prev_node, node])
                            tabu_km += d_km
                            tabu_time += t_min
                            tabu_energy += 0.436 * d_km + 0.002 * cum_load
                            
                            node_load = float(loads[node])
                            cum_load += node_load
                            tabu_load += node_load
                            prev_node = node
                        
                        d_km = float(D[prev_node, depot])
                        t_min = float(T[prev_node, depot])
                        tabu_km += d_km
                        tabu_time += t_min
                        tabu_energy += 0.436 * d_km + 0.002 * cum_load
                    
                    # GA statistics
                    ga_route = ga_routes[v] if v < len(ga_routes) else []
                    ga_km = 0.0
                    ga_time = 0.0
                    ga_load = 0.0
                    ga_energy = 0.0
                    
                    if ga_route:
                        prev_node = depot
                        cum_load = 0.0
                        
                        for node in ga_route:
                            if node >= len(loads):
                                continue
                            d_km = float(D[prev_node, node])
                            t_min = float(T[prev_node, node])
                            ga_km += d_km
                            ga_time += t_min
                            ga_energy += 0.436 * d_km + 0.002 * cum_load
                            
                            node_load = float(loads[node])
                            cum_load += node_load
                            ga_load += node_load
                            prev_node = node
                        
                        d_km = float(D[prev_node, depot])
                        t_min = float(T[prev_node, depot])
                        ga_km += d_km
                        ga_time += t_min
                        ga_energy += 0.436 * d_km + 0.002 * cum_load
                    
                    # Calculate improvements
                    km_improvement = ((tabu_km - ga_km) / tabu_km * 100) if tabu_km > 0 else 0
                    energy_improvement = ((tabu_energy - ga_energy) / tabu_energy * 100) if tabu_energy > 0 else 0
                    
                    comparison_data.append({
                        "Araç": f"Araç {v + 1}",
                        "Tabu KM": f"{tabu_km:.2f}",
                        "GA KM": f"{ga_km:.2f}",
                        "KM İyileşme %": f"{km_improvement:.1f}%",
                        "Tabu Enerji (kWh)": f"{tabu_energy:.3f}",
                        "GA Enerji (kWh)": f"{ga_energy:.3f}",
                        "Enerji İyileşme %": f"{energy_improvement:.1f}%",
                        "Tabu Süre (dk)": f"{tabu_time:.1f}",
                        "GA Süre (dk)": f"{ga_time:.1f}",
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                def color_improvement(val):
                    try:
                        num = float(val.replace('%', ''))
                        if num > 0:
                            return 'background-color: #d4edda'
                        elif num < 0:
                            return 'background-color: #f8d7da'
                        else:
                            return ''
                    except:
                        return ''
                
                styled_df = comparison_df.style.applymap(
                    color_improvement, 
                    subset=['KM İyileşme %', 'Enerji İyileşme %']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary metrics
                st.markdown("### 📈 Toplam Karşılaştırma")
                col1, col2, col3, col4 = st.columns(4)
                
                total_tabu_km = sum(float(row["Tabu KM"]) for row in comparison_data)
                total_ga_km = sum(float(row["GA KM"]) for row in comparison_data)
                total_tabu_energy = sum(float(row["Tabu Enerji (kWh)"]) for row in comparison_data)
                total_ga_energy = sum(float(row["GA Enerji (kWh)"]) for row in comparison_data)
                
                km_improvement = ((total_tabu_km - total_ga_km) / total_tabu_km * 100) if total_tabu_km > 0 else 0
                energy_improvement = ((total_tabu_energy - total_ga_energy) / total_tabu_energy * 100) if total_tabu_energy > 0 else 0
                
                with col1:
                    st.metric(
                        "Toplam Mesafe",
                        f"{total_ga_km:.2f} km",
                        f"{km_improvement:.1f}%"
                    )
                    st.caption(f"Tabu: {total_tabu_km:.2f} km")
                
                with col2:
                    st.metric(
                        "Toplam Enerji",
                        f"{total_ga_energy:.2f} kWh",
                        f"{energy_improvement:.1f}%"
                    )
                    st.caption(f"Tabu: {total_tabu_energy:.2f} kWh")
                
            elif has_tabu:
                # Only Tabu available
                st.markdown("### 🧠 Tabu Search Çözümü")
                st.info("Sadece Tabu Search çözümü mevcut. GA çözümü için '🧬 Genetik Algoritma' sekmesine gidin.")
                
                routing = tabu_result["routing"]
                manager = tabu_result["manager"]
                solution = tabu_result["solution"]
                time_dim = tabu_result["time_dim"]
                energy_dim = tabu_result["energy_dim"]
                
                # Extract all vehicle routes
                n_vehicles = data["num_vehicles"]
                all_routes = []
                for v in range(n_vehicles):
                    idx = routing.Start(v)
                    route = []
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        if node != data["depot"]:
                            route.append(node)
                        idx = solution.Value(routing.NextVar(idx))
                    all_routes.append(route)
                
                # Create two columns: checkboxes on left, map on right
                col_check, col_map = st.columns([1, 5])
                
                with col_check:
                    st.markdown("### 🚚 Araç Seçimi")
                    
                    if "select_all_state" not in st.session_state:
                        st.session_state.select_all_state = True
                    
                    if "vehicle_states" not in st.session_state:
                        st.session_state.vehicle_states = {v: True for v in range(n_vehicles)}
                    
                    select_all = st.checkbox(
                        "🔘 Tümünü Seç / Temizle",
                        value=st.session_state.select_all_state,
                        key="select_all_vehicles"
                    )
                    
                    if select_all != st.session_state.select_all_state:
                        st.session_state.select_all_state = select_all
                        for v in range(n_vehicles):
                            st.session_state.vehicle_states[v] = select_all
                        st.rerun()
                    
                    st.markdown("---")
                    
                    selected_vehicles = []
                    for v in range(n_vehicles):
                        num_stops = len(all_routes[v])
                        current_state = st.session_state.vehicle_states.get(v, True)
                        
                        is_selected = st.checkbox(
                            f"Araç {v+1} ({num_stops} müşteri)", 
                            value=current_state, 
                            key=f"vehicle_check_{v}"
                        )
                        
                        if is_selected != st.session_state.vehicle_states[v]:
                            st.session_state.vehicle_states[v] = is_selected
                        
                        if is_selected:
                            selected_vehicles.append(v)
                
                with col_map:
                    if not selected_vehicles:
                        st.warning("En az bir araç seçin.")
                    else:
                        filtered_data = data.copy()
                        filtered_data["num_vehicles"] = len(selected_vehicles)
                        
                        class FilteredSolution:
                            def __init__(self, original_routing, original_manager, original_solution, selected_v, all_r):
                                self.routing = original_routing
                                self.manager = original_manager
                                self.solution = original_solution
                                self.selected_vehicles = selected_v
                                self.all_routes = all_r
                            
                            def Start(self, v):
                                original_v = self.selected_vehicles[v]
                                return self.routing.Start(original_v)
                            
                            def IsEnd(self, idx):
                                return self.routing.IsEnd(idx)
                            
                            def NextVar(self, idx):
                                return self.routing.NextVar(idx)
                            
                            def Value(self, var):
                                return self.solution.Value(var)
                            
                            def get_original_vehicle_id(self, v):
                                return self.selected_vehicles[v]
                        
                        filtered_routing = FilteredSolution(routing, manager, solution, selected_vehicles, all_routes)
                        
                        with st.spinner("Harita oluşturuluyor..."):
                            m = visualize_routes_osrm(
                                depot_lat=DEPOT_LAT,
                                depot_lon=DEPOT_LON,
                                df_orders=df_orders,
                                data=filtered_data,
                                routing=filtered_routing,
                                manager=manager,
                                solution=solution,
                                time_dim=time_dim,
                                energy_dim=energy_dim,
                                osrm_client=osrm_client,
                                weekday=st.session_state.get("selected_weekday"),
                            )

                        st_folium(m, width=1200, height=800)
                        
                        # Statistics table
                        st.markdown("---")
                        st.subheader("📊 Araç İstatistikleri")
                        
                        vehicle_stats = []
                        D = np.array(data["distance_km"], dtype=float)
                        T = np.array(data["time_min"], dtype=float)
                        loads = np.array(data["demand_desi"], dtype=float)
                        depot = data["depot"]
                        battery_capacity = float(data.get("battery_capacity", 100.0))
                        vehicle_capacity = float(data.get("vehicle_cap_desi", 15000.0))
                        
                        for v_idx, original_v in enumerate(selected_vehicles):
                            route = all_routes[original_v]
                            
                            if not route:
                                continue
                            
                            total_km = 0.0
                            total_time = 0.0
                            total_load = 0.0
                            total_energy = 0.0
                            
                            prev_node = depot
                            cum_load = 0.0
                            
                            for node in route:
                                d_km = float(D[prev_node, node])
                                t_min = float(T[prev_node, node])
                                total_km += d_km
                                total_time += t_min
                                
                                energy_kwh = 0.436 * d_km + 0.002 * cum_load
                                total_energy += energy_kwh
                                
                                cum_load += loads[node]
                                total_load += loads[node]
                                
                                if node > 0 and (node - 1) < len(df_orders):
                                    service_time = float(df_orders.iloc[node - 1]["Servis Süresi (dk)"])
                                    total_time += service_time
                                
                                prev_node = node
                            
                            d_km = float(D[prev_node, depot])
                            t_min = float(T[prev_node, depot])
                            total_km += d_km
                            total_time += t_min
                            energy_kwh = 0.436 * d_km + 0.002 * cum_load
                            total_energy += energy_kwh
                            
                            energy_pct = (total_energy / battery_capacity) * 100.0
                            remaining_capacity = vehicle_capacity - total_load
                            
                            vehicle_stats.append({
                                "Araç": f"Araç {original_v + 1}",
                                "Toplam KM": f"{total_km:.2f}",
                                "Toplam Süre (dk)": f"{total_time:.1f}",
                                "Taşınan Yük (desi)": f"{total_load:.0f}",
                                "Boş Kapasite (desi)": f"{remaining_capacity:.0f}",
                                "Enerji (kWh)": f"{total_energy:.3f}",
                                "Enerji (%)": f"{energy_pct:.1f}",
                            })
                        
                        if vehicle_stats:
                            stats_df = pd.DataFrame(vehicle_stats)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            st.markdown("### 📈 Toplam Özet")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            total_km_all = sum(float(s["Toplam KM"]) for s in vehicle_stats)
                            total_time_all = sum(float(s["Toplam Süre (dk)"]) for s in vehicle_stats)
                            total_load_all = sum(float(s["Taşınan Yük (desi)"]) for s in vehicle_stats)
                            total_energy_kwh = sum(float(s["Enerji (kWh)"]) for s in vehicle_stats)
                            
                            with col1:
                                st.metric("Toplam Mesafe", f"{total_km_all:.2f} km")
                            with col2:
                                st.metric("Toplam Süre", f"{total_time_all:.1f} dk")
                            with col3:
                                st.metric("Toplam Yük", f"{total_load_all:.0f} desi")
                            with col4:
                                st.metric("Toplam Enerji", f"{total_energy_kwh:.2f} kWh")
                        else:
                            st.info("Seçili araçlar için istatistik hesaplanamadı.")
            
            elif has_ga:
                # Only GA available
                st.markdown("### 🧬 GA Çözümü")
                st.info("Sadece GA çözümü mevcut.")
                
                with st.spinner("Harita oluşturuluyor..."):
                    m_ga = visualize_routes_osrm(
                        depot_lat=DEPOT_LAT,
                        depot_lon=DEPOT_LON,
                        df_orders=df_orders,
                        data=data,
                        routing=None,
                        manager=None,
                        solution={"routes": ga_routes},
                        time_dim=None,
                        energy_dim=None,
                        osrm_client=osrm_client,
                        weekday=st.session_state.get("selected_weekday"),
                    )
                    st_folium(m_ga, width=1200, height=800)
                
                # GA Statistics
                st.markdown("---")
                st.subheader("📊 GA Çözüm İstatistikleri")
                
                D = np.array(data["distance_km"], dtype=float)
                T = np.array(data["time_min"], dtype=float)
                loads = np.array(data["demand_desi"], dtype=float)
                depot = data["depot"]
                battery_capacity = float(data.get("battery_capacity", 100.0))
                vehicle_capacity = float(data.get("vehicle_cap_desi", 15000.0))
                
                vehicle_stats = []
                
                for v, route in enumerate(ga_routes):
                    if not route:
                        continue
                    
                    total_km = 0.0
                    total_time = 0.0
                    total_load = 0.0
                    total_energy = 0.0
                    
                    prev_node = depot
                    cum_load = 0.0
                    
                    for node in route:
                        if node >= len(loads):
                            continue
                        
                        d_km = float(D[prev_node, node])
                        t_min = float(T[prev_node, node])
                        total_km += d_km
                        total_time += t_min
                        total_energy += 0.436 * d_km + 0.002 * cum_load
                        
                        cum_load += loads[node]
                        total_load += loads[node]
                        
                        if node > 0 and (node - 1) < len(df_orders):
                            service_time = float(df_orders.iloc[node - 1]["Servis Süresi (dk)"])
                            total_time += service_time
                        
                        prev_node = node
                    
                    d_km = float(D[prev_node, depot])
                    t_min = float(T[prev_node, depot])
                    total_km += d_km
                    total_time += t_min
                    total_energy += 0.436 * d_km + 0.002 * cum_load
                    
                    energy_pct = (total_energy / battery_capacity) * 100.0
                    
                    vehicle_stats.append({
                        "Araç": f"Araç {v + 1}",
                        "Toplam KM": f"{total_km:.2f}",
                        "Toplam Süre (dk)": f"{total_time:.1f}",
                        "Taşınan Yük (desi)": f"{total_load:.0f}",
                        "Enerji (kWh)": f"{total_energy:.3f}",
                        "Enerji (%)": f"{energy_pct:.1f}",
                    })
                
                if vehicle_stats:
                    stats_df = pd.DataFrame(vehicle_stats)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    st.markdown("### 📈 Toplam Özet")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_km_all = sum(float(s["Toplam KM"]) for s in vehicle_stats)
                    total_time_all = sum(float(s["Toplam Süre (dk)"]) for s in vehicle_stats)
                    total_load_all = sum(float(s["Taşınan Yük (desi)"]) for s in vehicle_stats)
                    total_energy_kwh = sum(float(s["Enerji (kWh)"]) for s in vehicle_stats)
                    
                    with col1:
                        st.metric("Toplam Mesafe", f"{total_km_all:.2f} km")
                    with col2:
                        st.metric("Toplam Süre", f"{total_time_all:.1f} dk")
                    with col3:
                        st.metric("Toplam Yük", f"{total_load_all:.0f} desi")
                    with col4:
                        st.metric("Toplam Enerji", f"{total_energy_kwh:.2f} kWh")
                else:
                    st.info("İstatistik hesaplanamadı.")

    # =========================================================
    # ⚡ ELEKTRİKLİ ARAÇLAR İÇİN OPTİMİZE ET - ARAÇ ATAMA
    # =========================================================
    st.markdown("---")
    st.header("⚡ Elektrikli Araç Atama Sistemi")

    # Import utility functions for shift reallocation
    from utils.shift_reallocation import (
        analyze_early_finishers,
        calculate_route_metrics,
        calculate_solution_metrics,
        two_phase_reallocation,
        clarke_wright_savings,
        apply_shift_reallocation,
    )

    st.info("""
    **💡 Nasıl Çalışır:**
    1. Tabu veya GA'dan gelen her rota **bir iş (job)** olarak kabul edilir
    2. Mevcut elektrikli araçları bu işlere atarız
    3. Her araç kısıtları (kapasite, batarya, çalışma saati) kontrol eder
    4. Sonuçları karşılaştırabilirsiniz
    """)

    opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs(
        [
            "🚚 Tabu → Araç Atama",
            "🧬 GA → Araç Atama",
            "🗺 Atama Haritaları",
            "📊 Karşılaştırma",
        ]
    )

    # ---------- OPT TAB 1: TABU ROUTES → VEHICLE ASSIGNMENT ----------
    with opt_tab1:
        st.subheader("🚚 Araç Atama (Tabu Tabanlı)")
        
        data = st.session_state.get("ortools_data")
        df_orders = st.session_state.get("orders_df")
        tabu_result = st.session_state.get("tabu_result")
        ga_routes = st.session_state.get("ga_best_routes")
    
        # Check if any solution exists
        has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
        has_ga = ga_routes is not None
        
        if not has_tabu and not has_ga:
            st.warning("⚠️ Önce '6️⃣ Problem Çözümü' sekmesinden Tabu veya GA çalıştırın.")
            st.info("👉 Problem Çözümü sekmesine gidin ve en az bir çözüm oluşturun.")
        elif df_orders is None or data is None:
            st.warning("⚠️ Problem verileri eksik.")
        else:
            # Route source selection
            st.markdown("### 📍 Rota Kaynağı Seçimi")
            
            route_sources = []
            if has_tabu:
                route_sources.append("Tabu Search")
            if has_ga:
                route_sources.append("Genetic Algorithm")
            
            selected_source = st.selectbox(
                "Hangi çözümün rotalarını kullanmak istersiniz?",
                route_sources,
            help="Araç ataması için kullanılacak rota kaynağını seçin",
            key="tabu_tab_route_source"
        )
        
        # Extract routes based on selection
        if selected_source == "Tabu Search":
            routing = tabu_result["routing"]
            manager = tabu_result["manager"]
            solution = tabu_result["solution"]

            source_routes = []
            for v in range(data["num_vehicles"]):
                r = []
                idx = routing.Start(v)
                while not routing.IsEnd(idx):
                    node = manager.IndexToNode(idx)
                    if node != 0:
                        r.append(node)
                    idx = solution.Value(routing.NextVar(idx))
                source_routes.append(r)
        else:  # Genetic Algorithm
            source_routes = ga_routes
        
        # Filter out empty routes
        jobs = [route for route in source_routes if route]
        num_jobs = len(jobs)
        
        st.success(f"✅ {selected_source}'ten {num_jobs} iş (rota) yüklendi!")
        
        # Display jobs
        st.markdown(f"### 📋 İşler ({selected_source} Rotaları)")
        for i, job in enumerate(jobs):
            st.write(f"**İş {i+1}:** {len(job)} müşteri - Düğümler: {job[:10]}{'...' if len(job) > 10 else ''}")
        
        st.markdown("---")
        
        # Vehicle assignment parameters
        st.markdown("### 🚗 Araç Atama Parametreleri")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            num_available_vehicles = st.number_input(
                "Mevcut Araç Sayısı",
                min_value=1,
                max_value=50,
                value=num_jobs,
                help="Kaç elektrikli araç mevcut?",
                key="tabu_num_vehicles"
            )
        with col_v2:
            assignment_strategy = st.selectbox(
                "Atama Stratejisi",
                ["Greedy (İlk Uygun)", "Optimal (Minimum Araç)", "Balanced (Dengeli Yük)"],
                help="İşleri araçlara nasıl atayalım?",
                key="tabu_assignment_strategy"
            )
        
        if st.button("🔄 Araçları Ata (Tabu)", key="assign_tabu_vehicles"):
            with st.spinner("Tabu rotalarına araçlar atanıyor..."):
                # Simple assignment: each job to a vehicle
                # In the future, this can be more sophisticated
                
                if num_available_vehicles >= num_jobs:
                    # Enough vehicles - one vehicle per job
                    st.success(f"✅ {num_jobs} işe {num_jobs} araç atandı (1:1 eşleme)")
                    
                    vehicle_assignments = []
                    for job_idx, job in enumerate(jobs):
                        vehicle_assignments.append({
                            "vehicle_id": job_idx + 1,
                            "job_id": job_idx + 1,
                            "route": job,
                            "num_customers": len(job)
                        })
                    
                    st.session_state["tabu_vehicle_assignments"] = vehicle_assignments
                    st.session_state["tabu_assignment_type"] = "1:1"
                    
                else:
                    # Not enough vehicles - need to combine jobs
                    st.warning(f"⚠️ {num_available_vehicles} araç ile {num_jobs} işi karşılamak gerekiyor!")
                    st.info("🔜 Gelecekte: İşler birleştirilerek araçlara atanacak.")
                    
                    # For now, assign as many as we can
                    vehicle_assignments = []
                    for v_idx in range(min(num_available_vehicles, num_jobs)):
                        vehicle_assignments.append({
                            "vehicle_id": v_idx + 1,
                            "job_id": v_idx + 1,
                            "route": jobs[v_idx],
                            "num_customers": len(jobs[v_idx])
                        })
                    
                    st.session_state["tabu_vehicle_assignments"] = vehicle_assignments
                    st.session_state["tabu_assignment_type"] = "partial"
        
        # Display assignments if available
        if "tabu_vehicle_assignments" in st.session_state:
            st.markdown("---")
            st.markdown("### ✅ Araç Atamaları")
            
            assignments = st.session_state["tabu_vehicle_assignments"]
            
            for assignment in assignments:
                st.write(f"🚗 **Araç {assignment['vehicle_id']}** → İş {assignment['job_id']} "
                        f"({assignment['num_customers']} müşteri)")
            
            st.markdown("---")
            
            # Calculate statistics
            D = np.array(data["distance_km"], dtype=float)
            T = np.array(data["time_min"], dtype=float)
            loads = np.array(data["demand_desi"], dtype=float)
            depot = data["depot"]
            
            assignment_stats = []
            for assignment in assignments:
                route = assignment["route"]
                
                total_km = 0.0
                total_time = 0.0
                total_load = 0.0
                total_energy = 0.0
                
                prev_node = depot
                cum_load = 0.0
                
                for node in route:
                    if node >= len(loads):
                        continue
                    
                    d_km = float(D[prev_node, node])
                    t_min = float(T[prev_node, node])
                    total_km += d_km
                    total_time += t_min
                    total_energy += 0.436 * d_km + 0.002 * cum_load
                    
                    cum_load += loads[node]
                    total_load += loads[node]
                    prev_node = node
                
                # Return to depot
                d_km = float(D[prev_node, depot])
                t_min = float(T[prev_node, depot])
                total_km += d_km
                total_time += t_min
                total_energy += 0.436 * d_km + 0.002 * cum_load
                
                assignment_stats.append({
                    "Araç": f"Araç {assignment['vehicle_id']}",
                    "İş": f"İş {assignment['job_id']}",
                    "Müşteri": assignment['num_customers'],
                    "KM": f"{total_km:.2f}",
                    "Süre (dk)": f"{total_time:.1f}",
                    "Yük (desi)": f"{total_load:.0f}",
                    "Enerji (kWh)": f"{total_energy:.2f}"
                })
            
            stats_df = pd.DataFrame(assignment_stats)
            st.dataframe(stats_df, use_container_width=True)


    # ---------- OPT TAB 2: GA ROUTES → VEHICLE ASSIGNMENT ----------
    with opt_tab2:
        st.subheader("🧬 Araç Atama (GA Tabanlı)")

        data = st.session_state.get("ortools_data")
        ga_routes = st.session_state.get("ga_best_routes")
        df_orders = st.session_state.get("orders_df")
        tabu_result = st.session_state.get("tabu_result")
        
        # Check if any solution exists
        has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
        has_ga = ga_routes is not None
        
        if not has_tabu and not has_ga:
            st.warning("⚠️ Önce '6️⃣ Problem Çözümü' sekmesinden Tabu veya GA çalıştırın.")
            st.info("👉 Problem Çözümü sekmesine gidin ve en az bir çözüm oluşturun.")
        elif df_orders is None or data is None:
            st.warning("⚠️ Problem verileri eksik.")
        else:
                # Route source selection
            st.markdown("### 📍 Rota Kaynağı Seçimi")
            
            route_sources = []
            if has_ga:
                route_sources.append("Genetic Algorithm")
            if has_tabu:
                route_sources.append("Tabu Search")
        
        selected_source = st.selectbox(
            "Hangi çözümün rotalarını kullanmak istersiniz?",
            route_sources,
            help="Araç ataması için kullanılacak rota kaynağını seçin",
            key="ga_tab_route_source"
        )
        
        # Extract routes based on selection
        if selected_source == "Genetic Algorithm":
            source_routes = ga_routes
        else:  # Tabu Search
            routing = tabu_result["routing"]
            manager = tabu_result["manager"]
            solution = tabu_result["solution"]

            source_routes = []
            for v in range(data["num_vehicles"]):
                r = []
                idx = routing.Start(v)
                while not routing.IsEnd(idx):
                    node = manager.IndexToNode(idx)
                    if node != 0:
                        r.append(node)
                    idx = solution.Value(routing.NextVar(idx))
                source_routes.append(r)
        
        # Filter out empty routes
        jobs = [route for route in source_routes if route]
        num_jobs = len(jobs)
        
        st.success(f"✅ {selected_source}'ten {num_jobs} iş (rota) yüklendi!")
        
        # Display jobs
        st.markdown(f"### 📋 İşler ({selected_source} Rotaları)")
        for i, job in enumerate(jobs):
            st.write(f"**İş {i+1}:** {len(job)} müşteri - Düğümler: {job[:10]}{'...' if len(job) > 10 else ''}")
        
        st.markdown("---")
        
        # Vehicle assignment parameters
        st.markdown("### 🚗 Araç Atama Parametreleri")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            num_available_vehicles = st.number_input(
                "Mevcut Araç Sayısı",
                min_value=1,
                max_value=50,
                value=num_jobs,
                help="Kaç elektrikli araç mevcut?",
                key="ga_num_vehicles"
            )
        with col_v2:
            assignment_strategy = st.selectbox(
                "Atama Stratejisi",
                ["Greedy (İlk Uygun)", "Optimal (Minimum Araç)", "Balanced (Dengeli Yük)"],
                help="İşleri araçlara nasıl atayalım?",
                key="ga_assignment_strategy"
            )
        
        if st.button("🔄 Araçları Ata (GA)", key="assign_ga_vehicles"):
            with st.spinner("GA rotalarına araçlar atanıyor..."):
                # Simple assignment: each job to a vehicle
                # In the future, this can be more sophisticated
                
                if num_available_vehicles >= num_jobs:
                    # Enough vehicles - one vehicle per job
                    st.success(f"✅ {num_jobs} işe {num_jobs} araç atandı (1:1 eşleme)")
                    
                    vehicle_assignments = []
                    for job_idx, job in enumerate(jobs):
                        vehicle_assignments.append({
                            "vehicle_id": job_idx + 1,
                            "job_id": job_idx + 1,
                            "route": job,
                            "num_customers": len(job)
                        })
                    
                    st.session_state["ga_vehicle_assignments"] = vehicle_assignments
                    st.session_state["ga_assignment_type"] = "1:1"
                    
                else:
                    # Not enough vehicles - need to combine jobs
                    st.warning(f"⚠️ {num_available_vehicles} araç ile {num_jobs} işi karşılamak gerekiyor!")
                    st.info("🔜 Gelecekte: İşler birleştirilerek araçlara atanacak.")
                    
                    # For now, assign as many as we can
                    vehicle_assignments = []
                    for v_idx in range(min(num_available_vehicles, num_jobs)):
                        vehicle_assignments.append({
                            "vehicle_id": v_idx + 1,
                            "job_id": v_idx + 1,
                            "route": jobs[v_idx],
                            "num_customers": len(jobs[v_idx])
                        })
                    
                    st.session_state["ga_vehicle_assignments"] = vehicle_assignments
                    st.session_state["ga_assignment_type"] = "partial"
        
        # Display assignments if available
        if "ga_vehicle_assignments" in st.session_state:
            st.markdown("---")
            st.markdown("### ✅ Araç Atamaları")
            
            assignments = st.session_state["ga_vehicle_assignments"]
            
            for assignment in assignments:
                st.write(f"🚗 **Araç {assignment['vehicle_id']}** → İş {assignment['job_id']} "
                        f"({assignment['num_customers']} müşteri)")
            
            st.markdown("---")
            
            # Calculate statistics
            D = np.array(data["distance_km"], dtype=float)
            T = np.array(data["time_min"], dtype=float)
            loads = np.array(data["demand_desi"], dtype=float)
            depot = data["depot"]
            
            assignment_stats = []
            for assignment in assignments:
                route = assignment["route"]
                
                total_km = 0.0
                total_time = 0.0
                total_load = 0.0
                total_energy = 0.0
                
                prev_node = depot
                cum_load = 0.0
                
                for node in route:
                    if node >= len(loads):
                        continue
                    
                    d_km = float(D[prev_node, node])
                    t_min = float(T[prev_node, node])
                    total_km += d_km
                    total_time += t_min
                    total_energy += 0.436 * d_km + 0.002 * cum_load
                    
                    cum_load += loads[node]
                    total_load += loads[node]
                    prev_node = node
                
                # Return to depot
                d_km = float(D[prev_node, depot])
                t_min = float(T[prev_node, depot])
                total_km += d_km
                total_time += t_min
                total_energy += 0.436 * d_km + 0.002 * cum_load
                
                assignment_stats.append({
                    "Araç": f"Araç {assignment['vehicle_id']}",
                    "İş": f"İş {assignment['job_id']}",
                    "Müşteri": assignment['num_customers'],
                    "KM": f"{total_km:.2f}",
                    "Süre (dk)": f"{total_time:.1f}",
                    "Yük (desi)": f"{total_load:.0f}",
                    "Enerji (kWh)": f"{total_energy:.2f}"
                })
            
            stats_df = pd.DataFrame(assignment_stats)
            st.dataframe(stats_df, use_container_width=True)


    # ---------- OPT TAB 3: ASSIGNMENT MAPS ----------
    with opt_tab3:
        st.subheader("🗺 Araç Atama Haritaları")

        df_orders = st.session_state.get("orders_df")
        data = st.session_state.get("ortools_data")
        osrm_client = st.session_state.get("osrm_client")
        
        tabu_assignments = st.session_state.get("tabu_vehicle_assignments")
        ga_assignments = st.session_state.get("ga_vehicle_assignments")
        
        has_tabu = tabu_assignments is not None
        has_ga = ga_assignments is not None

        if df_orders is None or data is None:
            st.info("Önce problem verilerini oluşturun.")
        elif not has_tabu and not has_ga:
            st.info("Henüz araç ataması yapılmadı. Önceki sekmelerde araçları atayın.")
        else:
            # Display based on what's available
            if has_tabu and has_ga:
                st.markdown("### 🔄 Tabu vs GA Atamaları")
                map_col1, map_col2 = st.columns(2)
                
                with map_col1:
                    st.markdown("#### 🚚 Tabu Ataması")
                    # Extract routes from tabu assignments
                    tabu_routes = [a["route"] for a in tabu_assignments]
                    
                    with st.spinner("Tabu haritası oluşturuluyor..."):
                        m_tabu = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=None,
                            manager=None,
                            solution={"routes": tabu_routes},
                            time_dim=None,
                            energy_dim=None,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                        )
                        st_folium(m_tabu, width=550, height=500, key="assignment_map_tabu")
                
                with map_col2:
                    st.markdown("#### 🧬 GA Ataması")
                    # Extract routes from GA assignments
                    ga_routes = [a["route"] for a in ga_assignments]
                    
                    with st.spinner("GA haritası oluşturuluyor..."):
                        m_ga = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=None,
                            manager=None,
                            solution={"routes": ga_routes},
                            time_dim=None,
                            energy_dim=None,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                        )
                        st_folium(m_ga, width=550, height=500, key="assignment_map_ga")
            
            elif has_tabu:
                st.markdown("### 🚚 Tabu Araç Ataması")
                tabu_routes = [a["route"] for a in tabu_assignments]
                
                with st.spinner("Harita oluşturuluyor..."):
                    m_tabu = visualize_routes_osrm(
                        depot_lat=DEPOT_LAT,
                        depot_lon=DEPOT_LON,
                        df_orders=df_orders,
                        data=data,
                        routing=None,
                        manager=None,
                        solution={"routes": tabu_routes},
                        time_dim=None,
                    energy_dim=None,
                    osrm_client=osrm_client,
                    weekday=st.session_state.get("selected_weekday"),
                )
                st_folium(m_tabu, width=1200, height=800, key="assignment_map_tabu_only")
            
            elif has_ga:
                # Only GA available
                st.markdown("### 🧬 GA Araç Ataması")
                ga_routes = [a["route"] for a in ga_assignments]
                
                with st.spinner("Harita oluşturuluyor..."):
                    m_ga = visualize_routes_osrm(
                        depot_lat=DEPOT_LAT,
                        depot_lon=DEPOT_LON,
                        df_orders=df_orders,
                        data=data,
                        routing=None,
                        manager=None,
                        solution={"routes": ga_routes},
                        time_dim=None,
                        energy_dim=None,
                        osrm_client=osrm_client,
                        weekday=st.session_state.get("selected_weekday"),
                    )
                    st_folium(m_ga, width=1200, height=800, key="assignment_map_ga_only")

    # ---------- OPT TAB 4: COMPARISON ----------
    with opt_tab4:
        st.subheader("📊 Tabu vs GA Araç Atama Karşılaştırması")
        
        tabu_assignments = st.session_state.get("tabu_vehicle_assignments")
        ga_assignments = st.session_state.get("ga_vehicle_assignments")
        data = st.session_state.get("ortools_data")
        
        has_tabu = tabu_assignments is not None
        has_ga = ga_assignments is not None
        
        if not has_tabu and not has_ga:
            st.info("Karşılaştırma için her iki çözümde de araç ataması yapın.")
        elif not has_tabu:
            st.warning("Sadece GA ataması mevcut. Karşılaştırma için Tabu ataması da yapın.")
        elif not has_ga:
            st.warning("Sadece Tabu ataması mevcut. Karşılaştırma için GA ataması da yapın.")
        else:
            st.success("✅ Her iki atama da mevcut - karşılaştırma yapılıyor!")
            
            # Calculate metrics for both
            D = np.array(data["distance_km"], dtype=float)
            T = np.array(data["time_min"], dtype=float)
            loads = np.array(data["demand_desi"], dtype=float)
            depot = data["depot"]
            
            def calculate_assignment_metrics(assignments):
                total_km = 0.0
                total_time = 0.0
                total_load = 0.0
                total_energy = 0.0
                num_vehicles = len(assignments)
                
                for assignment in assignments:
                    route = assignment["route"]
                    prev_node = depot
                    cum_load = 0.0
                    
                    for node in route:
                        if node >= len(loads):
                            continue
                        
                        d_km = float(D[prev_node, node])
                        t_min = float(T[prev_node, node])
                        total_km += d_km
                        total_time += t_min
                        total_energy += 0.436 * d_km + 0.002 * cum_load
                        
                        cum_load += loads[node]
                        total_load += loads[node]
                        prev_node = node
                    
                    # Return to depot
                    d_km = float(D[prev_node, depot])
                    t_min = float(T[prev_node, depot])
                    total_km += d_km
                    total_time += t_min
                    total_energy += 0.436 * d_km + 0.002 * cum_load
                
                return {
                    "vehicles": num_vehicles,
                    "total_km": total_km,
                    "total_time": total_time,
                    "total_load": total_load,
                    "total_energy": total_energy
                }
            
            tabu_metrics = calculate_assignment_metrics(tabu_assignments)
            ga_metrics = calculate_assignment_metrics(ga_assignments)
            
            # Display comparison
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Araç Sayısı", 
                         f"{tabu_metrics['vehicles']}",
                         delta=None)
                st.caption(f"GA: {ga_metrics['vehicles']}")
            
            with col2:
                km_diff = ga_metrics['total_km'] - tabu_metrics['total_km']
                st.metric("Toplam Mesafe (km)",
                         f"{tabu_metrics['total_km']:.2f}",
                         delta=f"{-km_diff:.2f}")
                st.caption(f"GA: {ga_metrics['total_km']:.2f}")
            
            with col3:
                energy_diff = ga_metrics['total_energy'] - tabu_metrics['total_energy']
                st.metric("Toplam Enerji (kWh)",
                         f"{tabu_metrics['total_energy']:.2f}",
                         delta=f"{-energy_diff:.2f}")
                st.caption(f"GA: {ga_metrics['total_energy']:.2f}")
            
            with col4:
                time_diff = ga_metrics['total_time'] - tabu_metrics['total_time']
                st.metric("Toplam Süre (dk)",
                         f"{tabu_metrics['total_time']:.1f}",
                         delta=f"{-time_diff:.1f}")
                st.caption(f"GA: {ga_metrics['total_time']:.1f}")
            
            # Determine winner
            st.markdown("---")
            st.markdown("### 🏆 Kazanan")
            
            if tabu_metrics['total_energy'] < ga_metrics['total_energy']:
                improvement_pct = ((ga_metrics['total_energy'] - tabu_metrics['total_energy']) / ga_metrics['total_energy']) * 100
                st.success(f"✅ **Tabu Search** daha iyi! {improvement_pct:.1f}% daha az enerji kullanımı")
            elif ga_metrics['total_energy'] < tabu_metrics['total_energy']:
                improvement_pct = ((tabu_metrics['total_energy'] - ga_metrics['total_energy']) / tabu_metrics['total_energy']) * 100
                st.success(f"✅ **Genetic Algorithm** daha iyi! {improvement_pct:.1f}% daha az enerji kullanımı")
            else:
                st.info("🤝 Her iki çözüm de aynı enerji kullanımına sahip!")

# =========================================================
# 7️⃣ ÇOKLU GÖREV (MULTI-TRIP) OPTİMİZASYONU
# =========================================================
with tab7:
    st.header("🚛 Çoklu Görev (Multi-Trip) Optimizasyonu")
    
    st.markdown("""
    ### 💡 Konsept: Araçların Birden Fazla Görev Yapması
    
    **Strateji:**
    - 🚚 Bir araç bir görevi tamamlar ve depoya döner
    - 🔄 Eğer zamanı ve kapasitesi varsa, başka bir görevi alır
    - ⚡ Toplam araç sayısını azaltarak maliyet tasarrufu sağlar
    
    **Potansiyel Faydalar:**
    - ✅ Daha az araç kullanımı (maliyet tasarrufu)
    - ✅ Daha iyi araç kullanım oranı
    - ✅ Operasyonel verimlilik artışı
    """)
    
    data = st.session_state.get("ortools_data")
    tabu_result = st.session_state.get("tabu_result")
    ga_routes = st.session_state.get("ga_best_routes")
    df_orders = st.session_state.get("orders_df")
    
    # Check if any solution exists
    has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
    has_ga = ga_routes is not None
    
    if data is None or df_orders is None:
        st.warning("⚠️ Önce EVRP modelini oluşturun.")
    elif not has_tabu and not has_ga:
        st.warning("⚠️ Önce '6️⃣ Problem Çözümü' sekmesinden Tabu veya GA çalıştırın.")
    else:
        # Solution selector
        st.markdown("---")
        st.subheader("📋 Çözüm Seçimi")
        
        col_sel1, col_sel2 = st.columns([2, 3])
        
        with col_sel1:
            solution_options = []
            if has_tabu:
                solution_options.append("Tabu Search")
            if has_ga:
                solution_options.append("Genetic Algorithm")
            
            selected_source = st.selectbox(
                "Hangi çözümü kullanmak istersiniz?",
                solution_options,
                help="Multi-trip optimizasyonu için baz çözüm",
                key="multitrip_source_selector"
            )
        
        with col_sel2:
            if selected_source == "Tabu Search":
                st.info("🚚 **Tabu Search** çözümü seçildi")
                # Extract Tabu routes
                routing = tabu_result["routing"]
                manager = tabu_result["manager"]
                solution = tabu_result["solution"]
                
                base_routes = []
                for v in range(data["num_vehicles"]):
                    route = []
                    idx = routing.Start(v)
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        if node != 0:
                            route.append(node)
                        idx = solution.Value(routing.NextVar(idx))
                    base_routes.append(route)
            else:
                st.info("🧬 **Genetic Algorithm** çözümü seçildi")
                base_routes = ga_routes
        
        # Filter out empty routes - these are our "jobs"
        jobs = [route for route in base_routes if route]
        num_jobs = len(jobs)
        
        st.success(f"✅ {num_jobs} görev (rota) tespit edildi!")
        
        # Parameters
        st.markdown("---")
        st.subheader("🔧 Multi-Trip Parametreleri")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            max_shift_duration = st.number_input(
                "Maksimum Vardiya Süresi (dakika)",
                min_value=240,
                max_value=720,
                value=540,  # 9 hours
                step=30,
                help="Bir aracın maksimum çalışma süresi"
            )
        
        with col_p2:
            depot_service_time = st.number_input(
                "Depo Servis Süresi (dakika)",
                min_value=0,
                max_value=60,
                value=15,
                step=5,
                help="Depoda yük boşaltma/yükleme süresi"
            )
        
        with col_p3:
            min_trips_per_vehicle = st.number_input(
                "Min. Görev/Araç (Hedef)",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Zorunlu bir kısıt değildir. Algoritma, zaman ve enerji kısıtları izin verdikçe araçlara ek görevler yükler. Bu değer sadece hedef/uyarı amaçlıdır."
            )
        
        # Calculate job metrics
        st.markdown("---")
        st.subheader("📊 Görev Analizi")
        
        D = np.array(data["distance_km"], dtype=float)
        T = np.array(data["time_min"], dtype=float)
        loads = np.array(data["demand_desi"], dtype=float)
        depot = data["depot"]
        
        job_metrics = []
        for job_idx, job in enumerate(jobs):
            total_time = 0.0
            total_km = 0.0
            total_load = 0.0
            total_energy = 0.0
            
            prev_node = depot
            cum_load = 0.0
            
            for node in job:
                if node >= len(loads):
                    continue
                
                d_km = float(D[prev_node, node])
                t_min = float(T[prev_node, node])
                total_km += d_km
                total_time += t_min
                total_energy += 0.436 * d_km + 0.002 * cum_load
                
                cum_load += loads[node]
                total_load += loads[node]
                prev_node = node
            
            # Return to depot
            d_km = float(D[prev_node, depot])
            t_min = float(T[prev_node, depot])
            total_km += d_km
            total_time += t_min
            total_energy += 0.436 * d_km + 0.002 * cum_load
            
            job_metrics.append({
                "job_id": job_idx,
                "route": job,
                "time_min": total_time,
                "distance_km": total_km,
                "load_desi": total_load,
                "energy_kwh": total_energy,
                "num_customers": len(job)
            })
        
        # Sort jobs by time (shortest first for better packing)
        job_metrics.sort(key=lambda x: x["time_min"])
        
        # Display job table
        job_display_data = []
        for jm in job_metrics:
            job_display_data.append({
                "Görev": f"Görev {jm['job_id'] + 1}",
                "Müşteri": jm['num_customers'],
                "Süre (dk)": f"{jm['time_min']:.1f}",
                "Mesafe (km)": f"{jm['distance_km']:.2f}",
                "Yük (desi)": f"{jm['load_desi']:.0f}",
                "Enerji (kWh)": f"{jm['energy_kwh']:.2f}"
            })
        
        job_df = pd.DataFrame(job_display_data)
        st.dataframe(job_df, use_container_width=True)
        
        if st.button("🚀 Multi-Trip Optimizasyonu Çalıştır", type="primary", key="run_multitrip"):
            with st.spinner("Görevler araçlara atanıyor..."):
                # Multi-trip assignment algorithm
                # Goal: Assign multiple jobs to vehicles while respecting time AND energy constraints
                
                battery_capacity = float(data.get("battery_capacity", 100.0))
                if battery_capacity <= 0:
                    st.error("Batarya kapasitesi 0 veya negatif. Lütfen geçerli bir batarya kapasitesi sağlayın.")
                    st.stop()

                vehicle_assignments = []  # List of vehicles, each with list of jobs

                # Filter out infeasible jobs to avoid infinite loops
                infeasible_jobs = []
                unassigned_jobs = []
                for idx, job in enumerate(job_metrics):
                    if job["time_min"] > max_shift_duration:
                        infeasible_jobs.append((idx, "Vardiya süresi kısıtı"))
                        continue
                    if job["energy_kwh"] > battery_capacity:
                        infeasible_jobs.append((idx, "Batarya kapasitesi kısıtı"))
                        continue
                    unassigned_jobs.append(idx)

                if infeasible_jobs:
                    msg_lines = []
                    for idx, reason in infeasible_jobs[:10]:
                        j = job_metrics[idx]
                        msg_lines.append(
                            f"- Görev {j['job_id'] + 1}: {reason} (süre={j['time_min']:.1f} dk, enerji={j['energy_kwh']:.2f} kWh)"
                        )
                    extra = "" if len(infeasible_jobs) <= 10 else f"\n... (+{len(infeasible_jobs) - 10} görev daha)"
                    st.warning(
                        "Bazı görevler mevcut kısıtlarla atanamaz ve çözüm dışı bırakıldı:\n" + "\n".join(msg_lines) + extra
                    )

                vehicle_id = 0
                
                while unassigned_jobs:
                    vehicle_jobs = []
                    vehicle_time = 0.0
                    vehicle_energy = 0.0
                    vehicle_distance = 0.0
                    vehicle_customers = 0
                    
                    # Try to assign jobs to this vehicle
                    jobs_assigned_this_iteration = []
                    
                    for job_idx in unassigned_jobs:
                        job = job_metrics[job_idx]
                        
                        # Check if adding this job exceeds shift duration
                        additional_time = job["time_min"]
                        if vehicle_jobs:  # Not first job, add depot service time
                            additional_time += depot_service_time
                        
                        if (vehicle_time + additional_time <= max_shift_duration) and (
                            vehicle_energy + job["energy_kwh"] <= battery_capacity
                        ):
                            # Assign this job to the vehicle
                            vehicle_jobs.append(job)
                            vehicle_time += additional_time
                            vehicle_energy += job["energy_kwh"]
                            vehicle_distance += job["distance_km"]
                            vehicle_customers += job["num_customers"]
                            jobs_assigned_this_iteration.append(job_idx)

                    # Safety: if nothing fits even for an empty vehicle, break to avoid infinite loop
                    if not jobs_assigned_this_iteration:
                        # This should not happen due to infeasible filtering, but keep as guard.
                        remaining = [job_metrics[i]["job_id"] + 1 for i in unassigned_jobs]
                        st.error(
                            "Bazı görevler mevcut kısıtlarla paketlenemedi (zaman/enerji). Kalan görevler: "
                            + ", ".join([f"Görev {j}" for j in remaining])
                        )
                        break
                    
                    # Remove assigned jobs from unassigned list
                    for job_idx in jobs_assigned_this_iteration:
                        unassigned_jobs.remove(job_idx)
                    
                    # Save vehicle assignment if it has jobs
                    if vehicle_jobs:
                        vehicle_assignments.append({
                            "vehicle_id": vehicle_id,
                            "jobs": vehicle_jobs,
                            "num_trips": len(vehicle_jobs),
                            "total_time": vehicle_time,
                            "total_energy": vehicle_energy,
                            "total_distance": vehicle_distance,
                            "total_customers": vehicle_customers
                        })
                        vehicle_id += 1
                
                # Calculate remaining energy for each vehicle
                for v_assign in vehicle_assignments:
                    remaining_energy = battery_capacity - v_assign["total_energy"]
                    # Guard against tiny floating errors; energy must never go below zero.
                    if remaining_energy < -1e-6:
                        st.error(
                            "Enerji kısıtı ihlal edildi (kalan enerji negatif). Lütfen parametreleri kontrol edin."
                        )
                        st.stop()
                    if remaining_energy < 0:
                        remaining_energy = 0.0
                    remaining_pct = (remaining_energy / battery_capacity * 100) if battery_capacity > 0 else 0
                    v_assign["remaining_energy"] = remaining_energy
                    v_assign["remaining_pct"] = remaining_pct

                # Informational warning if target min trips isn't met
                if vehicle_assignments:
                    below_target = [v for v in vehicle_assignments if v["num_trips"] < int(min_trips_per_vehicle)]
                    if below_target:
                        st.info(
                            f"Not: {len(below_target)} araç, 'Min. Görev/Araç (Hedef)' değerinin altında kaldı. "
                            "Bu değer zorunlu değildir; zaman/enerji kısıtları nedeniyle daha fazla görev eklenemedi."
                        )
                
                # Store in session state
                st.session_state["multitrip_assignments"] = vehicle_assignments
                st.session_state["multitrip_base_solution"] = selected_source
                st.session_state["multitrip_num_jobs"] = num_jobs
                st.session_state["multitrip_jobs"] = jobs
                st.session_state["multitrip_max_shift"] = max_shift_duration
                st.session_state["multitrip_battery_capacity"] = battery_capacity
                st.session_state["multitrip_min_trips_target"] = int(min_trips_per_vehicle)
        
        # Display results if available (outside button block so they persist)
        if "multitrip_assignments" in st.session_state and st.session_state["multitrip_assignments"]:
            vehicle_assignments = st.session_state["multitrip_assignments"]
            num_jobs = st.session_state["multitrip_num_jobs"]
            jobs = st.session_state["multitrip_jobs"]
            max_shift_duration = st.session_state["multitrip_max_shift"]
            battery_capacity = st.session_state["multitrip_battery_capacity"]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Optimizasyon Sonuçları")
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            total_vehicles_original = num_jobs  # Original: 1 vehicle per job
            total_vehicles_multitrip = len(vehicle_assignments)
            vehicle_reduction = total_vehicles_original - total_vehicles_multitrip
            reduction_pct = (vehicle_reduction / total_vehicles_original * 100) if total_vehicles_original > 0 else 0
            
            total_energy_multitrip = sum(v["total_energy"] for v in vehicle_assignments)
            total_distance_multitrip = sum(v["total_distance"] for v in vehicle_assignments)
            total_time_multitrip = sum(v["total_time"] for v in vehicle_assignments)
            
            with col_r1:
                st.metric(
                    "Kullanılan Araç",
                    total_vehicles_multitrip,
                    delta=f"{vehicle_reduction} araç azaldı" if vehicle_reduction > 0 else "Değişmedi",
                    delta_color="normal" if vehicle_reduction > 0 else "off"
                )
                st.caption(f"Orijinal: {total_vehicles_original}")
            
            with col_r2:
                st.metric(
                    "Toplam Enerji",
                    f"{total_energy_multitrip:.2f} kWh"
                )
            
            with col_r3:
                st.metric(
                    "Toplam Mesafe",
                    f"{total_distance_multitrip:.2f} km"
                )
            
            with col_r4:
                avg_trips = sum(v["num_trips"] for v in vehicle_assignments) / len(vehicle_assignments) if vehicle_assignments else 0
                st.metric(
                    "Ort. Görev/Araç",
                    f"{avg_trips:.1f}"
                )
            
            # Vehicle details
            st.markdown("---")
            st.markdown("### 🚚 Araç Detayları")
            
            vehicle_detail_data = []
            for v_assign in vehicle_assignments:
                job_ids = ", ".join([f"G{j['job_id']+1}" for j in v_assign["jobs"]])
                vehicle_detail_data.append({
                    "Araç": f"Araç {v_assign['vehicle_id'] + 1}",
                    "Görev Sayısı": v_assign["num_trips"],
                    "Görevler": job_ids,
                    "Müşteri": v_assign["total_customers"],
                    "Süre (dk)": f"{v_assign['total_time']:.1f}",
                    "Mesafe (km)": f"{v_assign['total_distance']:.2f}",
                    "Enerji (kWh)": f"{v_assign['total_energy']:.2f}",
                    "Kalan Enerji (kWh)": f"{v_assign['remaining_energy']:.2f}",
                    "Kalan %": f"{v_assign['remaining_pct']:.1f}%",
                    "Doluluk %": f"{(v_assign['total_time'] / max_shift_duration * 100):.1f}%"
                })
            
            vehicle_detail_df = pd.DataFrame(vehicle_detail_data)
            st.dataframe(vehicle_detail_df, use_container_width=True)
            
            # Energy summary
            st.markdown("---")
            st.markdown("### 🔋 Enerji Durumu")
            
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            
            avg_energy_used = total_energy_multitrip / len(vehicle_assignments) if vehicle_assignments else 0
            avg_remaining = sum(v["remaining_energy"] for v in vehicle_assignments) / len(vehicle_assignments) if vehicle_assignments else 0
            min_remaining = min(v["remaining_energy"] for v in vehicle_assignments) if vehicle_assignments else 0
            vehicles_needing_charge = sum(1 for v in vehicle_assignments if v["remaining_pct"] < 20)
            
            with col_e1:
                st.metric(
                    "Ort. Kullanılan Enerji",
                    f"{avg_energy_used:.2f} kWh",
                    help="Araç başına ortalama enerji tüketimi"
                )
            
            with col_e2:
                st.metric(
                    "Ort. Kalan Enerji",
                    f"{avg_remaining:.2f} kWh",
                    f"{(avg_remaining/battery_capacity*100):.1f}%"
                )
            
            with col_e3:
                st.metric(
                    "Min. Kalan Enerji",
                    f"{min_remaining:.2f} kWh",
                    delta="Kritik" if min_remaining < battery_capacity * 0.2 else "Normal",
                    delta_color="inverse" if min_remaining < battery_capacity * 0.2 else "normal"
                )
            
            with col_e4:
                st.metric(
                    "Şarj Gereken Araç",
                    vehicles_needing_charge,
                    help="Kalan enerjisi %20'nin altında olan araçlar"
                )
            
            # Summary
            if vehicle_reduction > 0:
                st.success(f"🎉 Multi-trip optimizasyonu ile **{vehicle_reduction}** araç tasarrufu sağlandı! "
                          f"(**{reduction_pct:.1f}%** azalma)")
            else:
                st.info("ℹ️ Mevcut görevler zaten optimal şekilde dağıtılmış.")
            
            # Show trip sequences
            st.markdown("---")
            st.markdown("### 🗓️ Görev Sıralamaları")
            
            for v_assign in vehicle_assignments:
                with st.expander(f"🚚 Araç {v_assign['vehicle_id'] + 1} - {v_assign['num_trips']} Görev"):
                    for trip_idx, job in enumerate(v_assign["jobs"], 1):
                        st.write(f"**{trip_idx}. Görev (G{job['job_id']+1}):** "
                               f"{job['num_customers']} müşteri, "
                               f"{job['time_min']:.1f} dk, "
                               f"{job['distance_km']:.2f} km")
                        st.caption(f"Rota: {job['route'][:8]}{'...' if len(job['route']) > 8 else ''}")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 🗺️ Rota Görselleştirme")
            
            osrm_client = st.session_state.get("osrm_client")
            
            if osrm_client is None:
                st.warning("OSRM client bulunamadı. Harita gösterilemez.")
            else:
                viz_col1, viz_col2 = st.columns(2)

                # Build selection UI in the right column first so it can drive both maps.
                with viz_col2:
                    st.markdown("#### 🚛 Multi-Trip Çözümü")

                    # Vehicle-based filtering (checkboxes on the right of the map)
                    map_col, filter_col = st.columns([4, 1])

                    filter_prefix = "multitrip_vehicle_visible_"

                    with filter_col:
                        st.markdown("**Filtre**")
                        st.caption("Haritada göstermek istediğiniz araçları seçin")

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Hepsi", key="multitrip_select_all"):
                                for v_assign in vehicle_assignments:
                                    st.session_state[f"{filter_prefix}{v_assign['vehicle_id']}"] = True
                        with c2:
                            if st.button("Temizle", key="multitrip_clear_all"):
                                for v_assign in vehicle_assignments:
                                    st.session_state[f"{filter_prefix}{v_assign['vehicle_id']}"] = False

                        selected_vehicle_ids = []
                        for v_assign in vehicle_assignments:
                            v_id = v_assign["vehicle_id"]
                            key = f"{filter_prefix}{v_id}"
                            visible = st.checkbox(
                                f"Araç {v_id + 1}",
                                value=st.session_state.get(key, True),
                                key=key,
                            )
                            if visible:
                                selected_vehicle_ids.append(v_id)
                            job_ids = ", ".join([f"G{j['job_id']+1}" for j in v_assign.get("jobs", [])])
                            if job_ids:
                                st.caption(job_ids)

                    # Flatten selected vehicles' jobs into routes for visualization
                    multitrip_vehicle_routes = []
                    multitrip_vehicle_route_labels = []
                    selected_vehicle_assignments = []
                    selected_original_routes = []
                    selected_original_route_labels = []

                    # If user already ran "Araç Atama Parametreleri" (Tab6), prefer those routes
                    # for the "Orijinal" map so Vehicle N shows the task assigned to it.
                    assignment_key = (
                        "tabu_vehicle_assignments" if selected_source == "Tabu Search" else "ga_vehicle_assignments"
                    )
                    assignment_solution = st.session_state.get(assignment_key)
                    assignment_routes_all = []
                    assignment_labels_all = []
                    assignment_by_vehicle = {}
                    if assignment_solution:
                        for a in assignment_solution:
                            if not a.get("route"):
                                continue
                            assignment_routes_all.append(a["route"])
                            assignment_labels_all.append(f"Araç {a['vehicle_id']} → İş {a['job_id']}")
                            assignment_by_vehicle[int(a["vehicle_id"])] = a

                    assignment_routes_selected = []
                    assignment_labels_selected = []
                    if assignment_by_vehicle:
                        for v_id in selected_vehicle_ids:
                            # Multi-trip UI uses 0-based IDs; vehicle assignment uses 1-based.
                            vehicle_number = int(v_id) + 1
                            a = assignment_by_vehicle.get(vehicle_number)
                            if a and a.get("route"):
                                assignment_routes_selected.append(a["route"])
                                assignment_labels_selected.append(
                                    f"Araç {a['vehicle_id']} → İş {a['job_id']}"
                                )

                    for v_assign in vehicle_assignments:
                        if v_assign["vehicle_id"] in selected_vehicle_ids:
                            selected_vehicle_assignments.append(v_assign)

                            # Build a single combined route for this multi-trip vehicle,
                            # returning to depot (0) between jobs.
                            combined = []
                            for job_idx, job in enumerate(v_assign["jobs"]):
                                combined.extend(job["route"])
                                if job_idx < len(v_assign["jobs"]) - 1:
                                    combined.append(0)  # depot sentinel between trips

                                # Track which original jobs this vehicle covers
                                selected_original_routes.append(job["route"])
                                selected_original_route_labels.append(f"G{job['job_id'] + 1}")

                            multitrip_vehicle_routes.append(combined)
                            multitrip_vehicle_route_labels.append(f"Araç {v_assign['vehicle_id'] + 1}")

                    with map_col:
                        if not multitrip_vehicle_routes:
                            st.info("Haritada rota göstermek için en az bir araç seçin.")
                        else:
                            with st.spinner("Multi-trip haritası oluşturuluyor..."):
                                m_multitrip = visualize_routes_osrm(
                                    depot_lat=DEPOT_LAT,
                                    depot_lon=DEPOT_LON,
                                    df_orders=df_orders,
                                    data=data,
                                    routing=None,
                                    manager=None,
                                    solution={"routes": multitrip_vehicle_routes},
                                    time_dim=None,
                                    energy_dim=None,
                                    osrm_client=osrm_client,
                                    weekday=st.session_state.get("selected_weekday"),
                                    route_labels=multitrip_vehicle_route_labels,
                                )
                                st_folium(m_multitrip, width=550, height=500, key="multitrip_optimized_map")

                        st.caption(
                            f"{len(selected_vehicle_assignments)}/{len(vehicle_assignments)} araç seçili, "
                            f"{len(selected_original_routes)} görev gösteriliyor"
                        )

                with viz_col1:
                    st.markdown("#### 📦 Orijinal Çözüm (Tek Görev/Araç)")
                    if assignment_solution:
                        st.caption("Araç Atama Parametreleri sonucuna göre gösterilir")

                    if not selected_vehicle_ids:
                        st.info("Orijinal haritada rota görmek için en az bir araç seçin.")
                    else:
                        # Prefer Vehicle Assignment output if present; otherwise fall back to base jobs.
                        if assignment_solution:
                            show_all_original = len(selected_vehicle_ids) == len(vehicle_assignments)
                            original_routes_to_show = (
                                assignment_routes_all if show_all_original else assignment_routes_selected
                            )
                            original_labels = (
                                assignment_labels_all if show_all_original else assignment_labels_selected
                            )

                            if not original_routes_to_show:
                                st.warning(
                                    "Seçili araçlar için Araç Atama sonucu bulunamadı. "
                                    "Önce '⚡ Elektrikli Araç Atama Sistemi' sekmesinde araç ataması yapın."
                                )
                        else:
                            # If everything is selected (default state), keep showing all base jobs.
                            show_all_original = len(selected_vehicle_ids) == len(vehicle_assignments)
                            original_routes_to_show = jobs if show_all_original else selected_original_routes
                            original_labels = None if show_all_original else selected_original_route_labels

                        with st.spinner("Orijinal harita oluşturuluyor..."):
                            m_original = visualize_routes_osrm(
                                depot_lat=DEPOT_LAT,
                                depot_lon=DEPOT_LON,
                                df_orders=df_orders,
                                data=data,
                                routing=None,
                                manager=None,
                                solution={"routes": original_routes_to_show},
                                time_dim=None,
                                energy_dim=None,
                                osrm_client=osrm_client,
                                weekday=st.session_state.get("selected_weekday"),
                                route_labels=original_labels,
                            )
                            st_folium(m_original, width=550, height=500, key="multitrip_original_map")

                        if show_all_original:
                            if assignment_solution:
                                st.caption(f"Toplam {len(assignment_routes_all)} rota")
                            else:
                                st.caption(f"Toplam {len(jobs)} rota")
                        else:
                            if assignment_solution:
                                shown = ", ".join(assignment_labels_selected) if assignment_labels_selected else "-"
                                st.caption(
                                    f"Seçili araçların atanmış işleri: {len(assignment_routes_selected)} rota (" + shown + ")"
                                )
                            else:
                                shown_jobs = ", ".join(selected_original_route_labels) if selected_original_route_labels else "-"
                                st.caption(
                                    f"Seçili araçların orijinal görevleri: {len(selected_original_routes)} rota (" + shown_jobs + ")"
                                )

