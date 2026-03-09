import unicodedata
from utils.depot_distance_filter import depot_distance_feasibility
from utils.normalization_ai import ascii_fallback
from utils.parser import parse_mahalle_regex, parse_cadde, parse_sokak
from utils.parser import (
    smart_mahalle_detector,
    parse_cadde,
    parse_sokak
)
from utils.normalization_ai import ai_normalize_address
from utils.energy_comparator import (
    compare_ortools_vs_ga,
    format_route_report,
    format_fleet_comparison,
)
from utils.traffic_time_matrices import build_time_matrices_with_traffic_optimized
from utils.traffic_osrm import osrm_route_with_traffic
from utils.ga_optimizer import ga_optimize_sequences, total_plan_cost
from utils.ortools_tabu_solver import solve_with_ortools_tabu
from utils.data_builder import (
    build_problem_and_data_from_globals,
    CAPACITY_DESI,
    BATTERY_CAPACITY,
    ENERGY_B,
    BASE_KWH_PER_100KM,
)
from utils.visualize_routes_osrm import visualize_routes_osrm
from utils.osrm_client import OSRMClient
from utils.ui_components import apply_custom_css, render_header, info_card, success_card, warning_card
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
OPENCAGE_API_KEY = _get_streamlit_secret(
    "OPENCAGE_API_KEY") or os.getenv("OPENCAGE_API_KEY")

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
        sections["capacity"].append(
            "❌ Aşağıdaki müşteriler kapasiteyi aşıyor:")
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
        sections["battery"].append(
            "❌ Batarya nedeniyle ulaşılamayan müşteriler:")
        for i, e1, e2 in too_far_nodes:
            sections["battery"].append(
                f" - Node {i}: gidiş {e1:.2f} kWh, dönüş {e2:.2f} kWh (batarya={battery})"
            )
    else:
        sections["battery"].append("✅ Batarya tüm müşteriler için yeterli.")

    min_energy = sum(
        D[depot, i] * BASE_KWH_PER_KM for i in range(n) if i != depot)
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
        st.dataframe(pd.DataFrame(
            st.session_state["single_results"]), use_container_width=True)
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

        required_cols = ["id", "il", "ilçe",
                         "adres", "desi", "tahmini servis süresi"]
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
            st.warning(
                f"❗ İstanbul dışı {removed_city_count} sipariş çıkarıldı.")

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
            st.warning(
                f"❗ Avrupa yakasından {removed_count} sipariş çıkarıldı.")

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

            required_cols = ["id", "enlem", "boylam",
                             "desi", "tahmini servis süresi"]
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
            st.warning(
                "Önce siparişleri ve OSRM matrislerini oluşturun (Tab 4).")
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

        selected_day = st.selectbox(
            "Gün Seç (Trafiğe Göre)", list(day_map.keys()))
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
                st.write("**Max single customer desi:**",
                         float(np.max(demand)))
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

                st.write(
                    "**Worst round-trip energy (depot → i → depot):**", worst_energy)
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
            st.warning(
                "Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        else:
            col_solver1, col_solver2, col_solver3 = st.columns(3)

            with col_solver1:
                time_limit = st.number_input(
                    "Zaman limiti (saniye)", min_value=1, value=10)
            with col_solver2:
                seed = st.number_input("Random Seed", min_value=0, value=42)
            with col_solver3:
                solver_mode = st.selectbox(
                    "Çözücü Modu",
                    ["Tek Tur (Tabu)", "Çoklu Tur (Multi-Trip)"],
                    help="Çoklu Tur: Araçlar yeterli enerji ve zaman varsa depoya dönüp yeni tur yapabilir"
                )

            st.markdown("---")

            allow_multitrip = (solver_mode == "Çoklu Tur (Multi-Trip)")
            if allow_multitrip:
                st.info("🔄 Multi-Trip solver etkin: araçlar depoda yeni göreve çıkabilir.")
            st.caption("Not: Tabu çözücüde enerji kısıtı mesafe bazlıdır. Formül bazlı (0.436×km + 0.002×desi) rota optimizasyonu için GA ve 7️⃣ Multi-Trip sekmesini kullanın.")

            if st.button("🚀 Çöz", key="evrp_tab2_run_solver"):
                import time
                start_time = time.time()

                if allow_multitrip:
                    from utils.multitrip_solver import solve_multitrip_ortools
                    with st.spinner("Multi-Trip solver çalışıyor..."):
                        result = solve_multitrip_ortools(
                            data,
                            time_limit_s=int(time_limit),
                            seed=int(seed),
                            allow_multi_trip=True,
                        )
                else:
                    with st.spinner("Tabu Search solver çalışıyor..."):
                        result = solve_with_ortools_tabu(
                            data,
                            time_limit_s=int(time_limit),
                            seed=int(seed),
                        )

                elapsed = time.time() - start_time
                st.session_state["tabu_result"] = result
                st.session_state["solver_mode"] = solver_mode

                if result.get("solution") is not None:
                    routes = extract_routes_from_solution(
                        data,
                        result["routing"],
                        result["manager"],
                        result["solution"],
                    )
                    st.session_state["ortools_routes"] = routes
                    served = sum(len(r) for r in routes)
                    st.success(f"✅ Çözüm bulundu. {served} müşteri servis edildi. (⏱️ {elapsed:.1f} sn)")
                else:
                    st.session_state["ortools_routes"] = None
                    st.error("❌ Çözüm bulunamadı.")

                st.text_area(
                    "Çözücü Log",
                    value=result.get("log", ""),
                    height=260,
                    key="evrp_tab2_solver_log",
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
            st.warning(
                "Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        else:
            st.info(
                "💡 GA, Tabu Search'ten **tamamen bağımsız** çalışır. Aynı siparişleri kullanır ama sıfırdan optimize eder.")
            st.info("🔧 GA enerji hesabı formül bazlıdır: 0.436×km + 0.002×desi. "
                    "Kapasite, batarya ve çalışma saati kısıtlarıyla optimize edilir.")

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
                    help="Energy: Formül bazlı enerji modeli (0.436×km + 0.002×desi)",
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
                st.markdown(
                    "### 📊 GA Başlangıç: Tüm Müşteriler (Bağımsız Çözüm)")
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
                        st.write(
                            f"  → Enerji (mesafe-based): {energy_dist:.3f} kWh")

                st.markdown("---")
                st.write(
                    f"**Toplam Enerji (mesafe-based):** {total_energy_distance_only:.3f} kWh")

                original_cost = total_plan_cost(data, base_routes, objective)
                st.write(
                    f"**Başlangıç Maliyeti ({objective}):** {original_cost:.4f}")

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
                    st.success(
                        f"🎉 GA ile **{improvement:.2f}%** iyileşme sağlandı! (⏱️ {ga_time:.1f} saniye)")
                elif improvement > 0:
                    st.info(
                        f"✅ GA ile **{improvement:.2f}%** küçük iyileşme sağlandı. (⏱️ {ga_time:.1f} saniye)")
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
                            st.write(
                                f"  Önce: {base_routes[v][:10]}{'...' if len(base_routes[v]) > 10 else ''}")
                            st.write(
                                f"  Sonra: {best_routes[v][:10]}{'...' if len(best_routes[v]) > 10 else ''}")

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
        has_tabu = tabu_result is not None and tabu_result.get(
            "solution") is not None
        has_ga = ga_routes is not None

        if data is None or df_orders is None:
            st.warning(
                "Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        elif not has_tabu and not has_ga:
            st.info("Önce Tabu Search veya GA çözümünü oluşturun.")
        else:
            # Display based on what's available
            if has_tabu and has_ga:
                st.markdown("### 🔄 Tabu vs GA Karşılaştırması")
                st.info("GA haritası, rota sayısı ve OSRM geometri çağrıları arttıkça yavaşlar. Aşağıdaki araç filtresi ile hızlanır.")

                routing = tabu_result["routing"]
                manager = tabu_result["manager"]
                solution = tabu_result["solution"]

                D = np.array(data["distance_km"], dtype=float)
                T = np.array(data["time_min"], dtype=float)
                loads = np.array(data["demand_desi"], dtype=float)
                depot = data["depot"]

                # Extract tabu routes once
                tabu_routes = []
                for v in range(data["num_vehicles"]):
                    idx = routing.Start(v)
                    route = []
                    while not routing.IsEnd(idx):
                        node = manager.IndexToNode(idx)
                        if node != depot:
                            route.append(node)
                        idx = solution.Value(routing.NextVar(idx))
                    tabu_routes.append(route)

                def route_metrics(route):
                    if not route:
                        return {
                            "km": 0.0,
                            "time": 0.0,
                            "load": 0.0,
                            "energy": 0.0,
                            "customers": 0,
                        }

                    km = 0.0
                    time = 0.0
                    load = 0.0
                    energy = 0.0
                    prev = depot
                    cum_load = 0.0

                    for node in route:
                        if node >= len(loads):
                            continue
                        d_km = float(D[prev, node])
                        t_min = float(T[prev, node])
                        node_load = float(loads[node])

                        km += d_km
                        time += t_min
                        energy += 0.436 * d_km + 0.002 * cum_load
                        load += node_load
                        cum_load += node_load
                        prev = node

                    d_km = float(D[prev, depot])
                    t_min = float(T[prev, depot])
                    km += d_km
                    time += t_min
                    energy += 0.436 * d_km + 0.002 * cum_load

                    return {
                        "km": km,
                        "time": time,
                        "load": load,
                        "energy": energy,
                        "customers": len(route),
                    }

                max_vehicles = max(len(tabu_routes), len(ga_routes))
                st.markdown("### 🚚 Araç Filtresi")
                f1, f2 = st.columns(2)
                with f1:
                    if st.button("Hepsini Seç", key="cmp_select_all"):
                        for i in range(max_vehicles):
                            st.session_state[f"cmp_vehicle_sel_{i}"] = True
                with f2:
                    if st.button("Hepsini Kaldır", key="cmp_clear_all"):
                        for i in range(max_vehicles):
                            st.session_state[f"cmp_vehicle_sel_{i}"] = False

                selected_vehicles = []
                filter_cols = st.columns(4)
                for i in range(max_vehicles):
                    key = f"cmp_vehicle_sel_{i}"
                    default_val = st.session_state.get(key, False)
                    with filter_cols[i % 4]:
                        if st.checkbox(f"Araç {i+1}", value=default_val, key=key):
                            selected_vehicles.append(i)

                if not selected_vehicles:
                    st.warning("En az bir araç seçin.")
                else:
                    map_col1, map_col2 = st.columns(2)

                    # TABU PANEL
                    with map_col1:
                        st.markdown("#### 🧠 Tabu Search")
                        selected_tabu_routes = [tabu_routes[i] for i in selected_vehicles if i < len(tabu_routes) and tabu_routes[i]]
                        if selected_tabu_routes:
                            with st.spinner("Tabu haritası oluşturuluyor..."):
                                m_tabu = visualize_routes_osrm(
                                    depot_lat=DEPOT_LAT,
                                    depot_lon=DEPOT_LON,
                                    df_orders=df_orders,
                                    data=data,
                                    routing=None,
                                    manager=None,
                                    solution={"routes": selected_tabu_routes},
                                    time_dim=None,
                                    energy_dim=None,
                                    osrm_client=osrm_client,
                                    weekday=st.session_state.get("selected_weekday"),
                                )
                                st_folium(m_tabu, width=550, height=500, key="comparison_map_tabu_filtered")

                        tabu_rows = []
                        for i in selected_vehicles:
                            route = tabu_routes[i] if i < len(tabu_routes) else []
                            m = route_metrics(route)
                            if route:
                                tabu_rows.append({
                                    "Araç": f"Araç {i+1}",
                                    "Müşteri": m["customers"],
                                    "Süre (dk)": round(m["time"], 1),
                                    "Mesafe (km)": round(m["km"], 2),
                                    "Yük (desi)": round(m["load"], 0),
                                    "Enerji (kWh)": round(m["energy"], 2),
                                })

                        st.markdown("**Özet**")
                        if tabu_rows:
                            tdf = pd.DataFrame(tabu_rows)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Toplam Mesafe", f"{tdf['Mesafe (km)'].sum():.2f} km")
                                st.metric("Toplam Süre", f"{tdf['Süre (dk)'].sum():.1f} dk")
                            with c2:
                                st.metric("Toplam Yük", f"{tdf['Yük (desi)'].sum():.0f} desi")
                                st.metric("Toplam Enerji", f"{tdf['Enerji (kWh)'].sum():.2f} kWh")
                            st.dataframe(tdf, use_container_width=True)
                        else:
                            st.info("Seçili araçlarda Tabu rota yok.")

                    # GA PANEL
                    with map_col2:
                        st.markdown("#### 🧬 Genetic Algorithm")
                        selected_ga_routes = [ga_routes[i] for i in selected_vehicles if i < len(ga_routes) and ga_routes[i]]
                        if selected_ga_routes:
                            with st.spinner("GA haritası oluşturuluyor..."):
                                m_ga = visualize_routes_osrm(
                                    depot_lat=DEPOT_LAT,
                                    depot_lon=DEPOT_LON,
                                    df_orders=df_orders,
                                    data=data,
                                    routing=None,
                                    manager=None,
                                    solution={"routes": selected_ga_routes},
                                    time_dim=None,
                                    energy_dim=None,
                                    osrm_client=osrm_client,
                                    weekday=st.session_state.get("selected_weekday"),
                                )
                                st_folium(m_ga, width=550, height=500, key="comparison_map_ga_filtered")

                        ga_rows = []
                        for i in selected_vehicles:
                            route = ga_routes[i] if i < len(ga_routes) else []
                            m = route_metrics(route)
                            if route:
                                ga_rows.append({
                                    "Araç": f"Araç {i+1}",
                                    "Müşteri": m["customers"],
                                    "Süre (dk)": round(m["time"], 1),
                                    "Mesafe (km)": round(m["km"], 2),
                                    "Yük (desi)": round(m["load"], 0),
                                    "Enerji (kWh)": round(m["energy"], 2),
                                })

                        st.markdown("**Özet**")
                        if ga_rows:
                            gdf = pd.DataFrame(ga_rows)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Toplam Mesafe", f"{gdf['Mesafe (km)'].sum():.2f} km")
                                st.metric("Toplam Süre", f"{gdf['Süre (dk)'].sum():.1f} dk")
                            with c2:
                                st.metric("Toplam Yük", f"{gdf['Yük (desi)'].sum():.0f} desi")
                                st.metric("Toplam Enerji", f"{gdf['Enerji (kWh)'].sum():.2f} kWh")
                            st.dataframe(gdf, use_container_width=True)
                        else:
                            st.info("Seçili araçlarda GA rota yok.")

                    st.markdown("### 📊 Tabu vs GA Fark Özeti (Seçili Araçlar)")
                    tabu_total = {"km": 0.0, "time": 0.0, "load": 0.0, "energy": 0.0}
                    ga_total = {"km": 0.0, "time": 0.0, "load": 0.0, "energy": 0.0}
                    for i in selected_vehicles:
                        tm = route_metrics(tabu_routes[i] if i < len(tabu_routes) else [])
                        gm = route_metrics(ga_routes[i] if i < len(ga_routes) else [])
                        for k in tabu_total:
                            tabu_total[k] += tm[k]
                            ga_total[k] += gm[k]

                    d1, d2, d3, d4 = st.columns(4)
                    with d1:
                        st.metric("Mesafe Farkı (GA-Tabu)", f"{ga_total['km'] - tabu_total['km']:.2f} km")
                    with d2:
                        st.metric("Süre Farkı (GA-Tabu)", f"{ga_total['time'] - tabu_total['time']:.1f} dk")
                    with d3:
                        st.metric("Yük Farkı (GA-Tabu)", f"{ga_total['load'] - tabu_total['load']:.0f} desi")
                    with d4:
                        st.metric("Enerji Farkı (GA-Tabu)", f"{ga_total['energy'] - tabu_total['energy']:.2f} kWh")

            elif has_tabu:
                # Only Tabu available
                st.markdown("### 🧠 Tabu Search Çözümü")
                st.info(
                    "Sadece Tabu Search çözümü mevcut. GA çözümü için '🧬 Genetik Algoritma' sekmesine gidin.")

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
                        st.session_state.vehicle_states = {
                            v: True for v in range(n_vehicles)}

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
                        current_state = st.session_state.vehicle_states.get(
                            v, True)

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

                        filtered_routing = FilteredSolution(
                            routing, manager, solution, selected_vehicles, all_routes)

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
                                weekday=st.session_state.get(
                                    "selected_weekday"),
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
                        battery_capacity = float(
                            data.get("battery_capacity", 100.0))
                        vehicle_capacity = float(
                            data.get("vehicle_cap_desi", 15000.0))

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
                                    service_time = float(
                                        df_orders.iloc[node - 1]["Servis Süresi (dk)"])
                                    total_time += service_time

                                prev_node = node

                            d_km = float(D[prev_node, depot])
                            t_min = float(T[prev_node, depot])
                            total_km += d_km
                            total_time += t_min
                            energy_kwh = 0.436 * d_km + 0.002 * cum_load
                            total_energy += energy_kwh

                            energy_pct = (
                                total_energy / battery_capacity) * 100.0
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

                            total_km_all = sum(
                                float(s["Toplam KM"]) for s in vehicle_stats)
                            total_time_all = sum(
                                float(s["Toplam Süre (dk)"]) for s in vehicle_stats)
                            total_load_all = sum(
                                float(s["Taşınan Yük (desi)"]) for s in vehicle_stats)
                            total_energy_kwh = sum(
                                float(s["Enerji (kWh)"]) for s in vehicle_stats)

                            with col1:
                                st.metric("Toplam Mesafe",
                                          f"{total_km_all:.2f} km")
                            with col2:
                                st.metric("Toplam Süre",
                                          f"{total_time_all:.1f} dk")
                            with col3:
                                st.metric("Toplam Yük",
                                          f"{total_load_all:.0f} desi")
                            with col4:
                                st.metric("Toplam Enerji",
                                          f"{total_energy_kwh:.2f} kWh")
                        else:
                            st.info(
                                "Seçili araçlar için istatistik hesaplanamadı.")

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
                            service_time = float(
                                df_orders.iloc[node - 1]["Servis Süresi (dk)"])
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

                    total_km_all = sum(float(s["Toplam KM"])
                                       for s in vehicle_stats)
                    total_time_all = sum(
                        float(s["Toplam Süre (dk)"]) for s in vehicle_stats)
                    total_load_all = sum(
                        float(s["Taşınan Yük (desi)"]) for s in vehicle_stats)
                    total_energy_kwh = sum(
                        float(s["Enerji (kWh)"]) for s in vehicle_stats)

                    with col1:
                        st.metric("Toplam Mesafe", f"{total_km_all:.2f} km")
                    with col2:
                        st.metric("Toplam Süre", f"{total_time_all:.1f} dk")
                    with col3:
                        st.metric("Toplam Yük", f"{total_load_all:.0f} desi")
                    with col4:
                        st.metric("Toplam Enerji",
                                  f"{total_energy_kwh:.2f} kWh")
                else:
                    st.info("İstatistik hesaplanamadı.")

    st.markdown("---")
    st.info("⚡ Elektrikli Araç Atama Sistemi kaldırıldı. Akış: Tabu/GA çözümü → 7️⃣ Multi-Trip Optimizasyonu (formül bazlı).")

# =========================================================
# 7️⃣ ÇOKLU GÖREV (MULTI-TRIP) OPTİMİZASYONU
# =========================================================
with tab7:
    st.header("🚛 Multi-Trip Optimizasyonu")

    data = st.session_state.get("ortools_data")
    tabu_result = st.session_state.get("tabu_result")
    ga_routes = st.session_state.get("ga_best_routes")
    df_orders = st.session_state.get("orders_df")

    tabu_ran = tabu_result is not None
    has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
    has_ga = ga_routes is not None

    if data is None:
        st.warning("⚠️ Önce EVRP modelini oluşturun.")
    elif not has_tabu and not has_ga:
        if tabu_ran:
            st.warning("⚠️ Tabu çalıştırıldı ancak geçerli çözüm bulunamadı. 6️⃣ Problem Çözümü sekmesindeki çözücü logunu kontrol edin veya araç sayısı/zaman limiti artırın.")
            if tabu_result.get("log"):
                with st.expander("Tabu Solver Log (Özet)"):
                    st.text(tabu_result.get("log", "")[:4000])
        else:
            st.warning("⚠️ Önce 6️⃣ Problem Çözümü sekmesinde Tabu veya GA çalıştırın.")
    else:
        D = np.array(data["distance_km"], dtype=float)
        T = np.array(data["time_min"], dtype=float)
        loads = np.array(data["demand_desi"], dtype=float)
        service = np.array(data.get("service_min", np.zeros(len(loads))), dtype=float)
        depot = int(data.get("depot", 0))
        battery_capacity = float(data.get("battery_capacity", 100.0))

        def _extract_routes_from_tabu(result, num_vehicles, dep):
            routes = []
            routing = result["routing"]
            manager = result["manager"]
            solution = result["solution"]
            for v in range(num_vehicles):
                idx = routing.Start(v)
                route = []
                while not routing.IsEnd(idx):
                    node = manager.IndexToNode(idx)
                    if node != dep:
                        route.append(node)
                    idx = solution.Value(routing.NextVar(idx))
                routes.append(route)
            return routes

        def _route_metrics(route):
            total_km = 0.0
            total_time = 0.0
            total_load = 0.0
            total_energy = 0.0
            cum_load = 0.0
            prev = depot

            for node in route:
                if node < 0 or node >= len(loads):
                    continue
                d_km = float(D[prev, node])
                t_min = float(T[prev, node])
                total_km += d_km
                total_time += t_min + float(service[node])
                total_energy += 0.436 * d_km + 0.002 * cum_load
                node_load = float(loads[node])
                cum_load += node_load
                total_load += node_load
                prev = node

            d_km = float(D[prev, depot])
            t_min = float(T[prev, depot])
            total_km += d_km
            total_time += t_min
            total_energy += 0.436 * d_km + 0.002 * cum_load

            return {
                "distance_km": total_km,
                "time_min": total_time,
                "load_desi": total_load,
                "energy_kwh": total_energy,
            }

        source_options = []
        if has_tabu:
            source_options.append("Tabu Search")
        if has_ga:
            source_options.append("Genetic Algorithm")

        selected_source = st.selectbox(
            "Baz çözüm",
            source_options,
            key="multitrip_source_selector",
            help="Multi-trip öncesi orijinal çözüm"
        )

        if selected_source == "Tabu Search":
            all_routes = _extract_routes_from_tabu(tabu_result, int(data["num_vehicles"]), depot)
        else:
            all_routes = ga_routes

        original_routes = [r for r in all_routes if r]
        original_rows = []
        original_jobs = []
        for i, route in enumerate(original_routes):
            m = _route_metrics(route)
            m["job_id"] = i + 1
            m["route"] = route
            original_jobs.append(m)
            original_rows.append({
                "Araç": f"Araç {i + 1}",
                "Müşteri": len(route),
                "Süre (dk)": round(m["time_min"], 1),
                "Mesafe (km)": round(m["distance_km"], 2),
                "Yük (desi)": round(m["load_desi"], 0),
                "Enerji (kWh)": round(m["energy_kwh"], 2),
            })

        if not original_jobs:
            st.info("Seçilen çözümde servis edilen rota bulunamadı.")
            st.stop()

        st.markdown("### Orijinal Çözüm (Tek Görev / Araç)")
        st.dataframe(pd.DataFrame(original_rows), use_container_width=True)

        st.markdown("### Multi-Trip Parametreleri")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_shift_duration = st.number_input(
                "Maksimum vardiya süresi (dk)",
                min_value=240,
                max_value=720,
                value=540,
                step=30,
                key="multitrip_max_shift"
            )
        with c2:
            depot_service_time = st.number_input(
                "Depo servis süresi (dk)",
                min_value=0,
                max_value=60,
                value=15,
                step=5,
                key="multitrip_depot_service"
            )
        with c3:
            min_return_pct = st.number_input(
                "Minimum dönüş şarjı (%)",
                min_value=0,
                max_value=90,
                value=20,
                step=5,
                key="multitrip_min_return_pct"
            )

        usable_energy = battery_capacity * (1.0 - float(min_return_pct) / 100.0)
        st.caption(
            f"Toplam batarya: {battery_capacity:.2f} kWh | Kullanılabilir enerji (rezerv sonrası): {usable_energy:.2f} kWh"
        )

        if st.button("🚀 Multi-Trip Optimizasyonu Çalıştır", type="primary", key="run_multitrip"):
            with st.spinner("Multi-trip araç atamaları hazırlanıyor..."):
                feasible_jobs = []
                infeasible = []
                for j in original_jobs:
                    if j["time_min"] > max_shift_duration:
                        infeasible.append((j["job_id"], "süre"))
                        continue
                    if j["energy_kwh"] > usable_energy:
                        infeasible.append((j["job_id"], "enerji"))
                        continue
                    feasible_jobs.append(dict(j))

                if infeasible:
                    msg = ", ".join([f"G{jid} ({reason})" for jid, reason in infeasible[:10]])
                    st.warning(
                        "Atanamaz görevler atlandı: "
                        f"{msg}. Bu görevler tek başına bile enerji/süre limitine sığmadığı için "
                        "multi-trip atamasına dahil edilmedi."
                    )

                remaining = sorted(feasible_jobs, key=lambda x: x["time_min"], reverse=True)
                vehicle_assignments = []
                vehicle_id = 1

                while remaining:
                    v_jobs = []
                    v_time = 0.0
                    v_energy = 0.0
                    v_dist = 0.0
                    v_load = 0.0
                    v_customers = 0

                    picked = []
                    for cand in list(remaining):
                        extra_time = cand["time_min"] + (depot_service_time if v_jobs else 0.0)
                        if (v_time + extra_time <= max_shift_duration) and (v_energy + cand["energy_kwh"] <= usable_energy):
                            v_jobs.append(cand)
                            v_time += extra_time
                            v_energy += cand["energy_kwh"]
                            v_dist += cand["distance_km"]
                            v_load += cand["load_desi"]
                            v_customers += len(cand["route"])
                            picked.append(cand)

                    if not picked:
                        hard = remaining.pop(0)
                        v_jobs = [hard]
                        v_time = hard["time_min"]
                        v_energy = hard["energy_kwh"]
                        v_dist = hard["distance_km"]
                        v_load = hard["load_desi"]
                        v_customers = len(hard["route"])
                    else:
                        for p in picked:
                            remaining.remove(p)

                    vehicle_assignments.append({
                        "vehicle_id": vehicle_id,
                        "jobs": v_jobs,
                        "num_trips": len(v_jobs),
                        "time_min": v_time,
                        "distance_km": v_dist,
                        "load_desi": v_load,
                        "energy_kwh": v_energy,
                        "remaining_energy_kwh": max(0.0, battery_capacity - v_energy),
                    })
                    vehicle_id += 1

                st.session_state["multitrip_assignments"] = vehicle_assignments
                st.session_state["multitrip_original_jobs"] = original_jobs
                st.session_state["multitrip_base_solution"] = selected_source
                st.session_state["multitrip_usable_energy"] = usable_energy

        saved_assignments = st.session_state.get("multitrip_assignments")
        saved_source = st.session_state.get("multitrip_base_solution")
        saved_original = st.session_state.get("multitrip_original_jobs")

        if saved_assignments and saved_source == selected_source and saved_original:
            st.markdown("---")
            st.markdown("### Sonuç Özeti")

            orig_total_time = sum(x["time_min"] for x in saved_original)
            orig_total_dist = sum(x["distance_km"] for x in saved_original)
            orig_total_load = sum(x["load_desi"] for x in saved_original)
            orig_total_energy = sum(x["energy_kwh"] for x in saved_original)

            mt_total_time = sum(x["time_min"] for x in saved_assignments)
            mt_total_dist = sum(x["distance_km"] for x in saved_assignments)
            mt_total_load = sum(x["load_desi"] for x in saved_assignments)
            mt_total_energy = sum(x["energy_kwh"] for x in saved_assignments)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Araç Sayısı", len(saved_assignments), delta=f"{len(saved_original) - len(saved_assignments)}")
            with k2:
                st.metric("Toplam Süre (dk)", f"{mt_total_time:.1f}", delta=f"{(mt_total_time - orig_total_time):.1f}")
            with k3:
                st.metric("Toplam Mesafe (km)", f"{mt_total_dist:.2f}", delta=f"{(mt_total_dist - orig_total_dist):.2f}")
            with k4:
                st.metric("Toplam Enerji (kWh)", f"{mt_total_energy:.2f}", delta=f"{(mt_total_energy - orig_total_energy):.2f}")

            st.markdown("### Grafik Karşılaştırma")
            chart_df = pd.DataFrame([
                {"Metrik": "Süre (dk)", "Orijinal": orig_total_time, "Multi-Trip": mt_total_time},
                {"Metrik": "Mesafe (km)", "Orijinal": orig_total_dist, "Multi-Trip": mt_total_dist},
                {"Metrik": "Yük (desi)", "Orijinal": orig_total_load, "Multi-Trip": mt_total_load},
                {"Metrik": "Enerji (kWh)", "Orijinal": orig_total_energy, "Multi-Trip": mt_total_energy},
            ]).set_index("Metrik")
            st.bar_chart(chart_df)

            st.markdown("### Araç Bazlı Orijinal vs Multi-Trip")
            max_rows = max(len(saved_original), len(saved_assignments))
            compare_rows = []
            for i in range(max_rows):
                o = saved_original[i] if i < len(saved_original) else {
                    "time_min": 0.0,
                    "distance_km": 0.0,
                    "load_desi": 0.0,
                    "energy_kwh": 0.0,
                }
                m = saved_assignments[i] if i < len(saved_assignments) else {
                    "num_trips": 0,
                    "time_min": 0.0,
                    "distance_km": 0.0,
                    "load_desi": 0.0,
                    "energy_kwh": 0.0,
                    "remaining_energy_kwh": battery_capacity,
                }
                compare_rows.append({
                    "Araç": f"Araç {i + 1}",
                    "Orijinal Süre (dk)": round(o["time_min"], 1),
                    "Multi-Trip Süre (dk)": round(m["time_min"], 1),
                    "Süre Farkı (dk)": round(m["time_min"] - o["time_min"], 1),
                    "Orijinal Mesafe (km)": round(o["distance_km"], 2),
                    "Multi-Trip Mesafe (km)": round(m["distance_km"], 2),
                    "Mesafe Farkı (km)": round(m["distance_km"] - o["distance_km"], 2),
                    "Orijinal Yük (desi)": round(o["load_desi"], 0),
                    "Multi-Trip Yük (desi)": round(m["load_desi"], 0),
                    "Yük Farkı (desi)": round(m["load_desi"] - o["load_desi"], 0),
                    "Orijinal Enerji (kWh)": round(o["energy_kwh"], 2),
                    "Multi-Trip Enerji (kWh)": round(m["energy_kwh"], 2),
                    "Enerji Farkı (kWh)": round(m["energy_kwh"] - o["energy_kwh"], 2),
                    "Multi-Trip Görev": int(m.get("num_trips", 0)),
                    "Kalan Enerji (kWh)": round(m.get("remaining_energy_kwh", 0.0), 2),
                })

            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

            st.markdown("### Multi-Trip Araç Detayı")
            detail_rows = []
            for v in saved_assignments:
                detail_rows.append({
                    "Araç": f"Araç {v['vehicle_id']}",
                    "Görev Sayısı": v["num_trips"],
                    "Görevler": ", ".join([f"G{j['job_id']}" for j in v["jobs"]]),
                    "Süre (dk)": round(v["time_min"], 1),
                    "Mesafe (km)": round(v["distance_km"], 2),
                    "Yük (desi)": round(v["load_desi"], 0),
                    "Enerji (kWh)": round(v["energy_kwh"], 2),
                    "Kalan Enerji (kWh)": round(v["remaining_energy_kwh"], 2),
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

            st.markdown("### Harita Karşılaştırması")
            osrm_client = st.session_state.get("osrm_client")

            if osrm_client is None or df_orders is None:
                st.info("Harita için gerekli veri eksik (orders/osrm_client).")
            else:
                map_col1, map_col2 = st.columns(2)

                original_routes_for_map = [j["route"] for j in saved_original if j.get("route")]

                multitrip_routes_for_map = []
                multitrip_labels = []
                for v in saved_assignments:
                    if not v.get("jobs"):
                        continue
                    combined_route = []
                    for i, j in enumerate(v["jobs"]):
                        combined_route.extend(j["route"])
                        if i < len(v["jobs"]) - 1:
                            combined_route.append(depot)
                    multitrip_routes_for_map.append(combined_route)
                    multitrip_labels.append(f"Araç {v['vehicle_id']}")

                with map_col1:
                    st.markdown("#### Orijinal Çözüm Haritası")
                    if original_routes_for_map:
                        m_original = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=None,
                            manager=None,
                            solution={"routes": original_routes_for_map},
                            time_dim=None,
                            energy_dim=None,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                        )
                        st_folium(m_original, width=550, height=500, key="multitrip_original_map_readded")
                    else:
                        st.info("Orijinal rota bulunamadı.")

                with map_col2:
                    st.markdown("#### Multi-Trip Çözüm Haritası")
                    if multitrip_routes_for_map:
                        m_multi = visualize_routes_osrm(
                            depot_lat=DEPOT_LAT,
                            depot_lon=DEPOT_LON,
                            df_orders=df_orders,
                            data=data,
                            routing=None,
                            manager=None,
                            solution={"routes": multitrip_routes_for_map},
                            time_dim=None,
                            energy_dim=None,
                            osrm_client=osrm_client,
                            weekday=st.session_state.get("selected_weekday"),
                            route_labels=multitrip_labels,
                        )
                        st_folium(m_multi, width=550, height=500, key="multitrip_optimized_map_readded")
                    else:
                        st.info("Multi-trip rota bulunamadı.")
