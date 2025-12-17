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
st.set_page_config(page_title="Adres → Koordinat", layout="wide")
st.title("📍 Adres → Koordinat Dönüştürücü (Structured + Hybrid Geocoder)")

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
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

if not OPENCAGE_API_KEY:
    raise RuntimeError("OPENCAGE_API_KEY not found in environment")
DATA_DIR = Path(__file__).parent / "Data"

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ Adres → Koordinat",
        "2️⃣ Sipariş Oluştur",
        "3️⃣ Siparişleri Haritada Göster",
        "4️⃣ OSRM Mesafe & Süre Matrisi",
        "5️⃣ Trafikli Süre Matrisleri",
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
                icon_shape="star",
                border_color="red",
                border_width=2,
                text_color="white",
                background_color="red",
                inner_icon_style="font-size:16px;padding-top:0px;",
                number="",
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
# 📦 PROBLEM ÇÖZÜMÜ (OR-Tools Tabu + Map)
# =========================================================
st.markdown("---")
st.header("📦 Problem Çözümü")

evrp_tab1, evrp_tab2, evrp_tab3 = st.tabs(
    [
        "📦 Problem Kurulumu",
        "🧠 Tabu Search",
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
    st.subheader("🧠 OR-Tools Tabu Çözücü")

    data = st.session_state.get("ortools_data")

    if data is None:
        st.warning("Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
    else:
        time_limit = st.number_input("Zaman limiti (saniye)", min_value=1, value=10)
        seed = st.number_input("Random Seed", min_value=0, value=42)

        if st.button("🚀 Tabu Search ile Çöz"):
            with st.spinner("OR-Tools Tabu Search çalışıyor..."):
                result = solve_with_ortools_tabu(
                    data,
                    time_limit_s=int(time_limit),
                    seed=int(seed),
                )

            st.session_state["tabu_result"] = result

            # extract routes (Option A: node indices)
            if result.get("solution") is not None:
                routes = extract_routes_from_solution(
                    data,
                    result["routing"],
                    result["manager"],
                    result["solution"],
                )
                st.session_state["ortools_routes"] = routes
                st.success("✅ Çözüm bulundu!")
                st.text("✅ Rotalar cache'lendi (GA için hazır).")
            else:
                st.session_state["ortools_routes"] = None
                st.error("❌ Çözüm bulunamadı.")

            st.text_area(
                "Çözüm Detayları",
                value=result.get("log", ""),
                height=400,
            )


# ---------- TAB 3: OR-Tools Solution Map ----------
with evrp_tab3:
    st.subheader("🗺 Çözümü Haritada Göster")

    tabu_result = st.session_state.get("tabu_result")
    data = st.session_state.get("ortools_data")
    df_orders = st.session_state.get("orders_df")
    osrm_client = st.session_state.get("osrm_client")

    if tabu_result is None or data is None or df_orders is None:
        st.warning("Önce Tabu Search çözümünü oluşturun.")
    elif tabu_result.get("solution") is None:
        st.error("Tabu çözümü bulunamadı, harita çizilemiyor.")
    else:
        routing = tabu_result["routing"]
        manager = tabu_result["manager"]
        solution = tabu_result["solution"]
        time_dim = tabu_result["time_dim"]
        energy_dim = tabu_result["energy_dim"]

        with st.spinner("Harita oluşturuluyor..."):
            m = visualize_routes_osrm(
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
            )

        st_folium(m, width=1200, height=800)


# =========================================================
# ⚡ ELEKTRİKLİ ARAÇLAR İÇİN OPTİMİZE ET (GA)
# =========================================================
st.markdown("---")
st.header("⚡ Elektrikli Araçlar için Optimize Et")

from utils.ga_optimizer import (
    ga_optimize_sequences,
    print_ga_detailed_solution,
    total_plan_cost,
)

opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs(
    [
        "🚚 OR-Tools Rotaları",
        "🧬 Genetik Algoritma Çözümü",
        "🗺 GA Çözüm Haritası",
        "⚡ Enerji Karşılaştırması",
    ]
)

# ---------- OPT TAB 1: SHOW OR-TOOLS ROUTES (DETAILED) ----------
with opt_tab1:
    st.subheader("🚚 OR-Tools Rota Özeti (Detaylı Çıktı)")

    data = st.session_state.get("ortools_data")
    df_orders = st.session_state.get("orders_df")
    tabu_result = st.session_state.get("tabu_result")

    if tabu_result is None or df_orders is None or data is None:
        st.info("Henüz OR-Tools çözümü yok. Önce Tabu Search çalıştırın.")
    else:
        routing = tabu_result["routing"]
        manager = tabu_result["manager"]
        solution = tabu_result["solution"]

        ortools_routes = []
        for v in range(data["num_vehicles"]):
            r = []
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    r.append(node)
                idx = solution.Value(routing.NextVar(idx))
            ortools_routes.append(r)

        st.session_state["ortools_routes"] = ortools_routes

        st.success("OR-Tools rotaları başarıyla yüklendi!")

        st.text(
            "📌 Rotalar (node index):\n"
            + "\n".join(f"Vehicle {v}: {r}" for v, r in enumerate(ortools_routes))
        )

        detailed_text = tabu_result["log"]
        st.text_area("Detaylı OR-Tools Çıktısı", value=detailed_text, height=600)


# ---------- OPT TAB 2: RUN GA + PRINT FULL TABLE ----------
with opt_tab2:
    st.subheader("🧬 Genetik Algoritma ile Rota Sıralamalarını İyileştir")

    data = st.session_state.get("ortools_data")
    routes = st.session_state.get("ortools_routes")
    df_orders = st.session_state.get("orders_df")

    if data is None or routes is None:
        st.info("Önce OR-Tools sonucunu alın.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pop_size = st.number_input(
                "Popülasyon boyutu",
                min_value=10,
                max_value=500,
                value=120,
                step=10,
            )
        with col2:
            generations = st.number_input(
                "Generasyon sayısı",
                min_value=50,
                max_value=2000,
                value=400,
                step=50,
            )
        with col3:
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)

        objective = st.selectbox("Amaç fonksiyonu", ["energy", "distance"], index=0)

        if st.button("🧬 GA Çalıştır"):
            with st.spinner("Genetik Algoritma çalışıyor..."):
                best_routes, best_fit = ga_optimize_sequences(
                    data=data,
                    base_routes=routes,
                    pop_size=int(pop_size),
                    generations=int(generations),
                    objective=objective,
                    elitism=2,
                    seed=int(seed),
                )

            st.session_state["ga_best_routes"] = best_routes
            st.session_state["ga_best_fitness"] = best_fit

            original_cost = total_plan_cost(data, routes, objective)
            improvement = (
                (original_cost - best_fit) / original_cost * 100
                if original_cost > 0
                else 0.0
            )

            st.success("🎉 GA tamamlandı!")

            st.markdown(
                f"**Başlangıç maliyeti ({objective}):** `{original_cost:.3f}`"
            )
            st.markdown(f"**GA sonrası en iyi maliyet:** `{best_fit:.3f}`")
            st.markdown(f"**İyileşme:** `{improvement:.2f}%`")

            txt_ga = print_ga_detailed_solution(
                data=data,
                routes=best_routes,
                df_orders=df_orders,
            )

            st.text_area("GA Detaylı Çıktı", txt_ga, height=600)


# ---------- OPT TAB 3: GA MAP USING OSRM ----------
with opt_tab3:
    st.subheader("🗺 GA Çözüm Haritası")

    df_orders = st.session_state.get("orders_df")
    data = st.session_state.get("ortools_data")
    ga_routes = st.session_state.get("ga_best_routes")

    if df_orders is None or data is None:
        st.info("Önce problem ve OR-Tools verilerini oluşturun.")
    elif ga_routes is None:
        st.info("Henüz GA çalıştırılmadı. İkinci sekmeden GA'yı çalıştırın.")
    else:
        with st.spinner("GA çözüm haritası oluşturuluyor..."):
            m_ga = visualize_routes_osrm(
                depot_lat=DEPOT_LAT,
                depot_lon=DEPOT_LON,
                df_orders=df_orders,
                data=data,
                routing=None,  # GA MODE
                manager=None,  # GA MODE
                solution={"routes": ga_routes},  # GA solution wrapper
                time_dim=None,
                energy_dim=None,
                osrm_client=st.session_state["osrm_client"],
            )

        st_folium(m_ga, width=1200, height=800)


# ---------- OPT TAB 4: ENERGY COMPARISON ----------
with opt_tab4:
    st.subheader("⚡ OR-Tools vs GA Enerji Karşılaştırması")

    data = st.session_state.get("ortools_data")
    ortools_routes = st.session_state.get("ortools_routes")
    ga_routes = st.session_state.get("ga_best_routes")

    if data is None:
        st.info("Önce EVRP modelini oluşturun.")
    elif ortools_routes is None:
        st.info("Önce OR-Tools Tabu Search çözümünü alın.")
    elif ga_routes is None:
        st.info("Henüz GA çalıştırılmadı.")
    else:
        result = compare_ortools_vs_ga(ortools_routes, ga_routes, data)

        st.markdown("### 📊 Enerji Özeti")
        st.code(format_fleet_comparison(result))

        st.markdown("### 🚚 OR-Tools Araç Enerji Raporu")
        for rep in result["ortools_vehicle_reports"]:
            st.code(format_route_report(rep))

        st.markdown("### 🧬 GA Araç Enerji Raporu")
        for rep in result["ga_vehicle_reports"]:
            st.code(format_route_report(rep))
