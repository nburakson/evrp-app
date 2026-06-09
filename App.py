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
import hashlib
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


def render_folium_safe(map_obj, width: int, height: int, key: str | None = None):
    """Render folium map with component fallback for environments where st_folium assets fail to load."""
    try:
        if key is None:
            st_folium(map_obj, width=width, height=height)
        else:
            st_folium(map_obj, width=width, height=height, key=key)
    except Exception:
        st.warning("Harita bileşeni yüklenemedi; HTML fallback ile gösteriliyor.")
        st.components.v1.html(map_obj._repr_html_(), width=width, height=height, scrolling=False)


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
            "ortools_routes", "ga_best_routes", "ga_best_fitness",
            "alns_result", "alns_routes", "gas_ga_routes",
            "gas_ga_summary", "gas_ga_best_distance"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "one_trip_cache" not in st.session_state:
    st.session_state["one_trip_cache"] = {}

for k in ["tabu_runtime_s", "ga_runtime_s", "alns_runtime_s", "gas_ga_runtime_s", "one_trip_signature"]:
    if k not in st.session_state:
        st.session_state[k] = None

if "opencage_warning_shown" not in st.session_state:
    st.session_state["opencage_warning_shown"] = False

if "product_service_map" not in st.session_state:
    st.session_state["product_service_map"] = None

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
GAS_LITERS_PER_100KM = 12.0
GAS_TANK_LITERS = 80.0


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


def standardize_excel_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Normalize uploaded Excel headers and map common variants to expected names."""
    column_aliases = {
        "id": "id",
        "il": "il",
        "ilce": "ilçe",
        "ilçe": "ilçe",
        "adres": "adres",
        "desi": "desi",
        "urun": "ürün",
        "ürün": "ürün",
        "enlem": "enlem",
        "lat": "enlem",
        "latitude": "enlem",
        "boylam": "boylam",
        "lon": "boylam",
        "lng": "boylam",
        "longitude": "boylam",
    }

    renamed_columns = {}
    for column in df.columns:
        normalized = normalize_tr(str(column))
        renamed_columns[column] = column_aliases.get(normalized, normalized)

    df = df.rename(columns=renamed_columns)

    if df.columns.duplicated().any():
        duplicate_columns = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(
            "Ayni anlama gelen birden fazla kolon bulundu: "
            + ", ".join(duplicate_columns)
        )

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Eksik kolonlar: " + ", ".join(missing)
        )

    return df


@st.cache_data
def load_product_service_map():
    """Load product -> service minutes map from Data/Ürün Servis Süreleri.xlsx."""
    path = DATA_DIR / "Ürün Servis Süreleri.xlsx"
    df = pd.read_excel(path)

    required_cols = {"Ürün", "Servis Süresi"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "Ürün servis süreleri dosyasında 'Ürün' ve 'Servis Süresi' sütunları olmalı."
        )

    product_map = {}
    for _, row in df.iterrows():
        product_raw = str(row.get("Ürün", "")).strip()
        if not product_raw:
            continue

        product_key = normalize_tr(product_raw)
        service_minutes = float(row.get("Servis Süresi", 0) or 0)
        product_map[product_key] = service_minutes

    return product_map


def get_product_service_map():
    """
    Get product service map from session state (if uploaded) or from default file.
    This function allows users to override the default product service times.
    """
    # Check if user has uploaded a custom product service map
    if "product_service_map" in st.session_state and st.session_state["product_service_map"]:
        return st.session_state["product_service_map"]
    
    # Otherwise, load from default file
    return load_product_service_map()


def split_product_list(product_list_text) -> list[str]:
    if product_list_text is None or pd.isna(product_list_text):
        return []

    return [
        item.strip()
        for item in re.split(r"\s*(?:,|\|)\s*", str(product_list_text))
        if item and item.strip()
    ]


def calculate_service_minutes_from_products(
    product_list_text,
    product_service_map,
    discount_per_extra_product: float = 0.05,
):
    """Calculate service time based on products with group discount.
    
    Logic:
    1. Get service time for each product
    2. If only 1 product: use full service time (no discount)
    3. If multiple products:
       - Calculate original sum (no discount)
       - Calculate discounted sum (each time * 0.05)
       - If discounted sum < 50% of original: use 50% of original
       - Otherwise: use discounted sum
    """
    raw_items = split_product_list(product_list_text)
    if not raw_items:
        return 0.0, []

    individual_times = []
    unknown_products = []

    for item in raw_items:
        key = normalize_tr(item)
        if key in product_service_map:
            individual_times.append(float(product_service_map[key]))
        else:
            unknown_products.append(item)
            individual_times.append(0.0)

    if not individual_times or sum(individual_times) == 0:
        return 0.0, unknown_products

    # If only 1 product, return full service time (no discount)
    if len(individual_times) == 1:
        return round(individual_times[0], 2), unknown_products

    # Multiple products: apply discount logic
    original_sum = sum(individual_times)
    discounted_sum = sum(t * discount_per_extra_product for t in individual_times)
    minimum_sum = original_sum * 0.5

    final_time = max(discounted_sum, minimum_sum)

    return round(final_time, 2), unknown_products


def merge_orders_by_coordinates(df_orders: pd.DataFrame, product_service_map=None):
    """Merge orders that share identical coordinates into a single stop."""
    if df_orders is None or df_orders.empty:
        return df_orders, 0, [], pd.DataFrame()

    def first_non_empty(series: pd.Series):
        for value in series:
            if pd.notna(value) and str(value).strip():
                return value
        return series.iloc[0] if not series.empty else None

    def sum_numeric(series: pd.Series) -> float:
        return pd.to_numeric(series, errors="coerce").fillna(0).sum()

    grouped_rows = []
    unknown_products = set()

    for _, group in df_orders.groupby(["Enlem", "Boylam"], dropna=False, sort=False):
        merged_row = {}

        for column in ["Street", "Mahalle", "Ilce", "Il"]:
            if column in group.columns:
                merged_row[column] = first_non_empty(group[column])

        merged_row["Enlem"] = group["Enlem"].iloc[0]
        merged_row["Boylam"] = group["Boylam"].iloc[0]

        if "Desi" in group.columns:
            merged_row["Desi"] = round(sum_numeric(group["Desi"]), 2)

        if "Ürün" in group.columns:
            merged_products = []
            for value in group["Ürün"]:
                merged_products.extend(split_product_list(value))

            merged_row["Ürün"] = ", ".join(merged_products)

            if product_service_map is not None:
                service_minutes, unknown = calculate_service_minutes_from_products(
                    merged_row["Ürün"],
                    product_service_map,
                )
                merged_row["Servis Süresi (dk)"] = service_minutes
                unknown_products.update(unknown)

        elif "Servis Süresi (dk)" in group.columns:
            merged_row["Servis Süresi (dk)"] = round(sum_numeric(group["Servis Süresi (dk)"]), 2)

        grouped_rows.append(merged_row)

    merged_df = pd.DataFrame(grouped_rows)
    if merged_df.empty:
        return merged_df, 0, [], pd.DataFrame()

    over_capacity_df = pd.DataFrame()
    if "Desi" in merged_df.columns:
        over_capacity_mask = pd.to_numeric(merged_df["Desi"], errors="coerce").fillna(0) > CAPACITY_DESI
        if over_capacity_mask.any():
            over_capacity_df = merged_df.loc[over_capacity_mask].copy()
            merged_df = merged_df.loc[~over_capacity_mask].copy()

    if merged_df.empty:
        return merged_df, len(df_orders) - len(grouped_rows), sorted(unknown_products, key=normalize_tr), over_capacity_df

    merged_df.insert(0, "OrderID", range(1, len(merged_df) + 1))

    ordered_columns = [
        column for column in df_orders.columns
        if column != "OrderID" and column in merged_df.columns
    ]
    for column in ["Ürün", "Servis Süresi (dk)", "Street", "Mahalle", "Ilce", "Il", "Enlem", "Boylam", "Desi"]:
        if column in merged_df.columns and column not in ordered_columns:
            ordered_columns.append(column)
    merged_df = merged_df[["OrderID", *ordered_columns]]

    if not over_capacity_df.empty:
        extra_columns = [column for column in merged_df.columns if column not in over_capacity_df.columns]
        for column in extra_columns:
            over_capacity_df[column] = None
        over_capacity_df = over_capacity_df[merged_df.columns]

    return merged_df, len(df_orders) - len(grouped_rows), sorted(unknown_products, key=normalize_tr), over_capacity_df


def gasoline_route_metrics(route, D, T, loads, service, depot=0):
    total_km = 0.0
    total_time = 0.0
    total_load = 0.0
    prev = depot

    for node in route:
        if node < 0 or node >= len(loads):
            continue
        total_km += float(D[prev, node])
        total_time += float(T[prev, node]) + float(service[node])
        total_load += float(loads[node])
        prev = node

    total_km += float(D[prev, depot])
    total_time += float(T[prev, depot])
    fuel_liters = total_km * (GAS_LITERS_PER_100KM / 100.0)

    return {
        "distance_km": total_km,
        "time_min": total_time,
        "load_desi": total_load,
        "fuel_liters": fuel_liters,
        "tank_pct": (fuel_liters / GAS_TANK_LITERS * 100.0) if GAS_TANK_LITERS > 0 else 0.0,
    }


def summarize_gasoline_routes(routes, data):
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    loads = np.array(data["demand_desi"], dtype=float)
    service = np.array(data.get("service_min", np.zeros(len(loads))), dtype=float)
    depot = int(data.get("depot", 0))

    vehicle_rows = []
    for vehicle_idx, route in enumerate(routes, start=1):
        if not route:
            continue
        metrics = gasoline_route_metrics(route, D, T, loads, service, depot=depot)
        vehicle_rows.append(
            {
                "Araç": f"Araç {vehicle_idx}",
                "Müşteri": len(route),
                "Mesafe (km)": round(metrics["distance_km"], 2),
                "Süre (dk)": round(metrics["time_min"], 1),
                "Yük (desi)": round(metrics["load_desi"], 0),
                "Benzin (lt)": round(metrics["fuel_liters"], 2),
                "Depo Kullanımı (%)": round(metrics["tank_pct"], 1),
            }
        )

    summary = {
        "used_vehicles": len(vehicle_rows),
        "total_km": round(sum(row["Mesafe (km)"] for row in vehicle_rows), 2),
        "total_time": round(sum(row["Süre (dk)"] for row in vehicle_rows), 1),
        "total_load": round(sum(row["Yük (desi)"] for row in vehicle_rows), 0),
        "total_fuel_liters": round(sum(row["Benzin (lt)"] for row in vehicle_rows), 2),
        "vehicle_rows": vehicle_rows,
    }
    return summary


def render_gasoline_ga_tab():
    st.header("9) Benzinli Araç Optimizasyonu")

    gas_data = st.session_state.get("ortools_data")
    gas_df_orders = st.session_state.get("orders_df")

    if gas_data is None or gas_df_orders is None:
        st.info("Önce 6️⃣ Problem Çözümü sekmesinde modeli oluşturun.")
        return

    st.caption(
        "Bu etap elektrik kısıtlarını kullanmaz. Amaç fonksiyonu yalnızca toplam km minimizasyonudur. "
        f"Yakıt tüketimi: {GAS_LITERS_PER_100KM:.0f} lt/100 km | Depo: {GAS_TANK_LITERS:.0f} lt"
    )
    st.caption(
        f"Maksimum teorik menzil: {(GAS_TANK_LITERS / GAS_LITERS_PER_100KM) * 100:.1f} km"
    )

    gg1, gg2, gg3, gg4 = st.columns(4)
    with gg1:
        gas_pop_size = st.number_input(
            "Popülasyon boyutu",
            min_value=20,
            max_value=500,
            value=150,
            step=10,
            key="tab9_gas_pop_size",
        )
    with gg2:
        gas_generations = st.number_input(
            "Generasyon sayısı",
            min_value=100,
            max_value=3000,
            value=600,
            step=50,
            key="tab9_gas_generations",
        )
    with gg3:
        gas_mutation_rate = st.slider(
            "Mutasyon oranı",
            min_value=0.01,
            max_value=0.5,
            value=0.15,
            step=0.05,
            key="tab9_gas_mutation_rate",
        )
    with gg4:
        gas_seed = st.number_input(
            "Random seed",
            min_value=0,
            value=321,
            key="tab9_gas_seed",
        )

    gas_improvement_mode = st.selectbox(
        "İyileştirme modu",
        ["none", "selective", "full"],
        format_func=lambda x: {
            "none": "Hızlı (Sadece GA)",
            "selective": "Dengeli (Seçici 2-opt)",
            "full": "Maksimum Kalite (Full 2-opt)",
        }[x],
        index=1,
        key="tab9_gas_improvement_mode",
    )

    if st.button("⛽ Benzinli GA Çalıştır", key="tab9_run_gas_ga"):
        import time

        all_customers = list(range(1, len(gas_df_orders) + 1))
        base_routes = [all_customers]
        sig = st.session_state.get("one_trip_signature")
        gas_cache_key = (
            "gas_ga",
            sig,
            int(gas_pop_size),
            int(gas_generations),
            float(gas_mutation_rate),
            int(gas_seed),
            str(gas_improvement_mode),
        )
        cached_gas = st.session_state["one_trip_cache"].get(gas_cache_key)

        if cached_gas is not None:
            gas_routes = cached_gas["routes"]
            gas_distance = float(cached_gas["best_distance"])
            gas_runtime = float(cached_gas["runtime_s"])
            gas_summary = cached_gas["summary"]
            st.info(f"Önbellekten yüklendi (Benzinli GA, {gas_runtime:.1f} sn).")
        else:
            start_time = time.time()
            with st.spinner("Benzinli araçlar için GA çalışıyor..."):
                gas_data_for_ga = dict(gas_data)
                gas_data_for_ga["fuel_liters_per_100km"] = GAS_LITERS_PER_100KM
                gas_data_for_ga["fuel_tank_liters"] = GAS_TANK_LITERS
                gas_routes, gas_distance = ga_optimize_sequences(
                    data=gas_data_for_ga,
                    base_routes=base_routes,
                    pop_size=int(gas_pop_size),
                    generations=int(gas_generations),
                    objective="distance",
                    elitism=2,
                    seed=int(gas_seed),
                    improvement_mode=gas_improvement_mode,
                    enforce_energy_constraints=False,
                    enforce_fuel_constraints=True,
                )
            gas_runtime = time.time() - start_time
            gas_summary = summarize_gasoline_routes(gas_routes, gas_data_for_ga)

            st.session_state["one_trip_cache"][gas_cache_key] = {
                "routes": gas_routes,
                "best_distance": gas_distance,
                "runtime_s": gas_runtime,
                "summary": gas_summary,
            }

        st.session_state["gas_ga_routes"] = gas_routes
        st.session_state["gas_ga_best_distance"] = gas_distance
        st.session_state["gas_ga_runtime_s"] = gas_runtime
        st.session_state["gas_ga_summary"] = gas_summary

    gas_ga_summary = st.session_state.get("gas_ga_summary")
    gas_ga_runtime = st.session_state.get("gas_ga_runtime_s")
    if gas_ga_summary:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Kullanılan Araç", gas_ga_summary["used_vehicles"])
        with m2:
            st.metric("Toplam Mesafe", f"{gas_ga_summary['total_km']:.2f} km")
        with m3:
            st.metric("Toplam Benzin", f"{gas_ga_summary['total_fuel_liters']:.2f} lt")
        with m4:
            st.metric("Çözüm Süresi", f"{float(gas_ga_runtime or 0.0):.1f} sn")

        if gas_ga_summary["vehicle_rows"]:
            st.dataframe(pd.DataFrame(gas_ga_summary["vehicle_rows"]), use_container_width=True)


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
        payload = r.json()
        if isinstance(payload, dict):
            payload["_http_status"] = r.status_code
        return payload, r.url
    except Exception:
        return None, None


def maybe_warn_opencage_issue(response_json):
    if not response_json or st.session_state.get("opencage_warning_shown"):
        return

    status = response_json.get("status") or {}
    status_code = status.get("code") or response_json.get("_http_status")
    message = status.get("message")

    if status_code and int(status_code) != 200:
        st.warning(
            f"OpenCage kullanılamadı: HTTP/API {status_code}"
            + (f" - {message}" if message else "")
            + ". Geocoding işlemi Nominatim fallback ile devam ediyor."
        )
        st.session_state["opencage_warning_shown"] = True


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
    maybe_warn_opencage_issue(oc_json)

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
    maybe_warn_opencage_issue(oc_json2)

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


def one_trip_signature(data: dict | None) -> str | None:
    """Create a stable signature for caching one-trip solutions."""
    if data is None:
        return None

    h = hashlib.md5()
    h.update(str(data.get("num_vehicles", "")).encode("utf-8"))
    h.update(str(data.get("vehicle_cap_desi", "")).encode("utf-8"))
    h.update(str(data.get("battery_capacity", data.get("battery_kwh", ""))).encode("utf-8"))
    h.update(str(data.get("depot", 0)).encode("utf-8"))

    for key in ["distance_km", "time_min", "demand_desi", "service_min"]:
        arr = np.array(data.get(key, []), dtype=float)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(np.round(arr, 3).tobytes())

    return h.hexdigest()


# =========================================================
# ⚡ ADVANCED EVRP FEASIBILITY ANALYZER
# =========================================================

BASE_KWH_PER_KM = 0.436
ENERGY_PER_DESI_KM = 0.00136


def evrp_feasibility_detailed(data, work_start_min=9*60, work_end_min=18*60):
    """
    EVRP Feasibility Debugger
    
    ⚠️ SERVICE TIME DEDUCTION: 
    Each customer visit requires travel_time + service_time. The available time window
    is (work_end_min - work_start_min) minutes. Service times are DEDUCTED from this
    available window in all calculations.
    
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "1️⃣ Adres → Koordinat",
        "2️⃣ Sipariş Oluştur",
        "3️⃣ Siparişleri Haritada Göster",
        "4️⃣ OSRM Mesafe & Süre Matrisi",
        "5️⃣ Trafikli Süre Matrisleri",
        "6️⃣ Problem Çözümü",
        "7️⃣ Çoklu Görev Optimizasyonu",
        "8️⃣ Sonuç Gösterimi",
        "9️⃣ Benzinli Araçlar",
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
        "Excel yükle (id, il, ilçe, adres, desi, ürün)",
        type=["xlsx"],
        key="bulk_upload_tab1",
    )

    if bulk_file:
        required_cols = ["id", "il", "ilçe", "adres", "desi", "ürün"]
        try:
            df_bulk = standardize_excel_columns(pd.read_excel(bulk_file), required_cols)
        except ValueError as exc:
            st.error(
                f"❌ Excel başlıkları okunamadı: {exc}. Gerekli sütunlar: {', '.join(required_cols)}"
            )
            st.stop()

        product_service_map = get_product_service_map()

        calc_results = df_bulk["ürün"].apply(
            lambda txt: calculate_service_minutes_from_products(
                txt,
                product_service_map,
            )
        )
        df_bulk["hesaplanan servis süresi"] = calc_results.apply(lambda x: x[0])
        df_bulk["_bilinmeyen_urunler"] = calc_results.apply(lambda x: x[1])

        unknown_set = sorted(
            {
                p
                for sublist in df_bulk["_bilinmeyen_urunler"]
                for p in sublist
                if p
            },
            key=lambda x: normalize_tr(x),
        )
        if unknown_set:
            st.warning(
                "⚠️ Ürün servis listesinde bulunamayan ürünler 0 dk kabul edildi: "
                + ", ".join(unknown_set)
            )

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
                "ürün": lambda x: " | ".join(x.astype(str)),
                "hesaplanan servis süresi": "mean",
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
                    "ürün": row["ürün"],
                    "hesaplanan servis süresi": round(float(row["hesaplanan servis süresi"]), 2),
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
            "ürün": ["yatak,baza", "beyaz ev tekstili,panel"],
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

    # -------- Excel Upload --------
    st.subheader("📤 Excel'den Servis Süresi Ekle")

    service_time_file = st.file_uploader(
        "Ürün Servis Süresi Excel dosyasını yükle (Ürün, Servis Süresi sütunları gerekli)",
        type=["xlsx"],
        key="service_time_upload",
    )

    if service_time_file is not None:
        try:
            df_service = pd.read_excel(service_time_file)
            
            # Check required columns
            required_cols = {"Ürün", "Servis Süresi"}
            if not required_cols.issubset(set(df_service.columns)):
                st.error(f"❌ Excel dosyasında '{', '.join(required_cols)}' sütunları olmalı.")
            else:
                # Create product service map from uploaded file
                product_map = {}
                for _, row in df_service.iterrows():
                    product_raw = str(row.get("Ürün", "")).strip()
                    if not product_raw:
                        continue
                    
                    product_key = normalize_tr(product_raw)
                    service_minutes = float(row.get("Servis Süresi", 0) or 0)
                    product_map[product_key] = service_minutes
                
                # Store in session state
                st.session_state["product_service_map"] = product_map
                
                st.success(f"✅ {len(product_map)} ürün servis süresi yüklendi!")
                st.dataframe(df_service, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Dosya okunamadı: {e}")
    else:
        if st.session_state.get("product_service_map"):
            st.info(f"ℹ️ {len(st.session_state['product_service_map'])} ürün servis süresi yüklü.")
        else:
            st.info("ℹ️ Ürün servis süresi dosyası yüklenmek üzere beklemede. Varsayılan değerler kullanılacaktır.")

    st.markdown("---")

    # -------- Excel Upload --------
    st.subheader("📤 Excel'den Sipariş Yükle")

    uploaded_file = st.file_uploader(
        "Excel yükle (id, enlem, boylam, desi, ürün)",
        type=["xlsx"],
        key="orders_upload",
    )

    if uploaded_file is not None:
        try:
            required_cols = ["id", "enlem", "boylam", "desi", "ürün"]
            df_up = standardize_excel_columns(pd.read_excel(uploaded_file), required_cols)

            df_orders = df_up.rename(
                columns={
                    "id": "OrderID",
                    "enlem": "Enlem",
                    "boylam": "Boylam",
                    "desi": "Desi",
                    "ürün": "Ürün",
                }
            )

            product_service_map = get_product_service_map()
            df_orders, merged_count, unknown_set, removed_over_capacity = merge_orders_by_coordinates(
                df_orders,
                product_service_map=product_service_map,
            )
            if unknown_set:
                st.warning(
                    "⚠️ Ürün servis listesinde bulunamayan ürünler 0 dk kabul edildi: "
                    + ", ".join(unknown_set)
                )

            if merged_count > 0:
                st.info(f"🔄 Aynı koordinatlı {merged_count} sipariş birleştirildi.")

            if not removed_over_capacity.empty:
                st.warning(
                    f"⚠️ Birleştirme sonrası desisi {CAPACITY_DESI} üstüne çıkan {len(removed_over_capacity)} sipariş silindi."
                )
                st.dataframe(removed_over_capacity, use_container_width=True)

            if df_orders.empty:
                st.error("❌ Birleştirme sonrası kapasiteye uygun sipariş kalmadı.")
                st.stop()

            df_orders = df_orders[
                ["OrderID", "Enlem", "Boylam", "Desi", "Ürün", "Servis Süresi (dk)"]
            ]

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
            df_orders, merged_count, _, removed_over_capacity = merge_orders_by_coordinates(df_orders)
            st.session_state["orders_df"] = df_orders

            if merged_count > 0:
                st.info(f"🔄 Aynı koordinatlı {merged_count} sipariş birleştirildi.")

            if not removed_over_capacity.empty:
                st.warning(
                    f"⚠️ Birleştirme sonrası desisi {CAPACITY_DESI} üstüne çıkan {len(removed_over_capacity)} sipariş silindi."
                )
                st.dataframe(removed_over_capacity, use_container_width=True)

            if df_orders.empty:
                st.error("❌ Birleştirme sonrası kapasiteye uygun sipariş kalmadı.")
                st.stop()

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
            render_folium_safe(m, width=1200, height=750)


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

    current_sig = one_trip_signature(st.session_state.get("ortools_data"))
    prev_sig = st.session_state.get("one_trip_signature")
    if current_sig != prev_sig:
        st.session_state["one_trip_cache"] = {}
        st.session_state["one_trip_signature"] = current_sig
        st.session_state["tabu_runtime_s"] = None
        st.session_state["ga_runtime_s"] = None
        st.session_state["alns_runtime_s"] = None
        st.session_state["gas_ga_runtime_s"] = None
        st.session_state["gas_ga_routes"] = None
        st.session_state["gas_ga_summary"] = None
        st.session_state["gas_ga_best_distance"] = None

    evrp_tab1, evrp_tab2, evrp_tab3, evrp_tab4, evrp_tab5 = st.tabs(
        [
            "📦 Problem Kurulumu",
            "🧠 Tabu Search",
            "🧬 Genetik Algoritma",
            "🧩 ALNS",
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

    # ================= CALCULATE MINIMUM VEHICLES FUNCTION ======================
    def calculate_minimum_vehicles(
        df_orders,
        D,
        T,
        max_capacity=CAPACITY_DESI,
        battery_capacity=BATTERY_CAPACITY,
        max_work_minutes=9*60,  # 09:00 to 18:00 = 540 minutes
    ):
        """
        Calculate minimum number of vehicles needed using greedy bin-packing approach.
        Considers capacity, time, and energy constraints.
        Returns the estimated minimum vehicles needed.
        """
        if df_orders is None or len(df_orders) == 0:
            return 1
        
        try:
            num_orders = len(df_orders)
            
            # Extract demands - handle different column names
            if "desi" in df_orders.columns:
                demands = np.array(df_orders["desi"], dtype=float)
            else:
                demands = np.ones(num_orders, dtype=float)  # Default to 1 if no desi column
            
            # Extract service times - handle different column names
            if "Servis Süresi (dk)" in df_orders.columns:
                service_times = np.array(df_orders["Servis Süresi (dk)"], dtype=float)
            else:
                service_times = np.zeros(num_orders, dtype=float)
            
            # Validate inputs
            if len(demands) == 0 or D is None or T is None:
                return 1
            
            # Start with 1 vehicle and greedily assign orders
            vehicles = []
            unassigned_orders = list(range(num_orders))
            depot = 0
            
            while unassigned_orders:
                # Create new vehicle
                vehicle_capacity = 0.0
                vehicle_time = 0.0
                vehicle_energy = 0.0
                vehicle_orders = []
                
                # Try to add orders to this vehicle (simple greedy)
                orders_to_remove = []
                for order_idx in unassigned_orders:
                    if order_idx >= len(demands):
                        continue
                    
                    order_demand = float(demands[order_idx])
                    order_service = float(service_times[order_idx]) if order_idx < len(service_times) else 0.0
                    
                    # Distance matrix indices: depot is 0, customers are 1 to num_orders
                    customer_idx = order_idx + 1
                    
                    if customer_idx >= D.shape[0] or customer_idx >= T.shape[0]:
                        continue
                    
                    # Distance to order and back to depot
                    dist_to_order = float(D[depot, customer_idx])
                    dist_from_order = float(D[customer_idx, depot])
                    total_dist = dist_to_order + dist_from_order
                    
                    # Time needed (travel + service)
                    time_needed = float(T[depot, customer_idx]) + order_service + float(T[customer_idx, depot])
                    
                    # Energy needed (0.436 per km + 0.002 per desi-km considering current load)
                    energy_needed = 0.436 * total_dist + 0.002 * (vehicle_capacity + order_demand) * total_dist / 1000
                    
                    # Check if order fits
                    can_fit_capacity = (vehicle_capacity + order_demand) <= max_capacity
                    can_fit_time = (vehicle_time + time_needed) <= max_work_minutes
                    can_fit_energy = (vehicle_energy + energy_needed) <= battery_capacity * 0.8  # 80% usable
                    
                    if can_fit_capacity and can_fit_time and can_fit_energy:
                        vehicle_capacity += order_demand
                        vehicle_time += time_needed
                        vehicle_energy += energy_needed
                        vehicle_orders.append(order_idx)
                        orders_to_remove.append(order_idx)
                
                # Remove assigned orders
                for order_idx in orders_to_remove:
                    unassigned_orders.remove(order_idx)
                
                vehicles.append({
                    "capacity": vehicle_capacity,
                    "time": vehicle_time,
                    "energy": vehicle_energy,
                    "orders": vehicle_orders
                })
                
                # Safety check: if no orders were assigned, force add the first unassigned
                if not orders_to_remove and unassigned_orders:
                    order_idx = unassigned_orders.pop(0)
                    vehicles[-1]["orders"].append(order_idx)
            
            return max(len(vehicles), 1)
        
        except Exception as e:
            st.warning(f"⚠️ Minimum araç hesaplaması hatası: {str(e)}. Varsayılan değer kullanılıyor.")
            return 1

    # ================= EVRP MODEL OLUŞTUR ======================
    if st.button("🚀 EVRP Modelini Derle"):

        # Calculate minimum vehicles needed based on constraints
        min_vehicles = calculate_minimum_vehicles(
            df_orders=df_orders,
            D=D,
            T=T_osrm if st.session_state.get("T_by_hour") is None else st.session_state.get("T_by_hour", {}).get(9, T_osrm),
            max_capacity=CAPACITY_DESI,
            battery_capacity=BATTERY_CAPACITY,
            max_work_minutes=9*60,
        )
        
        # Use the maximum of calculated minimum and user input
        # (but prefer minimum if it's feasible)
        actual_num_vehicles = max(min_vehicles, 1)
        
        st.info(f"📊 Hesaplanan minimum araç sayısı: **{min_vehicles}** | Kullanıcı girişi: {int(num_vehicles)} → **{actual_num_vehicles}** araç kullanılacak")

        T_by_hour = st.session_state.get("T_by_hour")

        if T_by_hour is not None:
            planning_hour = 9  # always start at 09:00
            problem, data = build_problem_and_data_from_globals(
                df_orders=df_orders,
                D=D,
                T=None,  # use T_by_hour
                num_vehicles=actual_num_vehicles,
                T_by_hour=T_by_hour,
                planning_hour=planning_hour,
            )
        else:
            problem, data = build_problem_and_data_from_globals(
                df_orders=df_orders,
                D=D,
                T=T_osrm,
                num_vehicles=actual_num_vehicles,
            )

        # ⚠️ SERVICE TIMES ARE INCLUDED: Each order's "Servis Süresi (dk)" is extracted from df_orders
        # and included in data["service_min"]. These service times are DEDUCTED from vehicle
        # operating hours (09:00–18:00) in all feasibility checks and optimization processes.

        # store
        st.session_state["evrp_problem"] = problem
        st.session_state["ortools_data"] = data
        st.session_state["calculated_min_vehicles"] = min_vehicles
        st.session_state["actual_num_vehicles"] = actual_num_vehicles
        st.session_state["tabu_result"] = None
        st.session_state["ortools_routes"] = None
        st.session_state["ga_best_routes"] = None
        st.session_state["ga_best_fitness"] = None
        st.session_state["alns_result"] = None
        st.session_state["alns_routes"] = None
        st.session_state["gas_ga_routes"] = None
        st.session_state["gas_ga_summary"] = None
        st.session_state["gas_ga_best_distance"] = None
        st.session_state["gas_ga_runtime_s"] = None

        st.success("EVRP modeli başarıyla oluşturuldu.")
        st.subheader("🧪 Detaylı Feasibility Analizi")
        st.caption("⚠️ Aşağıdaki analizde, her müşterinin servis süresi (Servis Süresi (dk)) araçların vardiya süresinden (09:00–18:00) çıkarılır.")

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
            col_solver1, col_solver2 = st.columns(2)

            with col_solver1:
                time_limit = st.number_input(
                    "Zaman limiti (saniye)", min_value=1, value=10)
            with col_solver2:
                seed = st.number_input("Random Seed", min_value=0, value=42)

            st.markdown("---")
            st.info("Bu sekmede yalnızca Tek Tur (Tabu) çözümü çalıştırılır.")

            if st.button("🚀 Çöz", key="evrp_tab2_run_solver"):
                import time
                sig = st.session_state.get("one_trip_signature")
                cache_key = ("tabu", sig, int(time_limit), int(seed))
                cached = st.session_state["one_trip_cache"].get(cache_key)

                if cached is not None:
                    result = cached["result"]
                    elapsed = float(cached["elapsed_s"])
                    st.info(f"Önbellekten yüklendi (Tabu, {elapsed:.1f} sn).")
                else:
                    start_time = time.time()
                    with st.spinner("Tabu Search solver çalışıyor..."):
                        result = solve_with_ortools_tabu(
                            data,
                            time_limit_s=int(time_limit),
                            seed=int(seed),
                        )
                    elapsed = time.time() - start_time
                    st.session_state["one_trip_cache"][cache_key] = {
                        "result": result,
                        "elapsed_s": elapsed,
                    }

                st.session_state["tabu_result"] = result
                st.session_state["tabu_runtime_s"] = elapsed
                st.session_state["solver_mode"] = "Tek Tur (Tabu)"
                st.session_state["tab6_multitrip_used"] = False

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
                sig = st.session_state.get("one_trip_signature")
                ga_cache_key = (
                    "ga",
                    sig,
                    int(pop_size),
                    int(generations),
                    float(mutation_rate),
                    int(ga_seed),
                    str(objective),
                    str(improvement_mode),
                )
                cached_ga = st.session_state["one_trip_cache"].get(ga_cache_key)

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

                if cached_ga is not None:
                    best_routes = cached_ga["best_routes"]
                    best_fit = float(cached_ga["best_fit"])
                    ga_time = float(cached_ga["ga_time"])
                    st.info(f"Önbellekten yüklendi (GA, {ga_time:.1f} sn).")
                else:
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

                    st.session_state["one_trip_cache"][ga_cache_key] = {
                        "best_routes": best_routes,
                        "best_fit": best_fit,
                        "ga_time": ga_time,
                    }

                st.session_state["ga_best_routes"] = best_routes
                st.session_state["ga_best_fitness"] = best_fit
                st.session_state["ga_original_cost"] = original_cost
                st.session_state["ga_runtime_s"] = ga_time

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

    # ---------- TAB 4: ALNS Single-Trip ----------
    with evrp_tab4:
        st.subheader("🧩 ALNS Çözücü (Tek Tur)")

        data = st.session_state.get("ortools_data")

        if data is None:
            st.warning("Önce EVRP modelini oluşturun.")
        else:
            from utils.alns_singletrip_solver import solve_with_alns_singletrip

            st.caption("Tabu/GA ile aynı EVRP kısıtlarını (kapasite, vardiya, batarya) kullanarak rota üretir.")

            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                alns_iterations = st.number_input(
                    "ALNS iterasyon",
                    min_value=50,
                    max_value=5000,
                    value=600,
                    step=50,
                    key="tab6_alns_iterations",
                )
            with ac2:
                alns_destroy_rate = st.slider(
                    "Destroy oranı",
                    min_value=0.05,
                    max_value=0.60,
                    value=0.20,
                    step=0.05,
                    key="tab6_alns_destroy_rate",
                )
            with ac3:
                alns_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=42,
                    key="tab6_alns_seed",
                )
            with ac4:
                alns_objective = st.selectbox(
                    "Amaç fonksiyonu",
                    ["distance", "energy"],
                    index=0,
                    key="tab6_alns_objective",
                )

            if st.button("🧩 ALNS Çalıştır", key="evrp_tab4_run_alns"):
                import time
                sig = st.session_state.get("one_trip_signature")
                alns_cache_key = (
                    "alns",
                    sig,
                    int(alns_iterations),
                    float(alns_destroy_rate),
                    int(alns_seed),
                    str(alns_objective),
                )
                cached_alns = st.session_state["one_trip_cache"].get(alns_cache_key)

                if cached_alns is not None:
                    alns_result = cached_alns["result"]
                    alns_elapsed = float(cached_alns["elapsed_s"])
                    st.info(f"Önbellekten yüklendi (ALNS, {alns_elapsed:.1f} sn).")
                else:
                    start_time = time.time()
                    with st.spinner("ALNS solver çalışıyor..."):
                        alns_result = solve_with_alns_singletrip(
                            data=data,
                            iterations=int(alns_iterations),
                            destroy_rate=float(alns_destroy_rate),
                            seed=int(alns_seed),
                            objective=str(alns_objective),
                        )
                    alns_elapsed = time.time() - start_time
                    st.session_state["one_trip_cache"][alns_cache_key] = {
                        "result": alns_result,
                        "elapsed_s": alns_elapsed,
                    }

                alns_routes = alns_result.get("routes", [])
                st.session_state["alns_result"] = alns_result
                st.session_state["alns_routes"] = alns_routes
                st.session_state["alns_runtime_s"] = alns_elapsed

                served = int(alns_result.get("served_customers", 0))
                n_customers = max(0, data["distance_km"].shape[0] - 1)
                unserved = alns_result.get("unserved_customers", [])

                if unserved:
                    st.warning(
                        f"ALNS çözümü tamamlandı. {served}/{n_customers} müşteri servis edildi. "
                        f"Servis edilemeyen: {unserved[:20]}"
                    )
                else:
                    st.success(f"ALNS çözümü bulundu. {served}/{n_customers} müşteri servis edildi. (⏱️ {alns_elapsed:.1f} sn)")

                st.text_area(
                    "ALNS Log",
                    value=alns_result.get("log", ""),
                    height=240,
                    key="evrp_tab4_alns_log",
                )

    # ---------- TAB 5: Solution Maps ----------
    with evrp_tab5:
        st.subheader("🗺 Çözümü Haritada Göster")

        tabu_result = st.session_state.get("tabu_result")
        ga_routes = st.session_state.get("ga_best_routes")
        alns_routes = st.session_state.get("alns_routes")
        mt_assignments = st.session_state.get("multitrip_assignments")
        data = st.session_state.get("ortools_data")
        df_orders = st.session_state.get("orders_df")
        osrm_client = st.session_state.get("osrm_client")

        # Check what solutions are available
        has_tabu = tabu_result is not None and tabu_result.get(
            "solution") is not None
        has_ga = ga_routes is not None
        has_alns = alns_routes is not None
        has_mt = mt_assignments is not None and len(mt_assignments) > 0

        if data is None or df_orders is None:
            st.warning(
                "Önce 'Problem Kurulumu' sekmesinde EVRP modelini oluşturun.")
        elif not has_tabu and not has_ga and not has_alns and not has_mt:
            st.info("Önce Tabu Search, GA, ALNS veya Multi-Trip çözümünü oluşturun.")
        else:
            # Display based on what's available
            if has_tabu or has_ga or has_alns or has_mt:
                st.markdown("### 🔄 Çözüm Karşılaştırması")
                st.info("Aşağıdan görmek istediğiniz çözüm türlerini seçin. Araç filtresi tüm seçili çözüm türleri için ortaktır.")

                # Debug: Check what solutions are actually available
                st.write("**Mevcut Çözümler:**")
                col_debug1, col_debug2, col_debug3, col_debug4 = st.columns(4)
                with col_debug1:
                    st.metric("Tabu", "✅" if has_tabu else "❌")
                with col_debug2:
                    st.metric("GA", "✅" if has_ga else "❌")
                with col_debug3:
                    st.metric("ALNS", "✅" if has_alns else "❌")
                with col_debug4:
                    st.metric("Multi-Trip", "✅" if has_mt else "❌")

                D = np.array(data["distance_km"], dtype=float)
                T = np.array(data["time_min"], dtype=float)
                loads = np.array(data["demand_desi"], dtype=float)
                depot = data["depot"]

                # Extract tabu routes once
                tabu_routes = []
                if has_tabu:
                    routing = tabu_result["routing"]
                    manager = tabu_result["manager"]
                    solution = tabu_result["solution"]
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

                    for node in route:
                        if node >= len(loads):
                            continue
                        d_km = float(D[prev, node])
                        t_min = float(T[prev, node])
                        node_load = float(loads[node])

                        km += d_km
                        time += t_min
                        from_node_desi = float(loads[prev]) if prev != depot else 0.0
                        energy += 0.436 * d_km + 0.002 * from_node_desi
                        load += node_load
                        prev = node

                    d_km = float(D[prev, depot])
                    t_min = float(T[prev, depot])
                    km += d_km
                    time += t_min
                    from_node_desi = float(loads[prev]) if prev != depot else 0.0
                    energy += 0.436 * d_km + 0.002 * from_node_desi

                    return {
                        "km": km,
                        "time": time,
                        "load": load,
                        "energy": energy,
                        "customers": len(route),
                    }

                mt_routes = []
                if has_mt:
                    for v in mt_assignments:
                        combined_route = []
                        jobs = v.get("jobs", [])
                        for i, j in enumerate(jobs):
                            combined_route.extend(j.get("route", []))
                            if i < len(jobs) - 1:
                                combined_route.append(depot)
                        mt_routes.append(combined_route)

                copt1, copt2, copt3, copt4 = st.columns(4)
                with copt1:
                    show_tabu = st.checkbox("Tabu", value=has_tabu, disabled=not has_tabu, key="map_show_tabu")
                with copt2:
                    show_ga = st.checkbox("GA", value=has_ga, disabled=not has_ga, key="map_show_ga")
                with copt3:
                    show_alns = st.checkbox("ALNS", value=has_alns, disabled=not has_alns, key="map_show_alns")
                with copt4:
                    show_mt = st.checkbox("Multi-Trip", value=has_mt, disabled=not has_mt, key="map_show_mt")

                solution_counts = []
                if show_tabu:
                    solution_counts.append(len(tabu_routes) if tabu_routes else 0)
                if show_ga:
                    solution_counts.append(len(ga_routes) if ga_routes else 0)
                if show_alns:
                    solution_counts.append(len(alns_routes) if alns_routes else 0)
                if show_mt:
                    solution_counts.append(len(mt_routes) if mt_routes else 0)

                if not solution_counts:
                    st.warning("En az bir çözüm türü seçin.")
                    max_vehicles = 0
                else:
                    max_vehicles = max(solution_counts)
                st.markdown("### 🚚 Görev Filtresi")
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
                if max_vehicles > 0:
                    filter_cols = st.columns(4)
                    for i in range(max_vehicles):
                        key = f"cmp_vehicle_sel_{i}"
                        if key not in st.session_state:
                            st.session_state[key] = False
                        with filter_cols[i % 4]:
                            if st.checkbox(f"Görev {i+1}", key=key):
                                selected_vehicles.append(i)

                if not selected_vehicles:
                    st.warning("En az bir görev seçin.")
                else:
                    selected_methods = []
                    if show_tabu:
                        selected_methods.append("tabu")
                    if show_ga:
                        selected_methods.append("ga")
                    if show_alns:
                        selected_methods.append("alns")
                    if show_mt:
                        selected_methods.append("mt")

                    method_cols = st.columns(len(selected_methods))
                    totals_rows = []

                    for m_idx, method_name in enumerate(selected_methods):
                        with method_cols[m_idx]:
                            if method_name == "tabu":
                                st.markdown("#### 🗂️ Tabu Search")
                                selected_routes = [tabu_routes[i] for i in selected_vehicles if tabu_routes and i < len(tabu_routes)]
                                panel_key = "comparison_map_tabu_filtered"
                                rows = []
                                for i in selected_vehicles:
                                    route = tabu_routes[i] if i < len(tabu_routes) else []
                                    rm = route_metrics(route)
                                    if route:
                                        rows.append({
                                            "Görev": f"Görev {i+1}",
                                            "Müşteri": rm["customers"],
                                            "Süre (dk)": round(rm["time"], 1),
                                            "Mesafe (km)": round(rm["km"], 2),
                                            "Yük (desi)": round(rm["load"], 0),
                                            "Kullanılan Enerji %": round(rm["energy"], 2),
                                        })
                            elif method_name == "ga":
                                st.markdown("#### 🧬 Genetic Algorithm")
                                selected_routes = [ga_routes[i] for i in selected_vehicles if ga_routes and i < len(ga_routes)]
                                panel_key = "comparison_map_ga_filtered"
                                rows = []
                                for i in selected_vehicles:
                                    route = ga_routes[i] if ga_routes and i < len(ga_routes) else []
                                    rm = route_metrics(route)
                                    if route:
                                        rows.append({
                                            "Görev": f"Görev {i+1}",
                                            "Müşteri": rm["customers"],
                                            "Süre (dk)": round(rm["time"], 1),
                                            "Mesafe (km)": round(rm["km"], 2),
                                            "Yük (desi)": round(rm["load"], 0),
                                            "Kullanılan Enerji %": round(rm["energy"], 2),
                                        })
                            elif method_name == "alns":
                                st.markdown("#### 🧩 ALNS")
                                selected_routes = [alns_routes[i] for i in selected_vehicles if alns_routes and i < len(alns_routes)]
                                panel_key = "comparison_map_alns_filtered"
                                rows = []
                                for i in selected_vehicles:
                                    route = alns_routes[i] if alns_routes and i < len(alns_routes) else []
                                    rm = route_metrics(route)
                                    if route:
                                        rows.append({
                                            "Görev": f"Görev {i+1}",
                                            "Müşteri": rm["customers"],
                                            "Süre (dk)": round(rm["time"], 1),
                                            "Mesafe (km)": round(rm["km"], 2),
                                            "Yük (desi)": round(rm["load"], 0),
                                            "Kullanılan Enerji %": round(rm["energy"], 2),
                                        })
                            else:
                                st.markdown("#### 🚛 Multi-Trip")
                                selected_routes = [mt_routes[i] for i in selected_vehicles if mt_routes and i < len(mt_routes)]
                                panel_key = "comparison_map_mt_filtered"
                                rows = []
                                for i in selected_vehicles:
                                    if mt_assignments and i < len(mt_assignments):
                                        a = mt_assignments[i]
                                        rows.append({
                                            "Görev": f"Görev {i+1}",
                                            "Müşteri": sum(len(j.get("route", [])) for j in a.get("jobs", [])),
                                            "Süre (dk)": round(a.get("time_min", 0.0), 1),
                                            "Mesafe (km)": round(a.get("distance_km", 0.0), 2),
                                            "Yük (desi)": round(a.get("load_desi", 0.0), 0),
                                            "Kullanılan Enerji %": round(a.get("energy_kwh", 0.0), 2),
                                        })

                            if selected_routes:
                                with st.spinner("Harita oluşturuluyor..."):
                                    m_panel = visualize_routes_osrm(
                                        depot_lat=DEPOT_LAT,
                                        depot_lon=DEPOT_LON,
                                        df_orders=df_orders,
                                        data=data,
                                        routing=None,
                                        manager=None,
                                        solution={"routes": selected_routes},
                                        time_dim=None,
                                        energy_dim=None,
                                        osrm_client=osrm_client,
                                        weekday=st.session_state.get("selected_weekday"),
                                    )
                                    render_folium_safe(m_panel, width=520, height=460, key=panel_key)

                            st.markdown("**Özet**")
                            if rows:
                                rdf = pd.DataFrame(rows)
                                st.dataframe(rdf, use_container_width=True)
                                totals_rows.append({
                                    "Çözüm": method_name.upper(),
                                    "Mesafe (km)": round(rdf["Mesafe (km)"].sum(), 2),
                                    "Süre (dk)": round(rdf["Süre (dk)"].sum(), 1),
                                    "Yük (desi)": round(rdf["Yük (desi)"].sum(), 0),
                                    "Kullanılan Enerji %": round(rdf["Kullanılan Enerji %"].sum(), 2),
                                })
                            else:
                                st.info("Seçili araçlarda rota yok.")

                    if totals_rows:
                        st.markdown("### 📊 Seçili Çözümler Toplam Karşılaştırma")
                        st.dataframe(pd.DataFrame(totals_rows), use_container_width=True)

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

                        render_folium_safe(m, width=1200, height=800)

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
                                "Kullanılan Enerji %": f"{total_energy:.3f}",
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
                                float(s["Kullanılan Enerji %"]) for s in vehicle_stats)

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
                    render_folium_safe(m_ga, width=1200, height=800)

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
                        "Kullanılan Enerji %": f"{total_energy:.3f}",
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
                        float(s["Kullanılan Enerji %"]) for s in vehicle_stats)

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

# =========================================================
# 7️⃣ ÇOKLU GÖREV (MULTI-TRIP) OPTİMİZASYONU
# =========================================================
with tab7:
    st.header("🚛 Multi-Trip Optimizasyonu")

    data = st.session_state.get("ortools_data")
    tabu_result = st.session_state.get("tabu_result")
    ga_routes = st.session_state.get("ga_best_routes")
    alns_routes = st.session_state.get("alns_routes")
    df_orders = st.session_state.get("orders_df")

    tabu_ran = tabu_result is not None
    has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
    has_ga = ga_routes is not None
    has_alns = alns_routes is not None
    if data is None:
        st.warning("⚠️ Önce EVRP modelini oluşturun.")
    elif not has_tabu and not has_ga and not has_alns:
        if tabu_ran:
            st.warning("⚠️ Tabu çalıştırıldı ancak geçerli çözüm bulunamadı. 6️⃣ Problem Çözümü sekmesindeki çözücü logunu kontrol edin veya araç sayısı/zaman limiti artırın.")
            if tabu_result.get("log"):
                with st.expander("Tabu Solver Log (Özet)"):
                    st.text(tabu_result.get("log", "")[:4000])
        else:
            st.warning("⚠️ Önce 6️⃣ Problem Çözümü sekmesinde Tabu, GA veya ALNS çalıştırın.")
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
            prev = depot

            for node in route:
                if node < 0 or node >= len(loads):
                    continue
                d_km = float(D[prev, node])
                t_min = float(T[prev, node])
                total_km += d_km
                total_time += t_min + float(service[node])
                from_node_desi = float(loads[prev]) if prev != depot else 0.0
                total_energy += 0.436 * d_km + 0.002 * from_node_desi
                node_load = float(loads[node])
                total_load += node_load
                prev = node

            d_km = float(D[prev, depot])
            t_min = float(T[prev, depot])
            total_km += d_km
            total_time += t_min
            from_node_desi = float(loads[prev]) if prev != depot else 0.0
            total_energy += 0.436 * d_km + 0.002 * from_node_desi

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
        if has_alns:
            source_options.append("ALNS")

        selected_source = st.selectbox(
            "Baz çözüm",
            source_options,
            key="multitrip_source_selector",
            help="Multi-trip öncesi orijinal çözüm"
        )

        if selected_source == "Tabu Search":
            all_routes = _extract_routes_from_tabu(tabu_result, int(data["num_vehicles"]), depot)
        elif selected_source == "ALNS":
            all_routes = alns_routes
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
                "Görev": f"Görev {i + 1}",
                "Müşteri": len(route),
                "Süre (dk)": round(m["time_min"], 1),
                "Mesafe (km)": round(m["distance_km"], 2),
                "Yük (desi)": round(m["load_desi"], 0),
                "Kullanılan Enerji %": round(m["energy_kwh"], 2),
            })

        if not original_jobs:
            st.info("Seçilen çözümde servis edilen rota bulunamadı.")
            st.warning("Tab 7 bu kaynakla hesaplanamadı; yine de Tab 8 kullanılabilir.")

        st.markdown("### Orijinal Çözüm (Tek Görev / Araç)")
        st.dataframe(pd.DataFrame(original_rows), use_container_width=True)

        st.markdown("### Multi-Trip Modu Seçimi")
        multitrip_mode = st.radio(
            "Görevleri nasıl işlemek istiyorsunuz?",
            options=["Rota Koruma Modu", "Görev Yeniden Atama"],
            index=0,
            help=(
                "**Rota Koruma Modu**: Araçlara atanan müşteri sırasını korur, "
                "sadece birden fazla rotayı tek araçta birleştirir. "
                "İki veya daha fazla rota bir araçta fit ise, bunları birleştirir ve "
                "toplam araç sayısını azaltır. "
                "Tercih edilen seçenek eğer atanmış rotalarınızdan memnunsanız.\n\n"
                "**Görev Yeniden Atama**: Orijinal rotaları görevler olarak ele alır "
                "ve onları araçlara yeniden atayabilir. "
                "Farklı araç kombinasyonları keşfetmek istiyorsanız kullanın."
            ),
            key="multitrip_mode_selector"
        )

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
        st.caption("⚠️ Vardiya süresi hesaplamasında, her görevin servis süresi (ürün kurulum süresi) vardiya saatlerinden çıkarılır.")

        st.markdown("#### Multi-Trip Optimizasyon Yöntemi")
        mt_optimizer = st.selectbox(
            "Yöntem",
            ["ALNS + MIP", "Tabu + MIP", "Genetik Algoritma + MIP"],
            index=0,
            key="multitrip_optimizer",
        )

        st.markdown("#### Seçili Yöntem Parametreleri")
        al1, al2, al3, al4 = st.columns(4)
        with al1:
            mt_alns_iterations = st.number_input(
                "İterasyon / Generasyon",
                min_value=50,
                max_value=5000,
                value=400,
                step=50,
                key="multitrip_alns_iterations",
            )
        with al2:
            mt_alns_destroy_rate = st.slider(
                "Destroy / Mutasyon oranı",
                min_value=0.10,
                max_value=0.60,
                value=0.25,
                step=0.05,
                key="multitrip_alns_destroy_rate",
            )
        with al3:
            mt_mip_time_limit = st.number_input(
                "MIP polishing süre (sn)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="multitrip_mip_time_limit",
            )
        with al4:
            mt_alns_seed = st.number_input(
                "Random seed",
                min_value=0,
                value=42,
                key="multitrip_alns_seed",
            )

        mt_ga_population = st.number_input(
            "GA popülasyon boyutu (sadece GA+MIP için)",
            min_value=8,
            max_value=200,
            value=30,
            step=2,
            key="multitrip_ga_population",
        )

        if st.button("🚀 Multi-Trip Optimizasyonu Çalıştır", type="primary", key="run_multitrip"):
            if multitrip_mode == "Rota Koruma Modu":
                # Route-preserving mode: merge routes into fewer vehicles
                from utils.multitrip_route_splitter import (
                    merge_routes_into_vehicles,
                )

                with st.spinner("Rotalar birleştiriliyor (müşteri sırası korunuyor)..."):
                    # Prepare data
                    D = st.session_state.get("osrm_D")
                    T = st.session_state.get("osrm_T")
                    
                    if D is None or T is None:
                        st.error("❌ OSRM matrisleri gerekli. Önce Tab 4'te matrisleri hesaplayın.")
                    else:
                        # Get constraint values
                        battery_capacity_val = float(data.get("battery_capacity", 100.0))
                        vehicle_capacity_val = float(data.get("vehicle_cap_desi", 4500))
                        
                        demand_arr = np.array(data.get("demand_desi", []))
                        service_arr = np.array(data.get("service_min", []))
                        
                        # Merge routes into fewer vehicles
                        merged_vehicles = merge_routes_into_vehicles(
                            routes=original_routes,
                            D=D,
                            T=T,
                            demand=demand_arr,
                            service_time=service_arr,
                            battery_capacity=battery_capacity_val,
                            vehicle_capacity=vehicle_capacity_val,
                            max_shift_duration=float(max_shift_duration),
                            min_return_battery_pct=float(min_return_pct),
                            depot=depot,
                        )
                        
                        # Convert to assignment format for display
                        assignments = []
                        for vehicle in merged_vehicles:
                            assignments.append({
                                "vehicle_id": vehicle["vehicle_id"],
                                "num_trips": vehicle["num_routes_merged"],
                                "original_routes": vehicle["original_route_indices"],
                                "combined_route": vehicle["combined_route"],
                                "time_min": vehicle["time_min"],
                                "distance_km": vehicle["distance_km"],
                                "energy_kwh": vehicle["energy_kwh"],
                                "load_desi": vehicle["load_desi"],
                                "trips": [{"nodes": vehicle["combined_route"]}],  # For compatibility
                            })
                        
                        # Store in session
                        st.session_state["multitrip_assignments"] = assignments
                        st.session_state["multitrip_original_jobs"] = original_jobs
                        st.session_state["multitrip_base_solution"] = selected_source
                        st.session_state["multitrip_usable_energy"] = battery_capacity_val * (1 - min_return_pct / 100)
                        st.session_state["multitrip_optimizer_used"] = "Rota Birleştirme Modu"
                        st.session_state["multitrip_mode"] = "route_preserving"
                        
                        st.success(f"✅ {len(original_routes)} rota → {len(assignments)} araca indirildi!")
                        
                        summary_lines = [
                            f"Orijinal araç: {len(original_routes)}",
                            f"Birleştirilmiş araç: {len(assignments)}",
                            f"Tasarruf: {len(original_routes) - len(assignments)} araç",
                            "---"
                        ]
                        for v in assignments:
                            summary_lines.append(
                                f"Araç {v['vehicle_id']}: {v['num_trips']} rota birleştirildi | "
                                f"{v['time_min']:.0f} dk | {v['distance_km']:.1f} km | "
                                f"{v['load_desi']:.0f} desi | {v['energy_kwh']:.1f} kWh"
                            )
                        
                        st.text_area(
                            "Özet",
                            value="\n".join(summary_lines),
                            height=150,
                            key="multitrip_summary",
                        )
            else:
                # Job reassignment mode: existing behavior
                from utils.alns_multitrip_solver import (
                    solve_multitrip_alns_mip,
                    solve_multitrip_tabu_mip,
                    solve_multitrip_ga_mip,
                )

                with st.spinner(f"{mt_optimizer} ile görevler araçlara atanıyor..."):
                    if mt_optimizer == "ALNS + MIP":
                        mt_result = solve_multitrip_alns_mip(
                            jobs=original_jobs,
                            max_shift_duration=float(max_shift_duration),
                            usable_energy=float(usable_energy),
                            depot_service_time=float(depot_service_time),
                            battery_capacity=float(battery_capacity),
                            iterations=int(mt_alns_iterations),
                            destroy_rate=float(mt_alns_destroy_rate),
                            seed=int(mt_alns_seed),
                            mip_time_limit_s=int(mt_mip_time_limit),
                        )
                    elif mt_optimizer == "Tabu + MIP":
                        mt_result = solve_multitrip_tabu_mip(
                            jobs=original_jobs,
                            max_shift_duration=float(max_shift_duration),
                            usable_energy=float(usable_energy),
                            depot_service_time=float(depot_service_time),
                            battery_capacity=float(battery_capacity),
                            iterations=int(mt_alns_iterations),
                            tabu_tenure=max(5, int(round(mt_alns_destroy_rate * 100))),
                            seed=int(mt_alns_seed),
                            mip_time_limit_s=int(mt_mip_time_limit),
                        )
                    else:
                        mt_result = solve_multitrip_ga_mip(
                            jobs=original_jobs,
                            max_shift_duration=float(max_shift_duration),
                            usable_energy=float(usable_energy),
                            depot_service_time=float(depot_service_time),
                            battery_capacity=float(battery_capacity),
                            iterations=int(mt_alns_iterations),
                            mutation_rate=float(mt_alns_destroy_rate),
                            seed=int(mt_alns_seed),
                            mip_time_limit_s=int(mt_mip_time_limit),
                            population_size=int(mt_ga_population),
                        )

                    vehicle_assignments = mt_result.get("assignments", [])
                    dropped_jobs = mt_result.get("dropped_jobs", [])

                    if dropped_jobs:
                        msg = ", ".join([f"G{j.get('job_id', '?')} (süre/enerji aşımı)" for j in dropped_jobs[:10]])
                        st.warning(
                            f"{len(dropped_jobs)} görev atanamadı: {msg}. "
                            "Bu görevler tek başına bile enerji/süre limitini aştığı için "
                            "multi-trip atamasına dahil edilmedi."
                        )

                    st.session_state["multitrip_assignments"] = vehicle_assignments
                    st.session_state["multitrip_original_jobs"] = original_jobs
                    st.session_state["multitrip_base_solution"] = selected_source
                    st.session_state["multitrip_usable_energy"] = usable_energy
                    st.session_state["multitrip_optimizer_used"] = mt_optimizer
                    st.session_state["multitrip_mode"] = "job_reassignment"

                    st.text_area(
                        f"{mt_optimizer} Log",
                        value=mt_result.get("log", ""),
                        height=180,
                        key="multitrip_alns_log",
                    )

        saved_assignments = st.session_state.get("multitrip_assignments")
        saved_source = st.session_state.get("multitrip_base_solution")
        saved_original = st.session_state.get("multitrip_original_jobs")
        multitrip_mode = st.session_state.get("multitrip_mode", "job_reassignment")

        if saved_assignments and saved_source == selected_source and saved_original:
            st.markdown("---")
            st.markdown("### Sonuç Özeti")
            used_optimizer = st.session_state.get("multitrip_optimizer_used", "ALNS + MIP")
            st.caption(f"Atama yöntemi: {used_optimizer}")

            orig_total_time = sum(x["time_min"] for x in saved_original)
            orig_total_dist = sum(x["distance_km"] for x in saved_original)
            orig_total_load = sum(x["load_desi"] for x in saved_original)
            orig_total_energy = sum(x["energy_kwh"] for x in saved_original)

            # Aggregate metrics based on mode
            if multitrip_mode == "route_preserving":
                mt_total_time = sum(x.get("time_min", 0.0) for x in saved_assignments)
                mt_total_dist = sum(x.get("distance_km", 0.0) for x in saved_assignments)
                mt_total_load = sum(x.get("load_desi", 0.0) for x in saved_assignments)
                mt_total_energy = sum(x.get("energy_kwh", 0.0) for x in saved_assignments)
            else:
                mt_total_time = sum(x.get("time_min", 0.0) for x in saved_assignments)
                mt_total_dist = sum(x.get("distance_km", 0.0) for x in saved_assignments)
                mt_total_load = sum(x.get("load_desi", 0.0) for x in saved_assignments)
                mt_total_energy = sum(x.get("energy_kwh", 0.0) for x in saved_assignments)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Araç Sayısı", len(saved_assignments), delta=f"{len(saved_original) - len(saved_assignments)}")
            with k2:
                st.metric("Toplam Süre (dk)", f"{mt_total_time:.1f}", delta=f"{(mt_total_time - orig_total_time):.1f}")
            with k3:
                st.metric("Toplam Mesafe (km)", f"{mt_total_dist:.2f}", delta=f"{(mt_total_dist - orig_total_dist):.2f}")
            with k4:
                st.metric("Toplam Kullanılan Enerji %", f"{mt_total_energy:.2f}", delta=f"{(mt_total_energy - orig_total_energy):.2f}")

            # Calculate capacity usage
            total_vehicle_capacity = len(saved_assignments) * CAPACITY_DESI
            capacity_usage_pct = (mt_total_load / total_vehicle_capacity * 100) if total_vehicle_capacity > 0 else 0

            k5, k6 = st.columns(2)
            with k5:
                st.metric("Toplam Taşınan Yük (desi)", f"{mt_total_load:.0f}", delta=f"{(mt_total_load - orig_total_load):.0f}")
            with k6:
                st.metric("Kapasite Kullanımı", f"{capacity_usage_pct:.1f}%")

            st.markdown("### Multi-Trip Araç / Görev Atamaları")
            assignment_rows = []
            multitrip_mode = st.session_state.get("multitrip_mode", "job_reassignment")
            
            for v in saved_assignments:
                if multitrip_mode == "route_preserving":
                    # Route-preserving (merge) format
                    original_routes = v.get('original_routes', [])
                    routes_str = ", ".join([f"Rota {r+1}" for r in original_routes])
                    assignment_rows.append({
                        "Araç": f"Araç {v.get('vehicle_id', '?')}",
                        "Birleştirilmiş Rotalar": routes_str if routes_str else "-",
                        "Rota Sayısı": v.get('num_trips', 0),
                        "Süre (dk)": round(v.get("time_min", 0.0), 1),
                        "Mesafe (km)": round(v.get("distance_km", 0.0), 2),
                        "Kullanılan Enerji (kWh)": round(v.get("energy_kwh", 0.0), 2),
                        "Taşınan Yük (desi)": round(v.get("load_desi", 0.0), 0),
                    })
                else:
                    # Job reassignment format
                    job_ids = [f"G{j.get('job_id', '?')}" for j in v.get("jobs", [])]
                    assignment_rows.append({
                        "Araç": f"Araç {v.get('vehicle_id', '?')}",
                        "Görevler": ", ".join(job_ids) if job_ids else "-",
                        "Süre (dk)": round(v.get("time_min", 0.0), 1),
                        "Kullanılan Enerji %": round(v.get("energy_kwh", 0.0), 2),
                        "Toplam taşınan desi": round(v.get("load_desi", 0.0), 0),
                    })
            st.dataframe(pd.DataFrame(assignment_rows), use_container_width=True)

            st.markdown("### Harita Karşılaştırması")
            osrm_client = st.session_state.get("osrm_client")

            if osrm_client is None or df_orders is None:
                st.info("Harita için gerekli veri eksik (orders/osrm_client).")
            else:
                original_vehicle_options = [f"Görev {i + 1}" for i in range(len(saved_original))]
                multitrip_vehicle_options = [f"Araç {v['vehicle_id']}" for v in saved_assignments]

                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                with btn_col1:
                    if st.button("Orijinal: Tümünü Seç", key="multitrip_original_select_all"):
                        st.session_state["multitrip_original_vehicle_filter"] = original_vehicle_options
                with btn_col2:
                    if st.button("Orijinal: Tümünü Kaldır", key="multitrip_original_clear_all"):
                        st.session_state["multitrip_original_vehicle_filter"] = []
                with btn_col3:
                    if st.button("Multi-Trip: Tümünü Seç", key="multitrip_select_all"):
                        st.session_state["multitrip_vehicle_filter"] = multitrip_vehicle_options
                with btn_col4:
                    if st.button("Multi-Trip: Tümünü Kaldır", key="multitrip_clear_all"):
                        st.session_state["multitrip_vehicle_filter"] = []

                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    selected_original_vehicles = st.multiselect(
                        "Orijinal çözüm araç filtresi",
                        options=original_vehicle_options,
                        default=original_vehicle_options,
                        key="multitrip_original_vehicle_filter",
                    )
                with filter_col2:
                    selected_multitrip_vehicles = st.multiselect(
                        "Multi-Trip çözüm araç filtresi",
                        options=multitrip_vehicle_options,
                        default=multitrip_vehicle_options,
                        key="multitrip_vehicle_filter",
                    )

                map_col1, map_col2 = st.columns(2)

                selected_original_ids = {
                    int(label.replace("Görev ", ""))
                    for label in selected_original_vehicles
                }
                original_routes_for_map = [
                    j["route"]
                    for idx, j in enumerate(saved_original, start=1)
                    if idx in selected_original_ids and j.get("route")
                ]

                multitrip_routes_for_map = []
                multitrip_labels = []
                for v in saved_assignments:
                    vehicle_label = f"Araç {v['vehicle_id']}"
                    if vehicle_label not in selected_multitrip_vehicles:
                        continue
                    
                    combined_route = []
                    
                    if multitrip_mode == "route_preserving":
                        # Route-preserving (merge) mode: use combined_route directly
                        route = v.get("combined_route", [])
                        if not route:
                            continue
                        combined_route = route
                    else:
                        # Job reassignment mode: combine jobs with depot
                        jobs = v.get("jobs", [])
                        if not jobs:
                            continue
                        for i, j in enumerate(jobs):
                            combined_route.extend(j["route"])
                            if i < len(jobs) - 1:
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
                        render_folium_safe(m_original, width=550, height=500, key="multitrip_original_map_readded")
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
                        render_folium_safe(m_multi, width=550, height=500, key="multitrip_optimized_map_readded")
                    else:
                        st.info("Multi-trip rota bulunamadı.")


# =========================================================
# 8️⃣ SONUÇ GÖSTERİMİ MATRİSİ
# =========================================================
with tab8:
    st.header("8) Sonuç Gösterimi")

    data = st.session_state.get("ortools_data")
    tabu_result = st.session_state.get("tabu_result")
    ga_routes = st.session_state.get("ga_best_routes")
    alns_routes = st.session_state.get("alns_routes")
    if data is None:
        st.warning("Önce EVRP modelini oluşturun.")
    else:
        D = np.array(data["distance_km"], dtype=float)
        T = np.array(data["time_min"], dtype=float)
        loads = np.array(data["demand_desi"], dtype=float)
        service = np.array(data.get("service_min", np.zeros(len(loads))), dtype=float)
        depot = int(data.get("depot", 0))
        battery_capacity = float(data.get("battery_capacity", 100.0))

        def _extract_routes_from_tabu_result(result, num_vehicles, dep):
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
            prev = depot

            for node in route:
                if node < 0 or node >= len(loads):
                    continue
                d_km = float(D[prev, node])
                t_min = float(T[prev, node])
                total_km += d_km
                total_time += t_min + float(service[node])
                from_node_desi = float(loads[prev]) if prev != depot else 0.0
                total_energy += 0.436 * d_km + 0.002 * from_node_desi
                total_load += float(loads[node])
                prev = node

            d_km = float(D[prev, depot])
            t_min = float(T[prev, depot])
            total_km += d_km
            total_time += t_min
            from_node_desi = float(loads[prev]) if prev != depot else 0.0
            total_energy += 0.436 * d_km + 0.002 * from_node_desi

            return {
                "distance_km": total_km,
                "time_min": total_time,
                "load_desi": total_load,
                "energy_kwh": total_energy,
            }

        one_trip_sources = {}

        has_tabu = tabu_result is not None and tabu_result.get("solution") is not None
        if has_tabu:
            one_trip_sources["Tabu"] = _extract_routes_from_tabu_result(
                tabu_result,
                int(data["num_vehicles"]),
                depot,
            )

        if ga_routes is not None:
            one_trip_sources["GA"] = ga_routes

        if alns_routes is not None:
            one_trip_sources["ALNS"] = alns_routes

        if not one_trip_sources:
            st.warning("Önce Tabu, GA veya ALNS tek-tur çözümü çalıştırın.")
        else:
            st.markdown("### Matris Parametreleri")
            p1, p2, p3 = st.columns(3)
            with p1:
                max_shift_duration = st.number_input(
                    "Maksimum vardiya süresi (dk)",
                    min_value=240,
                    max_value=720,
                    value=int(st.session_state.get("multitrip_max_shift", 540)),
                    step=30,
                    key="tab8_max_shift",
                )
            with p2:
                depot_service_time = st.number_input(
                    "Depo servis süresi (dk)",
                    min_value=0,
                    max_value=60,
                    value=int(st.session_state.get("multitrip_depot_service", 15)),
                    step=5,
                    key="tab8_depot_service",
                )
            with p3:
                min_return_pct = st.number_input(
                    "Minimum dönüş şarjı (%)",
                    min_value=0,
                    max_value=90,
                    value=int(st.session_state.get("multitrip_min_return_pct", 20)),
                    step=5,
                    key="tab8_min_return_pct",
                )

            q1, q2, q3, q4 = st.columns(4)
            with q1:
                mt_iterations = st.number_input(
                    "İterasyon / Generasyon",
                    min_value=50,
                    max_value=5000,
                    value=400,
                    step=50,
                    key="tab8_iterations",
                )
            with q2:
                mt_rate = st.slider(
                    "Destroy / Mutasyon oranı",
                    min_value=0.10,
                    max_value=0.60,
                    value=0.25,
                    step=0.05,
                    key="tab8_rate",
                )
            with q3:
                mt_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=42,
                    key="tab8_seed",
                )
            with q4:
                mt_mip_time = st.number_input(
                    "MIP polishing süre (sn)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    key="tab8_mip_time",
                )

            ga_population = st.number_input(
                "GA popülasyon boyutu (GA+MIP)",
                min_value=8,
                max_value=200,
                value=30,
                step=2,
                key="tab8_ga_population",
            )

            usable_energy = battery_capacity * (1.0 - float(min_return_pct) / 100.0)
            st.caption(
                f"Toplam batarya: {battery_capacity:.2f} kWh | Kullanılabilir enerji: {usable_energy:.2f} kWh"
            )

            if st.button("🧮 3x3 Sonuç Matrisini Hesapla", key="tab8_run_matrix", type="primary"):
                try:
                    # Check if we have one-trip solutions
                    st.info("🔍 Tab 7 çözümleri kontrol ediliyor...")
                    
                    if not one_trip_sources:
                        st.error("❌ Hata: Bir-tur çözümleri bulunamadı. Lütfen önce Tab 7'de en az bir optimizer (Tabu, GA veya ALNS) çalıştırın.")
                        st.stop()
                    
                    available_sources = list(one_trip_sources.keys())
                    st.success(f"✅ Bulunan bir-tur çözümleri: {', '.join(available_sources)}")

                    from utils.alns_multitrip_solver import (
                        solve_multitrip_alns_mip,
                        solve_multitrip_tabu_mip,
                        solve_multitrip_ga_mip,
                    )

                    optimizer_names = ["ALNS+MIP", "Tabu+MIP", "GA+MIP"]
                    matrix_data = {
                        src_name: {opt_name: "-" for opt_name in optimizer_names}
                        for src_name in ["Tabu", "GA", "ALNS"]
                    }
                    detail_rows = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_steps = len(one_trip_sources) * len(optimizer_names)
                    current_step = 0

                    for src_name, routes in one_trip_sources.items():
                        valid_routes = [r for r in routes if r]
                        if not valid_routes:
                            st.warning(f"⚠️ {src_name}: Geçerli rota yok, atlanıyor...")
                            continue

                        status_text.text(f"🔄 {src_name} çözümü işleniyor...")
                        st.info(f"📊 {src_name}: {len(valid_routes)} araç ile çalışılıyor")

                        ot_runtime_map = {
                            "Tabu": st.session_state.get("tabu_runtime_s"),
                            "GA": st.session_state.get("ga_runtime_s"),
                            "ALNS": st.session_state.get("alns_runtime_s"),
                        }
                        one_trip_runtime = ot_runtime_map.get(src_name)

                        one_jobs = []
                        for i, route in enumerate(valid_routes, start=1):
                            m = _route_metrics(route)
                            one_jobs.append(
                                {
                                    "job_id": i,
                                    "route": route,
                                    "time_min": m["time_min"],
                                    "distance_km": m["distance_km"],
                                    "load_desi": m["load_desi"],
                                    "energy_kwh": m["energy_kwh"],
                                }
                            )

                        one_trip_vehicles = len(valid_routes)
                        one_trip_energy = sum(j["energy_kwh"] for j in one_jobs)

                        import time
                        
                        # ALNS+MIP
                        try:
                            status_text.text(f"🔄 {src_name} + ALNS+MIP çalışıyor...")
                            mt_start = time.time()
                            res_alns = solve_multitrip_alns_mip(
                                jobs=one_jobs,
                                max_shift_duration=float(max_shift_duration),
                                usable_energy=float(usable_energy),
                                depot_service_time=float(depot_service_time),
                                battery_capacity=float(battery_capacity),
                                iterations=int(mt_iterations),
                                destroy_rate=float(mt_rate),
                                seed=int(mt_seed),
                                mip_time_limit_s=int(mt_mip_time),
                            )
                            rt_alns = time.time() - mt_start
                            current_step += 1
                            progress_bar.progress(min(current_step / total_steps, 1.0))
                        except Exception as e:
                            st.error(f"❌ {src_name} + ALNS+MIP hatası: {str(e)}")
                            rt_alns = 0
                            res_alns = {"assignments": []}

                        # Tabu+MIP
                        try:
                            status_text.text(f"🔄 {src_name} + Tabu+MIP çalışıyor...")
                            mt_start = time.time()
                            res_tabu = solve_multitrip_tabu_mip(
                                jobs=one_jobs,
                                max_shift_duration=float(max_shift_duration),
                                usable_energy=float(usable_energy),
                                depot_service_time=float(depot_service_time),
                                battery_capacity=float(battery_capacity),
                                iterations=int(mt_iterations),
                                tabu_tenure=max(5, int(round(float(mt_rate) * 100))),
                                seed=int(mt_seed),
                                mip_time_limit_s=int(mt_mip_time),
                            )
                            rt_tabu = time.time() - mt_start
                            current_step += 1
                            progress_bar.progress(min(current_step / total_steps, 1.0))
                        except Exception as e:
                            st.error(f"❌ {src_name} + Tabu+MIP hatası: {str(e)}")
                            rt_tabu = 0
                            res_tabu = {"assignments": []}

                        # GA+MIP
                        try:
                            status_text.text(f"🔄 {src_name} + GA+MIP çalışıyor...")
                            mt_start = time.time()
                            res_ga = solve_multitrip_ga_mip(
                                jobs=one_jobs,
                                max_shift_duration=float(max_shift_duration),
                                usable_energy=float(usable_energy),
                                depot_service_time=float(depot_service_time),
                                battery_capacity=float(battery_capacity),
                                iterations=int(mt_iterations),
                                mutation_rate=float(mt_rate),
                                seed=int(mt_seed),
                                mip_time_limit_s=int(mt_mip_time),
                                population_size=int(ga_population),
                            )
                            rt_ga = time.time() - mt_start
                            current_step += 1
                            progress_bar.progress(min(current_step / total_steps, 1.0))
                        except Exception as e:
                            st.error(f"❌ {src_name} + GA+MIP hatası: {str(e)}")
                            rt_ga = 0
                            res_ga = {"assignments": []}

                        results = {
                            "ALNS+MIP": (res_alns, rt_alns),
                            "Tabu+MIP": (res_tabu, rt_tabu),
                            "GA+MIP": (res_ga, rt_ga),
                        }

                        for opt_name, packed in results.items():
                            res, mt_runtime = packed
                            assignments = res.get("assignments", [])
                            if not assignments:
                                st.warning(f"⚠️ {src_name} + {opt_name}: Atama yok")
                                continue

                            mt_vehicles = len(assignments)
                            mt_energy = sum(float(a.get("energy_kwh", 0.0)) for a in assignments)
                            delta_vehicles = mt_vehicles - one_trip_vehicles
                            delta_energy = mt_energy - one_trip_energy
                            ot_runtime_text = f"{float(one_trip_runtime):.1f}" if one_trip_runtime is not None else "-"

                            matrix_data[src_name][opt_name] = (
                                f"OT: {one_trip_vehicles} araç | {one_trip_energy:.2f} kWh\n"
                                f"MT: {mt_vehicles} araç | {mt_energy:.2f} kWh\n"
                                f"OT Süre: {ot_runtime_text} sn | MT Süre: {mt_runtime:.1f} sn\n"
                                f"ΔAraç: {delta_vehicles:+d} | ΔEnerji: {delta_energy:+.2f} kWh"
                            )

                            detail_rows.append(
                                {
                                    "One-Trip": src_name,
                                    "Multi-Trip": opt_name,
                                    "OT Araç": one_trip_vehicles,
                                    "OT Kullanılan Enerji %": round(one_trip_energy, 2),
                                    "OT Süre (sn)": round(float(one_trip_runtime), 1) if one_trip_runtime is not None else None,
                                    "MT Araç": mt_vehicles,
                                    "MT Kullanılan Enerji %": round(mt_energy, 2),
                                    "MT Süre (sn)": round(mt_runtime, 1),
                                    "Toplam Süre (sn)": round((float(one_trip_runtime) if one_trip_runtime is not None else 0.0) + mt_runtime, 1),
                                    "ΔAraç": delta_vehicles,
                                    "ΔKullanılan Enerji %": round(delta_energy, 2),
                                }
                            )

                    status_text.text("✅ İşlem tamamlandı!")
                    progress_bar.progress(1.0)

                    if not detail_rows:
                        st.error("❌ Sonuç bulunamadı. Lütfen solver parametrelerini kontrol edin.")
                    else:
                        st.success(f"✅ {len(detail_rows)} başarılı kombinasyon bulundu!")

                    st.session_state["tab8_matrix_data"] = matrix_data
                    st.session_state["tab8_matrix_details"] = detail_rows

                except Exception as e:
                    st.error(f"❌ Beklenmeyen hata: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

            matrix_data = st.session_state.get("tab8_matrix_data")
            detail_rows = st.session_state.get("tab8_matrix_details")

            if matrix_data:
                st.markdown("### One-Trip vs Multi-Trip 3x3 Matris")
                matrix_df = pd.DataFrame.from_dict(matrix_data, orient="index")
                matrix_df.index.name = "One-Trip"
                st.dataframe(matrix_df, use_container_width=True)

            if detail_rows:
                st.markdown("### Detay Tablosu (Sadece Çalışan Kombinasyonlar)")
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


with tab9:
    render_gasoline_ga_tab()
