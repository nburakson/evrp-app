from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# =========================
# 🔧 GLOBAL VEHICLE CONSTANTS
# =========================
CAPACITY_DESI = 4500                 # max load (desi)
BATTERY_CAPACITY = 100                     # full battery energy
ENERGY_A = 0.436              # empty vehicle kWh per km
ENERGY_B = 0.00202         # load energy penalty

# convert for EVRPProblem model (which uses per 100km)
BASE_KWH_PER_100KM = ENERGY_A * 100   # = 43.6


# =========================
# 1) DATA CLASSES
# =========================

@dataclass
class Visit:
    index: int
    order_id: int
    lat: float
    lon: float
    desi: float
    service_min: float


@dataclass
class Vehicle:
    id: int
    capacity_desi: float
    battery_capacity: float


@dataclass
class EVRPProblem:
    depot_index: int
    visits: List[Visit]
    vehicles: List[Vehicle]
    D: np.ndarray
    # static time matrix used by OR-Tools (chosen planning hour)
    T: np.ndarray
    # optional full dynamic structure: hour -> time matrix
    T_by_hour: Optional[Dict[int, np.ndarray]]
    energy_A: float   # per km
    BASE_KWH_PER_100KM: float # per 100
    energy_B: float   # per desi
    battery_capacity: float



# =========================
# 2) BUILD EVRP PROBLEM
# =========================

def build_evrp_problem_from_globals(
    df_orders: pd.DataFrame,
    D: np.ndarray,
    T: Optional[np.ndarray],
    num_vehicles: int,
    depot_index: int = 0,
    T_by_hour: Optional[Dict[int, np.ndarray]] = None,
    planning_hour: int = 9,
) -> EVRPProblem:
    """
    Build EVRPProblem using GLOBAL constant parameters.

    Modes:
      • Static (old behaviour):
            D, T given; T_by_hour=None
      • Dynamic by hour:
            D given; T=None; T_by_hour={hour: matrix}, planning_hour selects
            which matrix is exposed as problem.T / data["time_min"].
    """
    n_orders = len(df_orders)
    expected = n_orders + 1  # depot + orders

    if D.shape != (expected, expected):
        raise ValueError(f"Distance matrix shape must be {(expected, expected)}, got {D.shape}")

    # Choose which T to expose as the "solver" matrix
    if T_by_hour is not None:
        if planning_hour not in T_by_hour:
            raise ValueError(
                f"planning_hour={planning_hour} not in T_by_hour keys={list(T_by_hour.keys())}"
            )
        T_planning = np.array(T_by_hour[planning_hour], dtype=float)
    else:
        if T is None:
            raise ValueError("Either T (static) or T_by_hour (dynamic) must be provided.")
        T_planning = np.array(T, dtype=float)

    if T_planning.shape != (expected, expected):
        raise ValueError(
            f"Duration matrix shape must be {(expected, expected)}, got {T_planning.shape}"
        )

    # -------- Build visits (1..n) --------
    visits: List[Visit] = []
    for _, row in df_orders.iterrows():
        matrix_idx = len(visits) + 1   # 1..n
        visits.append(
            Visit(
                index=matrix_idx,
                order_id=int(row["OrderID"]),
                lat=float(row["Enlem"]),
                lon=float(row["Boylam"]),
                desi=float(row["Desi"]),
                service_min=float(row["Servis Süresi (dk)"]),
            )
        )

    # -------- Build identical vehicles --------
    vehicles = [
        Vehicle(
            id=v,
            capacity_desi=CAPACITY_DESI,
            battery_capacity=BATTERY_CAPACITY,
        )
        for v in range(num_vehicles)
    ]

    problem = EVRPProblem(
        depot_index=depot_index,
        visits=visits,
        vehicles=vehicles,
        D=np.array(D, dtype=float),
        T=T_planning,
        T_by_hour=T_by_hour,
        energy_A=ENERGY_A,
        BASE_KWH_PER_100KM=BASE_KWH_PER_100KM,
        energy_B=ENERGY_B,
        battery_capacity=BATTERY_CAPACITY
    )
    return problem


# =========================
# 3) OR-TOOLS COMPATIBLE DATA
# =========================

def build_ortools_data(problem: EVRPProblem) -> dict:
    """Convert EVRPProblem → OR-Tools data dictionary."""
    n = problem.D.shape[0]

    demand = np.zeros(n)
    service = np.zeros(n)

    for v in problem.visits:
        demand[v.index] = v.desi
        service[v.index] = v.service_min

    vehicle_cap = problem.vehicles[0].capacity_desi
    battery = problem.vehicles[0].battery_capacity

    data = {
        "num_vehicles": len(problem.vehicles),
        "depot": problem.depot_index,

        "distance_km": problem.D,
        "time_min": problem.T,  # this is T at planning_hour

        "demand_desi": demand,
        "service_min": service,

        "vehicle_cap_desi": vehicle_cap,
        "battery_capacity": battery,

        "base_kwh_per_100km": problem.BASE_KWH_PER_100KM,
        "energy_per_desi_km": problem.energy_B,
    }

    # also expose full dynamic time matrices for inspection / GA / etc.
    if problem.T_by_hour is not None:
        data["time_min_by_hour"] = problem.T_by_hour

    return data


# =========================
# 4) ONE-SHOT BUILDER
# =========================

def build_problem_and_data_from_globals(
    df_orders: pd.DataFrame,
    D: np.ndarray,
    T: Optional[np.ndarray],
    num_vehicles: int,
    depot_index: int = 0,
    T_by_hour: Optional[Dict[int, np.ndarray]] = None,
    planning_hour: int = 9,
):
    """
    Convenience wrapper using global vehicle & energy constants.

    Old usage (still valid):
        problem, data = build_problem_and_data_from_globals(df, D, T, num_vehicles)

    New dynamic usage (with traffic-based T_by_hour):
        problem, data = build_problem_and_data_from_globals(
            df, D, T=None,
            num_vehicles=num_vehicles,
            T_by_hour=T_by_hour,
            planning_hour=9,
        )
    """
    problem = build_evrp_problem_from_globals(
        df_orders=df_orders,
        D=D,
        T=T,
        num_vehicles=num_vehicles,
        depot_index=depot_index,
        T_by_hour=T_by_hour,
        planning_hour=planning_hour,
    )
    data = build_ortools_data(problem)
    return problem, data
