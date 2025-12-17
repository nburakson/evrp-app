# utils/energy_comparator.py

import numpy as np
from dataclasses import dataclass


# ============================================================
# UNIVERSAL ENERGY FORMULA  (Wh)
# ============================================================

def energy_leg_wh(d_km: float, load_before: float) -> float:
    """
    Universal EVRP energy formula:
       E_ij (Wh) = d_ij * (0.436 + 0.002 * load_before)
    SAME formula for OR-Tools and GA.
    """
    return d_km * (0.436 + 0.002 * load_before)



# ============================================================
# PER-ROUTE ENERGY COMPUTATION
# ============================================================

@dataclass
class LegEnergy:
    start: int
    end: int
    distance_km: float
    load_before: float
    energy_wh: float
    cumulative_wh: float


@dataclass
class RouteEnergyReport:
    vehicle_id: int
    route: list
    legs: list  # List[LegEnergy]
    total_wh: float
    total_kwh: float



def compute_route_energy(route, D, loads, depot=0):
    """
    Compute energy for one route using the universal OR-Tools/GA formula.
    Returns a RouteEnergyReport object.
    """
    if not route:
        return RouteEnergyReport(0, [], [], 0.0, 0.0)

    legs = []
    total_wh = 0.0
    current_load = 0.0

    # depot → first
    prev = depot
    for node in route:
        d = float(D[prev, node])
        e = energy_leg_wh(d, current_load)
        total_wh += e

        legs.append(LegEnergy(
            start=prev,
            end=node,
            distance_km=d,
            load_before=current_load,
            energy_wh=e,
            cumulative_wh=total_wh
        ))

        current_load += loads[node]
        prev = node

    # final → depot
    d = float(D[prev, depot])
    e = energy_leg_wh(d, current_load)
    total_wh += e

    legs.append(LegEnergy(
        start=prev,
        end=depot,
        distance_km=d,
        load_before=current_load,
        energy_wh=e,
        cumulative_wh=total_wh
    ))

    return RouteEnergyReport(
        vehicle_id=0,
        route=route,
        legs=legs,
        total_wh=total_wh,
        total_kwh=total_wh 
    )



# ============================================================
# FLEET ENERGY
# ============================================================

def compute_fleet_energy(routes, D, loads):
    """
    Compute energy for all vehicles.
    """
    reports = []
    total = 0.0

    for v, route in enumerate(routes):
        rep = compute_route_energy(route, D, loads)
        rep.vehicle_id = v
        reports.append(rep)
        total += rep.total_wh

    return reports, total, total / 1000.0



# ============================================================
# OR-TOOLS vs GA COMPARISON
# ============================================================

def compare_ortools_vs_ga(ortools_routes, ga_routes, data):
    """
    Compares total and per-vehicle energy (kWh) between OR-Tools and GA.
    Returns comparison structure for Streamlit/UI.
    """

    D = np.array(data["distance_km"], dtype=float)
    loads = np.array(data["demand_desi"], dtype=float)

    # OR-Tools fleet
    ort_reports, ort_total_wh, ort_total_kwh = compute_fleet_energy(
        ortools_routes, D, loads
    )

    # GA fleet
    ga_reports, ga_total_wh, ga_total_kwh = compute_fleet_energy(
        ga_routes, D, loads
    )

    improvement = 0.0
    if ort_total_kwh > 0:
        improvement = (ort_total_kwh - ga_total_kwh) / ort_total_kwh * 100.0

    return {
        "ortools_total_kwh": ort_total_kwh,
        "ga_total_kwh": ga_total_kwh,
        "improvement_percent": improvement,
        "ortools_vehicle_reports": ort_reports,
        "ga_vehicle_reports": ga_reports,
    }


# ============================================================
# TEXT OUTPUT (pretty formatting)
# ============================================================

def format_route_report(report: RouteEnergyReport):
    """
    Produces a pretty OR-Tools style text block for a single vehicle.
    """
    lines = []
    lines.append(f"🚚 Vehicle {report.vehicle_id} route: {report.route}")
    lines.append("From→To | Dist(km) | Load | Energy(Wh) | Cum(Wh) | Cum(kWh)")
    lines.append("-" * 70)

    for L in report.legs:
        lines.append(
            f"{L.start:>2}→{L.end:<2} | "
            f"{L.distance_km:7.2f} | "
            f"{L.load_before:4.0f} | "
            f"{L.energy_wh:10.2f} | "
            f"{L.cumulative_wh:10.2f} | "
            f"{L.cumulative_wh:7.3f}"
        )

    lines.append("-" * 70)
    lines.append(f"TOTAL: {report.total_kwh:.3f} kWh")
    return "\n".join(lines)


def format_fleet_comparison(result):
    """
    Pretty final comparison text.
    """
    lines = []
    lines.append("============== ENERGY COMPARISON ==============")
    lines.append(f"OR-Tools total energy: {result['ortools_total_kwh']:.3f} kWh")
    lines.append(f"GA total energy:       {result['ga_total_kwh']:.3f} kWh")
    lines.append(f"Improvement:           {result['improvement_percent']:.2f}%")
    lines.append("================================================")
    return "\n".join(lines)
