"""
Single-trip ALNS solver for EVRP.

Builds customer routes directly (like Tabu/GA) under the same core constraints:
- Vehicle capacity
- Shift duration (09:00-18:00)
- Usable battery energy with reserve policy

This solver does not perform multi-trip packing. It returns one route per vehicle.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

WORK_START_MIN = 9 * 60
WORK_END_MIN = 18 * 60

BASE_KWH_PER_KM = 0.436
LOAD_TERM = 0.002


def _route_metrics(route: List[int], data: dict) -> Tuple[float, float, float, float]:
    """Return (distance_km, time_min, load_desi, energy_kwh) for a route."""
    if not route:
        return 0.0, 0.0, 0.0, 0.0

    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    service = np.array(data.get("service_min", np.zeros(D.shape[0])), dtype=float)
    demands = np.array(data["demand_desi"], dtype=float)
    depot = int(data["depot"])

    total_km = 0.0
    total_time = 0.0
    total_load = 0.0
    total_energy = 0.0

    prev = depot
    for node in route:
        d_km = float(D[prev, node])
        t_min = float(T[prev, node]) + float(service[node])
        from_node_desi = float(demands[prev]) if prev != depot else 0.0
        e_kwh = BASE_KWH_PER_KM * d_km + LOAD_TERM * from_node_desi

        total_km += d_km
        total_time += t_min
        total_energy += e_kwh
        total_load += float(demands[node])
        prev = node

    d_km = float(D[prev, depot])
    t_min = float(T[prev, depot])
    from_node_desi = float(demands[prev]) if prev != depot else 0.0
    e_kwh = BASE_KWH_PER_KM * d_km + LOAD_TERM * from_node_desi

    total_km += d_km
    total_time += t_min
    total_energy += e_kwh

    return total_km, total_time, total_load, total_energy


def _is_feasible_route(route: List[int], data: dict) -> bool:
    if not route:
        return True

    _, total_time, total_load, total_energy = _route_metrics(route, data)

    cap = float(data["vehicle_cap_desi"])
    battery_capacity = float(data.get("battery_capacity", data.get("battery_kwh", 100.0)))
    reserve_soc_pct = float(data.get("min_return_soc_pct", 0.0))
    usable_battery = battery_capacity * (1.0 - reserve_soc_pct / 100.0)
    max_shift = float(WORK_END_MIN - WORK_START_MIN)

    return (total_load <= cap + 1e-9) and (total_time <= max_shift + 1e-9) and (total_energy <= usable_battery + 1e-9)


def _total_cost(routes: List[List[int]], data: dict, objective: str = "distance") -> float:
    penalty_unserved = 1_000_000.0
    n = int(np.array(data["distance_km"]).shape[0])
    all_customers = set(range(1, n))
    served = {node for r in routes for node in r}
    unserved_count = len(all_customers - served)

    used_vehicles = sum(1 for r in routes if r)

    total_distance = 0.0
    total_energy = 0.0
    for r in routes:
        d, _, _, e = _route_metrics(r, data)
        total_distance += d
        total_energy += e

    main_cost = total_energy if objective == "energy" else total_distance
    return penalty_unserved * unserved_count + 1_000.0 * used_vehicles + main_cost


def _best_insertion(
    routes: List[List[int]],
    customer: int,
    data: dict,
    objective: str,
) -> Tuple[int, int] | None:
    best: Tuple[int, int] | None = None
    best_delta = float("inf")

    base_cost = _total_cost(routes, data, objective=objective)

    for vidx in range(len(routes)):
        route = routes[vidx]
        for pos in range(len(route) + 1):
            cand_routes = [list(r) for r in routes]
            cand_routes[vidx] = list(route)
            cand_routes[vidx].insert(pos, customer)

            if not _is_feasible_route(cand_routes[vidx], data):
                continue

            c = _total_cost(cand_routes, data, objective=objective)
            delta = c - base_cost
            if delta < best_delta:
                best_delta = delta
                best = (vidx, pos)

    return best


def solve_with_alns_singletrip(
    data: dict,
    iterations: int = 500,
    destroy_rate: float = 0.2,
    seed: int = 42,
    objective: str = "distance",
    candidate_vehicles: int = 6,
    max_insert_positions: int = 8,
) -> Dict[str, object]:
    """
    ALNS for single-trip route construction.

    Returns:
        {
          "routes": List[List[int]],
          "served_customers": int,
          "unserved_customers": List[int],
          "log": str,
          "objective": str,
        }
    """
    rng = random.Random(int(seed))
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    service = np.array(data.get("service_min", np.zeros(D.shape[0])), dtype=float)
    demands = np.array(data["demand_desi"], dtype=float)
    depot = int(data["depot"])

    cap = float(data["vehicle_cap_desi"])
    battery_capacity = float(data.get("battery_capacity", data.get("battery_kwh", 100.0)))
    reserve_soc_pct = float(data.get("min_return_soc_pct", 0.0))
    usable_battery = battery_capacity * (1.0 - reserve_soc_pct / 100.0)
    max_shift = float(WORK_END_MIN - WORK_START_MIN)

    n = int(D.shape[0])
    customers = list(range(1, n))
    num_vehicles = int(data["num_vehicles"])

    cand_veh = max(1, int(candidate_vehicles))
    pos_limit = max(3, int(max_insert_positions))

    route_cache: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {}

    def _route_metrics_fast(route: List[int]) -> Tuple[float, float, float, float]:
        if not route:
            return 0.0, 0.0, 0.0, 0.0

        key = tuple(route)
        if key in route_cache:
            return route_cache[key]

        total_km = 0.0
        total_time = 0.0
        total_load = 0.0
        total_energy = 0.0

        prev = depot
        for node in route:
            d_km = float(D[prev, node])
            t_min = float(T[prev, node]) + float(service[node])
            from_node_desi = float(demands[prev]) if prev != depot else 0.0
            e_kwh = BASE_KWH_PER_KM * d_km + LOAD_TERM * from_node_desi

            total_km += d_km
            total_time += t_min
            total_energy += e_kwh
            total_load += float(demands[node])
            prev = node

        d_km = float(D[prev, depot])
        t_min = float(T[prev, depot])
        from_node_desi = float(demands[prev]) if prev != depot else 0.0
        e_kwh = BASE_KWH_PER_KM * d_km + LOAD_TERM * from_node_desi

        total_km += d_km
        total_time += t_min
        total_energy += e_kwh

        out = (total_km, total_time, total_load, total_energy)
        route_cache[key] = out
        return out

    def _is_feasible_fast(route: List[int]) -> bool:
        if not route:
            return True
        _, total_time, total_load, total_energy = _route_metrics_fast(route)
        return (total_load <= cap + 1e-9) and (total_time <= max_shift + 1e-9) and (total_energy <= usable_battery + 1e-9)

    def _plan_cost_fast(routes: List[List[int]]) -> float:
        penalty_unserved = 1_000_000.0
        all_customers = set(range(1, n))
        served = {node for r in routes for node in r}
        unserved_count = len(all_customers - served)
        used_vehicles = sum(1 for r in routes if r)

        total_distance = 0.0
        total_energy = 0.0
        for r in routes:
            d, _, _, e = _route_metrics_fast(r)
            total_distance += d
            total_energy += e

        main_cost = total_energy if objective == "energy" else total_distance
        return penalty_unserved * unserved_count + 1_000.0 * used_vehicles + main_cost

    def _candidate_positions(route_len: int) -> List[int]:
        if route_len <= pos_limit:
            return list(range(route_len + 1))
        step = max(1, route_len // (pos_limit - 1))
        pos = list(range(0, route_len + 1, step))
        if route_len not in pos:
            pos.append(route_len)
        return sorted(set(pos))

    def _best_insertion_fast(routes: List[List[int]], customer: int) -> Tuple[int, int] | None:
        best: Tuple[int, int] | None = None
        best_delta = float("inf")
        base_cost = _plan_cost_fast(routes)

        veh_idx_sorted = sorted(
            range(len(routes)),
            key=lambda vidx: len(routes[vidx])
        )[: min(len(routes), cand_veh)]

        for vidx in veh_idx_sorted:
            route = routes[vidx]
            for pos in _candidate_positions(len(route)):
                cand_routes = [list(r) for r in routes]
                cand_routes[vidx] = list(route)
                cand_routes[vidx].insert(pos, customer)

                if not _is_feasible_fast(cand_routes[vidx]):
                    continue

                c = _plan_cost_fast(cand_routes)
                delta = c - base_cost
                if delta < best_delta:
                    best_delta = delta
                    best = (vidx, pos)

        return best

    log_lines: List[str] = []
    log_lines.append("ALNS single-trip çözümü başlatıldı.")
    log_lines.append(f"Müşteri sayısı: {len(customers)} | Araç sayısı: {num_vehicles}")

    # Greedy initial construction
    current_routes: List[List[int]] = [[] for _ in range(num_vehicles)]
    unserved: List[int] = []

    # Harder customers first (larger demand, then farther from depot)
    ordered = sorted(customers, key=lambda c: (demands[c], D[depot, c]), reverse=True)

    for c in ordered:
        ins = _best_insertion_fast(current_routes, c)
        if ins is None:
            unserved.append(c)
            continue
        vidx, pos = ins
        current_routes[vidx].insert(pos, c)

    best_routes = [list(r) for r in current_routes]
    best_unserved = list(unserved)
    best_score = _plan_cost_fast(best_routes) + 1_000_000.0 * len(best_unserved)

    temp = max(1.0, 0.02 * max(1, len(customers)))
    destroy_count = max(1, int(round(float(destroy_rate) * max(1, len(customers)))))

    for _ in range(max(1, int(iterations))):
        cand_routes = [list(r) for r in current_routes]

        served_pairs = [(v, i) for v, r in enumerate(cand_routes) for i in range(len(r))]
        rng.shuffle(served_pairs)
        to_remove = served_pairs[: min(destroy_count, len(served_pairs))]

        removed = []
        for v, i in sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True):
            removed.append(cand_routes[v].pop(i))

        repair_pool = removed + list(unserved)
        rng.shuffle(repair_pool)
        new_unserved: List[int] = []

        for c in repair_pool:
            ins = _best_insertion_fast(cand_routes, c)
            if ins is None:
                new_unserved.append(c)
                continue
            v, p = ins
            cand_routes[v].insert(p, c)

        cand_score = _plan_cost_fast(cand_routes) + 1_000_000.0 * len(new_unserved)
        cur_score = _plan_cost_fast(current_routes) + 1_000_000.0 * len(unserved)

        delta = cand_score - cur_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(1e-6, temp)):
            current_routes = cand_routes
            unserved = new_unserved

        if cand_score < best_score:
            best_score = cand_score
            best_routes = [list(r) for r in cand_routes]
            best_unserved = list(new_unserved)

        temp *= 0.999

    served_count = sum(len(r) for r in best_routes)
    log_lines.append(f"Servis edilen müşteri: {served_count}/{len(customers)}")
    if best_unserved:
        log_lines.append(f"Servis edilemeyen müşteriler: {best_unserved[:20]}")
    log_lines.append(f"Kullanılan araç sayısı: {sum(1 for r in best_routes if r)}")
    log_lines.append("ALNS single-trip çözümü tamamlandı.")

    return {
        "routes": best_routes,
        "served_customers": served_count,
        "unserved_customers": best_unserved,
        "log": "\n".join(log_lines),
        "objective": objective,
    }
