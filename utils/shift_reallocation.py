"""
Shift Reallocation Algorithm for Multi-Trip Optimization

This module implements a two-shift reallocation strategy:
- Morning shift: 09:00 - 12:00 (vehicles return and recharge)
- Afternoon shift: 13:00 - 18:00 (same vehicles can be reused)

Goal: Reduce total vehicle count and energy by reallocating orders
across two shifts for vehicles that finish early.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from copy import deepcopy


def analyze_early_finishers(
    routes: List[List[int]],
    data: Dict[str, Any],
    cutoff_time: float = 12 * 60  # 12:00 in minutes
) -> Dict[str, Any]:
    """
    Analyze which vehicles finish before cutoff time.
    
    Args:
        routes: List of routes (each route is list of customer node IDs)
        data: OR-Tools data dictionary with matrices
        cutoff_time: Time threshold in minutes (default 12:00 = 720 min)
    
    Returns:
        Dictionary with analysis results
    """
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    service = np.array(data["service_min"], dtype=float)
    depot = data["depot"]
    
    early_vehicles = []
    late_vehicles = []
    vehicle_stats = []
    
    for v_idx, route in enumerate(routes):
        if not route:
            continue
        
        # Calculate return time
        t_now = 9 * 60  # Start at 09:00
        prev = depot
        
        for node in route:
            # Travel time
            travel = T[prev, node]
            t_now += travel
            
            # Service time
            t_now += service[node]
            
            prev = node
        
        # Return to depot
        t_now += T[prev, depot]
        
        return_time = t_now
        return_hour = return_time / 60.0
        
        stats = {
            "vehicle_id": v_idx,
            "return_time_min": return_time,
            "return_hour": return_hour,
            "num_customers": len(route),
            "route": route.copy()
        }
        
        vehicle_stats.append(stats)
        
        if return_time <= cutoff_time:
            early_vehicles.append(v_idx)
        else:
            late_vehicles.append(v_idx)
    
    return {
        "early_vehicles": early_vehicles,
        "late_vehicles": late_vehicles,
        "vehicle_stats": vehicle_stats,
        "num_early": len(early_vehicles),
        "num_late": len(late_vehicles),
    }


def calculate_route_metrics(route: List[int], data: Dict[str, Any], start_time_min: float = 9*60) -> Dict[str, float]:
    """Calculate comprehensive metrics for a single route."""
    if not route:
        return {
            "distance_km": 0.0,
            "time_min": 0.0,
            "energy_kwh": 0.0,
            "return_time_min": start_time_min,
            "load_desi": 0.0,
        }
    
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    demand = np.array(data["demand_desi"], dtype=float)
    service = np.array(data["service_min"], dtype=float)
    depot = data["depot"]
    
    total_dist = 0.0
    total_time = 0.0
    total_energy = 0.0
    total_load = 0.0
    cum_load = 0.0
    
    prev = depot
    t_now = start_time_min
    
    for node in route:
        # Travel
        d = D[prev, node]
        t = T[prev, node]
        
        # Energy: 0.436 * distance + 0.002 * load_before_leg
        e = 0.436 * d + 0.002 * cum_load
        
        total_dist += d
        total_time += t
        total_energy += e
        t_now += t
        
        # Pickup
        cum_load += demand[node]
        total_load += demand[node]
        
        # Service
        t_now += service[node]
        total_time += service[node]
        
        prev = node
    
    # Return to depot
    d = D[prev, depot]
    t = T[prev, depot]
    e = 0.436 * d + 0.002 * cum_load
    
    total_dist += d
    total_time += t
    total_energy += e
    t_now += t
    
    return {
        "distance_km": total_dist,
        "time_min": total_time,
        "energy_kwh": total_energy,
        "return_time_min": t_now,
        "load_desi": total_load,
    }


def two_phase_reallocation(
    routes: List[List[int]],
    data: Dict[str, Any],
    morning_end: float = 12 * 60,  # 12:00
    afternoon_start: float = 13 * 60,  # 13:00
    afternoon_end: float = 18 * 60,  # 18:00
) -> Dict[str, Any]:
    """
    Two-phase greedy reallocation algorithm.
    
    Strategy:
    1. Identify vehicles finishing before morning_end
    2. Keep their morning routes
    3. Allocate remaining orders to afternoon shift (13:00-18:00)
    4. Try to use fewer vehicles by maximizing afternoon utilization
    
    Returns:
        Dictionary with reallocated routes and metrics
    """
    # Step 1: Analyze current solution
    analysis = analyze_early_finishers(routes, data, cutoff_time=morning_end)
    
    if analysis["num_early"] == 0:
        return {
            "success": False,
            "message": "Hiç araç 12:00'den önce dönmüyor. Yeniden atama yapılamaz.",
            "original_routes": routes,
            "reallocated_routes": routes,
        }
    
    # Step 2: Build morning and afternoon pools
    morning_routes = []
    afternoon_available_vehicles = []
    
    for v_idx, route in enumerate(routes):
        if v_idx in analysis["early_vehicles"]:
            morning_routes.append({
                "vehicle_id": v_idx,
                "route": route.copy(),
                "can_work_afternoon": True,
            })
            afternoon_available_vehicles.append(v_idx)
        else:
            morning_routes.append({
                "vehicle_id": v_idx,
                "route": route.copy(),
                "can_work_afternoon": False,
            })
    
    # Step 3: Collect all customers
    all_customers = set()
    for route in routes:
        all_customers.update(route)
    
    # Greedy afternoon assignment (simplified - just redistribute)
    # In practice, you'd want to solve a new VRP for afternoon shift
    
    afternoon_routes = {v: [] for v in afternoon_available_vehicles}
    
    # For now, keep original assignment (base case)
    # TODO: Implement actual reallocation logic
    
    return {
        "success": True,
        "message": f"{analysis['num_early']} araç sabah dönüyor ve öğleden sonra kullanılabilir.",
        "original_routes": routes,
        "reallocated_routes": routes,  # Placeholder
        "morning_vehicles": afternoon_available_vehicles,
        "analysis": analysis,
    }


def greedy_reassignment(
    unassigned_orders: List[int],
    available_vehicles: List[int],
    data: Dict[str, Any],
    start_time: float = 13 * 60,
    end_time: float = 18 * 60,
) -> List[List[int]]:
    """
    Greedy algorithm to assign unassigned orders to available vehicles.
    
    Uses nearest-neighbor heuristic with capacity and time constraints.
    """
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    demand = np.array(data["demand_desi"], dtype=float)
    service = np.array(data["service_min"], dtype=float)
    depot = data["depot"]
    capacity = data["vehicle_cap_desi"]
    
    # Initialize vehicle routes
    vehicle_routes = {v: [] for v in available_vehicles}
    vehicle_load = {v: 0.0 for v in available_vehicles}
    vehicle_time = {v: start_time for v in available_vehicles}
    vehicle_pos = {v: depot for v in available_vehicles}
    
    remaining = set(unassigned_orders)
    
    while remaining:
        best_insertion = None
        best_cost = float('inf')
        
        for v in available_vehicles:
            current_pos = vehicle_pos[v]
            current_load = vehicle_load[v]
            current_time = vehicle_time[v]
            
            for customer in remaining:
                # Check capacity
                if current_load + demand[customer] > capacity:
                    continue
                
                # Check time feasibility
                travel_time = T[current_pos, customer]
                service_time = service[customer]
                return_time = T[customer, depot]
                
                new_time = current_time + travel_time + service_time
                
                if new_time + return_time > end_time:
                    continue
                
                # Calculate insertion cost (distance)
                cost = D[current_pos, customer]
                
                if cost < best_cost:
                    best_cost = cost
                    best_insertion = (v, customer)
        
        if best_insertion is None:
            # No feasible insertion found
            break
        
        v, customer = best_insertion
        vehicle_routes[v].append(customer)
        vehicle_load[v] += demand[customer]
        vehicle_time[v] += T[vehicle_pos[v], customer] + service[customer]
        vehicle_pos[v] = customer
        remaining.remove(customer)
    
    # Convert to list format
    result_routes = [vehicle_routes[v] for v in available_vehicles]
    
    return result_routes


def calculate_solution_metrics(routes: List[List[int]], data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate total metrics for entire solution."""
    total_energy = 0.0
    total_distance = 0.0
    total_time = 0.0
    active_vehicles = 0
    
    for route in routes:
        if route:
            metrics = calculate_route_metrics(route, data)
            total_energy += metrics["energy_kwh"]
            total_distance += metrics["distance_km"]
            total_time += metrics["time_min"]
            active_vehicles += 1
    
    return {
        "total_energy_kwh": total_energy,
        "total_distance_km": total_distance,
        "total_time_min": total_time,
        "active_vehicles": active_vehicles,
    }


def clarke_wright_savings(
    customers: List[int],
    available_vehicles: List[int],
    data: Dict[str, Any],
    start_time: float = 13 * 60,
    end_time: float = 18 * 60,
) -> List[List[int]]:
    """
    Clarke-Wright Savings Algorithm for VRP.
    
    Classic algorithm that merges routes based on savings:
    s(i,j) = d(0,i) + d(0,j) - d(i,j)
    
    Args:
        customers: List of customer node IDs to assign
        available_vehicles: List of vehicle IDs available for afternoon shift
        data: OR-Tools data dictionary
        start_time: Shift start time in minutes (default 13:00)
        end_time: Shift end time in minutes (default 18:00)
    
    Returns:
        List of routes for afternoon shift
    """
    if not customers:
        return [[] for _ in available_vehicles]
    
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    demand = np.array(data["demand_desi"], dtype=float)
    service = np.array(data["service_min"], dtype=float)
    depot = data["depot"]
    capacity = data["vehicle_cap_desi"]
    battery_capacity = data["battery_capacity"]
    
    # Step 1: Initialize each customer in its own route (depot -> i -> depot)
    routes = {i: [i] for i in customers}
    route_loads = {i: demand[i] for i in customers}
    route_times = {i: 2 * T[depot, i] + service[i] for i in customers}
    route_energy = {i: 2 * 0.436 * D[depot, i] + 0.002 * demand[i] * D[depot, i] for i in customers}
    
    # Step 2: Calculate all savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    customer_list = list(customers)
    
    for i_idx, i in enumerate(customer_list):
        for j in customer_list[i_idx + 1:]:
            if i == j:
                continue
            
            # Savings calculation
            s = D[depot, i] + D[depot, j] - D[i, j]
            savings.append((s, i, j))
    
    # Step 3: Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Step 4: Merge routes based on savings
    for saving_value, i, j in savings:
        # Find which routes contain i and j
        route_i = None
        route_j = None
        
        for route_id, route in routes.items():
            if i in route:
                route_i = route_id
            if j in route:
                route_j = route_id
        
        # Skip if already in same route or routes not found
        if route_i is None or route_j is None or route_i == route_j:
            continue
        
        # Check if i is at the end of its route and j is at the start of its route
        # (or vice versa) - only merge if they are at the endpoints
        route_i_list = routes[route_i]
        route_j_list = routes[route_j]
        
        can_merge = False
        merge_order = None
        
        if route_i_list[-1] == i and route_j_list[0] == j:
            # i is last in route_i, j is first in route_j: merge i->j
            can_merge = True
            merge_order = "i_j"
        elif route_j_list[-1] == j and route_i_list[0] == i:
            # j is last in route_j, i is first in route_i: merge j->i
            can_merge = True
            merge_order = "j_i"
        elif route_i_list[0] == i and route_j_list[-1] == j:
            # i is first in route_i, j is last in route_j: merge j->i
            can_merge = True
            merge_order = "j_i"
        elif route_j_list[0] == j and route_i_list[-1] == i:
            # j is first in route_j, i is last in route_i: merge i->j
            can_merge = True
            merge_order = "i_j"
        
        if not can_merge:
            continue
        
        # Calculate merged route metrics
        if merge_order == "i_j":
            merged_route = route_i_list + route_j_list
        else:
            merged_route = route_j_list + route_i_list
        
        # Check capacity constraint
        merged_load = route_loads[route_i] + route_loads[route_j]
        if merged_load > capacity:
            continue
        
        # Check time constraint
        merged_metrics = calculate_route_metrics(merged_route, data, start_time_min=start_time)
        if merged_metrics["return_time_min"] > end_time:
            continue
        
        # Check battery constraint
        if merged_metrics["energy_kwh"] > battery_capacity:
            continue
        
        # Merge is feasible - update routes
        if merge_order == "i_j":
            routes[route_i] = merged_route
            route_loads[route_i] = merged_load
            route_times[route_i] = merged_metrics["time_min"]
            route_energy[route_i] = merged_metrics["energy_kwh"]
            del routes[route_j]
            del route_loads[route_j]
            del route_times[route_j]
            del route_energy[route_j]
        else:
            routes[route_j] = merged_route
            route_loads[route_j] = merged_load
            route_times[route_j] = merged_metrics["time_min"]
            route_energy[route_j] = merged_metrics["energy_kwh"]
            del routes[route_i]
            del route_loads[route_i]
            del route_times[route_i]
            del route_energy[route_i]
    
    # Step 5: Convert to list format
    final_routes = list(routes.values())
    
    # Pad with empty routes if we have more vehicles than routes
    while len(final_routes) < len(available_vehicles):
        final_routes.append([])
    
    return final_routes[:len(available_vehicles)]


def apply_shift_reallocation(
    original_routes: List[List[int]],
    data: Dict[str, Any],
    morning_cutoff: float = 12 * 60,
    afternoon_start: float = 13 * 60,
    afternoon_end: float = 18 * 60,
    strategy: str = "clarke_wright",
) -> Dict[str, Any]:
    """
    Apply two-shift reallocation to minimize total energy.
    
    Strategy:
    1. Early-finishing vehicles keep their morning routes
    2. All remaining orders are candidates for afternoon reallocation
    3. Try to assign afternoon orders to early vehicles to reduce total energy
    4. Each order served exactly once
    
    Example:
    - Vehicle 1: 5 orders (would take until 16:00)
    - Vehicle 2: 1 heavy order (finishes at 10:00)
    
    Reallocation:
    - Morning: Vehicle 2 does its 1 order (9:00-10:00)
    - Afternoon: Vehicle 2 takes 2-3 orders from Vehicle 1 (13:00-17:00)
    - Result: Maybe Vehicle 1 not needed, or serves fewer orders
    
    Args:
        original_routes: Original single-shift solution
        data: OR-Tools data dictionary
        morning_cutoff: Morning shift end time (minutes)
        afternoon_start: Afternoon shift start time (minutes)
        afternoon_end: Afternoon shift end time (minutes)
        strategy: Algorithm to use ("greedy", "clarke_wright", "2opt")
    
    Returns:
        Dictionary with reallocation results
    """
    # Analyze early finishers
    analysis = analyze_early_finishers(original_routes, data, cutoff_time=morning_cutoff)
    
    if analysis["num_early"] == 0:
        return {
            "success": False,
            "message": "Hiç araç erken dönmüyor. Yeniden atama yapılamaz.",
            "original_metrics": calculate_solution_metrics(original_routes, data),
        }
    
    # Step 1: Early vehicles keep their morning routes
    morning_routes_early = []
    morning_customers_early = set()
    
    for v_idx in analysis["early_vehicles"]:
        if v_idx < len(original_routes):
            route = original_routes[v_idx].copy()
            morning_routes_early.append(route)
            morning_customers_early.update(route)
    
    # Step 2: All other customers are candidates for reallocation
    all_customers = set()
    for route in original_routes:
        all_customers.update(route)
    
    remaining_customers = all_customers - morning_customers_early
    
    # Step 3: Try to allocate remaining customers to afternoon shift
    if strategy == "clarke_wright":
        afternoon_routes = clarke_wright_savings(
            list(remaining_customers),
            analysis["early_vehicles"],
            data,
            start_time=afternoon_start,
            end_time=afternoon_end,
        )
    elif strategy == "greedy":
        afternoon_routes = greedy_reassignment(
            list(remaining_customers),
            analysis["early_vehicles"],
            data,
            start_time=afternoon_start,
            end_time=afternoon_end,
        )
    else:
        afternoon_routes = greedy_reassignment(
            list(remaining_customers),
            analysis["early_vehicles"],
            data,
            start_time=afternoon_start,
            end_time=afternoon_end,
        )
    
    # Step 4: Check which customers were successfully assigned to afternoon
    assigned_afternoon = set()
    for route in afternoon_routes:
        assigned_afternoon.update(route)
    
    unassigned = remaining_customers - assigned_afternoon
    
    # Step 5: Unassigned customers need to be served in morning by late vehicles
    # OR by additional morning routes
    if unassigned:
        # Assign to late vehicles or create new morning routes
        late_morning_routes = []
        
        # First, try to use late vehicles' original routes if they exist
        for v_idx in analysis["late_vehicles"]:
            if v_idx < len(original_routes):
                route = original_routes[v_idx].copy()
                # Only include customers that are in unassigned set
                filtered_route = [c for c in route if c in unassigned]
                if filtered_route:
                    late_morning_routes.append(filtered_route)
                    unassigned -= set(filtered_route)
        
        # If still unassigned, create new routes
        if unassigned:
            extra_routes = greedy_reassignment(
                list(unassigned),
                list(range(len(original_routes), len(original_routes) + 10)),  # Use extra vehicles if needed
                data,
                start_time=9 * 60,
                end_time=morning_cutoff,
            )
            late_morning_routes.extend([r for r in extra_routes if r])
        
        # Combine all morning routes
        morning_routes = morning_routes_early + late_morning_routes
    else:
        # All remaining customers fit in afternoon
        morning_routes = morning_routes_early
    
    # Step 6: Verify no duplicates
    all_morning_customers = set()
    for route in morning_routes:
        all_morning_customers.update(route)
    
    all_afternoon_customers = set()
    for route in afternoon_routes:
        all_afternoon_customers.update(route)
    
    duplicates = all_morning_customers & all_afternoon_customers
    if duplicates:
        return {
            "success": False,
            "message": f"Hata: {len(duplicates)} müşteri hem sabah hem öğleden sonra atandı. Algoritma hatası.",
            "original_metrics": calculate_solution_metrics(original_routes, data),
        }
    
    # Step 7: Verify all customers are served
    all_served = all_morning_customers | all_afternoon_customers
    missing = all_customers - all_served
    
    if missing:
        return {
            "success": False,
            "message": f"Hata: {len(missing)} müşteri atanmadı: {list(missing)[:10]}. Daha fazla araç gerekli.",
            "original_metrics": calculate_solution_metrics(original_routes, data),
        }
    
    # Step 8: Calculate metrics
    original_metrics = calculate_solution_metrics(original_routes, data)
    morning_metrics = calculate_solution_metrics(morning_routes, data)
    afternoon_metrics = calculate_solution_metrics(afternoon_routes, data)
    
    # Combined metrics
    total_energy = morning_metrics["total_energy_kwh"] + afternoon_metrics["total_energy_kwh"]
    total_distance = morning_metrics["total_distance_km"] + afternoon_metrics["total_distance_km"]
    
    # Count actual vehicles used (not route count, but unique physical vehicles)
    # Morning: early vehicles + any late vehicles actually used
    # Afternoon: early vehicles only (they work both shifts)
    total_vehicles_used = len([r for r in morning_routes if r])
    # Don't double-count early vehicles working afternoon
    unique_vehicles = len([r for r in morning_routes if r])  # This might overcount, but is safe upper bound
    
    return {
        "success": True,
        "message": f"Yeniden atama tamamlandı ({strategy}).",
        "original_metrics": original_metrics,
        "morning_routes": morning_routes,
        "afternoon_routes": afternoon_routes,
        "morning_metrics": morning_metrics,
        "afternoon_metrics": afternoon_metrics,
        "combined_metrics": {
            "total_energy_kwh": total_energy,
            "total_distance_km": total_distance,
            "total_vehicles_used": total_vehicles_used,
        },
        "early_vehicles": analysis["early_vehicles"],
        "late_vehicles": analysis["late_vehicles"],
        "customers_in_morning": len(all_morning_customers),
        "customers_in_afternoon": len(all_afternoon_customers),
        "total_customers_served": len(all_served),
        "early_morning_customers": len(morning_customers_early),
    }
