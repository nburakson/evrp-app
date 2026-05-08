"""
Route-preserving multi-trip merger.

Takes assigned routes from optimization (Tabu, GA, ALNS) and merges them into
fewer vehicles WITHOUT changing customer order or removing customers.

Routes are merged greedily: try to add each route to a vehicle while feasible.
When a route can't fit, assign to a new vehicle.

Each combined trip respects:
- Battery capacity (with return margin)
- Time constraints (working hours)
- Load capacity
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Any


def merge_routes_into_vehicles(
    routes: List[List[int]],
    D: np.ndarray,
    T: np.ndarray,
    demand: np.ndarray,
    service_time: np.ndarray,
    battery_capacity: float,
    vehicle_capacity: float,
    max_shift_duration: float,
    min_return_battery_pct: float = 20.0,
    depot: int = 0,
) -> List[Dict[str, Any]]:
    """
    Merge routes into fewer vehicles greedily while preserving route order.

    Each vehicle can carry multiple complete routes if they fit within constraints.
    Routes are combined in sequence: depot → route1 → route2 → ... → depot

    Args:
        routes: list of routes, where routes[i] = route for vehicle i
        D: distance matrix (km)
        T: time matrix (minutes)
        demand: demand array where demand[i] = demand of node i
        service_time: service time array where service_time[i] = service time at node i
        battery_capacity: total battery capacity (kWh)
        vehicle_capacity: vehicle capacity (desi)
        max_shift_duration: max shift time in minutes
        min_return_battery_pct: minimum battery % to return to depot (safety margin)
        depot: depot node index (default 0)

    Returns:
        List of vehicles, each with:
        - vehicle_id: new vehicle id (1-indexed)
        - trips: list of original route indices that this vehicle handles
        - combined_route: the merged route [all customers from all trips]
        - time_min, distance_km, energy_kwh, load_desi: metrics for combined trip
    """
    if not routes:
        return []

    usable_battery = battery_capacity * (1.0 - min_return_battery_pct / 100.0)
    vehicles = []
    current_vehicle_routes = []
    current_load = 0.0
    current_distance = 0.0
    current_time = 0.0
    current_energy = 0.0
    current_combined = []

    def can_add_route(
        route: List[int],
        current_combined: List[int],
        current_load: float,
        current_distance: float,
        current_time: float,
        current_energy: float,
    ) -> bool:
        """Check if adding a route keeps vehicle feasible."""
        if not route:
            return True

        route_demand = sum(demand[n] for n in route)
        route_service = sum(service_time[n] for n in route)

        # Check capacity
        new_load = current_load + route_demand
        if new_load > vehicle_capacity:
            return False

        # Calculate distance and time for this route
        route_distance = 0.0
        route_time = 0.0
        route_energy = 0.0

        if not current_combined:
            # First route: depot -> route -> depot
            prev_node = depot
            for node in route:
                route_distance += D[prev_node, node]
                route_time += T[prev_node, node]
                # Energy with partial load
                load_so_far = sum(demand[n] for n in route[:route.index(node) + 1])
                route_energy += D[prev_node, node] * (0.436 + 0.00136 * load_so_far)
                route_time += service_time[node]
                prev_node = node
            route_distance += D[prev_node, depot]
            route_time += T[prev_node, depot]
            route_energy += D[prev_node, depot] * 0.436
        else:
            # Subsequent route: current_last -> route -> depot (no return to depot first)
            prev_node = current_combined[-1]
            for node in route:
                route_distance += D[prev_node, node]
                route_time += T[prev_node, node]
                # Energy: current load + new customers
                load_so_far = current_load + sum(demand[n] for n in route[:route.index(node) + 1])
                route_energy += D[prev_node, node] * (0.436 + 0.00136 * load_so_far)
                route_time += service_time[node]
                prev_node = node
            route_distance += D[prev_node, depot]
            route_time += T[prev_node, depot]
            route_energy += D[prev_node, depot] * 0.436

        # Check time
        new_time = current_time + route_time
        if new_time > max_shift_duration:
            return False

        # Check energy
        new_energy = current_energy + route_energy
        if new_energy > usable_battery:
            return False

        return True

    # Greedy merge: try to add each route to current vehicle
    for route_idx, route in enumerate(routes):
        if not route:
            continue

        if can_add_route(route, current_combined, current_load, current_distance, current_time, current_energy):
            # Add this route to current vehicle
            current_vehicle_routes.append(route_idx)
            
            # Update metrics
            route_demand = sum(demand[n] for n in route)
            current_load += route_demand

            if not current_combined:
                # First route in vehicle
                prev_node = depot
                for node in route:
                    current_distance += D[prev_node, node]
                    current_time += T[prev_node, node]
                    load_so_far = sum(demand[n] for n in route[:route.index(node) + 1])
                    current_energy += D[prev_node, node] * (0.436 + 0.00136 * load_so_far)
                    current_time += service_time[node]
                    current_combined.append(node)
                    prev_node = node
                current_distance += D[prev_node, depot]
                current_time += T[prev_node, depot]
                current_energy += D[prev_node, depot] * 0.436
            else:
                # Subsequent route in vehicle
                prev_node = current_combined[-1]
                for node in route:
                    current_distance += D[prev_node, node]
                    current_time += T[prev_node, node]
                    load_so_far = current_load - route_demand + sum(demand[n] for n in route[:route.index(node) + 1])
                    current_energy += D[prev_node, node] * (0.436 + 0.00136 * load_so_far)
                    current_time += service_time[node]
                    current_combined.append(node)
                    prev_node = node
                # Remove old return to depot edge, add new one
                # (Recalculate from last node)
                current_distance += D[prev_node, depot]
                current_time += T[prev_node, depot]
                current_energy += D[prev_node, depot] * 0.436
        else:
            # Route doesn't fit, save current vehicle and start new one
            if current_vehicle_routes:
                vehicles.append({
                    "vehicle_id": len(vehicles) + 1,
                    "original_route_indices": current_vehicle_routes,
                    "combined_route": current_combined,
                    "time_min": current_time,
                    "distance_km": current_distance,
                    "energy_kwh": current_energy,
                    "load_desi": current_load,
                    "num_routes_merged": len(current_vehicle_routes),
                })

            # Start new vehicle with this route
            current_vehicle_routes = [route_idx]
            current_load = sum(demand[n] for n in route)
            current_combined = list(route)
            
            # Calculate metrics for this single route
            prev_node = depot
            current_distance = 0.0
            current_time = 0.0
            current_energy = 0.0
            
            for node in route:
                current_distance += D[prev_node, node]
                current_time += T[prev_node, node]
                load_so_far = sum(demand[n] for n in route[:route.index(node) + 1])
                current_energy += D[prev_node, node] * (0.436 + 0.00136 * load_so_far)
                current_time += service_time[node]
                prev_node = node
            
            current_distance += D[prev_node, depot]
            current_time += T[prev_node, depot]
            current_energy += D[prev_node, depot] * 0.436

    # Add last vehicle
    if current_vehicle_routes:
        vehicles.append({
            "vehicle_id": len(vehicles) + 1,
            "original_route_indices": current_vehicle_routes,
            "combined_route": current_combined,
            "time_min": current_time,
            "distance_km": current_distance,
            "energy_kwh": current_energy,
            "load_desi": current_load,
            "num_routes_merged": len(current_vehicle_routes),
        })

    return vehicles
