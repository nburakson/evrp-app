# utils/multitrip_route_extractor.py
"""
Extract multi-trip routes from OR-Tools solution.
Identifies separate trips by detecting depot revisits.
"""

def extract_multitrip_routes(data, routing, manager, solution):
    """
    Extract routes with multi-trip detection.
    
    Returns:
        trips_per_vehicle: List[List[List[int]]]
            trips_per_vehicle[v][trip_num] = [node1, node2, ...]
            
    Example:
        Vehicle 0 makes 2 trips:
        trips_per_vehicle[0] = [[1, 3, 5], [7, 9]]
        
        Vehicle 1 makes 1 trip:
        trips_per_vehicle[1] = [[2, 4, 6]]
    """
    trips_per_vehicle = []
    n_vehicles = data["num_vehicles"]
    depot = data["depot"]
    
    for v in range(n_vehicles):
        idx = routing.Start(v)
        vehicle_trips = []
        current_trip = []
        
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            
            if node == depot:
                # Depot visit - check if it's a trip separator
                if current_trip:
                    # End of a trip
                    vehicle_trips.append(current_trip)
                    current_trip = []
            else:
                # Customer visit
                current_trip.append(node)
            
            idx = solution.Value(routing.NextVar(idx))
        
        # Add last trip if any
        if current_trip:
            vehicle_trips.append(current_trip)
        
        trips_per_vehicle.append(vehicle_trips)
    
    return trips_per_vehicle


def flatten_multitrip_routes(trips_per_vehicle):
    """
    Flatten multi-trip routes to simple route format for compatibility.
    
    Returns: List[List[int]] - one route per vehicle (all trips concatenated)
    """
    return [
        [node for trip in vehicle_trips for node in trip]
        for vehicle_trips in trips_per_vehicle
    ]


def get_trip_statistics(trips_per_vehicle, data):
    """
    Calculate statistics for each trip.
    
    Returns:
        stats: List[List[Dict]]
            stats[v][trip_num] = {
                'trip_num': int,
                'nodes': List[int],
                'distance_km': float,
                'energy_kwh': float,
                'load_desi': float,
                'time_min': float,
            }
    """
    import numpy as np
    
    D = np.array(data["distance_km"], dtype=float)
    T = np.array(data["time_min"], dtype=float)
    loads = np.array(data["demand_desi"], dtype=float)
    depot = data["depot"]
    
    all_stats = []
    
    for v, vehicle_trips in enumerate(trips_per_vehicle):
        vehicle_stats = []
        
        for trip_num, trip_nodes in enumerate(vehicle_trips, 1):
            if not trip_nodes:
                continue
            
            # Calculate trip metrics
            total_km = 0.0
            total_energy = 0.0
            total_time = 0.0
            total_load = 0.0
            
            prev = depot
            cum_load = 0.0
            
            for node in trip_nodes:
                # Distance and time
                d_km = float(D[prev, node])
                t_min = float(T[prev, node])
                total_km += d_km
                total_time += t_min
                
                # Energy with load
                empty_energy = 0.436 * d_km
                load_energy = 0.002 * cum_load
                total_energy += empty_energy + load_energy
                
                # Load
                node_load = float(loads[node])
                cum_load += node_load
                total_load += node_load
                
                prev = node
            
            # Return to depot
            d_km = float(D[prev, depot])
            t_min = float(T[prev, depot])
            total_km += d_km
            total_time += t_min
            
            empty_energy = 0.436 * d_km
            load_energy = 0.002 * cum_load
            total_energy += empty_energy + load_energy
            
            vehicle_stats.append({
                'trip_num': trip_num,
                'nodes': trip_nodes,
                'num_customers': len(trip_nodes),
                'distance_km': total_km,
                'energy_kwh': total_energy,
                'load_desi': total_load,
                'time_min': total_time,
            })
        
        all_stats.append(vehicle_stats)
    
    return all_stats
