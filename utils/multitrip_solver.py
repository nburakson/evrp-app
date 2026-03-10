# utils/multitrip_solver.py
"""
Multi-Trip Vehicle Routing Solver

Allows vehicles to make multiple trips from depot if they have:
- Sufficient remaining energy
- Sufficient remaining time in working day
"""

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
from typing import List, Dict, Tuple

# Workday: 09:00–18:00
WORK_START_MIN = 9 * 60
WORK_END_MIN = 18 * 60


def solve_multitrip_ortools(
    data: dict, 
    time_limit_s: int = 30, 
    seed: int = 42,
    allow_multi_trip: bool = True
):
    """
    Solve VRP with multi-trip capability.
    
    When allow_multi_trip=True:
    - Each vehicle can make multiple round trips
    - After returning to depot, vehicle can start another route if:
        * Remaining energy >= energy needed for shortest possible round trip
        * Remaining time >= time needed for shortest possible round trip
    """
    
    log_lines: list[str] = []
    
    def log(msg: str = ""):
        log_lines.append(str(msg))
    
    def _fmt_time(minutes_from_start: float, start_hour: int = 9) -> str:
        total_min = start_hour * 60 + minutes_from_start
        h = int(total_min // 60)
        m = int(total_min % 60)
        return f"{h:02d}:{m:02d}"
    
    log("🧩 Building Multi-Trip OR-Tools model...")
    log(f"Multi-trip enabled: {allow_multi_trip}")
    
    # ----------------------------------------
    # MODEL SETUP
    # ----------------------------------------
    n = data["distance_km"].shape[0]
    num_vehicles = data["num_vehicles"]
    depot = data["depot"]
    
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    # ----------------------------------------
    # Distance callback (in meters)
    # ----------------------------------------
    dist_m = (data["distance_km"] * 1000.0).round().astype(int)
    
    def distance_cb(i_idx, j_idx):
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        return int(dist_m[i, j])
    
    transit_dist = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_dist)
    
    # ----------------------------------------
    # TIME DIMENSION
    # ----------------------------------------
    time_min = np.array(data["time_min"], dtype=float)
    service_min = np.array(data["service_min"], dtype=float)
    
    def time_cb(i_idx, j_idx):
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        travel = time_min[i, j]
        service = service_min[i] if i != depot else 0.0
        return int(round(travel + service))
    
    transit_time = routing.RegisterTransitCallback(time_cb)
    
    routing.AddDimension(
        transit_time,
        int(WORK_END_MIN - WORK_START_MIN),  # slack allows depot revisits
        int(WORK_END_MIN - WORK_START_MIN),
        True,  # force start at 0
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    
    # Start times = 0
    for v in range(num_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetValue(0)
    
    # ----------------------------------------
    # CAPACITY DIMENSION (Pickup model with depot reset)
    # ----------------------------------------
    demands = np.array(data["demand_desi"], dtype=int)
    
    def demand_cb(i_idx):
        node = manager.IndexToNode(i_idx)
        return int(demands[node])
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        int(data["vehicle_cap_desi"]) if allow_multi_trip else 0,  # slack allows unloading at depot
        [int(data["vehicle_cap_desi"])] * num_vehicles,
        True,  # vehicle starts empty
        "Capacity",
    )
    capacity_dim = routing.GetDimensionOrDie("Capacity")
    
    # Allow capacity to reset at depot (unloading)
    if allow_multi_trip:
        for v in range(num_vehicles):
            depot_idx = routing.Start(v)
            capacity_dim.SlackVar(depot_idx).SetValue(0)
    
    # ----------------------------------------
    # ENERGY DIMENSION (with depot recharge)
    # ----------------------------------------
    BASE_KWH_PER_KM = 0.436
    LOAD_TERM = 0.002
    battery_capacity = float(data.get("battery_capacity", 100.0))
    reserve_soc_pct = float(data.get("min_return_soc_pct", 0.0))
    usable_battery = battery_capacity * (1.0 - reserve_soc_pct / 100.0)
    
    def energy_cb(i_idx, j_idx):
        i = manager.IndexToNode(i_idx)
        j = manager.IndexToNode(j_idx)
        d_km = float(data["distance_km"][i, j])
        from_node_desi = float(demands[i]) if i != depot else 0.0
        e_kwh = BASE_KWH_PER_KM * d_km + LOAD_TERM * from_node_desi
        return int(round(e_kwh))
    
    energy_transit = routing.RegisterTransitCallback(energy_cb)
    
    routing.AddDimension(
        energy_transit,
        int(round(usable_battery)) if allow_multi_trip else 0,  # slack allows recharge at depot
        int(round(usable_battery)),
        True,  # starts with 0 energy consumed
        "Energy",
    )
    energy_dim = routing.GetDimensionOrDie("Energy")
    
    # Allow energy to reset at depot (recharging)
    if allow_multi_trip:
        for v in range(num_vehicles):
            depot_idx = routing.Start(v)
            energy_dim.SlackVar(depot_idx).SetValue(0)
    
    # ----------------------------------------
    # DISJUNCTIONS (optional: make visits optional)
    # ----------------------------------------
    penalty = 1000000  # high penalty for unserved customers
    for node in range(1, n):  # skip depot
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # ----------------------------------------
    # Search parameters
    # ----------------------------------------
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))
    params.log_search = False
    
    try:
        params.use_randomization = True
        params.random_seed = seed
    except Exception:
        pass
    
    log(f"🚀 Solving (Multi-Trip Mode, {time_limit_s}s)...")
    solution = routing.SolveWithParameters(params)
    
    log("")
    log("================ MULTI-TRIP SOLVER RESULT ================")
    
    # ----------------------------------------
    # NO SOLUTION
    # ----------------------------------------
    if solution is None:
        log("❌ Çözüm bulunamadı.")
        log("🔍 Öneriler:")
        log("   • Zaman limitini artırın")
        log("   • Araç sayısını artırın")
        log("   • Kapasite veya batarya kısıtlarını gevşetin")
        log("=======================================================")
        return {
            "routing": routing,
            "manager": manager,
            "solution": None,
            "time_dim": time_dim,
            "energy_dim": energy_dim,
            "capacity_dim": capacity_dim,
            "log": "\n".join(log_lines),
            "multi_trip": allow_multi_trip,
        }
    
    log("✅ Çözüm bulundu!")
    log(f"🔢 Vehicles: {num_vehicles} | Nodes: {n}\n")
    
    # Track dropped nodes
    dropped = []
    for node in range(1, n):
        idx = manager.NodeToIndex(node)
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue
        if solution.Value(routing.NextVar(idx)) == idx:
            dropped.append(node)
    
    if dropped:
        log(f"⚠️  Servis edilemeyen müşteriler: {dropped}\n")
    
    # ----------------------------------------
    # ROUTE OUTPUT with TRIP DETECTION
    # ----------------------------------------
    total_trips = 0
    
    for v in range(num_vehicles):
        idx = routing.Start(v)
        
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            log(f"\n🚚 Araç {v}: Boş rota")
            continue
        
        trip_num = 1
        vehicle_total_dist = 0.0
        vehicle_total_energy = 0.0
        vehicle_total_time = 0.0
        
        log(f"\n🚚 Araç {v} rotası:")
        
        while not routing.IsEnd(idx):
            # Detect trip start
            node = manager.IndexToNode(idx)
            
            if node == depot:
                # Check if this is a depot revisit (multi-trip)
                next_idx = solution.Value(routing.NextVar(idx))
                if not routing.IsEnd(next_idx):
                    next_node = manager.IndexToNode(next_idx)
                    if next_node != depot:
                        # Starting a new trip
                        trip_time = solution.Value(time_dim.CumulVar(idx))
                        trip_energy = solution.Value(energy_dim.CumulVar(idx))
                        trip_load = solution.Value(capacity_dim.CumulVar(idx))
                        
                        log(f"\n  🔄 TUR {trip_num} başlangıcı - Saat: {_fmt_time(trip_time)} | "
                            f"Tüketilen Enerji: {trip_energy:.1f} kWh | Yük: {trip_load:.0f} desi")
                        trip_num += 1
                        total_trips += 1
            
            # Process current node
            if node != depot:
                arr_time = solution.Value(time_dim.CumulVar(idx))
                cum_energy = solution.Value(energy_dim.CumulVar(idx))
                cum_load = solution.Value(capacity_dim.CumulVar(idx))
                
                log(f"    Müşteri {node:2d} | Saat: {_fmt_time(arr_time)} | "
                    f"Enerji: {cum_energy:5.1f} kWh | Yük: {cum_load:6.0f} desi | "
                    f"Servis: {service_min[node]:.0f} dk")
            
            idx = solution.Value(routing.NextVar(idx))
        
        # Final depot arrival
        arr_time = solution.Value(time_dim.CumulVar(idx))
        cum_energy = solution.Value(energy_dim.CumulVar(idx))
        
        log(f"\n  ✅ Depoya dönüş - Saat: {_fmt_time(arr_time)} | "
            f"Toplam Enerji: {cum_energy:.2f} kWh")
        
        if trip_num > 1:
            log(f"  📊 Araç {v} toplam {trip_num - 1} tur yaptı")
    
    log(f"\n📈 Toplam tur sayısı (tüm araçlar): {total_trips}")
    log("=======================================================")
    
    return {
        "routing": routing,
        "manager": manager,
        "solution": solution,
        "time_dim": time_dim,
        "energy_dim": energy_dim,
        "capacity_dim": capacity_dim,
        "log": "\n".join(log_lines),
        "multi_trip": allow_multi_trip,
        "total_trips": total_trips,
    }
