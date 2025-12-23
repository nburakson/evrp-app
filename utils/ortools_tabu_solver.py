# utils/ortools_tabu_solver.py

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# Workday: 09:00–18:00
WORK_START_MIN = 9 * 60
WORK_END_MIN = 18 * 60


def solve_with_ortools_tabu(data, time_limit_s: int = 10, seed: int = 42):
    """
    Run OR-Tools VRP with Tabu Search, using 'data' dict from build_ortools_data.

    This solver enforces:
    - Pickup model (load starts at 0 and increases with each visit)
    - Energy constraint: distance * 0.436  (no load component)
    - Normal time + service time constraints
    """

    log_lines: list[str] = []

    def log(msg: str = ""):
        log_lines.append(str(msg))

    def _fmt_time(minutes_from_start: float, start_hour: int = 9) -> str:
        total_min = start_hour * 60 + minutes_from_start
        h = int(total_min // 60)
        m = int(total_min % 60)
        return f"{h:02d}:{m:02d}"

    log("🧩 Building OR-Tools model...")

    # ----------------------------------------
    # MODEL SETUP
    # ----------------------------------------
    n = data["distance_km"].shape[0]
    manager = pywrapcp.RoutingIndexManager(n, data["num_vehicles"], data["depot"])
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
        service = service_min[i] if i != data["depot"] else 0.0
        return int(round(travel + service))

    transit_time = routing.RegisterTransitCallback(time_cb)

    routing.AddDimension(
        transit_time,
        0,  # no slack
        int(WORK_END_MIN - WORK_START_MIN),
        True,  # force start at 0
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Start times = 0
    for v in range(data["num_vehicles"]):
        time_dim.CumulVar(routing.Start(v)).SetValue(0)

    # Allow no route to exceed working horizon
    for idx in range(routing.Size()):
        time_dim.CumulVar(idx).SetRange(0, int(WORK_END_MIN - WORK_START_MIN))

    # ----------------------------------------
    # CAPACITY DIMENSION  (Pickup model)
    # ----------------------------------------
    demands = np.array(data["demand_desi"], dtype=int)

    def demand_cb(i_idx):
        node = manager.IndexToNode(i_idx)
        return int(demands[node])  # load increases as customers are visited

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)

    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,  # no slack
        [int(data["vehicle_cap_desi"])] * data["num_vehicles"],
        True,  # vehicle starts empty
        "Capacity",
    )
    capacity_dim = routing.GetDimensionOrDie("Capacity")

    # ----------------------------------------
    # ENERGY DIMENSION (distance-only model)
    # ----------------------------------------
    BASE_KWH_PER_KM = 0.436
    battery_capacity = float(
        data.get("battery_capacity", data.get("battery_kwh", 100.0))
    )

    def energy_cb(i_idx, j_idx):
        i = manager.IndexToNode(i_idx)
        j = manager.IndexToNode(j_idx)
        d_km = float(data["distance_km"][i, j])
        return int(round(d_km * BASE_KWH_PER_KM ))

    energy_transit = routing.RegisterTransitCallback(energy_cb)

    routing.AddDimension(
        energy_transit,
        0,
        int(round(battery_capacity)),  # battery units = kWh allowed
        True,  # starts with 0 energy consumed
        "Energy",
    )
    energy_dim = routing.GetDimensionOrDie("Energy")

    # ----------------------------------------
    # Search parameters
    # ----------------------------------------
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))
    params.log_search = False

    try:
        params.use_randomization = True
        params.random_seed = seed
    except Exception:
        pass

    log(f"🚀 Solving (Tabu Search, {time_limit_s}s)...")
    solution = routing.SolveWithParameters(params)

    log("")
    log("================ OR-TOOLS TABU RESULT ================")

    # ----------------------------------------
    # NO SOLUTION
    # ----------------------------------------
    if solution is None:
        log("❌ Çözüm bulunamadı (time limit içinde).")
        log("🔍 Şunları deneyebilirsiniz:")
        log("   • Zaman limitini artır (time_limit_s)")
        log("   • Araç kapasitesini artır (vehicle_cap_desi)")
        log("   • Çalışma süresini gevşet (09:00–18:00 sınırı)")
        log("   • Batarya kapasitesini artır (şu an ~= 100 kWh)")
        log("====================================================")
        return {
            "routing": routing,
            "manager": manager,
            "solution": None,
            "time_dim": time_dim,
            "energy_dim": energy_dim,
            "capacity_dim": capacity_dim,
            "log": "\n".join(log_lines),
        }

    log("✅ Çözüm bulundu!")
    log(f"🔢 Vehicles: {data['num_vehicles']} | Nodes: {n}\n")

    # ----------------------------------------
    # ROUTE OUTPUT
    # ----------------------------------------
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)

        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # empty route

        log(f"\n🚚 Araç {v} rotası:")
        log("From->To | Dist(km) | Time(min) | Speed(km/h) |  Arr  |  Dep  | Desi | Serv(min) | CumDesi | Energy(kWh)")
        log("-" * 110)

        total_dist = 0.0
        total_energy = 0.0
        total_service = 0.0
        total_travel = 0.0

        prev_node = None
        prev_idx = None

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            next_idx = solution.Value(routing.NextVar(idx))

            arr_time = solution.Value(time_dim.CumulVar(idx))
            service_t = float(service_min[node] if node != data["depot"] else 0.0)
            dep_time = arr_time + service_t
            cum_load_at_node = solution.Value(capacity_dim.CumulVar(idx))

            if prev_node is not None:
                leg_d = float(data["distance_km"][prev_node, node])
                leg_t = float(time_min[prev_node, node])
                avg_speed = (leg_d / (leg_t / 60.0)) if leg_t > 0 else 0.0

                total_dist += leg_d
                total_travel += leg_t
                total_service += service_t

                # Reporting formula (not used for constraints)
                load_before_leg = solution.Value(capacity_dim.CumulVar(prev_idx))
                en_used = leg_d * (0.436 + 0.002 * (load_before_leg / 1000.0))
                total_energy += en_used

                log(
                    f"{prev_node:>2}->{node:<2} | {leg_d:8.2f} | {leg_t:9.1f} | {avg_speed:11.1f} | "
                    f"{_fmt_time(arr_time):>6} | {_fmt_time(dep_time):>6} | "
                    f"{(0 if node == data['depot'] else int(data['demand_desi'][node])):5} | "
                    f"{(0.0 if node == data['depot'] else service_t):8.1f} | "
                    f"{cum_load_at_node:8.0f} | {en_used:10.3f}"
                )

            prev_node = node
            prev_idx = idx
            idx = next_idx

        # Last leg to depot
        node = manager.IndexToNode(idx)
        arr_time = solution.Value(time_dim.CumulVar(idx))
        cum_load_at_depot = solution.Value(capacity_dim.CumulVar(idx))

        leg_d = float(data["distance_km"][prev_node, node])
        leg_t = float(time_min[prev_node, node])
        avg_speed = (leg_d / (leg_t / 60.0)) if leg_t > 0 else 0.0

        load_before_leg = solution.Value(capacity_dim.CumulVar(prev_idx))
        en_used = leg_d * (0.436 + 0.002 * (load_before_leg / 1000.0))

        total_dist += leg_d
        total_travel += leg_t
        total_energy += en_used

        log(
            f"{prev_node:>2}->{node:<2} | {leg_d:8.2f} | {leg_t:9.1f} | {avg_speed:11.1f} | "
            f"{_fmt_time(arr_time):>6} | {'-':>6} | "
            f"{'-':>5} | {'-':>8} | "
            f"{cum_load_at_depot:8.0f} | {en_used:10.3f}"
        )

        log("-" * 110)
        log(f"🧾 Araç {v} özeti:")
        log(f"   • Toplam mesafe: {total_dist:.2f} km")
        log(f"   • Toplam seyahat süresi: {total_travel:.1f} dk")
        log(f"   • Toplam servis süresi: {total_service:.1f} dk")
        log(f"   • Toplam çalışma süresi: {(total_travel + total_service):.1f} dk")
        log(f"   • Tahmini enerji tüketimi: {total_energy:.3f} kWh\n")

    return {
        "routing": routing,
        "manager": manager,
        "solution": solution,
        "time_dim": time_dim,
        "energy_dim": energy_dim,
        "capacity_dim": capacity_dim,
        "log": "\n".join(log_lines),
    }
