import numpy as np

BASE_KWH_PER_KM = 0.436      # empty vehicle energy model (matching OR-Tools)


def evrp_feasibility_detailed(data, work_start_min=9*60, work_end_min=18*60):
    """
    Highly detailed EVRP feasibility checker.
    Provides:
        ✔ Capacity feasibility (customer + fleet)
        ✔ Time feasibility (per-node)
        ✔ Battery feasibility (distance-only OR-Tools model)
        ✔ Pairwise time feasibility
        ✔ Lists all problematic nodes
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

    report = []
    feasible = True

    report.append("========== 🔍 EVRP FEASIBILITY DEBUGGER ==========\n")

    # =====================================================================
    # 1) CAPACITY CHECK ----------------------------------------------------
    # =====================================================================
    report.append("### 1) CAPACITY CHECK\n")

    # A) Customers exceeding vehicle capacity
    oversized = np.where(demand > cap)[0]
    if len(oversized) > 0:
        feasible = False
        report.append("❌ Customers exceeding vehicle capacity:")
        for idx in oversized:
            report.append(f"   - Node {idx}: {demand[idx]} desi > capacity {cap}")
    else:
        report.append("✅ No customer exceeds vehicle capacity.")

    # B) Total fleet capacity
    total_demand = demand.sum()
    total_capacity = num_vehicles * cap

    if total_capacity < total_demand:
        feasible = False
        report.append(
            f"❌ Total demand {total_demand:.1f} > fleet capacity {total_capacity:.1f}"
        )
    else:
        report.append("✅ Fleet capacity is sufficient.")

    lb_cap = int(np.ceil(total_demand / cap))
    report.append(f"ℹ️ Minimum vehicles (capacity): {lb_cap}\n")

    # =====================================================================
    # 2) TIME FEASIBILITY -------------------------------------------------
    # =====================================================================
    report.append("### 2) TIME HORIZON CHECK\n")

    impossible_time_nodes = []

    for i in range(n):
        if i == depot:
            continue

        travel_out = T[depot, i]
        travel_back = T[i, depot]
        required = travel_out + service[i] + travel_back

        if required > horizon:
            impossible_time_nodes.append((i, required))

    if impossible_time_nodes:
        feasible = False
        report.append("❌ Nodes impossible to serve within time horizon:")
        for node, mins in impossible_time_nodes:
            report.append(f"   - Node {node}: requires {mins:.1f} min > {horizon} min")
    else:
        report.append("✅ All nodes can be served within working hours.")

    # Lower bound by total minimal travel
    min_travel = [min(T[depot, i], T[i, depot]) for i in range(n) if i != depot]
    approx_total_min = sum(min_travel) + sum(service)
    lb_time = int(np.ceil(approx_total_min / horizon))

    report.append(f"ℹ️ Minimum vehicles (time): {lb_time}\n")

    # =====================================================================
    # 3) ENERGY FEASIBILITY — DISTANCE ONLY (MATCHES OR-TOOLS)
    # =====================================================================
    report.append("### 🔋 ENERGY CHECK (distance-based, matching OR-Tools)\n")

    unreachable_orders = []
    worst_round_trip_energy = 0.0
    worst_round_trip_node = None

    for i in range(n):
        if i == depot:
            continue

        dist_out  = D[depot, i]
        dist_back = D[i, depot]

        e_out  = dist_out  * BASE_KWH_PER_KM
        e_back = dist_back * BASE_KWH_PER_KM
        e_round = e_out + e_back

        # track worst round trip
        if e_round > worst_round_trip_energy:
            worst_round_trip_energy = e_round
            worst_round_trip_node = i

        # *** THIS is what matters for OR-Tools: full tour energy ***
        if e_round > battery:
            unreachable_orders.append(
                (i, dist_out, dist_back, e_out, e_back, e_round)
            )

    if unreachable_orders:
        feasible = False
        report.append("❌ Customers whose *round trip* exceeds battery:")
        for (node, d_out, d_back, e_out, e_back, e_round) in unreachable_orders:
            report.append(
                f" - Node {node}: out {d_out:.2f} km ({e_out:.1f} kWh) + "
                f"back {d_back:.2f} km ({e_back:.1f} kWh) = "
                f"{e_round:.1f} kWh > battery {battery:.1f} kWh"
            )
    else:
        report.append("✅ All customers feasible w.r.t round-trip battery.\n")

    # Lower bound by energy (using minimal round-trips)
    total_min_energy = sum([
        (min(D[depot, i], D[i, depot]) * 2.0) * BASE_KWH_PER_KM
        for i in range(n) if i != depot
    ])
    lb_energy = int(np.ceil(total_min_energy / battery))
    report.append(f"ℹ️ Minimum vehicles (energy): {lb_energy}\n")

    # =====================================================================
    # 4) PAIRWISE ROUTE FEASIBILITY --------------------------------------
    # =====================================================================
    report.append("### 4) PAIRWISE TIME FEASIBILITY\n")

    bad_pairs = []

    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or j == i:
                continue

            # travel: depot -> i -> j -> depot
            travel = (
                T[depot, i]
                + service[i]
                + T[i, j]
                + service[j]
                + T[j, depot]
            )

            if travel > horizon:
                bad_pairs.append((i, j, travel))

    if bad_pairs:
        feasible = False
        report.append("❌ Customer pairs that cannot be on the same route:")
        for i, j, mins in bad_pairs:
            report.append(
                f"   - ({i}, {j}) requires {mins:.1f} min > horizon {horizon} min"
            )
    else:
        report.append("✅ All customer pairs fit within the time horizon.\n")

    # =====================================================================
    # SUMMARY --------------------------------------------------------------
    # =====================================================================
    report.append("========== SUMMARY ==========")

    required = max(lb_cap, lb_time, lb_energy)
    report.append(f"➡️ Required minimum vehicles: {required}")
    report.append(f"➡️ Current fleet size: {num_vehicles}")

    if num_vehicles < required:
        report.append("❌ Fleet size insufficient.")
        feasible = False
    else:
        report.append("🎉 Fleet size sufficient!")

    return feasible, "\n".join(report)
