import random
import numpy as np
from copy import deepcopy

# ============================================================
# ⏱️ WORKDAY CONSTANTS
# ============================================================
WORK_START_MIN = 9 * 60
WORK_END_MIN = 18 * 60
MAX_ROUTE_MIN = WORK_END_MIN - WORK_START_MIN  # 540


# ============================================================
# ⚡ ENERGY FORMULA — Distance + Load model
# Units: kWh (will be converted to % when displaying)
# ============================================================
def leg_energy_kwh(d_km: float, load_before: float) -> float:
    """
    Energy model (kWh) - DISTANCE + LOAD:
      E = 0.436 * d_km + 0.002 * load_before

    Returns energy consumption in kWh.
    """
    return 0.436 * float(d_km) + 0.002 * float(load_before)


def leg_energy_kwh_node_proxy(d_km: float, from_node_desi: float) -> float:
    """
    Tabu-compatible proxy for formula-based energy.
    Uses known from-node desi term so both Tabu and GA can enforce the same model.
    """
    return 0.436 * float(d_km) + 0.002 * float(from_node_desi)


def _base_energy_per_km(data) -> float:
    """
    Base kWh/km used by OR-Tools energy dimension (distance-only model).
    Falls back to legacy constant if not provided.
    """

    if "base_kwh_per_100km" in data:
        return float(data["base_kwh_per_100km"]) / 100.0
    return 0.436

def route_energy_objective_pickup(route, D, demand, depot=0) -> float:
    """
        Tabu-compatible energy proxy model:
            Energy = 0.436 * distance + 0.002 * from_node_desi
    
    Returns total energy consumption in kWh.
    """
    if not route:
        return 0.0

    E = 0.0

    # depot -> first
    first = route[0]
    E += leg_energy_kwh_node_proxy(D[depot, first], 0.0)

    # internal legs
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        E += leg_energy_kwh_node_proxy(D[a, b], demand[a])

    # last -> depot
    last = route[-1]
    E += leg_energy_kwh_node_proxy(D[last, depot], demand[last])
    return E


# ============================================================
# ⚙️ FAST CONTEXT BUILD (important for 300+ nodes)
# ============================================================
def _build_cost_context(data):
    """
    Precompute reusable arrays / structures once for speed.
    """
    D = np.asarray(data["distance_km"], dtype=float)
    depot = int(data.get("depot", 0))
    demand = np.asarray(data["demand_desi"], dtype=float)
    service = np.asarray(data["service_min"], dtype=float)
    vehicle_cap = float(data.get("vehicle_cap_desi", 0.0))
    battery_cap = float(data.get("battery_capacity", 0.0))
    fuel_liters_per_100km = float(data.get("fuel_liters_per_100km", 0.0))
    fuel_tank_liters = float(data.get("fuel_tank_liters", 0.0))

    # Time matrices
    T_static = None
    if data.get("time_min") is not None:
        T_static = np.asarray(data["time_min"], dtype=float)

    T_by_hour = data.get("time_min_by_hour", None)
    hours_sorted = None
    if T_by_hour is not None:
        hours_sorted = sorted(T_by_hour.keys())  # cache once

    return {
        "D": D,
        "depot": depot,
        "demand": demand,
        "service": service,
        "vehicle_cap": vehicle_cap,
        "battery_cap": battery_cap,
        "fuel_liters_per_100km": fuel_liters_per_100km,
        "fuel_tank_liters": fuel_tank_liters,
        "num_vehicles": int(data.get("num_vehicles", 0)),
        "base_energy_per_km": _base_energy_per_km(data),
        "T_static": T_static,
        "T_by_hour": T_by_hour,
        "hours_sorted": hours_sorted,
    }


def _leg_time_min(a: int, b: int, t_since_start: float, ctx) -> float:
    """
    Returns travel time (minutes) for leg a->b given t since 09:00.
    Uses T_by_hour if available, else T_static.
    """
    T_by_hour = ctx["T_by_hour"]
    if T_by_hour is None:
        # Static
        return float(ctx["T_static"][a, b])

    # Dynamic hour selection (clamped)
    abs_min = WORK_START_MIN + t_since_start
    hour = int(abs_min // 60)

    hours = ctx["hours_sorted"]
    if hour <= hours[0]:
        h_sel = hours[0]
    elif hour >= hours[-1]:
        h_sel = hours[-1]
    else:
        # Fast clamp-to-nearest: linear is fine for 9..18 (only ~10 keys)
        # If you have many keys, replace with bisect.
        h_sel = min(hours, key=lambda h: abs(h - hour))

    T_h = T_by_hour[h_sel]
    return float(T_h[a, b])


def route_total_time_min(route, ctx, depot=0) -> float:
    if not route:
        return 0.0

    service = ctx["service"]
    t = 0.0

    # depot -> first
    first = route[0]
    t += _leg_time_min(depot, first, t, ctx)
    t += float(service[first])

    # internal
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        t += _leg_time_min(a, b, t, ctx)
        t += float(service[b])

    # last -> depot
    last = route[-1]
    t += _leg_time_min(last, depot, t, ctx)

    return t


def route_time_penalty(route, ctx, depot=0, penalty_per_min=1e5) -> float:
    total_t = route_total_time_min(route, ctx, depot)
    overflow = max(0.0, total_t - MAX_ROUTE_MIN)
    return float(overflow) * float(penalty_per_min)


def route_fuel_liters(route, ctx, depot=0) -> float:
    if not route:
        return 0.0

    D = ctx["D"]
    liters_per_km = float(ctx.get("fuel_liters_per_100km", 0.0)) / 100.0
    if liters_per_km <= 0:
        return 0.0

    total_km = float(D[depot, route[0]])
    for i in range(len(route) - 1):
        total_km += float(D[route[i], route[i + 1]])
    total_km += float(D[route[-1], depot])
    return total_km * liters_per_km


def _build_routes_from_sequence(
    sequence,
    ctx,
    enforce_energy_constraints=True,
    enforce_fuel_constraints=False,
):
    """
    Convert a flat customer permutation into vehicle routes while enforcing
    OR-Tools style constraints (capacity, battery, workday).

    Returns (routes, penalties).
    """
    num_vehicles = ctx["num_vehicles"]
    cap = ctx["vehicle_cap"]
    battery = ctx["battery_cap"]
    fuel_tank = ctx.get("fuel_tank_liters", 0.0)
    fuel_per_km = float(ctx.get("fuel_liters_per_100km", 0.0)) / 100.0
    base_energy = ctx["base_energy_per_km"]
    D = ctx["D"]
    depot = ctx["depot"]
    demand = ctx["demand"]
    service = ctx["service"]

    routes = []
    penalties = 0.0

    v = 0
    current = []
    load = 0.0
    energy = 0.0
    fuel_liters = 0.0
    t = 0.0
    prev = depot

    def finish_route():
        nonlocal current, load, energy, fuel_liters, t, prev
        routes.append(current)
        current = []
        load = 0.0
        energy = 0.0
        fuel_liters = 0.0
        t = 0.0
        prev = depot

    for node in sequence:
        # If we already exceeded vehicle count, keep adding to last vehicle but penalize heavily.
        if v >= num_vehicles:
            penalties += 1e6

        travel = _leg_time_min(prev, node, t, ctx)
        leg_dist = float(D[prev, node])
        from_node_desi = float(demand[prev]) if prev != depot else 0.0
        leg_energy = leg_energy_kwh_node_proxy(leg_dist, from_node_desi)
        svc = float(service[node])

        # Predict constraints including return to depot
        projected_load = load + demand[node]
        projected_return_energy = leg_energy_kwh_node_proxy(float(D[node, depot]), float(demand[node]))
        projected_energy = energy + leg_energy + projected_return_energy
        leg_fuel = leg_dist * fuel_per_km
        projected_fuel = fuel_liters + leg_fuel + float(D[node, depot]) * fuel_per_km
        projected_time = t + travel + svc + _leg_time_min(node, depot, t + travel + svc, ctx)

        violates = (
            projected_load > cap
            or projected_time > MAX_ROUTE_MIN
            or (enforce_energy_constraints and projected_energy > battery)
            or (enforce_fuel_constraints and fuel_tank > 0 and projected_fuel > fuel_tank)
        )

        if violates and v + 1 < num_vehicles:
            finish_route()
            v += 1
            travel = _leg_time_min(prev, node, t, ctx)
            leg_dist = float(D[prev, node])
            from_node_desi = float(demand[prev]) if prev != depot else 0.0
            leg_energy = leg_energy_kwh_node_proxy(leg_dist, from_node_desi)
            svc = float(service[node])

        # Update state
        load += demand[node]
        energy += leg_energy
        fuel_liters += leg_fuel
        t += travel + svc
        current.append(node)
        prev = node

    # finalize last route
    finish_route()

    # pad with empty routes if needed
    while len(routes) < num_vehicles:
        routes.append([])

    # penalties for constraint overflows in each route
    cap_penalty = 0.0
    energy_penalty = 0.0
    fuel_penalty = 0.0
    for r in routes:
        if not r:
            continue
        load_total = sum(demand[n] for n in r)
        if load_total > cap:
            cap_penalty += (load_total - cap) * 1e5

        if enforce_energy_constraints:
            used_kwh = leg_energy_kwh_node_proxy(float(D[depot, r[0]]), 0.0)
            for i in range(len(r) - 1):
                a, b = r[i], r[i + 1]
                used_kwh += leg_energy_kwh_node_proxy(float(D[a, b]), float(demand[a]))
            used_kwh += leg_energy_kwh_node_proxy(float(D[r[-1], depot]), float(demand[r[-1]]))
            if used_kwh > battery:
                energy_penalty += (used_kwh - battery) * 1e5

        if enforce_fuel_constraints and fuel_tank > 0:
            used_fuel = route_fuel_liters(r, ctx, depot=depot)
            if used_fuel > fuel_tank:
                fuel_penalty += (used_fuel - fuel_tank) * 1e5

    penalties += cap_penalty + energy_penalty + fuel_penalty
    return routes[:num_vehicles], penalties


# ============================================================
# ⚙️ TOTAL PLAN COST (FAST)
# ============================================================
def total_plan_cost(data, routes, objective="energy"):
    ctx = _build_cost_context(data)
    D = ctx["D"]
    depot = ctx["depot"]

    if objective == "distance":
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += D[depot, r[0]]
            for i in range(len(r) - 1):
                total += D[r[i], r[i + 1]]
            total += D[r[-1], depot]
        return float(total)

    if objective == "energy":
        demand = ctx["demand"]

        total_E = 0.0
        total_penalty = 0.0

        for r in routes:
            if not r:
                continue
            total_E += route_energy_objective_pickup(r, D, demand, depot=depot)
            total_penalty += route_time_penalty(r, ctx, depot=depot)

        return float(total_E + total_penalty)

    raise ValueError("objective must be 'energy' or 'distance'")


# ============================================================
# 🧬 GA OPERATORS (SAFE)
# ============================================================
def ox_crossover(parent1, parent2):
    """
    Order crossover (OX) — SAFE
    Handles length 0/1 routes (common in VRP).
    """
    n = len(parent1)
    if n <= 1:
        return parent1[:]  # copy

    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = parent1[a:b + 1]

    fill = [x for x in parent2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[j]
            j += 1

    return child


def mutation_swap(seq, p=0.2):
    seq = seq[:]
    if len(seq) >= 2 and random.random() < p:
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def mutation_2opt(seq, p=0.5):
    seq = seq[:]
    if len(seq) >= 4 and random.random() < p:
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i:j + 1] = reversed(seq[i:j + 1])
    return seq


def mutate_route(seq):
    if len(seq) <= 1:
        return seq[:]
    return mutation_2opt(mutation_swap(seq), p=0.7)


def crossover_plan(p1, p2):
    assert len(p1) == len(p2), "Chromosomes must have equal length"
    return ox_crossover(p1, p2)


def mutate_plan(plan):
    return mutate_route(plan)


def tournament_select(pop, fitness, k=3):
    cand = random.sample(range(len(pop)), k)
    cand = min(cand, key=lambda i: fitness[i])
    return deepcopy(pop[cand])


# ============================================================
# 🔧 2-OPT LOCAL SEARCH
# ============================================================
def two_opt_improve_route(route, D, max_iterations=50):
    """
    Apply 2-opt local search to improve a single route.
    Returns improved route.
    """
    if len(route) <= 2:
        return route
    
    improved = True
    iteration = 0
    best_route = route[:]
    
    while improved and iteration < max_iterations:
        improved = False
        best_dist = 0.0
        
        # Calculate current distance
        for i in range(len(best_route) - 1):
            best_dist += D[best_route[i], best_route[i + 1]]
        
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                # Try reversing segment [i+1:j+1]
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                
                # Calculate new distance
                new_dist = 0.0
                for k in range(len(new_route) - 1):
                    new_dist += D[new_route[k], new_route[k + 1]]
                
                if new_dist < best_dist:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            
            if improved:
                break
        
        iteration += 1
    
    return best_route


def apply_2opt_to_routes(routes, D, mode="full"):
    """
    Apply 2-opt to routes based on mode.
    
    mode: 
        - "none": No 2-opt
        - "selective": Apply to best 30% of routes
        - "full": Apply to all routes
    """
    if mode == "none":
        return routes
    
    improved_routes = []
    
    for route in routes:
        if not route:
            improved_routes.append(route)
            continue
        
        if mode == "selective":
            # Apply 2-opt with 30% probability
            if random.random() < 0.3:
                improved_route = two_opt_improve_route(route, D, max_iterations=20)
            else:
                improved_route = route
        else:  # full
            improved_route = two_opt_improve_route(route, D, max_iterations=50)
        
        improved_routes.append(improved_route)
    
    return improved_routes


# ============================================================
# 🧠 GA SOLVER (SCALES BETTER)
# ============================================================
def ga_optimize_sequences(
    data,
    base_routes,
    pop_size=120,
    generations=400,
    objective="energy",
    elitism=2,
    seed=42,
    improvement_mode="none",
    enforce_energy_constraints=True,
    enforce_fuel_constraints=False,
):
    random.seed(seed)

    # Build context ONCE and reuse in evaluate
    ctx = _build_cost_context(data)

    customers = [n for r in base_routes for n in r]

    def evaluate(seq):
        routes, penalties = _build_routes_from_sequence(
            seq,
            ctx,
            enforce_energy_constraints=enforce_energy_constraints,
            enforce_fuel_constraints=enforce_fuel_constraints,
        )
        D = ctx["D"]
        depot = ctx["depot"]

        if objective == "distance":
            total = 0.0
            for r in routes:
                if not r:
                    continue
                total += D[depot, r[0]]
                for i in range(len(r) - 1):
                    total += D[r[i], r[i + 1]]
                total += D[r[-1], depot]
            # Enforce workday with penalties (same as Tabu constraint)
            total_time_pen = sum(route_time_penalty(r, ctx, depot=depot) for r in routes)
            return float(total + penalties + total_time_pen)

        if objective == "energy":
            demand = ctx["demand"]
            total_E = 0.0
            total_pen = penalties
            for r in routes:
                if not r:
                    continue
                total_E += route_energy_objective_pickup(r, D, demand, depot=depot)
                total_pen += route_time_penalty(r, ctx, depot=depot)
            return float(total_E + total_pen)

        raise ValueError("objective must be 'energy' or 'distance'")

    # init population (permutation of all customers)
    base_seq = customers[:]
    population = [deepcopy(base_seq)]
    for _ in range(pop_size - 1):
        seq = base_seq[:]
        random.shuffle(seq)
        population.append(seq)

    fitness = [evaluate(p) for p in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_seq = deepcopy(population[best_idx])
    best_fit = float(fitness[best_idx])

    for _ in range(generations):
        ranked = sorted(range(len(population)), key=lambda i: fitness[i])
        new_pop = [deepcopy(population[i]) for i in ranked[:elitism]]

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitness)
            p2 = tournament_select(population, fitness)
            child = mutate_plan(crossover_plan(p1, p2))
            new_pop.append(child)

        population = new_pop
        fitness = [evaluate(p) for p in population]

        g_best = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[g_best] < best_fit:
            best_fit = float(fitness[g_best])
            best_seq = deepcopy(population[g_best])

    # Apply 2-opt improvement to final routes based on mode
    best_routes, _ = _build_routes_from_sequence(best_seq, ctx)
    if improvement_mode != "none":
        best_routes = apply_2opt_to_routes(best_routes, ctx["D"], mode=improvement_mode)
        # Recalculate fitness after 2-opt
        # Rebuild sequence from improved routes
        improved_seq = [n for r in best_routes for n in r]
        best_fit = evaluate(improved_seq)
    
    return best_routes, best_fit


# ============================================================
# 📄 OR-TOOLS STYLE PRINTER FOR GA
# ============================================================
def print_ga_detailed_solution(data, routes, df_orders):
    ctx = _build_cost_context(data)
    D = ctx["D"]
    depot = ctx["depot"]
    demand = ctx["demand"]
    service = ctx["service"]

    def fmt_time(mins_since_start):
        mins = int(mins_since_start)
        h = 9 + mins // 60
        m = mins % 60
        return f"{h:02d}:{m:02d}"

    txt = []
    txt.append("\n================== GA ENERGY RESULT ==================\n")

    fleet_energy = 0.0

    for v, route in enumerate(routes):
        if not route:
            txt.append(f"🚚 Araç {v}: (boş rota)\n")
            continue

        txt.append(f"\n🚚 Araç {v} rotası:")
        txt.append("From->To | Dist(km) |  Arr  |  Dep  | Desi | Serv(min) | CumDesi | Energy(kWh)")
        txt.append("-" * 85)

        load = 0.0
        total_E = 0.0
        total_dist = 0.0
        t_now = 0.0

        # depot -> first
        f = route[0]
        d = float(D[depot, f])
        travel_t = _leg_time_min(depot, f, t_now, ctx)
        E = leg_energy_kwh_node_proxy(d, 0.0)
        svc = float(service[f])

        arr = t_now + travel_t
        dep = arr + svc

        txt.append(
            f"{depot:>2}->{f:<2} | {d:8.2f} | "
            f"{fmt_time(arr):>6} | {fmt_time(dep):>6} | "
            f"{int(demand[f]):5} | {svc:8.0f} | "
            f"{load:8.0f} | {E:10.3f}"
        )

        load += demand[f]
        t_now = dep
        total_E += E
        total_dist += d

        # internal
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            d = float(D[a, b])
            travel_t = _leg_time_min(a, b, t_now, ctx)
            E = leg_energy_kwh_node_proxy(d, float(demand[a]))
            svc = float(service[b])

            arr = t_now + travel_t
            dep = arr + svc

            txt.append(
                f"{a:>2}->{b:<2} | {d:8.2f} | "
                f"{fmt_time(arr):>6} | {fmt_time(dep):>6} | "
                f"{int(demand[b]):5} | {svc:8.0f} | "
                f"{load:8.0f} | {E:10.3f}"
            )

            load += demand[b]
            t_now = dep
            total_E += E
            total_dist += d

        # last -> depot
        last = route[-1]
        d = float(D[last, depot])
        travel_t = _leg_time_min(last, depot, t_now, ctx)
        E = leg_energy_kwh_node_proxy(d, float(demand[last]))
        arr = t_now + travel_t

        txt.append(
            f"{last:>2}->{depot:<2} | {d:8.2f} | "
            f"{fmt_time(arr):>6} | {'-':>6} | "
            f"{'-':>5} | {'-':>8} | "
            f"{load:8.0f} | {E:10.3f}"
        )

        total_E += E
        total_dist += d

        txt.append("-" * 85)
        txt.append(f"🧾 Araç {v} özeti:")
        txt.append(f"   • Toplam mesafe: {total_dist:.2f} km")
        txt.append(f"   • Tahmini enerji tüketimi: {total_E:.3f} kWh\n")

        fleet_energy += total_E

    txt.append("======================================================")
    txt.append(f"🏆 Toplam filo enerjisi: {fleet_energy:.3f} kWh")
    txt.append("======================================================\n")

    return "\n".join(txt)
