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
# ⚡ ENERGY FORMULA — Pickup Type (same as OR-Tools)
# Units: kWh (despite docstring in your older version)
# ============================================================
def leg_energy_kwh(d_km: float, load_before: float) -> float:
    """
    Energy model (kWh):
      E = d_km * (0.436 + 0.002 * load_before)

    NOTE: load_before here is in "desi" in your pipeline; keep consistent.
    """
    return float(d_km) * (0.436 + 0.002 * float(load_before))


def route_energy_objective_pickup(route, D, demand, depot=0) -> float:
    """
    Pickup model:
      load starts at 0 and increases at each visit.
    """
    if not route:
        return 0.0

    load = 0.0
    E = 0.0

    # depot -> first
    first = route[0]
    E += leg_energy_kwh(D[depot, first], load)
    load += demand[first]

    # internal legs
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        E += leg_energy_kwh(D[a, b], load)
        load += demand[b]

    # last -> depot
    last = route[-1]
    E += leg_energy_kwh(D[last, depot], load)

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
    assert len(p1) == len(p2), "Plans must have the same vehicle count"
    return [ox_crossover(p1[v], p2[v]) for v in range(len(p1))]


def mutate_plan(plan):
    return [mutate_route(r) for r in plan]


def tournament_select(pop, fitness, k=3):
    cand = random.sample(range(len(pop)), k)
    cand = min(cand, key=lambda i: fitness[i])
    return deepcopy(pop[cand])


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
    seed=42
):
    random.seed(seed)

    # Build context ONCE and reuse in evaluate
    ctx = _build_cost_context(data)

    def evaluate(plan):
        # Inline fast cost (avoid rebuilding ctx)
        D = ctx["D"]
        depot = ctx["depot"]

        if objective == "distance":
            total = 0.0
            for r in plan:
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
            total_pen = 0.0
            for r in plan:
                if not r:
                    continue
                total_E += route_energy_objective_pickup(r, D, demand, depot=depot)
                total_pen += route_time_penalty(r, ctx, depot=depot)
            return float(total_E + total_pen)

        raise ValueError("objective must be 'energy' or 'distance'")

    # init population
    population = [deepcopy(base_routes)]
    for _ in range(pop_size - 1):
        population.append([mutate_route(r) for r in base_routes])

    fitness = [evaluate(p) for p in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_plan = deepcopy(population[best_idx])
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
            best_plan = deepcopy(population[g_best])

    return best_plan, best_fit


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
        E = leg_energy_kwh(d, load)
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
            E = leg_energy_kwh(d, load)
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
        E = leg_energy_kwh(d, load)
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
