"""
ALNS + MIP polishing solver for multi-trip assignment.

This solver takes precomputed "jobs" (each job is a full route/trip with metrics)
and reassigns them to vehicles under shift-duration and usable-energy limits.

Pipeline:
1) Construct a greedy feasible assignment.
2) Improve assignment with ALNS destroy/repair iterations.
3) Run a small MIP polishing model to rebalance and possibly reduce vehicles.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

try:
    from ortools.linear_solver import pywraplp
except Exception:  # pragma: no cover
    pywraplp = None


@dataclass
class _Job:
    idx: int
    time_min: float
    energy_kwh: float
    distance_km: float
    load_desi: float
    payload: Dict[str, Any]


def _vehicle_metrics(
    vehicle_jobs: List[_Job],
    depot_service_time: float,
) -> Tuple[float, float, float, float]:
    """Return (time_min, energy_kwh, distance_km, load_desi) for one vehicle."""
    if not vehicle_jobs:
        return 0.0, 0.0, 0.0, 0.0

    t = sum(j.time_min for j in vehicle_jobs)
    # Extra depot handling only between consecutive trips.
    t += depot_service_time * max(0, len(vehicle_jobs) - 1)
    e = sum(j.energy_kwh for j in vehicle_jobs)
    d = sum(j.distance_km for j in vehicle_jobs)
    l = sum(j.load_desi for j in vehicle_jobs)
    return t, e, d, l


def _is_feasible_vehicle(
    vehicle_jobs: List[_Job],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
) -> bool:
    t, e, _, _ = _vehicle_metrics(vehicle_jobs, depot_service_time)
    return (t <= max_shift_duration + 1e-9) and (e <= usable_energy + 1e-9)


def _score_solution(
    assignment: List[List[_Job]],
    depot_service_time: float,
) -> float:
    """
    Lexicographic-like scalar score:
    - primary: number of used vehicles
    - secondary: total time
    - tertiary: workload balance
    """
    used = [v for v in assignment if v]
    if not used:
        return 0.0

    per_vehicle_time = [_vehicle_metrics(v, depot_service_time)[0] for v in used]
    total_time = sum(per_vehicle_time)
    mean_t = total_time / len(per_vehicle_time)
    variance = sum((x - mean_t) ** 2 for x in per_vehicle_time) / len(per_vehicle_time)
    std_t = math.sqrt(variance)

    return 1_000_000.0 * len(used) + 10.0 * total_time + std_t


def _deepcopy_assignment(assignment: List[List[_Job]]) -> List[List[_Job]]:
    return [list(v) for v in assignment]


def _cleanup_empty_vehicles(assignment: List[List[_Job]]) -> List[List[_Job]]:
    return [v for v in assignment if v]


def _build_greedy_initial(
    jobs: List[_Job],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
) -> List[List[_Job]]:
    """First-fit decreasing by time."""
    assignment: List[List[_Job]] = []
    for job in sorted(jobs, key=lambda x: x.time_min, reverse=True):
        placed = False
        for v in assignment:
            cand = v + [job]
            if _is_feasible_vehicle(cand, max_shift_duration, usable_energy, depot_service_time):
                v.append(job)
                placed = True
                break
        if not placed:
            assignment.append([job])
    return _cleanup_empty_vehicles(assignment)


def _repair_best_insertion(
    assignment: List[List[_Job]],
    removed_jobs: List[_Job],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
) -> List[List[_Job]]:
    """Insert removed jobs one by one in best feasible place, else open new vehicle."""
    for job in sorted(removed_jobs, key=lambda x: x.time_min, reverse=True):
        best_vidx = None
        best_score = float("inf")

        for vidx, v in enumerate(assignment):
            cand = v + [job]
            if not _is_feasible_vehicle(cand, max_shift_duration, usable_energy, depot_service_time):
                continue

            before = _score_solution(assignment, depot_service_time)
            old = assignment[vidx]
            assignment[vidx] = cand
            after = _score_solution(assignment, depot_service_time)
            assignment[vidx] = old

            delta = after - before
            if delta < best_score:
                best_score = delta
                best_vidx = vidx

        if best_vidx is None:
            assignment.append([job])
        else:
            assignment[best_vidx].append(job)

    return _cleanup_empty_vehicles(assignment)


def _random_destroy(
    assignment: List[List[_Job]],
    destroy_count: int,
    rng: random.Random,
) -> Tuple[List[List[_Job]], List[_Job]]:
    """Randomly remove jobs from current assignment."""
    non_empty = [(vidx, jidx) for vidx, v in enumerate(assignment) for jidx, _ in enumerate(v)]
    if not non_empty:
        return assignment, []

    destroy_count = max(1, min(destroy_count, len(non_empty)))
    rng.shuffle(non_empty)
    to_remove = non_empty[:destroy_count]

    removed: List[_Job] = []
    for vidx, jidx in sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True):
        removed.append(assignment[vidx].pop(jidx))

    return _cleanup_empty_vehicles(assignment), removed


def _try_random_swap(
    assignment: List[List[_Job]],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
    rng: random.Random,
) -> List[List[_Job]]:
    """Try a random cross-vehicle swap if it improves score."""
    if len(assignment) < 2:
        return assignment

    v1, v2 = rng.sample(range(len(assignment)), 2)
    if not assignment[v1] or not assignment[v2]:
        return assignment

    j1 = rng.randrange(len(assignment[v1]))
    j2 = rng.randrange(len(assignment[v2]))

    cand = _deepcopy_assignment(assignment)
    cand[v1][j1], cand[v2][j2] = cand[v2][j2], cand[v1][j1]

    if not _is_feasible_vehicle(cand[v1], max_shift_duration, usable_energy, depot_service_time):
        return assignment
    if not _is_feasible_vehicle(cand[v2], max_shift_duration, usable_energy, depot_service_time):
        return assignment

    if _score_solution(cand, depot_service_time) < _score_solution(assignment, depot_service_time):
        return _cleanup_empty_vehicles(cand)

    return assignment


def _mip_polish(
    jobs: List[_Job],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
    max_vehicles: int,
    time_limit_s: int,
) -> Tuple[List[List[_Job]], str]:
    """Small MIP polishing over fixed candidate vehicle pool."""
    if pywraplp is None:
        return [], "MIP polishing atlandı: ortools linear solver yüklenemedi."

    if not jobs:
        return [], "MIP polishing atlandı: iş listesi boş."

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return [], "MIP polishing atlandı: SCIP backend oluşturulamadı."

    solver.SetTimeLimit(int(max(1, time_limit_s) * 1000))

    n = len(jobs)
    v_count = max(1, max_vehicles)

    x = {}
    z = {}
    for v in range(v_count):
        z[v] = solver.BoolVar(f"z_{v}")
        for j in range(n):
            x[j, v] = solver.BoolVar(f"x_{j}_{v}")

    max_time = solver.NumVar(0.0, solver.infinity(), "max_time")
    max_energy = solver.NumVar(0.0, solver.infinity(), "max_energy")

    # Every job must be assigned exactly once.
    for j in range(n):
        solver.Add(sum(x[j, v] for v in range(v_count)) == 1)

    # Vehicle capacity constraints in time and energy.
    for v in range(v_count):
        n_jobs_v = sum(x[j, v] for j in range(n))
        time_expr = (
            sum(jobs[j].time_min * x[j, v] for j in range(n))
            + depot_service_time * (n_jobs_v - z[v])
        )
        energy_expr = sum(jobs[j].energy_kwh * x[j, v] for j in range(n))

        solver.Add(time_expr <= max_shift_duration + 1e-9)
        solver.Add(energy_expr <= usable_energy + 1e-9)

        # Link usage variable and assignments.
        solver.Add(n_jobs_v <= n * z[v])
        solver.Add(z[v] <= n_jobs_v)

        solver.Add(time_expr <= max_time)
        solver.Add(energy_expr <= max_energy)

    # Allow MIP to reduce active vehicle count vs ALNS result.
    solver.Add(sum(z[v] for v in range(v_count)) <= max_vehicles)

    # Minimize used vehicles first, then balance workloads.
    objective = solver.Objective()
    for v in range(v_count):
        objective.SetCoefficient(z[v], 1000.0)
    objective.SetCoefficient(max_time, 1.0)
    objective.SetCoefficient(max_energy, 0.2)
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return [], "MIP polishing uygun çözüm bulamadı; ALNS sonucu korunuyor."

    polished: List[List[_Job]] = [[] for _ in range(v_count)]
    for j in range(n):
        for v in range(v_count):
            if x[j, v].solution_value() > 0.5:
                polished[v].append(jobs[j])
                break

    polished = _cleanup_empty_vehicles(polished)
    return polished, "MIP polishing uygulandı."


def solve_multitrip_alns_mip(
    jobs: List[Dict[str, Any]],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
    battery_capacity: float,
    iterations: int = 400,
    destroy_rate: float = 0.25,
    seed: int = 42,
    mip_time_limit_s: int = 5,
) -> Dict[str, Any]:
    """
    Multi-trip assignment optimization with ALNS + MIP polishing.

    Args:
        jobs: list of dicts with at least {job_id, time_min, energy_kwh, distance_km, load_desi, route}
        max_shift_duration: max allowed shift time per vehicle (minutes)
        usable_energy: max allowed usable battery per vehicle (kWh)
        depot_service_time: service/reload overhead between trips (minutes)
        battery_capacity: full battery capacity (kWh), used for reporting only
        iterations: ALNS iterations
        destroy_rate: fraction of jobs removed in destroy phase
        seed: rng seed
        mip_time_limit_s: small time limit for polishing stage
    """
    rng = random.Random(int(seed))

    candidate_jobs: List[_Job] = []
    dropped_jobs: List[Dict[str, Any]] = []

    for i, raw in enumerate(jobs):
        t = float(raw.get("time_min", 0.0))
        e = float(raw.get("energy_kwh", 0.0))
        d = float(raw.get("distance_km", 0.0))
        l = float(raw.get("load_desi", 0.0))

        if t > max_shift_duration + 1e-9 or e > usable_energy + 1e-9:
            dropped_jobs.append(dict(raw))
            continue

        candidate_jobs.append(
            _Job(
                idx=i,
                time_min=t,
                energy_kwh=e,
                distance_km=d,
                load_desi=l,
                payload=dict(raw),
            )
        )

    log_lines = []
    log_lines.append("ALNS + MIP multi-trip optimizasyonu başlatıldı.")
    log_lines.append(f"Toplam iş: {len(jobs)} | Uygun iş: {len(candidate_jobs)} | Hariç tutulan: {len(dropped_jobs)}")

    if not candidate_jobs:
        return {
            "assignments": [],
            "dropped_jobs": dropped_jobs,
            "log": "\n".join(log_lines + ["Uygun iş yok; çözüm üretilemedi."]),
            "stats": {
                "used_vehicles": 0,
                "score": 0.0,
                "mip_applied": False,
            },
        }

    current = _build_greedy_initial(
        jobs=candidate_jobs,
        max_shift_duration=max_shift_duration,
        usable_energy=usable_energy,
        depot_service_time=depot_service_time,
    )
    best = _deepcopy_assignment(current)

    best_score = _score_solution(best, depot_service_time)
    log_lines.append(f"Başlangıç (greedy) araç sayısı: {len(best)}")

    destroy_count = max(1, int(round(destroy_rate * len(candidate_jobs))))
    temp = max(0.5, 0.02 * len(candidate_jobs))

    for _ in range(max(1, int(iterations))):
        cand = _deepcopy_assignment(current)
        cand, removed = _random_destroy(cand, destroy_count=destroy_count, rng=rng)
        cand = _repair_best_insertion(
            assignment=cand,
            removed_jobs=removed,
            max_shift_duration=max_shift_duration,
            usable_energy=usable_energy,
            depot_service_time=depot_service_time,
        )

        if rng.random() < 0.25:
            cand = _try_random_swap(
                assignment=cand,
                max_shift_duration=max_shift_duration,
                usable_energy=usable_energy,
                depot_service_time=depot_service_time,
                rng=rng,
            )

        s_cur = _score_solution(current, depot_service_time)
        s_new = _score_solution(cand, depot_service_time)
        delta = s_new - s_cur

        if delta <= 0 or rng.random() < math.exp(-delta / max(1e-6, temp)):
            current = cand

        if s_new < best_score:
            best = _deepcopy_assignment(cand)
            best_score = s_new

        temp *= 0.999

    log_lines.append(f"ALNS sonrası araç sayısı: {len(best)}")

    # Small MIP polishing over ALNS vehicle budget.
    polished, mip_msg = _mip_polish(
        jobs=candidate_jobs,
        max_shift_duration=max_shift_duration,
        usable_energy=usable_energy,
        depot_service_time=depot_service_time,
        max_vehicles=len(best),
        time_limit_s=int(mip_time_limit_s),
    )
    log_lines.append(mip_msg)

    final_assignment = polished if polished else best
    final_score = _score_solution(final_assignment, depot_service_time)

    # Build API-compatible assignment payload.
    assignments: List[Dict[str, Any]] = []
    for vid, vjobs in enumerate(final_assignment, start=1):
        t, e, d, l = _vehicle_metrics(vjobs, depot_service_time)
        payload_jobs = [dict(j.payload) for j in vjobs]
        assignments.append(
            {
                "vehicle_id": vid,
                "jobs": payload_jobs,
                "num_trips": len(payload_jobs),
                "time_min": t,
                "distance_km": d,
                "load_desi": l,
                "energy_kwh": e,
                "remaining_energy_kwh": max(0.0, battery_capacity - e),
            }
        )

    log_lines.append(f"Nihai araç sayısı: {len(assignments)}")
    log_lines.append("ALNS + MIP optimizasyonu tamamlandı.")

    return {
        "assignments": assignments,
        "dropped_jobs": dropped_jobs,
        "log": "\n".join(log_lines),
        "stats": {
            "used_vehicles": len(assignments),
            "score": final_score,
            "mip_applied": bool(polished),
        },
    }


def solve_multitrip_tabu_mip(
    jobs: List[Dict[str, Any]],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
    battery_capacity: float,
    iterations: int = 400,
    tabu_tenure: int = 20,
    seed: int = 42,
    mip_time_limit_s: int = 5,
) -> Dict[str, Any]:
    """Multi-trip assignment optimization with Tabu Search + MIP polishing."""
    rng = random.Random(int(seed))

    candidate_jobs: List[_Job] = []
    dropped_jobs: List[Dict[str, Any]] = []

    for i, raw in enumerate(jobs):
        t = float(raw.get("time_min", 0.0))
        e = float(raw.get("energy_kwh", 0.0))
        d = float(raw.get("distance_km", 0.0))
        l = float(raw.get("load_desi", 0.0))

        if t > max_shift_duration + 1e-9 or e > usable_energy + 1e-9:
            dropped_jobs.append(dict(raw))
            continue

        candidate_jobs.append(
            _Job(
                idx=i,
                time_min=t,
                energy_kwh=e,
                distance_km=d,
                load_desi=l,
                payload=dict(raw),
            )
        )

    log_lines: List[str] = []
    log_lines.append("Tabu + MIP multi-trip optimizasyonu başlatıldı.")
    log_lines.append(f"Toplam iş: {len(jobs)} | Uygun iş: {len(candidate_jobs)} | Hariç tutulan: {len(dropped_jobs)}")

    if not candidate_jobs:
        return {
            "assignments": [],
            "dropped_jobs": dropped_jobs,
            "log": "\n".join(log_lines + ["Uygun iş yok; çözüm üretilemedi."]),
            "stats": {"used_vehicles": 0, "score": 0.0, "mip_applied": False},
        }

    current = _build_greedy_initial(
        jobs=candidate_jobs,
        max_shift_duration=max_shift_duration,
        usable_energy=usable_energy,
        depot_service_time=depot_service_time,
    )
    best = _deepcopy_assignment(current)
    best_score = _score_solution(best, depot_service_time)
    log_lines.append(f"Başlangıç (greedy) araç sayısı: {len(best)}")

    tabu_until: Dict[Tuple[int, int, int], int] = {}
    iters = max(1, int(iterations))
    tenure = max(2, int(tabu_tenure))

    for it in range(iters):
        best_cand = None
        best_cand_score = float("inf")
        best_move_key = None

        trials = max(12, 3 * max(1, len(candidate_jobs)))
        for _ in range(trials):
            if not current:
                break
            cand = _deepcopy_assignment(current)

            if len(cand) >= 2 and rng.random() < 0.65:
                src = rng.randrange(len(cand))
                if not cand[src]:
                    continue
                jpos = rng.randrange(len(cand[src]))
                job = cand[src].pop(jpos)

                if not cand[src]:
                    cand.pop(src)
                if not cand:
                    cand = [[job]]
                    move_key = (job.idx, src, 0)
                else:
                    dst = rng.randrange(len(cand))
                    cand[dst].append(job)
                    move_key = (job.idx, src, dst)

                cand = _cleanup_empty_vehicles(cand)
                if any(not _is_feasible_vehicle(v, max_shift_duration, usable_energy, depot_service_time) for v in cand):
                    continue
            else:
                if len(cand) < 2:
                    continue
                v1, v2 = rng.sample(range(len(cand)), 2)
                if not cand[v1] or not cand[v2]:
                    continue
                j1 = rng.randrange(len(cand[v1]))
                j2 = rng.randrange(len(cand[v2]))
                cand[v1][j1], cand[v2][j2] = cand[v2][j2], cand[v1][j1]
                move_key = (cand[v1][j1].idx, v1, v2)

                if not _is_feasible_vehicle(cand[v1], max_shift_duration, usable_energy, depot_service_time):
                    continue
                if not _is_feasible_vehicle(cand[v2], max_shift_duration, usable_energy, depot_service_time):
                    continue

            cand_score = _score_solution(cand, depot_service_time)
            is_tabu = tabu_until.get(move_key, -1) > it
            aspiration = cand_score < best_score
            if is_tabu and not aspiration:
                continue

            if cand_score < best_cand_score:
                best_cand = cand
                best_cand_score = cand_score
                best_move_key = move_key

        if best_cand is None:
            # Diversification step
            cand = _deepcopy_assignment(current)
            cand, removed = _random_destroy(cand, destroy_count=max(1, len(candidate_jobs) // 10), rng=rng)
            cand = _repair_best_insertion(
                assignment=cand,
                removed_jobs=removed,
                max_shift_duration=max_shift_duration,
                usable_energy=usable_energy,
                depot_service_time=depot_service_time,
            )
            current = cand
        else:
            current = best_cand
            if best_move_key is not None:
                tabu_until[best_move_key] = it + tenure

        cur_score = _score_solution(current, depot_service_time)
        if cur_score < best_score:
            best = _deepcopy_assignment(current)
            best_score = cur_score

    log_lines.append(f"Tabu sonrası araç sayısı: {len(best)}")

    polished, mip_msg = _mip_polish(
        jobs=candidate_jobs,
        max_shift_duration=max_shift_duration,
        usable_energy=usable_energy,
        depot_service_time=depot_service_time,
        max_vehicles=len(best),
        time_limit_s=int(mip_time_limit_s),
    )
    log_lines.append(mip_msg)

    final_assignment = polished if polished else best
    final_score = _score_solution(final_assignment, depot_service_time)

    assignments: List[Dict[str, Any]] = []
    for vid, vjobs in enumerate(final_assignment, start=1):
        t, e, d, l = _vehicle_metrics(vjobs, depot_service_time)
        payload_jobs = [dict(j.payload) for j in vjobs]
        assignments.append(
            {
                "vehicle_id": vid,
                "jobs": payload_jobs,
                "num_trips": len(payload_jobs),
                "time_min": t,
                "distance_km": d,
                "load_desi": l,
                "energy_kwh": e,
                "remaining_energy_kwh": max(0.0, battery_capacity - e),
            }
        )

    log_lines.append(f"Nihai araç sayısı: {len(assignments)}")
    log_lines.append("Tabu + MIP optimizasyonu tamamlandı.")

    return {
        "assignments": assignments,
        "dropped_jobs": dropped_jobs,
        "log": "\n".join(log_lines),
        "stats": {
            "used_vehicles": len(assignments),
            "score": final_score,
            "mip_applied": bool(polished),
        },
    }


def solve_multitrip_ga_mip(
    jobs: List[Dict[str, Any]],
    max_shift_duration: float,
    usable_energy: float,
    depot_service_time: float,
    battery_capacity: float,
    iterations: int = 250,
    mutation_rate: float = 0.20,
    seed: int = 42,
    mip_time_limit_s: int = 5,
    population_size: int = 30,
) -> Dict[str, Any]:
    """Multi-trip assignment optimization with GA + MIP polishing."""
    rng = random.Random(int(seed))

    candidate_jobs: List[_Job] = []
    dropped_jobs: List[Dict[str, Any]] = []

    for i, raw in enumerate(jobs):
        t = float(raw.get("time_min", 0.0))
        e = float(raw.get("energy_kwh", 0.0))
        d = float(raw.get("distance_km", 0.0))
        l = float(raw.get("load_desi", 0.0))

        if t > max_shift_duration + 1e-9 or e > usable_energy + 1e-9:
            dropped_jobs.append(dict(raw))
            continue

        candidate_jobs.append(
            _Job(
                idx=i,
                time_min=t,
                energy_kwh=e,
                distance_km=d,
                load_desi=l,
                payload=dict(raw),
            )
        )

    log_lines: List[str] = []
    log_lines.append("GA + MIP multi-trip optimizasyonu başlatıldı.")
    log_lines.append(f"Toplam iş: {len(jobs)} | Uygun iş: {len(candidate_jobs)} | Hariç tutulan: {len(dropped_jobs)}")

    if not candidate_jobs:
        return {
            "assignments": [],
            "dropped_jobs": dropped_jobs,
            "log": "\n".join(log_lines + ["Uygun iş yok; çözüm üretilemedi."]),
            "stats": {"used_vehicles": 0, "score": 0.0, "mip_applied": False},
        }

    def _randomized_initial() -> List[List[_Job]]:
        asg: List[List[_Job]] = []
        ordered = list(candidate_jobs)
        rng.shuffle(ordered)
        for job in ordered:
            feasible_slots = []
            for vidx, veh in enumerate(asg):
                cand = veh + [job]
                if _is_feasible_vehicle(cand, max_shift_duration, usable_energy, depot_service_time):
                    feasible_slots.append(vidx)

            if feasible_slots:
                asg[rng.choice(feasible_slots)].append(job)
            else:
                asg.append([job])
        return _cleanup_empty_vehicles(asg)

    def _repair_full(assignment: List[List[_Job]]) -> List[List[_Job]]:
        seen = set()
        clean: List[List[_Job]] = []
        dupes: List[_Job] = []
        for v in assignment:
            vv: List[_Job] = []
            for j in v:
                if j.idx in seen:
                    dupes.append(j)
                    continue
                seen.add(j.idx)
                vv.append(j)
            if vv:
                clean.append(vv)

        missing = [j for j in candidate_jobs if j.idx not in seen]
        pool = dupes + missing
        return _repair_best_insertion(
            assignment=clean,
            removed_jobs=pool,
            max_shift_duration=max_shift_duration,
            usable_energy=usable_energy,
            depot_service_time=depot_service_time,
        )

    def _crossover(a: List[List[_Job]], b: List[List[_Job]]) -> List[List[_Job]]:
        child: List[List[_Job]] = []
        used = set()

        parents = [a, b]
        rng.shuffle(parents)
        for p in parents:
            for v in p:
                if not v:
                    continue
                if rng.random() > 0.45:
                    continue
                cand = [j for j in v if j.idx not in used]
                if not cand:
                    continue
                if _is_feasible_vehicle(cand, max_shift_duration, usable_energy, depot_service_time):
                    child.append(cand)
                    used.update(j.idx for j in cand)

        remaining = [j for j in candidate_jobs if j.idx not in used]
        child = _repair_best_insertion(
            assignment=child,
            removed_jobs=remaining,
            max_shift_duration=max_shift_duration,
            usable_energy=usable_energy,
            depot_service_time=depot_service_time,
        )
        return _repair_full(child)

    def _mutate(assignment: List[List[_Job]]) -> List[List[_Job]]:
        cand = _deepcopy_assignment(assignment)
        if rng.random() < 0.5:
            cand = _try_random_swap(
                assignment=cand,
                max_shift_duration=max_shift_duration,
                usable_energy=usable_energy,
                depot_service_time=depot_service_time,
                rng=rng,
            )
        else:
            cand, removed = _random_destroy(cand, destroy_count=max(1, len(candidate_jobs) // 12), rng=rng)
            cand = _repair_best_insertion(
                assignment=cand,
                removed_jobs=removed,
                max_shift_duration=max_shift_duration,
                usable_energy=usable_energy,
                depot_service_time=depot_service_time,
            )
        return _repair_full(cand)

    pop_size = max(8, int(population_size))
    generations = max(1, int(iterations))
    mut = max(0.01, min(0.8, float(mutation_rate)))

    population: List[List[List[_Job]]] = [_build_greedy_initial(candidate_jobs, max_shift_duration, usable_energy, depot_service_time)]
    for _ in range(pop_size - 1):
        population.append(_randomized_initial())

    def _fitness(asg: List[List[_Job]]) -> float:
        return _score_solution(_repair_full(asg), depot_service_time)

    best = min(population, key=_fitness)
    best_score = _fitness(best)
    log_lines.append(f"Başlangıç en iyi araç sayısı: {len(_cleanup_empty_vehicles(best))}")

    for _ in range(generations):
        ranked = sorted(population, key=_fitness)
        elite_count = max(2, pop_size // 5)
        new_pop = [
            _repair_full(_deepcopy_assignment(ind))
            for ind in ranked[:elite_count]
        ]

        def _pick_parent() -> List[List[_Job]]:
            contenders = rng.sample(ranked[: max(elite_count * 2, 4)], k=min(3, len(ranked[: max(elite_count * 2, 4)])))
            return min(contenders, key=_fitness)

        while len(new_pop) < pop_size:
            p1 = _pick_parent()
            p2 = _pick_parent()
            child = _crossover(p1, p2)
            if rng.random() < mut:
                child = _mutate(child)
            new_pop.append(_repair_full(child))

        population = new_pop
        gen_best = min(population, key=_fitness)
        gen_score = _fitness(gen_best)
        if gen_score < best_score:
            best = _deepcopy_assignment(gen_best)
            best_score = gen_score

    best = _cleanup_empty_vehicles(_repair_full(best))
    log_lines.append(f"GA sonrası araç sayısı: {len(best)}")

    polished, mip_msg = _mip_polish(
        jobs=candidate_jobs,
        max_shift_duration=max_shift_duration,
        usable_energy=usable_energy,
        depot_service_time=depot_service_time,
        max_vehicles=len(best),
        time_limit_s=int(mip_time_limit_s),
    )
    log_lines.append(mip_msg)

    final_assignment = polished if polished else best
    final_score = _score_solution(final_assignment, depot_service_time)

    assignments: List[Dict[str, Any]] = []
    for vid, vjobs in enumerate(final_assignment, start=1):
        t, e, d, l = _vehicle_metrics(vjobs, depot_service_time)
        payload_jobs = [dict(j.payload) for j in vjobs]
        assignments.append(
            {
                "vehicle_id": vid,
                "jobs": payload_jobs,
                "num_trips": len(payload_jobs),
                "time_min": t,
                "distance_km": d,
                "load_desi": l,
                "energy_kwh": e,
                "remaining_energy_kwh": max(0.0, battery_capacity - e),
            }
        )

    log_lines.append(f"Nihai araç sayısı: {len(assignments)}")
    log_lines.append("GA + MIP optimizasyonu tamamlandı.")

    return {
        "assignments": assignments,
        "dropped_jobs": dropped_jobs,
        "log": "\n".join(log_lines),
        "stats": {
            "used_vehicles": len(assignments),
            "score": final_score,
            "mip_applied": bool(polished),
        },
    }
