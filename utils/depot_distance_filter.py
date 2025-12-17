import numpy as np

BASE_KWH_PER_KM = 0.436

def depot_distance_feasibility(
    D,
    demand,
    battery_kwh,
    max_one_way_km=110,
    depot=0,
):
    """
    Distance-only feasibility check (traffic independent)

    Returns:
        feasible_nodes: list[int]
        removed_records: list[dict]
    """

    feasible_nodes = []
    removed_records = []

    n = D.shape[0]

    for i in range(n):
        if i == depot:
            continue

        d_out = float(D[depot, i])
        d_back = float(D[i, depot])
        round_trip_energy = (d_out + d_back) * BASE_KWH_PER_KM

        distance_ok = d_out <= max_one_way_km
        energy_ok = round_trip_energy <= battery_kwh

        if distance_ok and energy_ok:
            feasible_nodes.append(i)
        else:
            removed_records.append({
                "node_index": i,
                "distance_depot_km": round(d_out, 2),
                "round_trip_energy_kwh": round(round_trip_energy, 2),
                "distance_ok": distance_ok,
                "energy_ok": energy_ok,
            })

    return feasible_nodes, removed_records
