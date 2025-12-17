# utils/visualize_routes_osrm.py

import folium
import numpy as np
import matplotlib.colors as mcolors
from folium.plugins import BeautifyIcon


def visualize_routes_osrm(
    depot_lat,
    depot_lon,
    df_orders,
    data,
    routing,
    manager,
    solution,
    time_dim,
    energy_dim,
    osrm_client
):
    """
    Visualize EVRP routes (OR-Tools or GA) using OSRM polylines.

    Supports:
        • OR-Tools snapshot matrix (data["time_min"])
        • Dynamic T_by_hour matrices (data["time_min_by_hour"])
        • Load-dependent EV energy model:
              E = 0.436 * distance + 0.002 * load_before_leg
        • Battery-before/after, cumulative load, colored polylines.
    """

    # ============================================================
    # READ DATA
    # ============================================================
    color_list = list(mcolors.TABLEAU_COLORS.values())

    n_vehicles = data["num_vehicles"]
    D = np.array(data["distance_km"], dtype=float)
    T_static = np.array(data["time_min"], dtype=float)
    T_by_hour = data.get("time_min_by_hour", None)
    loads = np.array(data["demand_desi"], dtype=float)
    depot = data["depot"]
    # ✅ FIXED BATTERY READ (aligned with OR-Tools)
    battery_capacity = float(
       data.get("battery_capacity", data.get("battery_capacity", 100.0))
    )
    # ============================================================
    # CREATE MAP
    # ============================================================
    center_lat = float(np.mean([depot_lat] + df_orders["Enlem"].tolist()))
    center_lon = float(np.mean([depot_lon] + df_orders["Boylam"].tolist()))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    # depot marker
    folium.Marker(
        [depot_lat, depot_lon],
        tooltip="🚩 <b>Depot (Start/End)</b>",
        icon=BeautifyIcon(
            icon_shape="star",
            border_color="red",
            background_color="red",
            text_color="white",
            border_width=2,
        )
    ).add_to(m)

    # ============================================================
    # GET VEHICLE ROUTES
    # ============================================================
    is_ga = routing is None

    if is_ga:
        vehicle_routes = solution["routes"]  # format: [[1,3,5], ...]
    else:
        vehicle_routes = []
        for v in range(n_vehicles):
            idx = routing.Start(v)
            route = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != depot:
                    route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            vehicle_routes.append(route)

    # coords for OSRM calls
    coords = [(depot_lon, depot_lat)] + [
        (row["Boylam"], row["Enlem"]) for _, row in df_orders.iterrows()
    ]

    # ============================================================
    # HELPERS
    # ============================================================
    def get_leg_time(prev_node, node, t_now):
        """Return travel minutes using dynamic T_by_hour or static matrix."""
        if T_by_hour is None:
            return float(T_static[prev_node, node])

        hour = int((t_now + 540) // 60)  # +540 offset = 9:00
        nearest = min(T_by_hour.keys(), key=lambda h: abs(h - hour))
        return float(T_by_hour[nearest][prev_node, node])

    def energy_color(E):
        if E < 0.15: return "#00cc44"
        if E < 0.35: return "#ffcc00"
        return "#ff3300"

    # ============================================================
    # DRAW VEHICLE ROUTES
    # ============================================================
    all_points = [(depot_lat, depot_lon)]

    for v, route in enumerate(vehicle_routes):
        if not route:
            continue

        prev_node = depot
        t_now = 0.0
        cum_load = 0.0
        remaining_batt = battery_capacity

        for node in route:
            row = df_orders.iloc[node - 1]
            d_km = float(D[prev_node, node])

            # -----------------------
            # LOAD BEFORE LEG
            # -----------------------
            load_before_leg = cum_load

            # -----------------------
            # GA MODE
            # -----------------------
            if is_ga:
                travel_min = get_leg_time(prev_node, node, t_now)
                arr = t_now + travel_min
                service = float(row["Servis Süresi (dk)"])
                dep = arr + service
                t_now = dep

            # -----------------------
            # OR-TOOLS MODE
            # -----------------------
            else:
                idx = manager.NodeToIndex(node)
                arr = solution.Value(time_dim.CumulVar(idx))
                service = float(row["Servis Süresi (dk)"])
                dep = arr + service
                travel_min = float(T_static[prev_node, node])

            # -----------------------
            # UPDATE LOAD
            # -----------------------
            cum_load += loads[node]

            # -----------------------
            # ENERGY MODEL
            # -----------------------
            empty_energy = 0.436 * d_km
            load_energy = 0.002 * load_before_leg
            total_energy = empty_energy + load_energy

            batt_before = remaining_batt
            remaining_batt -= total_energy
            batt_after = remaining_batt

            # -----------------------
            # MARKER
            # -----------------------
            folium.Marker(
                [row["Enlem"], row["Boylam"]],
                tooltip=(
                    f"<b>Order ID:</b> {row['OrderID']}<br>"
                    f"<b>Araç:</b> {v}<br>"
                    f"<b>Mesafe:</b> {d_km:.2f} km<br>"
                    f"<b>Süre:</b> {travel_min:.1f} dk<br>"
                    f"<b>LoadBefore:</b> {load_before_leg:.0f}<br>"
                    f"<b>CumLoad:</b> {cum_load:.0f}<br>"
                    f"<b>Enerji(Empty):</b> {empty_energy:.3f} kWh<br>"
                    f"<b>Enerji(Load):</b> {load_energy:.3f} kWh<br>"
                    f"<b>Enerji(Total):</b> {total_energy:.3f} kWh<br>"
                    f"<b>BatteryBefore:</b> {batt_before/battery_capacity*100:.1f}%<br>"
                    f"<b>BatteryAfter:</b> {batt_after/battery_capacity*100:.1f}%"
                ),
                icon=BeautifyIcon(
                    number=str(row["OrderID"]),
                    background_color=color_list[v % len(color_list)],
                    text_color="white",
                    border_color="black",
                    border_width=2,
                )
            ).add_to(m)

            all_points.append((row["Enlem"], row["Boylam"]))

            # -----------------------
            # POLYLINE (OSRM)
            # -----------------------
            seg = osrm_client.route(coords[prev_node], coords[node])
            if seg:
                folium.PolyLine(
                    seg,
                    color=energy_color(total_energy),
                    weight=6,
                    opacity=0.85,
                    tooltip=folium.Tooltip(
                        f"<b>Araç {v}</b><br>"
                        f"{prev_node} → {node}<br>"
                        f"Mesafe: {d_km:.2f} km<br>"
                        f"LoadBefore: {load_before_leg:.0f}<br>"
                        f"CumLoad: {cum_load:.0f}<br>"
                        f"Enerji(Total): {total_energy:.3f} kWh"
                    )
                ).add_to(m)

            prev_node = node

        # ========================================================
        # RETURN TO DEPOT
        # ========================================================
        load_before_leg = cum_load
        d_km = D[prev_node, depot]
        travel_min = T_static[prev_node, depot]

        empty_energy = 0.436 * d_km
        load_energy = 0.002 * load_before_leg
        total_energy = empty_energy + load_energy

        seg = osrm_client.route(coords[prev_node], coords[depot])
        if seg:
            folium.PolyLine(
                seg,
                color=energy_color(total_energy),
                weight=6,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>Araç {v}</b><br>"
                    f"{prev_node} → Depot<br>"
                    f"Mesafe: {d_km:.2f} km<br>"
                    f"LoadBefore: {load_before_leg:.0f}<br>"
                    f"Enerji(Total): {total_energy:.3f} kWh"
                )
            ).add_to(m)

    m.fit_bounds(all_points)
    return m
