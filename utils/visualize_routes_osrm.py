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
    osrm_client,
    weekday=None
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
    # Battery capacity in kWh
    battery_capacity_kwh = float(data.get("battery_capacity", 100.0))  # Default 100 kWh
    # ============================================================
    # CREATE MAP
    # ============================================================
    center_lat = float(np.mean([depot_lat] + df_orders["Enlem"].tolist()))
    center_lon = float(np.mean([depot_lon] + df_orders["Boylam"].tolist()))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")
    
    # Add weekday label if provided
    if weekday is not None:
        day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        day_name = day_names[weekday] if 0 <= weekday < 7 else "Bilinmeyen"
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; 
                    left: 50px; 
                    width: 250px; 
                    height: 50px; 
                    background-color: white; 
                    border:2px solid grey; 
                    z-index:9999; 
                    font-size:16px;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                    ">
        <b>🗓️ Trafik Günü:</b> {day_name}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

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
        if E < 0.0025: return "#00cc44"
        if E < 0.35: return "#ffcc00"
        return "#ff3300"

    # ============================================================
    # DRAW VEHICLE ROUTES
    # ============================================================
    all_points = [(depot_lat, depot_lon)]

    for v, route in enumerate(vehicle_routes):
        if not route:
            continue
        
        # Get original vehicle ID for color mapping (if FilteredSolution)
        if hasattr(routing, 'get_original_vehicle_id'):
            original_v = routing.get_original_vehicle_id(v)
        else:
            original_v = v
        
        vehicle_color = color_list[original_v % len(color_list)]

        prev_node = depot
        t_now = 0.0
        cum_load = 0.0
        remaining_batt = battery_capacity_kwh  # Start with full battery in kWh

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
                # Calculate traffic-aware time for tooltip
                traffic_time = get_leg_time(prev_node, node, t_now)
                t_now = arr + service  # Update for next iteration

            # -----------------------
            # UPDATE LOAD
            # -----------------------
            cum_load += loads[node]

            # -----------------------
            # ENERGY MODEL (kWh, then convert to %)
            # Formula: 0.436 * distance + 0.002 * load_before
            # -----------------------
            energy_kwh = 0.436 * d_km + 0.002 * load_before_leg
            energy_pct = (energy_kwh / battery_capacity_kwh) * 100.0  # Convert to percentage

            batt_before_pct = (remaining_batt / battery_capacity_kwh) * 100.0
            remaining_batt -= energy_kwh  # Subtract kWh
            batt_after_pct = (remaining_batt / battery_capacity_kwh) * 100.0
            
            # Calculate average speed (km/h)
            avg_speed_kmh = (d_km / travel_min * 60) if travel_min > 0 else 0
            
            # Convert arrival and departure times to HH:MM format
            arr_hours = int((arr + 540) // 60)  # +540 = 9:00 start
            arr_mins = int((arr + 540) % 60)
            dep_hours = int((dep + 540) // 60)
            dep_mins = int((dep + 540) % 60)
            arr_time_str = f"{arr_hours:02d}:{arr_mins:02d}"
            dep_time_str = f"{dep_hours:02d}:{dep_mins:02d}"
            
            # Picked load at this node
            picked_load = loads[node]

            # -----------------------
            # MARKER
            # -----------------------
            folium.Marker(
                [row["Enlem"], row["Boylam"]],
                tooltip=(
                    f"<b>Order ID:</b> {row['OrderID']}<br>"
                    f"<b>Araç:</b> {original_v + 1}<br>"
                    f"<b>Varış Saati:</b> {arr_time_str}<br>"
                    f"<b>Servis Süresi:</b> {service:.1f} dk<br>"
                    f"<b>Çıkış Saati:</b> {dep_time_str}<br>"
                    f"<b>Kalan Batarya:</b> {batt_after_pct:.1f}%<br>"
                    f"<b>Alınan Yük:</b> {picked_load:.0f} desi<br>"
                    f"<b>Toplam Yük:</b> {cum_load:.0f} desi"
                ),
                icon=BeautifyIcon(
                    number=str(row["OrderID"]),
                    background_color=vehicle_color,
                    text_color="white",
                    border_color="black",
                    border_width=2,
                )
            ).add_to(m)

            all_points.append((row["Enlem"], row["Boylam"]))

            # -----------------------
            # POLYLINE (OSRM) - Use vehicle color instead of energy color
            # -----------------------
            # Get OSRM baseline time (without traffic)
            osrm_time = float(T_static[prev_node, node])
            
            # Get traffic-aware time (uses T_by_hour if available)
            if is_ga:
                traffic_time_display = travel_min
            else:
                traffic_time_display = traffic_time
            
            seg = osrm_client.route(coords[prev_node], coords[node])
            if seg:
                folium.PolyLine(
                    seg,
                    color=vehicle_color,
                    weight=6,
                    opacity=0.85,
                    tooltip=folium.Tooltip(
                        f"<b>Rota:</b> {prev_node} → {node}<br>"
                        f"<b>Araç:</b> {original_v + 1}<br>"
                        f"<b>Mesafe:</b> {d_km:.2f} km<br>"
                        f"<b>Ort. Hız:</b> {avg_speed_kmh:.1f} km/h<br>"
                        f"<b>OSRM Süre:</b> {osrm_time:.1f} dk<br>"
                        f"<b>Trafikli Süre:</b> {traffic_time_display:.1f} dk<br>"
                        f"<b>Taşınan Yük:</b> {cum_load:.0f} desi<br>"
                        f"<b>Enerji:</b> {energy_kwh:.3f} kWh ({energy_pct:.1f}%)"
                    )
                ).add_to(m)

            prev_node = node

        # ========================================================
        # RETURN TO DEPOT
        # ========================================================
        load_before_leg = cum_load
        d_km = D[prev_node, depot]
        osrm_time = float(T_static[prev_node, depot])
        
        # Calculate traffic-aware time for return leg
        if is_ga:
            traffic_time_return = get_leg_time(prev_node, depot, t_now)
        else:
            traffic_time_return = get_leg_time(prev_node, depot, t_now)
        
        avg_speed_kmh = (d_km / traffic_time_return * 60) if traffic_time_return > 0 else 0

        energy_kwh = 0.436 * d_km + 0.002 * load_before_leg
        energy_pct = (energy_kwh / battery_capacity_kwh) * 100.0

        seg = osrm_client.route(coords[prev_node], coords[depot])
        if seg:
            folium.PolyLine(
                seg,
                color=vehicle_color,
                weight=6,
                opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>Rota:</b> {prev_node} → Depot<br>"
                    f"<b>Araç:</b> {original_v + 1}<br>"
                    f"<b>Mesafe:</b> {d_km:.2f} km<br>"
                    f"<b>Ort. Hız:</b> {avg_speed_kmh:.1f} km/h<br>"
                    f"<b>OSRM Süre:</b> {osrm_time:.1f} dk<br>"
                    f"<b>Trafikli Süre:</b> {traffic_time_return:.1f} dk<br>"
                    f"<b>Taşınan Yük:</b> {cum_load:.0f} desi<br>"
                    f"<b>Enerji:</b> {energy_kwh:.3f} kWh ({energy_pct:.1f}%)"
                )
            ).add_to(m)

    m.fit_bounds(all_points)
    return m
