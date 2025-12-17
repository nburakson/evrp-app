##utils/traffic_osrm.py


import requests
import numpy as np
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_traffic_data(path: str) -> pd.DataFrame:
    """
    Load and clean the traffic CSV once.
    Expected columns:
      LATITUDE,LONGITUDE,HOUR,DAY_OF_WEEK,AVG_SPEED_CLEAN,...
    """
    df = pd.read_csv(path)
    df.columns = [c.upper().strip() for c in df.columns]

    keep_cols = ["LATITUDE", "LONGITUDE", "HOUR", "DAY_OF_WEEK", "AVG_SPEED_CLEAN"]
    df = df[keep_cols].copy()

    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    df["HOUR"] = df["HOUR"].astype(int)
    df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype(int)
    df["AVG_SPEED_CLEAN"] = df["AVG_SPEED_CLEAN"].astype(float)

    return df


def _nearest_csv_speed_for_snapped_point(
    traffic: pd.DataFrame,
    road_lat: float,
    road_lon: float,
    hour: int,
    day_of_week: int,
):
    """
    Given a road coordinate (lat, lon) that is already snapped to the road
    and a (hour, day_of_week), find the nearest traffic sensor and return its speed.
    """
    subset = traffic[
        (traffic["HOUR"] == hour) & (traffic["DAY_OF_WEEK"] == day_of_week)
    ]
    if subset.empty:
        return np.nan, np.nan, np.nan

    # vectorized distance (road point to all sensors of that hour/day)
    lat_arr = subset["LATITUDE"].to_numpy()
    lon_arr = subset["LONGITUDE"].to_numpy()

    dlat = lat_arr - road_lat
    dlon = lon_arr - road_lon
    dists = np.sqrt(dlat * dlat + dlon * dlon)  # degrees

    best_idx = int(np.argmin(dists))
    best_dist_deg = float(dists[best_idx])
    best_dist_m = best_dist_deg * 111_000.0  # rough conversion

    speed = float(subset["AVG_SPEED_CLEAN"].iloc[best_idx])
    csv_index = int(subset.index[best_idx])

    return speed, best_dist_m, csv_index


def osrm_route_with_traffic(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    day_of_week: int,
    start_minute: int,
    traffic: pd.DataFrame,
    osrm_host: str = "https://router.project-osrm.org",
):
    """
    Query OSRM for the real route between (start_lat, start_lon) and
    (end_lat, end_lon), then adjust each segment's duration using
    the nearest traffic sensor for that segment.

    Returns:
        df_segments: DataFrame with per-segment details
        total_dist_km: float (sum of OSRM distances)
        osrm_total_min: float (sum of OSRM durations)
        traffic_total_min: float (sum of traffic-adjusted durations)
    """
    # 1) Get full OSRM route with geometry + per-segment distance/duration
    route_url = (
        f"{osrm_host}/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        "?overview=full&geometries=geojson&annotations=distance,duration"
    )

    r = requests.get(route_url, timeout=60, verify=False)
    r.raise_for_status()
    data = r.json()

    route = data["routes"][0]
    leg = route["legs"][0]
    ann = leg["annotation"]

    coords = route["geometry"]["coordinates"]  # [ [lon, lat], ... ]
    distances = np.array(ann["distance"], dtype=float) / 1000.0  # km
    durations = np.array(ann["duration"], dtype=float) / 60.0    # minutes

    segments = []
    current_min = float(start_minute)

    for seg_idx, (lon, lat) in enumerate(coords[:-1]):  # segment start points
        hour = int(current_min // 60)

        # 2) Snap this geometry point to true road center using OSRM /nearest
        nearest_url = (
            f"{osrm_host}/nearest/v1/driving/{lon},{lat}?number=1"
        )
        r_near = requests.get(nearest_url, timeout=30, verify=False)
        r_near.raise_for_status()
        near_data = r_near.json()
        road_lon, road_lat = near_data["waypoints"][0]["location"]  # lon, lat

        # 3) Match snapped road point to nearest traffic sensor for that hour+day
        csv_speed, dist_m, csv_row_idx = _nearest_csv_speed_for_snapped_point(
            traffic=traffic,
            road_lat=road_lat,
            road_lon=road_lon,
            hour=hour,
            day_of_week=day_of_week,
        )

        dist_km = float(distances[seg_idx])
        osrm_dur = float(durations[seg_idx])

        if not np.isnan(csv_speed) and csv_speed > 0:
            # time (min) = distance(km) / speed(km/h) * 60
            adj_dur = dist_km / csv_speed * 60.0
        else:
            adj_dur = np.nan

        segments.append({
            "SEG": seg_idx,
            "HOUR": hour,
            "OSRM_GEOM_LAT": lat,
            "OSRM_GEOM_LON": lon,
            "SNAP_ROAD_LAT": road_lat,
            "SNAP_ROAD_LON": road_lon,
            "CSV_SPEED": csv_speed,
            "SENSOR_DIST_M": dist_m,
            "CSV_ROW_IDX": csv_row_idx,
            "OSRM_DIST_KM": dist_km,
            "OSRM_DUR_MIN": osrm_dur,
            "TRAFFIC_DUR_MIN": adj_dur,
        })

        current_min += osrm_dur

    df_segments = pd.DataFrame(segments)

    total_dist_km = float(distances.sum())
    osrm_total_min = float(durations.sum())

    valid = df_segments.dropna(subset=["TRAFFIC_DUR_MIN"])
    traffic_total_min = float(valid["TRAFFIC_DUR_MIN"].sum()) if not valid.empty else np.nan

    return df_segments, total_dist_km, osrm_total_min, traffic_total_min
