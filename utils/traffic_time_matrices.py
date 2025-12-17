import os
import pickle
import numpy as np
from scipy.spatial import cKDTree

# ---- OPTIONAL GPU SUPPORT (CuPy) ----
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _build_speed_matrix_for_weekday(depot, customers, traffic, weekday, hours):
    """
    Build speed[hour_index, node_index] in km/h:
        - node_index = 0 is depot, 1..N are customers
        - hours = iterable of hours, e.g. range(9, 19)
    """
    n = len(customers) + 1
    hours = list(hours)

    # Lat/lon arrays for all nodes
    lats = np.array([depot.enlem] + [c.enlem for c in customers])
    lons = np.array([depot.boylam] + [c.boylam for c in customers])

    speed = np.zeros((len(hours), n), dtype=float)

    for hi, h in enumerate(hours):
        subset = traffic[(traffic["HOUR"] == h) & (traffic["DAY_OF_WEEK"] == weekday)]
        if subset.empty:
            # no data for this hour/day → fallback to 30 km/h everywhere
            speed[hi, :] = 30.0
            continue

        pts = subset[["LATITUDE", "LONGITUDE"]].to_numpy()
        tree = cKDTree(pts)

        # query all nodes at once
        dists, idx = tree.query(np.column_stack([lats, lons]), k=1)

        vals = subset.iloc[idx]["AVG_SPEED_CLEAN"].to_numpy()
        vals = np.where((vals <= 0) | ~np.isfinite(vals), 30.0, vals)

        speed[hi, :] = vals

    return speed


def _compute_T_for_hour(args):
    """
    Worker for a single hour (can run inside or outside multiprocessing).
    args = (hour_index, hour_value, D, speed_vec, use_gpu)
    """
    hi, h, D, speed_vec, use_gpu = args

    if use_gpu and HAS_CUPY:
        D_gpu = cp.asarray(D)
        spd_gpu = cp.asarray(speed_vec)
        spd_mat_gpu = cp.tile(spd_gpu, (D_gpu.shape[0], 1))
        T_gpu = (D_gpu / spd_mat_gpu) * 60.0
        T = cp.asnumpy(T_gpu)
    else:
        spd_mat = np.tile(speed_vec, (D.shape[0], 1))  # broadcast dst-speed by column
        T = (D / spd_mat) * 60.0

    np.fill_diagonal(T, 0.0)
    return h, T


def build_time_matrices_with_traffic_optimized(
    D,
    depot,
    customers,
    traffic,
    weekday,
    *,
    hours=range(9, 19),
    cache_path=None,
    use_gpu=False,
    use_multiprocessing=False,
    progress_callback=None,
):
    """
    Build dict {hour -> time_matrix(minutes)} using traffic speeds.

    Features:
    - vectorized nearest-speed computation
    - optional GPU via CuPy
    - optional multiprocessing per hour
    - optional progress callback (for Streamlit progress bar)
    - optional disk cache to avoid recomputing

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (km), shape (N, N)
    depot : object
        Has attributes .enlem and .boylam
    customers : list
        List of customer objects with .enlem, .boylam
    traffic : pd.DataFrame
        Must contain LATITUDE, LONGITUDE, HOUR, DAY_OF_WEEK, AVG_SPEED_CLEAN
    weekday : int
        0=Mon, 1=Tue, ..., 6=Sun
    hours : iterable[int]
        Hours to compute, default 9..18
    cache_path : str or None
        If set, will try to load/save {hour: T} dictionary from this file
    use_gpu : bool
        Use CuPy if installed
    use_multiprocessing : bool
        Parallelize over hours (not recommended inside Streamlit on Windows)
    progress_callback : callable or None
        Called as progress_callback(done, total)

    Returns
    -------
    dict[int, np.ndarray]
        {hour: time_matrix_minutes}
    """
    hours = list(hours)
    n = D.shape[0]

    # ===== 1) Try load from cache =====
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("weekday") == weekday
                and payload.get("n_nodes") == n
                and payload.get("hours") == hours
            ):
                T_by_hour = payload["T_by_hour"]
                # progress: fully done
                if progress_callback:
                    progress_callback(len(hours), len(hours))
                return T_by_hour
        except Exception:
            # Ignore cache errors → recompute
            pass

    # ===== 2) Build speed table speed[hour_index, node_index] =====
    speed = _build_speed_matrix_for_weekday(depot, customers, traffic, weekday, hours)

    # ===== 3) Compute each hour's matrix (optionally parallel) =====
    tasks = [(hi, h, D, speed[hi], use_gpu) for hi, h in enumerate(hours)]
    total = len(tasks)
    T_by_hour = {}

    if use_multiprocessing:
        import multiprocessing as mp
        with mp.Pool() as pool:
            for done, (h, T) in enumerate(pool.imap_unordered(_compute_T_for_hour, tasks), start=1):
                T_by_hour[h] = T
                if progress_callback:
                    progress_callback(done, total)
    else:
        for done, args in enumerate(tasks, start=1):
            h, T = _compute_T_for_hour(args)
            T_by_hour[h] = T
            if progress_callback:
                progress_callback(done, total)

    # sort by hour
    T_by_hour = dict(sorted(T_by_hour.items()))

    # ===== 4) Save to cache =====
    if cache_path:
        try:
            payload = {
                "weekday": weekday,
                "n_nodes": n,
                "hours": hours,
                "T_by_hour": T_by_hour,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"⚠️ Could not write cache: {e}")

    return T_by_hour
