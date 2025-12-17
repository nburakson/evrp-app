import os
import json
import hashlib
import requests
import urllib3
import numpy as np

from functools import lru_cache
from polyline import decode
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================
# 🔐 FILESYSTEM CACHE CONFIG
# =============================================================
CACHE_DIR = "osrm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(lon1, lat1, lon2=None, lat2=None):
    """
    If lon2/lat2 are None, this can still be used for other caching later.
    For routes we always use 4 coords.
    """
    if lon2 is None or lat2 is None:
        raw = f"{lon1:.6f},{lat1:.6f}"
    else:
        raw = f"{lon1:.6f},{lat1:.6f}-{lon2:.6f},{lat2:.6f}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")


def save_to_disk(key, data):
    """Save data (coords, etc.) to disk as JSON."""
    try:
        with open(_cache_path(key), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_from_disk(key):
    """Load data if exists, otherwise None."""
    try:
        with open(_cache_path(key), "r") as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================
# 🚀 OSRM CLIENT
# =============================================================
class OSRMClient:
    def __init__(self, host="https://router.project-osrm.org", profile="driving", chunk_size=80):
        self.host = host.rstrip("/")
        self.profile = profile
        self.chunk_size = chunk_size

    # ---------------------------------------------------------
    # 🧠 INTERNAL LRU CACHE (fastest lookup) FOR ROUTE POLYLINES
    # ---------------------------------------------------------
    @lru_cache(maxsize=50000)
    def _cached_route(self, lon1, lat1, lon2, lat2):
        """
        Internal cached OSRM call → polyline decode → list[(lat, lon)].
        Uses both in-memory LRU and on-disk cache.
        """
        key = _cache_key(lon1, lat1, lon2, lat2)

        # 1) Filesystem cache lookup
        disk_data = load_from_disk(key)
        if disk_data is not None:
            return tuple(tuple(p) for p in disk_data)

        # 2) Network fetch
        url = (
            f"{self.host}/route/v1/{self.profile}/"
            f"{lon1},{lat1};{lon2},{lat2}"
            "?overview=full&geometries=polyline"
        )

        try:
            r = requests.get(url, timeout=60, verify=False)
            r.raise_for_status()
            j = r.json()

            if "routes" not in j or not j["routes"]:
                return ()

            poly = j["routes"][0]["geometry"]
            coords = decode(poly)  # list[(lat, lon)]
            coords = [(lat, lon) for lat, lon in coords]

            # Save persistent cache
            save_to_disk(key, coords)

            return tuple(coords)

        except Exception as e:
            print(f"⚠ OSRM polyline fetch error: {e}")
            return ()

    # ---------------------------------------------------------
    # PUBLIC WRAPPER
    # ---------------------------------------------------------
    def route(self, start, end):
        """
        start = (lon, lat), end = (lon, lat)
        Returns list[(lat, lon)] along the polyline.
        """
        lon1, lat1 = start
        lon2, lat2 = end
        return list(self._cached_route(lon1, lat1, lon2, lat2))

    # ---------------------------------------------------------
    # ⚡ PARALLEL PREFETCHER
    # ---------------------------------------------------------
    def prefetch_many(self, node_pairs, max_workers=12):
        """
        node_pairs = [((lon1,lat1),(lon2,lat2)), ...]
        Preloads OSRM cache in parallel for route() calls.
        """

        def fetch(pair):
            (lon1, lat1), (lon2, lat2) = pair
            self.route((lon1, lat1), (lon2, lat2))

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for pair in node_pairs:
                futures.append(ex.submit(fetch, pair))

            for _ in as_completed(futures):
                pass

        print(f"⚡ Prefetched {len(node_pairs)} OSRM legs.")

    # ---------------------------------------------------------
    # 🧮 MATRIX BUILDING (D & T) VIA OSRM TABLE API
    # ---------------------------------------------------------
    def _osrm_table_chunk(self, src_indices, dst_indices, lats, lons):
        """
        Calls OSRM table API for a subset of coordinates.
        Returns duration (sec) and distance (m) matrices for src×dst.
        """
        coords = [
            f"{lons[i]:.6f},{lats[i]:.6f}"
            for i in (src_indices + dst_indices)
        ]
        coord_str = ";".join(coords)

        src_param = ";".join(str(i) for i in range(len(src_indices)))
        dst_param = ";".join(
            str(i + len(src_indices)) for i in range(len(dst_indices))
        )

        url = (
            f"{self.host}/table/v1/{self.profile}/{coord_str}"
            f"?sources={src_param}&destinations={dst_param}"
            "&annotations=duration,distance"
        )

        r = requests.get(url, timeout=120, verify=False)
        r.raise_for_status()
        data = r.json()

        durations = np.array(data["durations"], dtype=float)
        distances = np.array(data["distances"], dtype=float)
        return durations, distances

    def build_matrices_from_latlon(self, lats, lons):
        """
        Build full OSRM distance & time matrices using chunking.
        Returns:
            D (km) shape (N,N)
            T (min) shape (N,N)
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        total = len(lats)

        print(
            f"🧭 Building OSRM matrices for {total} points "
            f"with chunk size {self.chunk_size}..."
        )

        full_D = np.zeros((total, total), dtype=float)
        full_T = np.zeros((total, total), dtype=float)

        ranges = []
        i = 0
        while i < total:
            ranges.append(list(range(i, min(i + self.chunk_size, total))))
            i += self.chunk_size

        print(f"🔁 Chunk count = {len(ranges)}")

        for si, src_block in enumerate(ranges):
            for di, dst_block in enumerate(ranges):
                print(
                    f"   🔹 Fetching block S{si} → D{di} "
                    f"({len(src_block)} × {len(dst_block)})"
                )

                durations_sec, distances_m = self._osrm_table_chunk(
                    src_block, dst_block, lats, lons
                )

                full_T[np.ix_(src_block, dst_block)] = durations_sec
                full_D[np.ix_(src_block, dst_block)] = distances_m

        D = full_D / 1000.0   # meters → km
        T = full_T / 60.0     # seconds → minutes

        D = np.nan_to_num(D, nan=99, posinf=99, neginf=99)
        T = np.nan_to_num(T, nan=9999, posinf=9999, neginf=9999)

        print(f"✅ Matrix build complete → {D.shape}")
        return D, T

    # ---------------------------------------------------------
    # CONVENIENCE HELPER
    # ---------------------------------------------------------
    def build_matrices_from_orders(self, depot, customers):
        """
        depot: object with .enlem, .boylam
        customers: list of objects with .enlem, .boylam
        """
        lats = [depot.enlem] + [c.enlem for c in customers]
        lons = [depot.boylam] + [c.boylam for c in customers]
        return self.build_matrices_from_latlon(lats, lons)
