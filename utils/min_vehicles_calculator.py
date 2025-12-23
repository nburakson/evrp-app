# utils/min_vehicles_calculator.py
"""
Calculate minimum number of vehicles needed for multi-trip EVRP
"""

import numpy as np


def calculate_min_vehicles_multitrip(
    D,
    T,
    demands,
    depot=0,
    vehicle_capacity=4500,
    battery_capacity=100.0,
    work_start_min=540,  # 09:00
    work_end_min=1080,   # 18:00
    service_times=None,
):
    """
    Calculate theoretical minimum vehicles needed for multi-trip routing.
    
    Returns:
        dict with keys:
            - min_vehicles_capacity: Based on total demand
            - min_vehicles_time: Based on time constraints
            - min_vehicles_energy: Based on energy/distance
            - recommended_min: Maximum of above (conservative estimate)
            - explanation: String explaining the calculation
    """
    
    n_customers = len(demands) - 1  # exclude depot
    total_demand = np.sum(demands[1:])  # exclude depot
    
    if service_times is None:
        service_times = np.zeros(len(demands))
    
    # ========================================
    # 1. CAPACITY-BASED MINIMUM
    # ========================================
    # Each vehicle can make multiple trips, each trip limited by capacity
    # Minimum = ceil(total_demand / vehicle_capacity)
    min_vehicles_capacity = int(np.ceil(total_demand / vehicle_capacity))
    
    # ========================================
    # 2. TIME-BASED MINIMUM
    # ========================================
    # Estimate average trip time considering:
    # - Average customer is at distance from depot
    # - Average service time
    # - Need to return to depot between trips
    
    # Average distance from depot to customers
    avg_dist_to_depot = np.mean([D[depot, i] for i in range(1, len(D))])
    avg_time_to_depot = np.mean([T[depot, i] for i in range(1, len(T))])
    avg_service_time = np.mean(service_times[1:])
    
    # Estimate customers per trip (capacity-limited)
    avg_demand_per_customer = total_demand / n_customers if n_customers > 0 else 1
    customers_per_trip = int(vehicle_capacity / avg_demand_per_customer) if avg_demand_per_customer > 0 else 1
    customers_per_trip = max(1, min(customers_per_trip, n_customers))
    
    # Estimate time per trip:
    # - Travel to first customer: avg_time_to_depot
    # - Service N customers: customers_per_trip * avg_service_time
    # - Travel between customers: estimate as 70% of depot distance
    # - Return to depot: avg_time_to_depot
    est_time_per_trip = (
        avg_time_to_depot +  # outbound
        customers_per_trip * (avg_service_time + avg_time_to_depot * 0.7) +  # customers
        avg_time_to_depot  # return
    )
    
    # Available working time per vehicle
    working_minutes = work_end_min - work_start_min
    
    # Trips per vehicle per day
    trips_per_vehicle = int(working_minutes / est_time_per_trip) if est_time_per_trip > 0 else 1
    trips_per_vehicle = max(1, trips_per_vehicle)
    
    # Total trips needed
    total_trips_needed = int(np.ceil(n_customers / customers_per_trip))
    
    # Minimum vehicles based on time
    min_vehicles_time = int(np.ceil(total_trips_needed / trips_per_vehicle))
    
    # ========================================
    # 3. ENERGY-BASED MINIMUM
    # ========================================
    # Similar to time, but based on battery capacity
    
    # Energy per km (simplified - no load component for estimate)
    energy_per_km = 0.436
    
    # Estimate energy per trip
    est_dist_per_trip = (
        avg_dist_to_depot +  # outbound
        customers_per_trip * avg_dist_to_depot * 0.7 +  # between customers
        avg_dist_to_depot  # return
    )
    est_energy_per_trip = est_dist_per_trip * energy_per_km
    
    # Trips per vehicle per battery charge
    trips_per_battery = int(battery_capacity / est_energy_per_trip) if est_energy_per_trip > 0 else 1
    trips_per_battery = max(1, trips_per_battery)
    
    # With multi-trip, vehicle can recharge between trips
    # So this is less limiting than time typically
    # But we still need enough vehicles to complete all trips
    min_vehicles_energy = int(np.ceil(total_trips_needed / trips_per_battery))
    
    # ========================================
    # 4. RECOMMENDED MINIMUM
    # ========================================
    recommended_min = max(min_vehicles_capacity, min_vehicles_time, min_vehicles_energy)
    
    # ========================================
    # 5. EXPLANATION
    # ========================================
    explanation = f"""
📊 Minimum Vehicle Calculation (Multi-Trip Mode)

📦 Kapasite Bazlı:
   • Toplam talep: {total_demand:.0f} desi
   • Araç kapasitesi: {vehicle_capacity:.0f} desi
   • Minimum araç: {min_vehicles_capacity} (tek seferde taşınabilecek maksimum)

⏱️ Zaman Bazlı:
   • Çalışma süresi: {working_minutes:.0f} dakika ({(working_minutes/60):.1f} saat)
   • Tahmini tur süresi: {est_time_per_trip:.1f} dakika
   • Araç başına tur sayısı: ~{trips_per_vehicle} tur
   • Toplam gerekli tur: ~{total_trips_needed} tur
   • Minimum araç: {min_vehicles_time}

🔋 Enerji Bazlı:
   • Batarya kapasitesi: {battery_capacity:.1f} kWh
   • Tahmini tur başına enerji: {est_energy_per_trip:.1f} kWh
   • Batarya başına tur sayısı: ~{trips_per_battery} tur
   • Minimum araç: {min_vehicles_energy}

✅ ÖNERİLEN MİNİMUM: {recommended_min} araç

💡 Not: Bu teorik bir hesaplamadır. Gerçek rotalar ve trafik 
   durumuna göre daha fazla araç gerekebilir. Güvenli bir başlangıç 
   için {recommended_min + 1} araç deneyebilirsiniz.
"""
    
    return {
        'min_vehicles_capacity': min_vehicles_capacity,
        'min_vehicles_time': min_vehicles_time,
        'min_vehicles_energy': min_vehicles_energy,
        'recommended_min': recommended_min,
        'trips_per_vehicle': trips_per_vehicle,
        'total_trips_needed': total_trips_needed,
        'customers_per_trip': customers_per_trip,
        'est_time_per_trip': est_time_per_trip,
        'est_energy_per_trip': est_energy_per_trip,
        'explanation': explanation,
    }


def calculate_min_vehicles_single_trip(
    demands,
    vehicle_capacity=4500,
):
    """
    Calculate minimum vehicles for single-trip mode.
    Much simpler - just based on capacity.
    """
    
    total_demand = np.sum(demands[1:])  # exclude depot
    min_vehicles = int(np.ceil(total_demand / vehicle_capacity))
    
    explanation = f"""
📊 Minimum Vehicle Calculation (Single-Trip Mode)

📦 Kapasite Bazlı:
   • Toplam talep: {total_demand:.0f} desi
   • Araç kapasitesi: {vehicle_capacity:.0f} desi
   • Minimum araç: {min_vehicles}

⚠️ Tek tur modunda her araç sadece bir kez çıkabilir.
   Mesafe ve enerji kısıtları nedeniyle daha fazla araç gerekebilir.

✅ ÖNERİLEN MİNİMUM: {min_vehicles} araç
"""
    
    return {
        'recommended_min': min_vehicles,
        'explanation': explanation,
    }
