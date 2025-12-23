# Multi-Trip Vehicle Routing - Feature Documentation

## Overview
This feature enables vehicles to make multiple trips during the working day (09:00-18:00) if they have sufficient remaining:
- **Energy** (battery capacity)
- **Time** (within working hours)
- **Capacity** (can unload at depot)

## How It Works

### 1. Multi-Trip Solver (`utils/multitrip_solver.py`)
The new solver extends OR-Tools with multi-trip capabilities:

**Key Features:**
- Vehicles can return to depot mid-route
- Battery recharging at depot (energy dimension resets)
- Cargo unloading at depot (capacity dimension resets)
- Time continues accumulating (no reset)
- Uses Guided Local Search metaheuristic for better multi-trip solutions

**Constraints:**
- Each trip must be completable within remaining battery
- Each trip must be completable within remaining working hours
- Capacity resets at depot (unloading)
- Energy resets at depot (recharging)

### 2. Route Extraction (`utils/multitrip_route_extractor.py`)
Utilities to extract and analyze multi-trip routes:

**Functions:**
- `extract_multitrip_routes()`: Identifies separate trips per vehicle
- `flatten_multitrip_routes()`: Converts to simple route format
- `get_trip_statistics()`: Calculates metrics per trip

### 3. UI Integration (App.py)
Added in the "Tabu Search" tab:

**Solver Mode Selection:**
- "Tek Tur (Tabu)": Traditional single-trip routing
- "Çoklu Tur (Multi-Trip)": Allows multiple trips per vehicle

**Trip Statistics Display:**
Shows for each vehicle:
- Number of trips made
- Per-trip metrics:
  - Number of customers
  - Distance (km)
  - Energy consumption (kWh)
  - Load (desi)
  - Time (minutes)

## Usage Example

### Scenario
You have:
- 50 customers to serve
- 3 vehicles available
- Each vehicle: 4500 desi capacity, 100 kWh battery
- Working hours: 09:00 - 18:00

### Single-Trip Mode
- Vehicle 1: 20 customers (limited by capacity/battery)
- Vehicle 2: 18 customers
- Vehicle 3: 12 customers
- **Total: 50 customers in 3 trips**

### Multi-Trip Mode
- Vehicle 1: Trip 1 (15 customers) + Trip 2 (12 customers)
- Vehicle 2: Trip 1 (14 customers) + Trip 2 (9 customers)
- Vehicle 3: Not needed
- **Total: 50 customers in 4 trips using only 2 vehicles**

## Benefits

1. **Reduced Fleet Size**: Fewer vehicles needed to serve same customers
2. **Better Resource Utilization**: Vehicles work closer to full capacity
3. **Cost Savings**: Lower vehicle operating costs
4. **Flexibility**: Adapts to varying customer densities

## Technical Details

### Energy Model
Each leg's energy consumption:
```
E = 0.436 * distance_km + 0.002 * cumulative_load_desi
```

### Depot Behavior
When vehicle returns to depot:
- Energy counter resets to 0 (battery recharged)
- Capacity counter resets to 0 (cargo unloaded)
- Time counter continues (represents actual clock time)

### Optimization Strategy
The solver uses:
- Guided Local Search (better than Tabu for multi-trip)
- Longer time limits recommended (30+ seconds)
- Disjunctions for optional customers (if infeasible)

## Configuration

### In App.py (Tabu Search Tab):
```python
solver_mode = st.selectbox(
    "Çözücü Modu",
    ["Tek Tur (Tabu)", "Çoklu Tur (Multi-Trip)"]
)
```

### Solver Parameters:
- `time_limit_s`: Increased to 30s for multi-trip (default: 10s for single-trip)
- `allow_multi_trip`: Boolean flag to enable/disable feature
- `seed`: Random seed for reproducibility

## Validation

The solver ensures:
1. ✅ No trip exceeds battery capacity
2. ✅ No trip exceeds vehicle capacity (before unloading)
3. ✅ All work completes before 18:00
4. ✅ Each customer visited exactly once
5. ✅ All trips start and end at depot

## Visualization

Multi-trip routes are displayed on the map with:
- Same vehicle color for all trips
- Depot markers show trip transitions
- Statistics table shows per-trip breakdown

## Future Enhancements

Potential improvements:
1. **Variable recharge times**: Model actual battery charging duration
2. **Partial recharging**: Allow departing with <100% battery
3. **Time windows**: Add customer time window constraints
4. **Dynamic pricing**: Different costs per trip
5. **Driver breaks**: Model required rest periods

## Troubleshooting

**Problem**: No multi-trip solution found
**Solutions**:
- Increase time limit (try 60+ seconds)
- Reduce number of customers
- Increase battery capacity
- Extend working hours

**Problem**: Vehicles making too many short trips
**Solutions**:
- Add minimum trip size constraint
- Adjust cost function to penalize depot returns
- Use longer time limits for better optimization

## Performance Notes

- Multi-trip problems are computationally harder (NP-hard++)
- Recommended time limits: 30-60 seconds
- More vehicles = easier to find solution
- Fewer vehicles + multi-trip = harder but more efficient
