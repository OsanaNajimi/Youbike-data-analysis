# YouBike Regression Analysis

Simple project to model bike availability as a function of time of day using regression.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Data
```bash
python collect_data.py
```

Enter duration in minutes (e.g., 20 for debugging, 1440 for 24 hours).

### 3. Analyze with Regression
```bash
python analyze_regression.py
```

This will train regression models and generate visualizations.

## Files

- **`trash.py`** - Quick demo of current bike data
- **`collect_data.py`** - Collect bike data over time
- **`analyze_regression.py`** - Run regression analysis
- **`data_collector.py`** - Data collection module
- **`regression_models.py`** - Regression modeling module
- **`extras/`** - Additional analyses (clustering, weekday/weekend, etc.)

## Manual Usage

### Collect Data
```python
from data_collector import BikeDataCollector

collector = BikeDataCollector()

# Option 1: Single snapshot
df = collector.collect_once()

# Option 2: Collect over time - duration in minutes
collector.collect_continuous(interval_minutes=5, duration_minutes=20)

# Option 3: Collect over time - duration in hours
collector.collect_continuous(interval_minutes=10, duration_hours=24)
```

### Run Regression
```python
from data_collector import BikeDataCollector
from regression_models import BikeRegressionModels

# Load data
collector = BikeDataCollector()
df = collector.load_all_data()

# Train regression model for a station
regressor = BikeRegressionModels(df)
station = "YouBike2.0_捷運市政府站(3號出口)"
results = regressor.train_time_regression(station)

# View results
print(regressor.get_model_summary(station))
regressor.plot_model_comparison(station)
regressor.analyze_feature_importance(station)
```

## What the Regression Does

- **Models**: Linear, Ridge, Lasso, Random Forest
- **Features**: Hour of day, day of week, weekend indicator, rush hours
- **Target**: Available bikes at a station
- **Output**: R² score, RMSE, predictions vs actual, feature importance

## Data Collection Tips

- **For debugging**: 20-60 minutes is enough to test the code
- **For good results**: Collect for at least 24-48 hours
- More data = better predictions
- Data is saved to `bike_data/` folder automatically

## Example: 20-Minute Debug Collection

```bash
# Step 1: Collect data
python collect_data.py
# Choose option 2
# Duration: 20 minutes
# Interval: 5 minutes

# Step 2: Analyze
python analyze_regression.py
```
