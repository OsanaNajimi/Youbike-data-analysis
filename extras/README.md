# YouBike Data Analysis Project

A comprehensive statistical analysis of Taipei's YouBike bike-sharing system for a statistics course final project.

## Project Overview

This project analyzes YouBike station data to understand:
- **Temporal patterns**: How bike availability changes throughout the day
- **Weekday vs Weekend differences**: Different usage patterns between working days and leisure days
- **Spatial patterns**: How nearby stations behave and geographic clustering
- **Predictive modeling**: Regression models to forecast bike availability
- **Station clustering**: Identifying stations with similar usage patterns

## Data Source

Data is collected from Taipei's official YouBike API:
`https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json`

## Project Structure

```
.
├── data_collector.py           # Data collection and management
├── visualizations.py            # Exploratory visualizations
├── regression_models.py         # Time-based regression models
├── clustering_spatial.py        # Clustering and spatial analysis
├── weekday_weekend_analysis.py  # Weekday/weekend comparisons
├── main_analysis.py            # Main script to run all analyses
├── requirements.txt            # Python dependencies
├── bike_data/                  # Folder for collected data (created automatically)
└── results/                    # Folder for analysis outputs (created automatically)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Complete Analysis

```bash
python main_analysis.py
```

This will:
1. Load existing data (or collect new data if none exists)
2. Run all analysis modules
3. Generate visualizations and save to `results/` folder
4. Create a summary report

### Option 1: Collect Data Once

```python
from data_collector import BikeDataCollector

collector = BikeDataCollector()
df = collector.collect_once()
```

### Option 2: Collect Data Over Time (Recommended for Time-Series Analysis)

```python
from data_collector import BikeDataCollector

collector = BikeDataCollector()

# Collect data every 10 minutes for 24 hours
collector.collect_continuous(interval_minutes=10, duration_hours=24)
```

**Note**: For best results, collect data over several days to capture weekday and weekend patterns!

### Run Individual Analyses

#### 1. Exploratory Visualization

```python
from visualizations import BikeVisualizer

viz = BikeVisualizer(df)

# Hourly patterns (shows morning/evening rush)
viz.plot_hourly_patterns()

# Compare top stations
viz.plot_top_stations_comparison(n_stations=10)

# Heatmap by area
viz.plot_heatmap_by_area()

# Time series for specific station
viz.plot_time_series_station("YouBike2.0_捷運市政府站(3號出口)")
```

#### 2. Regression Models

```python
from regression_models import BikeRegressionModels

regressor = BikeRegressionModels(df)

# Train models for a specific station
station_name = "YouBike2.0_捷運市政府站(3號出口)"
results = regressor.train_time_regression(station_name)

# View model performance
print(regressor.get_model_summary(station_name))

# Visualize predictions
regressor.plot_model_comparison(station_name)
regressor.analyze_feature_importance(station_name)

# Predict future hours
predictions = regressor.predict_future_hours(station_name, hours_ahead=24)
```

#### 3. Clustering & Spatial Analysis

```python
from clustering_spatial import BikeSpatialAnalysis

spatial = BikeSpatialAnalysis(df)

# Create station profiles
spatial.create_station_profiles()

# Cluster stations by patterns
spatial.cluster_by_availability_patterns(n_clusters=5)

# Visualize clusters
spatial.plot_cluster_patterns()
spatial.plot_spatial_clusters()

# PCA analysis
spatial.pca_analysis()

# Find nearby stations
nearby = spatial.find_nearby_stations("YouBike2.0_捷運市政府站(3號出口)", radius_km=1.0)

# Analyze if nearby stations run out together
spatial.plot_correlation_analysis(max_distance_km=1.0)
```

#### 4. Weekday vs Weekend Analysis

```python
from weekday_weekend_analysis import WeekdayWeekendAnalysis

analyzer = WeekdayWeekendAnalysis(df)

# Overall comparison
analyzer.compare_overall_patterns()

# Rush hour analysis
analyzer.compare_rush_hour_patterns()

# Statistical tests
analyzer.statistical_comparison()

# Peak usage times
analyzer.peak_usage_analysis()

# By area comparison
analyzer.compare_by_area()
```

## Research Questions Addressed

### 1. Time-Series Patterns
- **Morning rush vs. Evening rush**: Visualizations show clear peaks during commute hours (7-9 AM, 5-7 PM)
- **Hourly patterns**: Occupancy rates vary significantly throughout the day

### 2. Regression Models
- **Multiple models tested**: Linear, Ridge, Lasso, Random Forest
- **Time-based features**: Hour of day, day of week, cyclical encoding
- **Prediction accuracy**: R² scores and RMSE metrics for each model
- **Feature importance**: Which time features matter most?

### 3. Clustering & Spatial Analysis
- **Behavioral clusters**: Stations grouped by similar availability patterns
- **Geographic distribution**: How clusters are distributed across Taipei
- **MRT vs Park vs University**: Cluster analysis reveals different station types
- **Nearby station correlation**: Do adjacent stations run out together?

### 4. Weekday vs Weekend
- **Usage pattern differences**: Statistical tests show significant differences
- **Rush hour shifts**: Different peak times on weekends
- **Area-specific patterns**: Some areas show bigger weekend changes

## Analysis Outputs

All results are saved to the `results/` folder:

- **Visualizations**: PNG files of all plots
- **CSV files**: Numerical results and summaries
- **Analysis report**: Text summary of key findings

## Tips for Your Final Project

1. **Collect data over multiple days**: Run the collector continuously to get rich temporal data
2. **Focus on interesting patterns**: Morning rush, MRT stations, weekend leisure patterns
3. **Statistical rigor**: Use the statistical tests provided to support your claims
4. **Visual presentation**: The generated plots are publication-ready
5. **Reproducibility**: All code is modular and well-documented

## Example Analysis Workflow

```python
# Step 1: Collect data over time
collector = BikeDataCollector()
collector.collect_continuous(interval_minutes=15, duration_hours=48)  # 2 days

# Step 2: Load and run complete analysis
from main_analysis import ComprehensiveBikeAnalysis
analysis = ComprehensiveBikeAnalysis()
analysis.run_complete_analysis()

# Step 3: Review results in 'results/' folder
```

## Advanced Customization

Each module can be customized:
- Change number of clusters
- Adjust distance thresholds for spatial analysis
- Try different regression models
- Focus on specific areas or stations

## Contact & Support

For questions about the code or analysis methods, refer to the inline documentation in each module.

Good luck with your final project! 🚲📊

