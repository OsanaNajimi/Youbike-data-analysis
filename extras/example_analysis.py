"""
Example Analysis Script
Step-by-step demonstration of key analyses
"""

from data_collector import BikeDataCollector
from visualizations import BikeVisualizer
from regression_models import BikeRegressionModels
from clustering_spatial import BikeSpatialAnalysis
from weekday_weekend_analysis import WeekdayWeekendAnalysis

def main():
    print("="*70)
    print(" "*20 + "EXAMPLE ANALYSIS WORKFLOW")
    print("="*70)
    
    # Step 1: Load or collect data
    print("\n### STEP 1: Data Collection ###")
    collector = BikeDataCollector()
    
    # Try to load existing data first
    df = collector.load_all_data()
    
    if df is None or len(df) == 0:
        print("No existing data found. Collecting current snapshot...")
        df = collector.collect_once()
    else:
        print(f"Loaded existing data: {len(df):,} records")
    
    if df is None:
        print("Error: Could not load data. Please check your internet connection.")
        return
    
    # Step 2: Basic Exploration
    print("\n### STEP 2: Basic Exploration ###")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"\nFirst few rows:")
    print(df[['sna', 'sarea', 'available_rent_bikes', 'Quantity', 'occupancy_rate']].head())
    
    # Step 3: Time-series Visualization
    print("\n### STEP 3: Exploratory Visualizations ###")
    viz = BikeVisualizer(df)
    
    print("Generating hourly patterns plot...")
    viz.plot_hourly_patterns()
    
    # Step 4: Station Analysis
    print("\n### STEP 4: Station-Level Analysis ###")
    
    # Get top station
    top_station = df.groupby('sna')['Quantity'].mean().nlargest(1).index[0]
    print(f"Analyzing top station: {top_station}")
    
    # If we have temporal data, show time series
    if len(df['timestamp'].unique()) > 1:
        viz.plot_time_series_station(top_station)
    
    # Step 5: Clustering Analysis
    print("\n### STEP 5: Clustering Analysis ###")
    spatial = BikeSpatialAnalysis(df)
    
    print("Creating station profiles...")
    spatial.create_station_profiles()
    
    print("Clustering stations (k=5)...")
    spatial.cluster_by_availability_patterns(n_clusters=5)
    
    print("Visualizing clusters...")
    spatial.plot_cluster_patterns()
    
    print("\nCluster distribution by area:")
    print(spatial.analyze_cluster_by_area())
    
    # Step 6: Spatial Analysis
    print("\n### STEP 6: Spatial Correlation Analysis ###")
    print("Analyzing if nearby stations run out together...")
    spatial.plot_correlation_analysis(max_distance_km=1.0)
    
    # Step 7: Regression (if sufficient temporal data)
    if len(df['timestamp'].unique()) >= 10:
        print("\n### STEP 7: Regression Modeling ###")
        regressor = BikeRegressionModels(df)
        
        print(f"Training regression models for: {top_station}")
        results = regressor.train_time_regression(top_station)
        
        if results:
            print("\nModel Performance:")
            print(regressor.get_model_summary(top_station))
            
            print("\nVisualizing model comparison...")
            regressor.plot_model_comparison(top_station)
            
            print("\nAnalyzing feature importance...")
            regressor.analyze_feature_importance(top_station)
    else:
        print("\n### STEP 7: Regression Modeling ###")
        print("Insufficient temporal data for regression.")
        print("Collect data over time using: collector.collect_continuous()")
    
    # Step 8: Weekday/Weekend comparison (if available)
    has_weekday = (df['is_weekend'] == 0).any()
    has_weekend = (df['is_weekend'] == 1).any()
    
    if has_weekday and has_weekend:
        print("\n### STEP 8: Weekday vs Weekend Analysis ###")
        analyzer = WeekdayWeekendAnalysis(df)
        
        print("Comparing overall patterns...")
        analyzer.compare_overall_patterns()
        
        print("\nStatistical comparison:")
        analyzer.statistical_comparison()
        
        print("\nPeak usage analysis:")
        analyzer.peak_usage_analysis()
    else:
        print("\n### STEP 8: Weekday vs Weekend Analysis ###")
        print("Need data from both weekdays and weekends.")
        print("Collect data over multiple days for this analysis.")
    
    print("\n" + "="*70)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Run data collection over multiple days for best results")
    print("2. Use main_analysis.py for comprehensive automated analysis")
    print("3. All modules are customizable - see README.md for details")
    print("="*70)


if __name__ == "__main__":
    main()

