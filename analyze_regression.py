"""
Regression Analysis Script
Model bike availability as a function of time of day
"""

from data_collector import BikeDataCollector
from regression_models import BikeRegressionModels
import os

def main():
    print("="*70)
    print(" "*20 + "REGRESSION ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n📊 Loading collected data...")
    collector = BikeDataCollector()
    df = collector.load_all_data()
    
    if df is None or len(df) == 0:
        print("\n❌ No data found!")
        print("   Please run 'python collect_data.py' first to collect data")
        return
    
    # Show data summary
    n_timepoints = len(df['timestamp'].unique())
    print(f"\n✓ Data loaded successfully:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Unique stations: {df['sna'].nunique()}")
    print(f"  - Time points: {n_timepoints}")
    print(f"  - Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if n_timepoints < 5:
        print(f"\n⚠️  Warning: Only {n_timepoints} time points available")
        print("   Regression may not be accurate. Recommend collecting more data.")
        proceed = input("\nContinue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting. Collect more data with 'python collect_data.py'")
            return
    
    # Select stations to analyze
    print("\n" + "="*70)
    print("STATION SELECTION")
    print("="*70)
    
    print("\nHow many stations to analyze?")
    try:
        n_stations = int(input("Enter number (e.g., 3): ").strip())
        n_stations = max(1, min(n_stations, 10))  # Between 1 and 10
    except ValueError:
        n_stations = 3
        print(f"Using default: {n_stations} stations")
    
    # Get top stations by capacity
    top_stations = df.groupby('sna')['Quantity'].mean().nlargest(n_stations).index.tolist()
    
    print(f"\nAnalyzing top {n_stations} stations:")
    for i, station in enumerate(top_stations, 1):
        print(f"  {i}. {station}")
    
    # Create output folder
    output_folder = 'regression_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Run regression analysis
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODELS")
    print("="*70)
    
    regressor = BikeRegressionModels(df)
    
    for i, station in enumerate(top_stations, 1):
        print(f"\n{'─'*70}")
        print(f"Station {i}/{n_stations}: {station}")
        print(f"{'─'*70}")
        
        # Train models
        results = regressor.train_time_regression(station)
        
        if results is None:
            print(f"⚠️  Insufficient data for {station}")
            continue
        
        # Show results
        summary = regressor.get_model_summary(station)
        print("\nModel Performance:")
        print(summary.to_string(index=False))
        
        # Find best model
        best_model = None
        best_r2 = -999
        for model_name, result in results.items():
            if result['test_r2'] > best_r2:
                best_r2 = result['test_r2']
                best_model = model_name
        
        print(f"\n✨ Best Model: {best_model} (Test R² = {best_r2:.4f})")
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        
        regressor.plot_model_comparison(
            station, 
            save_path=os.path.join(output_folder, f'station_{i}_models.png')
        )
        print(f"   ✓ Saved: {output_folder}/station_{i}_models.png")
        
        regressor.analyze_feature_importance(
            station,
            save_path=os.path.join(output_folder, f'station_{i}_features.png')
        )
        print(f"   ✓ Saved: {output_folder}/station_{i}_features.png")
        
        regressor.plot_predictions_over_time(
            station,
            model_name=best_model,
            save_path=os.path.join(output_folder, f'station_{i}_predictions.png')
        )
        print(f"   ✓ Saved: {output_folder}/station_{i}_predictions.png")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_folder}/")
    print("\nWhat the regression models do:")
    print("  • Predict bike availability based on time of day")
    print("  • Use features: hour, day of week, weekend, rush hours")
    print("  • Compare 4 models: Linear, Ridge, Lasso, Random Forest")
    print("\nCheck the plots to see:")
    print("  • How well each model predicts (R² score)")
    print("  • Which time features are most important")
    print("  • Actual vs predicted bike availability")
    print("="*70)


if __name__ == "__main__":
    main()

