"""
Main Analysis Script
Comprehensive bike sharing analysis combining all modules
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data_collector import BikeDataCollector
from visualizations import BikeVisualizer
from regression_models import BikeRegressionModels
from clustering_spatial import BikeSpatialAnalysis
from weekday_weekend_analysis import WeekdayWeekendAnalysis


class ComprehensiveBikeAnalysis:
    def __init__(self, data_folder='bike_data', output_folder='results'):
        """
        Initialize comprehensive analysis
        
        Parameters:
        - data_folder: Folder containing collected bike data
        - output_folder: Folder to save analysis results
        """
        self.collector = BikeDataCollector(data_folder=data_folder)
        self.output_folder = output_folder
        self.df = None
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def load_or_collect_data(self, collect_new=False):
        """
        Load existing data or collect new data
        """
        if collect_new:
            print("Collecting new data snapshot...")
            self.df = self.collector.collect_once()
        else:
            print("Loading existing data...")
            self.df = self.collector.load_all_data()
            
            if self.df is None or len(self.df) == 0:
                print("No existing data found. Collecting new data...")
                self.df = self.collector.collect_once()
        
        if self.df is not None:
            print(f"\nData loaded successfully!")
            print(f"  - Total records: {len(self.df):,}")
            print(f"  - Unique stations: {self.df['sna'].nunique()}")
            print(f"  - Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            print(f"  - Total hours covered: {(self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600:.1f}")
        
        return self.df
    
    def run_exploratory_analysis(self):
        """
        Run exploratory visualizations
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        viz = BikeVisualizer(self.df)
        
        # Overall hourly patterns
        print("\n1. Generating hourly patterns visualization...")
        viz.plot_hourly_patterns(save_path=os.path.join(self.output_folder, 'hourly_patterns.png'))
        
        # Top stations comparison
        print("2. Comparing top busiest stations...")
        viz.plot_top_stations_comparison(n_stations=10, 
                                         save_path=os.path.join(self.output_folder, 'top_stations.png'))
        
        # Heatmap by area
        print("3. Creating heatmap by area...")
        viz.plot_heatmap_by_area(save_path=os.path.join(self.output_folder, 'area_heatmap.png'))
        
        # Get station summary
        print("4. Generating station summary...")
        summary = viz.get_station_summary()
        summary.to_csv(os.path.join(self.output_folder, 'station_summary.csv'))
        print(f"   Saved to: {self.output_folder}/station_summary.csv")
        
        # Time series for popular station if we have temporal data
        if len(self.df['timestamp'].unique()) > 1:
            popular_station = self.df.groupby('sna')['Quantity'].mean().nlargest(1).index[0]
            print(f"5. Plotting time series for popular station: {popular_station}")
            viz.plot_time_series_station(popular_station, 
                                         save_path=os.path.join(self.output_folder, 'time_series_example.png'))
    
    def run_regression_analysis(self):
        """
        Run regression models
        """
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS")
        print("="*60)
        
        if len(self.df['timestamp'].unique()) < 10:
            print("\nInsufficient temporal data for regression analysis.")
            print("Please collect data over multiple time points.")
            return
        
        regressor = BikeRegressionModels(self.df)
        
        # Select top 5 stations for modeling
        top_stations = self.df.groupby('sna')['Quantity'].mean().nlargest(5).index.tolist()
        
        model_results = []
        
        for i, station in enumerate(top_stations, 1):
            print(f"\n{i}. Training models for: {station}")
            results = regressor.train_time_regression(station)
            
            if results:
                # Get summary
                summary = regressor.get_model_summary(station)
                print(summary)
                
                # Save visualizations
                regressor.plot_model_comparison(
                    station, 
                    save_path=os.path.join(self.output_folder, f'regression_{i}_{station[:20]}.png')
                )
                
                regressor.analyze_feature_importance(
                    station,
                    save_path=os.path.join(self.output_folder, f'feature_importance_{i}_{station[:20]}.png')
                )
                
                model_results.append({
                    'station': station,
                    'best_model': 'Random Forest',
                    'test_r2': results['Random Forest']['test_r2']
                })
        
        # Save model results summary
        if model_results:
            results_df = pd.DataFrame(model_results)
            results_df.to_csv(os.path.join(self.output_folder, 'regression_results.csv'), index=False)
            print(f"\nRegression results saved to: {self.output_folder}/regression_results.csv")
    
    def run_clustering_analysis(self):
        """
        Run clustering and spatial analysis
        """
        print("\n" + "="*60)
        print("CLUSTERING & SPATIAL ANALYSIS")
        print("="*60)
        
        spatial = BikeSpatialAnalysis(self.df)
        
        # Create station profiles
        print("\n1. Creating station behavioral profiles...")
        spatial.create_station_profiles()
        
        # Cluster stations
        print("2. Clustering stations by availability patterns...")
        spatial.cluster_by_availability_patterns(n_clusters=5)
        
        # Visualize clusters
        print("3. Generating cluster visualizations...")
        spatial.plot_cluster_patterns(save_path=os.path.join(self.output_folder, 'cluster_patterns.png'))
        spatial.plot_spatial_clusters(save_path=os.path.join(self.output_folder, 'spatial_clusters.png'))
        
        # PCA analysis
        print("4. Running PCA analysis...")
        spatial.pca_analysis(save_path=os.path.join(self.output_folder, 'pca_analysis.png'))
        
        # Analyze cluster distribution by area
        print("5. Analyzing cluster distribution by area...")
        cluster_area = spatial.analyze_cluster_by_area()
        print(cluster_area)
        cluster_area.to_csv(os.path.join(self.output_folder, 'cluster_by_area.csv'))
        
        # Correlation analysis
        print("6. Analyzing nearby station correlations...")
        spatial.plot_correlation_analysis(
            max_distance_km=1.0,
            save_path=os.path.join(self.output_folder, 'correlation_analysis.png')
        )
    
    def run_weekday_weekend_analysis(self):
        """
        Run weekday vs weekend comparison
        """
        print("\n" + "="*60)
        print("WEEKDAY vs WEEKEND ANALYSIS")
        print("="*60)
        
        # Check if we have both weekday and weekend data
        has_weekday = (self.df['is_weekend'] == 0).any()
        has_weekend = (self.df['is_weekend'] == 1).any()
        
        if not (has_weekday and has_weekend):
            print("\nInsufficient data for weekday/weekend comparison.")
            print("Please collect data across both weekdays and weekends.")
            return
        
        analyzer = WeekdayWeekendAnalysis(self.df)
        
        # Overall patterns
        print("\n1. Comparing overall patterns...")
        analyzer.compare_overall_patterns(
            save_path=os.path.join(self.output_folder, 'weekday_weekend_overall.png')
        )
        
        # Rush hour analysis
        print("2. Analyzing rush hour differences...")
        analyzer.compare_rush_hour_patterns(
            save_path=os.path.join(self.output_folder, 'weekday_weekend_rush.png')
        )
        
        # Statistical comparison
        print("3. Running statistical tests...")
        stats_results = analyzer.statistical_comparison()
        
        # Save statistics
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv(os.path.join(self.output_folder, 'weekday_weekend_stats.csv'), index=False)
        
        # Peak usage analysis
        print("4. Identifying peak usage times...")
        peak_results = analyzer.peak_usage_analysis()
        
        # Area comparison
        print("5. Comparing by area...")
        area_comparison = analyzer.compare_by_area(
            save_path=os.path.join(self.output_folder, 'weekday_weekend_by_area.png')
        )
        area_comparison.to_csv(os.path.join(self.output_folder, 'area_comparison.csv'))
    
    def generate_report(self):
        """
        Generate a summary report
        """
        report_path = os.path.join(self.output_folder, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("YOUBIKE DATA ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Unique Stations: {self.df['sna'].nunique()}\n")
            f.write(f"Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}\n")
            f.write(f"Hours Covered: {(self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600:.1f}\n")
            f.write(f"Areas Covered: {self.df['sarea'].nunique()}\n\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            # Overall statistics
            f.write(f"Average Occupancy Rate: {self.df['occupancy_rate'].mean()*100:.1f}%\n")
            f.write(f"Average Available Bikes: {self.df['available_rent_bikes'].mean():.1f}\n")
            
            # Peak hours
            hourly_avg = self.df.groupby('hour')['occupancy_rate'].mean()
            peak_hour = hourly_avg.idxmax()
            f.write(f"Peak Hour: {peak_hour}:00 ({hourly_avg[peak_hour]*100:.1f}% occupancy)\n")
            
            # Busiest stations
            top_stations = self.df.groupby('sna')['available_rent_bikes'].mean().nlargest(3)
            f.write(f"\nTop 3 Busiest Stations:\n")
            for i, (station, bikes) in enumerate(top_stations.items(), 1):
                f.write(f"  {i}. {station}: {bikes:.1f} avg bikes\n")
            
            # Weekday vs weekend if available
            has_both = (self.df['is_weekend'] == 0).any() and (self.df['is_weekend'] == 1).any()
            if has_both:
                weekday_avg = self.df[self.df['is_weekend'] == 0]['occupancy_rate'].mean()
                weekend_avg = self.df[self.df['is_weekend'] == 1]['occupancy_rate'].mean()
                f.write(f"\nWeekday Avg Occupancy: {weekday_avg*100:.1f}%\n")
                f.write(f"Weekend Avg Occupancy: {weekend_avg*100:.1f}%\n")
                f.write(f"Difference: {(weekend_avg - weekday_avg)*100:.1f}%\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("ANALYSIS OUTPUTS\n")
            f.write("="*60 + "\n\n")
            f.write("All visualizations and detailed results have been saved to:\n")
            f.write(f"{self.output_folder}/\n\n")
            f.write("Files generated:\n")
            
            # List all output files
            output_files = sorted([f for f in os.listdir(self.output_folder) if f != 'analysis_report.txt'])
            for file in output_files:
                f.write(f"  - {file}\n")
        
        print(f"\nAnalysis report saved to: {report_path}")
        
        # Also print to console
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())
    
    def run_complete_analysis(self, collect_new=False):
        """
        Run all analysis modules
        """
        print("\n" + "="*70)
        print(" "*20 + "COMPREHENSIVE BIKE ANALYSIS")
        print("="*70)
        
        # Load or collect data
        self.load_or_collect_data(collect_new=collect_new)
        
        if self.df is None:
            print("Error: No data available for analysis!")
            return
        
        # Run all analyses
        self.run_exploratory_analysis()
        self.run_regression_analysis()
        self.run_clustering_analysis()
        self.run_weekday_weekend_analysis()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print(" "*20 + "ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_folder}/")


if __name__ == "__main__":
    # Initialize comprehensive analysis
    analysis = ComprehensiveBikeAnalysis(
        data_folder='bike_data',
        output_folder='results'
    )
    
    # Run complete analysis
    # Set collect_new=True to collect fresh data
    # Set collect_new=False to use existing data
    analysis.run_complete_analysis(collect_new=False)
    
    print("\n📊 Analysis complete! Check the 'results' folder for all outputs.")

