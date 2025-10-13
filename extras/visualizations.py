"""
Exploratory Visualization Module
Time-series plots and visual analysis of bike availability patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BikeVisualizer:
    def __init__(self, df):
        """
        Initialize visualizer with bike data
        
        Parameters:
        - df: DataFrame with bike station data including timestamps
        """
        self.df = df.copy()
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 6)
    
    def plot_time_series_station(self, station_name, save_path=None):
        """
        Plot time series for a specific station
        Shows bike availability over time
        """
        station_data = self.df[self.df['sna'] == station_name].sort_values('timestamp')
        
        if len(station_data) == 0:
            print(f"No data found for station: {station_name}")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot available bikes
        axes[0].plot(station_data['timestamp'], station_data['available_rent_bikes'], 
                    marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[0].set_title(f'Available Bikes - {station_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Number of Available Bikes')
        axes[0].grid(True, alpha=0.3)
        
        # Plot occupancy rate
        axes[1].plot(station_data['timestamp'], station_data['occupancy_rate'] * 100, 
                    marker='o', linestyle='-', linewidth=2, markersize=4, color='green')
        axes[1].set_title(f'Occupancy Rate - {station_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Occupancy Rate (%)')
        axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hourly_patterns(self, save_path=None):
        """
        Plot average bike availability by hour of day
        Shows morning and evening rush patterns
        """
        hourly_avg = self.df.groupby('hour').agg({
            'available_rent_bikes': 'mean',
            'occupancy_rate': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average bikes available by hour
        axes[0].bar(hourly_avg['hour'], hourly_avg['available_rent_bikes'], 
                   color='steelblue', alpha=0.7)
        axes[0].set_title('Average Available Bikes by Hour of Day', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Average Available Bikes')
        axes[0].set_xticks(range(24))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Mark rush hours
        axes[0].axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
        axes[0].axvspan(17, 19, alpha=0.2, color='orange', label='Evening Rush')
        axes[0].legend()
        
        # Occupancy rate by hour
        axes[1].plot(hourly_avg['hour'], hourly_avg['occupancy_rate'] * 100, 
                    marker='o', linewidth=3, markersize=8, color='green')
        axes[1].set_title('Average Occupancy Rate by Hour of Day', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Occupancy Rate (%)')
        axes[1].set_xticks(range(24))
        axes[1].grid(True, alpha=0.3)
        axes[1].axvspan(7, 9, alpha=0.2, color='red')
        axes[1].axvspan(17, 19, alpha=0.2, color='orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weekday_vs_weekend(self, save_path=None):
        """
        Compare bike availability patterns between weekdays and weekends
        """
        # Calculate hourly patterns for weekdays vs weekends
        weekday_data = self.df[self.df['is_weekend'] == 0].groupby('hour').agg({
            'available_rent_bikes': 'mean',
            'occupancy_rate': 'mean'
        }).reset_index()
        
        weekend_data = self.df[self.df['is_weekend'] == 1].groupby('hour').agg({
            'available_rent_bikes': 'mean',
            'occupancy_rate': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Available bikes comparison
        axes[0].plot(weekday_data['hour'], weekday_data['available_rent_bikes'], 
                    marker='o', linewidth=3, markersize=8, label='Weekday', color='steelblue')
        axes[0].plot(weekend_data['hour'], weekend_data['available_rent_bikes'], 
                    marker='s', linewidth=3, markersize=8, label='Weekend', color='coral')
        axes[0].set_title('Weekday vs Weekend: Available Bikes', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Average Available Bikes')
        axes[0].set_xticks(range(24))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Occupancy rate comparison
        axes[1].plot(weekday_data['hour'], weekday_data['occupancy_rate'] * 100, 
                    marker='o', linewidth=3, markersize=8, label='Weekday', color='steelblue')
        axes[1].plot(weekend_data['hour'], weekend_data['occupancy_rate'] * 100, 
                    marker='s', linewidth=3, markersize=8, label='Weekend', color='coral')
        axes[1].set_title('Weekday vs Weekend: Occupancy Rate', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Occupancy Rate (%)')
        axes[1].set_xticks(range(24))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_stations_comparison(self, n_stations=10, save_path=None):
        """
        Compare bike availability across top N busiest stations
        """
        # Find busiest stations by total capacity
        top_stations = self.df.groupby('sna')['Quantity'].mean().nlargest(n_stations)
        
        # Get data for these stations
        station_hourly = self.df[self.df['sna'].isin(top_stations.index)].groupby(['sna', 'hour']).agg({
            'available_rent_bikes': 'mean'
        }).reset_index()
        
        plt.figure(figsize=(16, 8))
        
        for station in top_stations.index:
            station_data = station_hourly[station_hourly['sna'] == station]
            plt.plot(station_data['hour'], station_data['available_rent_bikes'], 
                    marker='o', linewidth=2, label=station, alpha=0.7)
        
        plt.title(f'Hourly Patterns: Top {n_stations} Busiest Stations', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Available Bikes')
        plt.xticks(range(24))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heatmap_by_area(self, save_path=None):
        """
        Create heatmap of bike availability by area and hour
        """
        # Calculate average availability by area and hour
        heatmap_data = self.df.groupby(['sarea', 'hour'])['occupancy_rate'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='sarea', columns='hour', values='occupancy_rate')
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_pivot * 100, cmap='RdYlGn', annot=False, fmt='.1f', 
                   cbar_kws={'label': 'Occupancy Rate (%)'})
        plt.title('Bike Occupancy Rate Heatmap by Area and Hour', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Area')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_station_summary(self):
        """Get summary statistics for all stations"""
        summary = self.df.groupby('sna').agg({
            'available_rent_bikes': ['mean', 'std', 'min', 'max'],
            'occupancy_rate': ['mean', 'std'],
            'Quantity': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'sarea': 'first'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.sort_values('available_rent_bikes_mean', ascending=False)
        
        return summary


if __name__ == "__main__":
    # Example usage
    from data_collector import BikeDataCollector
    
    collector = BikeDataCollector()
    
    # Try to load existing data, or collect new data
    df = collector.load_all_data()
    if df is None or len(df) == 0:
        print("No existing data found. Collecting new data...")
        df = collector.collect_once()
    
    if df is not None:
        viz = BikeVisualizer(df)
        
        # Generate visualizations
        print("\nGenerating hourly patterns plot...")
        viz.plot_hourly_patterns()
        
        if len(df['timestamp'].unique()) > 1:
            print("\nGenerating weekday vs weekend comparison...")
            viz.plot_weekday_vs_weekend()
        
        print("\nStation summary statistics:")
        print(viz.get_station_summary().head(10))

