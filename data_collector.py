"""
YouBike Data Collection System
Collects bike availability data over time for analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json

class BikeDataCollector:
    def __init__(self, data_folder='bike_data'):
        self.url = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
        self.data_folder = data_folder
        
        # Create data folder if it doesn't exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
    
    def fetch_data(self):
        """Fetch current bike data from API"""
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Add timestamp to each record
            timestamp = datetime.now()
            for record in data:
                record['timestamp'] = timestamp.isoformat()
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def clean_data(self, data):
        """Convert raw data to clean DataFrame"""
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        numeric_cols = ["Quantity", "available_rent_bikes", "available_return_bikes", 
                       "latitude", "longitude"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter out inactive stations
        df = df[df['act'] == '1'].copy()
        
        # Extract useful features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['date'] = df['timestamp'].dt.date
        
        # Calculate occupancy rate
        df['occupancy_rate'] = df['available_rent_bikes'] / df['Quantity']
        
        return df
    
    def save_data(self, df, filename=None):
        """Save data to CSV with timestamp"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'bike_data_{timestamp}.csv'
        
        filepath = os.path.join(self.data_folder, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def collect_once(self):
        """Collect and save one snapshot of data"""
        data = self.fetch_data()
        if data:
            df = self.clean_data(data)
            filepath = self.save_data(df)
            print(f"Collected {len(df)} station records")
            return df
        return None
    
    def collect_continuous(self, interval_minutes=10, duration_minutes=None, duration_hours=None):
        """
        Collect data continuously at regular intervals
        
        Parameters:
        - interval_minutes: How often to collect data (default: 10 minutes)
        - duration_minutes: How long to collect for in minutes (e.g., 20)
        - duration_hours: How long to collect for in hours (e.g., 24)
        Note: Specify either duration_minutes OR duration_hours, not both
        """
        if duration_minutes is not None:
            total_minutes = duration_minutes
        elif duration_hours is not None:
            total_minutes = duration_hours * 60
        else:
            total_minutes = 24 * 60  # Default: 24 hours
        
        intervals = int(total_minutes / interval_minutes)
        print(f"Starting data collection: {intervals} samples over {total_minutes} minutes ({total_minutes/60:.1f} hours)")
        
        for i in range(intervals):
            print(f"\n--- Collection {i+1}/{intervals} ---")
            self.collect_once()
            
            if i < intervals - 1:  # Don't sleep after last collection
                print(f"Waiting {interval_minutes} minutes until next collection...")
                time.sleep(interval_minutes * 60)
        
        print("\nData collection complete!")
    
    def load_all_data(self):
        """Load and combine all collected data files"""
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print("No data files found!")
            return None
        
        dfs = []
        for file in csv_files:
            filepath = os.path.join(self.data_folder, file)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').drop_duplicates()
        
        print(f"Loaded {len(csv_files)} files with {len(combined_df)} total records")
        return combined_df


if __name__ == "__main__":
    collector = BikeDataCollector()
    
    # Example: Collect data once
    print("Collecting single snapshot...")
    df = collector.collect_once()
    
    # Example: Collect continuously (uncomment to use)
    # collector.collect_continuous(interval_minutes=10, duration_hours=24)
    
    # Display sample data
    if df is not None:
        print("\nSample of collected data:")
        print(df[['sna', 'sarea', 'available_rent_bikes', 'Quantity', 'occupancy_rate', 'hour']].head(10))

