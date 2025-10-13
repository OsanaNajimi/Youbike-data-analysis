"""
Simple Demo Script - YouBike Data Analysis
This demonstrates basic data fetching and exploration
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch data from YouBike API
url = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
print("Fetching YouBike data...")
data = requests.get(url).json()
df = pd.DataFrame(data)

# Convert numeric columns
for col in ["Quantity", "available_rent_bikes", "available_return_bikes", "latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter active stations only
df = df[df["act"] == '1'].copy()

print(f"\n{'='*60}")
print(f"YOUBIKE DATA SNAPSHOT")
print(f"{'='*60}")
print(f"Total Active Stations: {len(df)}")
print(f"Total Bikes Available: {df['available_rent_bikes'].sum():.0f}")
print(f"Total Capacity: {df['Quantity'].sum():.0f}")
print(f"Average Occupancy Rate: {(df['available_rent_bikes'].sum() / df['Quantity'].sum())*100:.1f}%")
print(f"Areas Covered: {df['sarea'].nunique()}")

# Show top 10 stations by capacity
print(f"\n{'='*60}")
print("TOP 10 LARGEST STATIONS")
print(f"{'='*60}")
top_stations = df.nlargest(10, 'Quantity')[['sna', 'sarea', 'Quantity', 'available_rent_bikes']]
for idx, row in enumerate(top_stations.itertuples(), 1):
    occupancy = (row.available_rent_bikes / row.Quantity * 100) if row.Quantity > 0 else 0
    print(f"{idx:2d}. {row.sna[:40]:<40} | Capacity: {row.Quantity:3.0f} | Available: {row.available_rent_bikes:3.0f} | {occupancy:5.1f}%")

# Show areas with most stations
print(f"\n{'='*60}")
print("STATIONS BY AREA")
print(f"{'='*60}")
area_counts = df['sarea'].value_counts().head(10)
for area, count in area_counts.items():
    print(f"{area:<30} {count:3d} stations")

print(f"\n{'='*60}")
print("NEXT STEPS:")
print(f"{'='*60}")
print("1. Run 'python collect_data.py' to collect bike data over time")
print("2. Run 'python analyze_regression.py' to train regression models")
print("3. Check 'extras/' folder for additional analyses (clustering, etc.)")
print(f"{'='*60}\n")