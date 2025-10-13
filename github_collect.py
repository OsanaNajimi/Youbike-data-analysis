"""
GitHub Actions Data Collection Script
Collects a single snapshot each time it runs (every 10 minutes via GitHub Actions)
"""

from data_collector import BikeDataCollector
import sys

def main():
    print("="*70)
    print("GitHub Actions - Data Collection")
    print("="*70)
    
    try:
        collector = BikeDataCollector()
        df = collector.collect_once()
        
        if df is not None:
            print(f"✓ Successfully collected data for {len(df)} stations")
            print(f"  Data saved to bike_data/ folder")
            sys.exit(0)
        else:
            print("✗ Failed to collect data")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

