"""
Data Collection Script
Collect bike availability data over time
"""

from data_collector import BikeDataCollector

def main():
    print("="*70)
    print(" "*25 + "DATA COLLECTION")
    print("="*70)
    
    collector = BikeDataCollector()
    
    print("\nHow do you want to collect data?")
    print("1. Single snapshot (for quick testing)")
    print("2. Continuous collection (for regression analysis)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        print("\n📸 Collecting single snapshot...")
        df = collector.collect_once()
        
        if df is not None:
            print(f"\n✅ Success! Collected {len(df)} stations")
            print(f"   Data saved to bike_data/ folder")
    
    elif choice == "2":
        print("\n⏱️  Continuous collection setup:")
        
        try:
            duration = int(input("Duration in minutes (e.g., 20): "))
            interval = int(input("Interval in minutes (e.g., 5): "))
        except ValueError:
            print("Invalid input. Using defaults: 20 minutes, 5 min interval")
            duration = 20
            interval = 5
        
        samples = int(duration / interval)
        
        print(f"\n🔄 Starting collection...")
        print(f"   - Every {interval} minutes")
        print(f"   - For {duration} minutes")
        print(f"   - Total samples: {samples}")
        print("\n   Press Ctrl+C to stop early\n")
        
        try:
            collector.collect_continuous(interval_minutes=interval, duration_minutes=duration)
            print("\n✅ Collection complete!")
            
            # Show summary
            df = collector.load_all_data()
            if df is not None:
                print(f"\nCollected data summary:")
                print(f"  - Total records: {len(df):,}")
                print(f"  - Unique stations: {df['sna'].nunique()}")
                print(f"  - Time points: {len(df['timestamp'].unique())}")
                print(f"  - Data saved to: bike_data/ folder")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Collection stopped by user")
            print("   Partial data has been saved to bike_data/ folder")
    
    else:
        print("\n❌ Invalid choice")
        return
    
    print("\n" + "="*70)
    print("Next step: Run 'python analyze_regression.py' to analyze the data")
    print("="*70)


if __name__ == "__main__":
    main()

