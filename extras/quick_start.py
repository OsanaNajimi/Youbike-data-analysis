"""
Quick Start Script
Simple way to get started with bike data analysis
"""

import sys

def main():
    print("="*70)
    print(" "*20 + "YOUBIKE DATA ANALYSIS")
    print(" "*25 + "Quick Start")
    print("="*70)
    
    print("\nWhat would you like to do?\n")
    print("1. Collect single snapshot of current bike data")
    print("2. Collect data continuously (recommended for full analysis)")
    print("3. Run analysis on existing data")
    print("4. Run complete analysis (collect + analyze)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        from data_collector import BikeDataCollector
        print("\nCollecting current bike data...")
        collector = BikeDataCollector()
        df = collector.collect_once()
        if df is not None:
            print(f"\n✓ Success! Collected data for {len(df)} stations")
            print("Data saved to 'bike_data/' folder")
    
    elif choice == "2":
        from data_collector import BikeDataCollector
        print("\nContinuous data collection setup:")
        
        try:
            interval = int(input("Collection interval in minutes (e.g., 10): "))
            duration = int(input("Total duration in hours (e.g., 24): "))
        except ValueError:
            print("Invalid input. Using defaults: 10 minutes interval, 24 hours")
            interval = 10
            duration = 24
        
        print(f"\nStarting data collection...")
        print(f"  - Every {interval} minutes")
        print(f"  - For {duration} hours")
        print(f"  - Total samples: {int(duration * 60 / interval)}")
        print("\nThis will take a while. Press Ctrl+C to stop early.\n")
        
        collector = BikeDataCollector()
        try:
            collector.collect_continuous(interval_minutes=interval, duration_hours=duration)
            print("\n✓ Data collection complete!")
        except KeyboardInterrupt:
            print("\n\nCollection stopped by user.")
            print("Partial data has been saved to 'bike_data/' folder")
    
    elif choice == "3":
        from main_analysis import ComprehensiveBikeAnalysis
        print("\nRunning analysis on existing data...")
        analysis = ComprehensiveBikeAnalysis()
        analysis.run_complete_analysis(collect_new=False)
    
    elif choice == "4":
        from main_analysis import ComprehensiveBikeAnalysis
        print("\nRunning complete analysis (will collect fresh data)...")
        analysis = ComprehensiveBikeAnalysis()
        analysis.run_complete_analysis(collect_new=True)
    
    elif choice == "5":
        print("\nGoodbye!")
        sys.exit(0)
    
    else:
        print("\nInvalid choice. Please run the script again.")
    
    print("\n" + "="*70)
    print("\nNext steps:")
    print("  - Check 'results/' folder for visualizations and reports")
    print("  - Run individual modules for specific analyses")
    print("  - See README.md for detailed usage instructions")
    print("="*70)

if __name__ == "__main__":
    main()

