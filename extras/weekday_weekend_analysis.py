"""
Weekday vs Weekend Comparison Analysis
Detailed analysis of differences in bike usage patterns between weekdays and weekends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class WeekdayWeekendAnalysis:
    def __init__(self, df):
        """
        Initialize weekday/weekend analysis
        
        Parameters:
        - df: DataFrame with bike station data
        """
        self.df = df.copy()
        self.df['day_name'] = self.df['timestamp'].dt.day_name()
    
    def compare_overall_patterns(self, save_path=None):
        """
        Compare overall usage patterns between weekdays and weekends
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Hourly availability comparison
        weekday_hourly = self.df[self.df['is_weekend'] == 0].groupby('hour')['available_rent_bikes'].mean()
        weekend_hourly = self.df[self.df['is_weekend'] == 1].groupby('hour')['available_rent_bikes'].mean()
        
        axes[0, 0].plot(weekday_hourly.index, weekday_hourly.values, 
                       marker='o', linewidth=3, markersize=8, label='Weekday', color='steelblue')
        axes[0, 0].plot(weekend_hourly.index, weekend_hourly.values, 
                       marker='s', linewidth=3, markersize=8, label='Weekend', color='coral')
        axes[0, 0].set_title('Average Available Bikes by Hour', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Available Bikes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(24))
        
        # 2. Occupancy rate comparison
        weekday_occ = self.df[self.df['is_weekend'] == 0].groupby('hour')['occupancy_rate'].mean()
        weekend_occ = self.df[self.df['is_weekend'] == 1].groupby('hour')['occupancy_rate'].mean()
        
        axes[0, 1].plot(weekday_occ.index, weekday_occ.values * 100, 
                       marker='o', linewidth=3, markersize=8, label='Weekday', color='steelblue')
        axes[0, 1].plot(weekend_occ.index, weekend_occ.values * 100, 
                       marker='s', linewidth=3, markersize=8, label='Weekend', color='coral')
        axes[0, 1].set_title('Average Occupancy Rate by Hour', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Occupancy Rate (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(24))
        
        # 3. Distribution comparison
        weekday_data = self.df[self.df['is_weekend'] == 0]['occupancy_rate'].dropna()
        weekend_data = self.df[self.df['is_weekend'] == 1]['occupancy_rate'].dropna()
        
        axes[1, 0].hist(weekday_data, bins=50, alpha=0.6, label='Weekday', 
                       color='steelblue', density=True)
        axes[1, 0].hist(weekend_data, bins=50, alpha=0.6, label='Weekend', 
                       color='coral', density=True)
        axes[1, 0].set_title('Distribution of Occupancy Rates', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Occupancy Rate')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Box plot comparison by day
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = [self.df[self.df['day_name'] == day]['occupancy_rate'].dropna() 
                   for day in day_order if day in self.df['day_name'].values]
        
        bp = axes[1, 1].boxplot(day_data, labels=[d[:3] for d in day_order if d in self.df['day_name'].values],
                               patch_artist=True)
        
        # Color weekdays and weekends differently
        colors = ['steelblue'] * 5 + ['coral'] * 2
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1, 1].set_title('Occupancy Rate by Day of Week', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Occupancy Rate')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Weekday vs Weekend: Overall Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_rush_hour_patterns(self, save_path=None):
        """
        Detailed analysis of rush hour differences
        """
        morning_rush = [7, 8, 9]
        evening_rush = [17, 18, 19]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Morning rush - weekday vs weekend
        weekday_morning = self.df[(self.df['is_weekend'] == 0) & 
                                 (self.df['hour'].isin(morning_rush))]
        weekend_morning = self.df[(self.df['is_weekend'] == 1) & 
                                 (self.df['hour'].isin(morning_rush))]
        
        weekday_morning_hourly = weekday_morning.groupby('hour')['occupancy_rate'].mean()
        weekend_morning_hourly = weekend_morning.groupby('hour')['occupancy_rate'].mean()
        
        axes[0, 0].bar([x - 0.2 for x in weekday_morning_hourly.index], 
                      weekday_morning_hourly.values * 100, width=0.4, 
                      label='Weekday', color='steelblue', alpha=0.7)
        axes[0, 0].bar([x + 0.2 for x in weekend_morning_hourly.index], 
                      weekend_morning_hourly.values * 100, width=0.4, 
                      label='Weekend', color='coral', alpha=0.7)
        axes[0, 0].set_title('Morning Rush Hours (7-9 AM)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Occupancy Rate (%)')
        axes[0, 0].legend()
        axes[0, 0].set_xticks(morning_rush)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Evening rush - weekday vs weekend
        weekday_evening = self.df[(self.df['is_weekend'] == 0) & 
                                 (self.df['hour'].isin(evening_rush))]
        weekend_evening = self.df[(self.df['is_weekend'] == 1) & 
                                 (self.df['hour'].isin(evening_rush))]
        
        weekday_evening_hourly = weekday_evening.groupby('hour')['occupancy_rate'].mean()
        weekend_evening_hourly = weekend_evening.groupby('hour')['occupancy_rate'].mean()
        
        axes[0, 1].bar([x - 0.2 for x in weekday_evening_hourly.index], 
                      weekday_evening_hourly.values * 100, width=0.4, 
                      label='Weekday', color='steelblue', alpha=0.7)
        axes[0, 1].bar([x + 0.2 for x in weekend_evening_hourly.index], 
                      weekend_evening_hourly.values * 100, width=0.4, 
                      label='Weekend', color='coral', alpha=0.7)
        axes[0, 1].set_title('Evening Rush Hours (5-7 PM)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Occupancy Rate (%)')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(evening_rush)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Morning rush distribution
        axes[1, 0].hist(weekday_morning['occupancy_rate'].dropna(), bins=30, 
                       alpha=0.6, label='Weekday', color='steelblue', density=True)
        axes[1, 0].hist(weekend_morning['occupancy_rate'].dropna(), bins=30, 
                       alpha=0.6, label='Weekend', color='coral', density=True)
        axes[1, 0].set_title('Morning Rush Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Occupancy Rate')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Evening rush distribution
        axes[1, 1].hist(weekday_evening['occupancy_rate'].dropna(), bins=30, 
                       alpha=0.6, label='Weekday', color='steelblue', density=True)
        axes[1, 1].hist(weekend_evening['occupancy_rate'].dropna(), bins=30, 
                       alpha=0.6, label='Weekend', color='coral', density=True)
        axes[1, 1].set_title('Evening Rush Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Occupancy Rate')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Rush Hour Analysis: Weekday vs Weekend', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_comparison(self):
        """
        Perform statistical tests to compare weekday vs weekend
        """
        weekday_data = self.df[self.df['is_weekend'] == 0]['occupancy_rate'].dropna()
        weekend_data = self.df[self.df['is_weekend'] == 1]['occupancy_rate'].dropna()
        
        # T-test
        t_stat, t_pvalue = stats.ttest_ind(weekday_data, weekend_data)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(weekday_data, weekend_data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(weekday_data, weekend_data)
        
        results = {
            'weekday_mean': weekday_data.mean(),
            'weekend_mean': weekend_data.mean(),
            'weekday_std': weekday_data.std(),
            'weekend_std': weekend_data.std(),
            'weekday_median': weekday_data.median(),
            'weekend_median': weekend_data.median(),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }
        
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON: WEEKDAY vs WEEKEND")
        print("="*60)
        print(f"\nDescriptive Statistics:")
        print(f"  Weekday - Mean: {results['weekday_mean']:.4f}, Std: {results['weekday_std']:.4f}, Median: {results['weekday_median']:.4f}")
        print(f"  Weekend - Mean: {results['weekend_mean']:.4f}, Std: {results['weekend_std']:.4f}, Median: {results['weekend_median']:.4f}")
        print(f"  Difference in means: {results['weekday_mean'] - results['weekend_mean']:.4f}")
        
        print(f"\nHypothesis Tests:")
        print(f"  T-test: t={results['t_statistic']:.4f}, p-value={results['t_pvalue']:.6f}")
        print(f"  Mann-Whitney U: U={results['u_statistic']:.2f}, p-value={results['u_pvalue']:.6f}")
        print(f"  Kolmogorov-Smirnov: KS={results['ks_statistic']:.4f}, p-value={results['ks_pvalue']:.6f}")
        
        if results['t_pvalue'] < 0.05:
            print(f"\n  → The difference between weekday and weekend is STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print(f"\n  → No statistically significant difference found (p >= 0.05)")
        
        print("="*60 + "\n")
        
        return results
    
    def compare_by_area(self, save_path=None):
        """
        Compare weekday vs weekend patterns by area
        """
        area_comparison = self.df.groupby(['sarea', 'is_weekend']).agg({
            'occupancy_rate': 'mean',
            'available_rent_bikes': 'mean'
        }).reset_index()
        
        # Pivot for easier plotting
        area_pivot = area_comparison.pivot(index='sarea', 
                                           columns='is_weekend', 
                                           values='occupancy_rate')
        area_pivot.columns = ['Weekday', 'Weekend']
        area_pivot['Difference'] = area_pivot['Weekend'] - area_pivot['Weekday']
        area_pivot = area_pivot.sort_values('Difference', ascending=False)
        
        # Plot top areas with biggest differences
        top_n = min(15, len(area_pivot))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Grouped bar chart
        x = np.arange(top_n)
        width = 0.35
        
        axes[0].bar(x - width/2, area_pivot['Weekday'].head(top_n) * 100, 
                   width, label='Weekday', color='steelblue', alpha=0.7)
        axes[0].bar(x + width/2, area_pivot['Weekend'].head(top_n) * 100, 
                   width, label='Weekend', color='coral', alpha=0.7)
        axes[0].set_xlabel('Area')
        axes[0].set_ylabel('Occupancy Rate (%)')
        axes[0].set_title(f'Top {top_n} Areas: Weekday vs Weekend Occupancy', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(area_pivot.head(top_n).index, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Difference plot
        colors = ['green' if x > 0 else 'red' for x in area_pivot['Difference'].head(top_n)]
        axes[1].barh(range(top_n), area_pivot['Difference'].head(top_n) * 100, color=colors, alpha=0.7)
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(area_pivot.head(top_n).index)
        axes[1].set_xlabel('Difference in Occupancy Rate (Weekend - Weekday) %')
        axes[1].set_title('Biggest Weekend/Weekday Differences by Area', 
                         fontsize=12, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Area-Level Weekday vs Weekend Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return area_pivot
    
    def peak_usage_analysis(self, save_path=None):
        """
        Identify and compare peak usage times
        """
        # Find peak hours for weekday and weekend
        weekday_hourly = self.df[self.df['is_weekend'] == 0].groupby('hour')['occupancy_rate'].mean()
        weekend_hourly = self.df[self.df['is_weekend'] == 1].groupby('hour')['occupancy_rate'].mean()
        
        weekday_peak = weekday_hourly.idxmax()
        weekend_peak = weekend_hourly.idxmax()
        
        weekday_low = weekday_hourly.idxmin()
        weekend_low = weekend_hourly.idxmin()
        
        print("\n" + "="*60)
        print("PEAK USAGE ANALYSIS")
        print("="*60)
        print(f"\nWeekday Peak Usage:")
        print(f"  Peak hour: {weekday_peak}:00 (Occupancy: {weekday_hourly[weekday_peak]*100:.1f}%)")
        print(f"  Lowest hour: {weekday_low}:00 (Occupancy: {weekday_hourly[weekday_low]*100:.1f}%)")
        
        print(f"\nWeekend Peak Usage:")
        print(f"  Peak hour: {weekend_peak}:00 (Occupancy: {weekend_hourly[weekend_peak]*100:.1f}%)")
        print(f"  Lowest hour: {weekend_low}:00 (Occupancy: {weekend_hourly[weekend_low]*100:.1f}%)")
        
        print(f"\nPeak Hour Shift: {abs(weekday_peak - weekend_peak)} hours difference")
        print("="*60 + "\n")
        
        return {
            'weekday_peak': weekday_peak,
            'weekend_peak': weekend_peak,
            'weekday_low': weekday_low,
            'weekend_low': weekend_low
        }


if __name__ == "__main__":
    from data_collector import BikeDataCollector
    
    collector = BikeDataCollector()
    df = collector.load_all_data()
    
    if df is None or len(df) == 0:
        print("No data available. Please collect data first.")
    else:
        analyzer = WeekdayWeekendAnalysis(df)
        
        print("Generating weekday vs weekend visualizations...")
        analyzer.compare_overall_patterns()
        analyzer.compare_rush_hour_patterns()
        
        print("\nPerforming statistical comparison...")
        analyzer.statistical_comparison()
        
        print("\nAnalyzing peak usage times...")
        analyzer.peak_usage_analysis()
        
        print("\nComparing patterns by area...")
        analyzer.compare_by_area()

