"""
Clustering and Spatial Analysis Module
Analyze station patterns and geographic relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class BikeSpatialAnalysis:
    def __init__(self, df):
        """
        Initialize spatial analysis with bike data
        
        Parameters:
        - df: DataFrame with bike station data
        """
        self.df = df.copy()
        self.station_profiles = None
        self.clusters = None
    
    def create_station_profiles(self):
        """
        Create behavioral profiles for each station based on hourly patterns
        """
        # Calculate hourly average availability for each station
        hourly_profiles = self.df.groupby(['sna', 'hour'])['occupancy_rate'].mean().reset_index()
        
        # Pivot to create feature matrix (stations x hours)
        profile_matrix = hourly_profiles.pivot(index='sna', columns='hour', values='occupancy_rate')
        
        # Add aggregate statistics
        station_stats = self.df.groupby('sna').agg({
            'occupancy_rate': ['mean', 'std'],
            'available_rent_bikes': ['mean', 'std'],
            'Quantity': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'sarea': 'first',
            'is_weekend': 'mean',
            'is_morning_rush': 'mean',
            'is_evening_rush': 'mean'
        })
        
        station_stats.columns = ['_'.join(col).strip() for col in station_stats.columns.values]
        
        # Combine hourly profiles with statistics
        self.station_profiles = profile_matrix.join(station_stats)
        
        return self.station_profiles
    
    def cluster_by_availability_patterns(self, n_clusters=5, method='kmeans'):
        """
        Cluster stations based on their availability patterns
        
        Parameters:
        - n_clusters: Number of clusters
        - method: 'kmeans' or 'dbscan'
        """
        if self.station_profiles is None:
            self.create_station_profiles()
        
        # Select hourly features (columns 0-23)
        hourly_features = self.station_profiles.iloc[:, :24].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(hourly_features)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features_scaled)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features_scaled)
        else:
            raise ValueError("Method must be 'kmeans' or 'dbscan'")
        
        self.station_profiles['cluster'] = cluster_labels
        self.clusters = cluster_labels
        
        return cluster_labels
    
    def plot_cluster_patterns(self, save_path=None):
        """
        Visualize the hourly patterns for each cluster
        """
        if 'cluster' not in self.station_profiles.columns:
            print("Please run cluster_by_availability_patterns first")
            return
        
        n_clusters = len(self.station_profiles['cluster'].unique())
        
        fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))
        if n_clusters == 1:
            axes = [axes]
        
        for cluster_id in range(n_clusters):
            cluster_stations = self.station_profiles[
                self.station_profiles['cluster'] == cluster_id
            ].iloc[:, :24]
            
            # Plot each station's pattern
            for idx, row in cluster_stations.iterrows():
                axes[cluster_id].plot(range(24), row.values, alpha=0.3, color='steelblue')
            
            # Plot cluster average
            avg_pattern = cluster_stations.mean(axis=0)
            axes[cluster_id].plot(range(24), avg_pattern.values, 
                                 linewidth=3, color='red', label='Cluster Average')
            
            axes[cluster_id].set_title(f'Cluster {cluster_id}\n({len(cluster_stations)} stations)')
            axes[cluster_id].set_xlabel('Hour of Day')
            axes[cluster_id].set_ylabel('Occupancy Rate')
            axes[cluster_id].legend()
            axes[cluster_id].grid(True, alpha=0.3)
            axes[cluster_id].set_xticks(range(0, 24, 3))
        
        plt.suptitle('Station Clusters by Availability Patterns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spatial_clusters(self, save_path=None):
        """
        Plot clusters on a geographic map
        """
        if 'cluster' not in self.station_profiles.columns:
            print("Please run cluster_by_availability_patterns first")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Get unique clusters and create color map
        clusters = self.station_profiles['cluster'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for i, cluster_id in enumerate(clusters):
            cluster_data = self.station_profiles[self.station_profiles['cluster'] == cluster_id]
            
            plt.scatter(cluster_data['longitude_first'], 
                       cluster_data['latitude_first'],
                       c=[colors[i]], 
                       s=100, 
                       alpha=0.6, 
                       label=f'Cluster {cluster_id} (n={len(cluster_data)})',
                       edgecolors='black',
                       linewidth=0.5)
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('Geographic Distribution of Station Clusters', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_cluster_by_area(self):
        """
        Analyze how clusters distribute across different areas
        """
        if 'cluster' not in self.station_profiles.columns:
            print("Please run cluster_by_availability_patterns first")
            return None
        
        cluster_area_dist = pd.crosstab(
            self.station_profiles['sarea_first'], 
            self.station_profiles['cluster'],
            margins=True
        )
        
        return cluster_area_dist
    
    def find_nearby_stations(self, station_name, radius_km=1.0):
        """
        Find stations within a certain radius of a given station
        
        Parameters:
        - station_name: Name of the reference station
        - radius_km: Search radius in kilometers
        """
        if self.station_profiles is None:
            self.create_station_profiles()
        
        # Get reference station coordinates
        ref_station = self.station_profiles.loc[station_name]
        ref_lat = ref_station['latitude_first']
        ref_lon = ref_station['longitude_first']
        
        # Calculate distances using Haversine formula approximation
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        distances = []
        for idx, row in self.station_profiles.iterrows():
            dist = haversine_distance(ref_lat, ref_lon, 
                                     row['latitude_first'], 
                                     row['longitude_first'])
            distances.append({'station': idx, 'distance_km': dist})
        
        nearby_df = pd.DataFrame(distances).sort_values('distance_km')
        nearby_df = nearby_df[nearby_df['distance_km'] <= radius_km]
        
        return nearby_df
    
    def analyze_nearby_correlation(self, max_distance_km=1.0):
        """
        Analyze if nearby stations tend to run out of bikes together
        """
        if self.station_profiles is None:
            self.create_station_profiles()
        
        # Create distance matrix between all stations
        coords = self.station_profiles[['latitude_first', 'longitude_first']].values
        
        def haversine_vectorized(coords1, coords2):
            R = 6371
            lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
            lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        n_stations = len(coords)
        distance_matrix = np.zeros((n_stations, n_stations))
        
        for i in range(n_stations):
            for j in range(i+1, n_stations):
                dist = haversine_vectorized(coords[i:i+1], coords[j:j+1])[0]
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Calculate correlation matrix for occupancy rates
        hourly_data = self.station_profiles.iloc[:, :24].T
        correlation_matrix = hourly_data.corr()
        
        # Find pairs of nearby stations
        nearby_pairs = []
        far_pairs = []
        
        for i in range(n_stations):
            for j in range(i+1, n_stations):
                if distance_matrix[i, j] <= max_distance_km:
                    nearby_pairs.append(correlation_matrix.iloc[i, j])
                elif distance_matrix[i, j] > max_distance_km * 2:  # Compare with far stations
                    far_pairs.append(correlation_matrix.iloc[i, j])
        
        return {
            'nearby_correlations': np.array(nearby_pairs),
            'far_correlations': np.array(far_pairs),
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix
        }
    
    def plot_correlation_analysis(self, max_distance_km=1.0, save_path=None):
        """
        Visualize correlation between nearby vs distant stations
        """
        analysis = self.analyze_nearby_correlation(max_distance_km)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram comparison
        axes[0].hist(analysis['nearby_correlations'], bins=30, alpha=0.7, 
                    label=f'Nearby (<{max_distance_km}km)', color='green', density=True)
        axes[0].hist(analysis['far_correlations'], bins=30, alpha=0.7, 
                    label=f'Far (>{max_distance_km*2}km)', color='red', density=True)
        axes[0].set_xlabel('Correlation Coefficient')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Availability Correlation: Nearby vs Far Stations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [analysis['nearby_correlations'], analysis['far_correlations']]
        axes[1].boxplot(data_to_plot, labels=[f'Nearby\n(<{max_distance_km}km)', 
                                              f'Far\n(>{max_distance_km*2}km)'])
        axes[1].set_ylabel('Correlation Coefficient')
        axes[1].set_title('Distribution of Correlations')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        nearby_mean = np.mean(analysis['nearby_correlations'])
        far_mean = np.mean(analysis['far_correlations'])
        
        axes[1].text(0.5, 0.95, 
                    f'Nearby mean: {nearby_mean:.3f}\nFar mean: {far_mean:.3f}',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Do Nearby Stations Run Out Together?', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nCorrelation Analysis Results:")
        print(f"Nearby stations (<{max_distance_km}km) - Mean correlation: {nearby_mean:.3f}")
        print(f"Far stations (>{max_distance_km*2}km) - Mean correlation: {far_mean:.3f}")
        print(f"Difference: {nearby_mean - far_mean:.3f}")
    
    def pca_analysis(self, n_components=2, save_path=None):
        """
        Perform PCA to visualize station patterns in lower dimensions
        """
        if self.station_profiles is None:
            self.create_station_profiles()
        
        # Use hourly features
        hourly_features = self.station_profiles.iloc[:, :24].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(hourly_features)
        
        # PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(features_scaled)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        if 'cluster' in self.station_profiles.columns:
            # Color by cluster if available
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=self.station_profiles['cluster'], 
                                cmap='Set3', s=100, alpha=0.6, edgecolors='black')
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                       s=100, alpha=0.6, edgecolors='black')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA of Station Availability Patterns', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPCA Results:")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
        
        return pca, pca_result


if __name__ == "__main__":
    from data_collector import BikeDataCollector
    
    collector = BikeDataCollector()
    df = collector.load_all_data()
    
    if df is None or len(df) == 0:
        print("No data available. Please collect data first.")
    else:
        spatial = BikeSpatialAnalysis(df)
        
        print("Creating station profiles...")
        spatial.create_station_profiles()
        
        print("\nClustering stations by availability patterns...")
        spatial.cluster_by_availability_patterns(n_clusters=5)
        
        print("\nGenerating cluster visualizations...")
        spatial.plot_cluster_patterns()
        spatial.plot_spatial_clusters()
        
        print("\nAnalyzing nearby station correlations...")
        spatial.plot_correlation_analysis(max_distance_km=1.0)

