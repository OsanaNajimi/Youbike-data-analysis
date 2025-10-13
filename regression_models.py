"""
Regression Models Module
Time-based regression models for predicting bike availability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

class BikeRegressionModels:
    def __init__(self, df):
        """
        Initialize regression models with bike data
        
        Parameters:
        - df: DataFrame with bike station data
        """
        self.df = df.copy()
        self.models = {}
        self.results = {}
    
    def prepare_features(self, station_name=None):
        """
        Prepare features for regression modeling
        
        Parameters:
        - station_name: If specified, prepare features for a specific station
        """
        if station_name:
            data = self.df[self.df['sna'] == station_name].copy()
        else:
            data = self.df.copy()
        
        # Sort by timestamp
        data = data.sort_values('timestamp').copy()
        
        # Create time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['minute'] = data['timestamp'].dt.minute
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_morning_rush'] = data['hour'].isin([7, 8, 9]).astype(int)
        data['is_evening_rush'] = data['hour'].isin([17, 18, 19]).astype(int)
        
        # Cyclical encoding for hour (to capture circular nature of time)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Cyclical encoding for day of week
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Time since start (for trend analysis)
        data['time_index'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600
        
        return data
    
    def train_time_regression(self, station_name, target='available_rent_bikes'):
        """
        Train regression models to predict bike availability based on time features
        
        Parameters:
        - station_name: Station to model
        - target: Target variable ('available_rent_bikes' or 'occupancy_rate')
        """
        data = self.prepare_features(station_name)
        
        if len(data) < 10:
            print(f"Insufficient data for station: {station_name}")
            return None
        
        # Define features
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_morning_rush', 
                       'is_evening_rush', 'hour_sin', 'hour_cos', 'day_sin', 
                       'day_cos', 'time_index']
        
        X = data[feature_cols].values
        y = data[target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            results[name] = {
                'model': model,
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'y_test': y_test,
                'y_pred': y_pred_test,
                'X_test': X_test
            }
        
        self.models[station_name] = models
        self.results[station_name] = results
        
        return results
    
    def plot_model_comparison(self, station_name, save_path=None):
        """
        Plot comparison of different regression models for a station
        """
        if station_name not in self.results:
            print(f"No results found for station: {station_name}")
            return
        
        results = self.results[station_name]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            # Scatter plot: actual vs predicted
            ax.scatter(result['y_test'], result['y_pred'], alpha=0.5)
            
            # Perfect prediction line
            min_val = min(result['y_test'].min(), result['y_pred'].min())
            max_val = max(result['y_test'].max(), result['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{name}\nR² = {result["test_r2"]:.3f}, RMSE = {result["test_rmse"]:.2f}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Regression Model Comparison - {station_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_over_time(self, station_name, model_name='Random Forest', save_path=None):
        """
        Plot predictions vs actual values over time
        """
        if station_name not in self.results:
            print(f"No results found for station: {station_name}")
            return
        
        result = self.results[station_name][model_name]
        
        plt.figure(figsize=(16, 6))
        
        # Extract hour from test data for x-axis
        test_indices = range(len(result['y_test']))
        
        plt.plot(test_indices, result['y_test'], 'o-', label='Actual', alpha=0.7, markersize=6)
        plt.plot(test_indices, result['y_pred'], 's-', label='Predicted', alpha=0.7, markersize=6)
        
        plt.title(f'Predictions vs Actual - {station_name} ({model_name})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Available Bikes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, station_name, save_path=None):
        """
        Analyze and visualize feature importance using Random Forest
        """
        if station_name not in self.results:
            print(f"No results found for station: {station_name}")
            return
        
        rf_model = self.results[station_name]['Random Forest']['model']
        
        feature_names = ['hour', 'day_of_week', 'is_weekend', 'is_morning_rush', 
                        'is_evening_rush', 'hour_sin', 'hour_cos', 'day_sin', 
                        'day_cos', 'time_index']
        
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices], alpha=0.7)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Feature Importance - {station_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print importance scores
        print(f"\nFeature Importances for {station_name}:")
        for idx in indices:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    def get_model_summary(self, station_name):
        """
        Get summary of all models for a station
        """
        if station_name not in self.results:
            return None
        
        results = self.results[station_name]
        
        summary = []
        for name, result in results.items():
            summary.append({
                'Model': name,
                'Train R²': f"{result['train_r2']:.4f}",
                'Test R²': f"{result['test_r2']:.4f}",
                'Train RMSE': f"{result['train_rmse']:.2f}",
                'Test RMSE': f"{result['test_rmse']:.2f}",
                'Train MAE': f"{result['train_mae']:.2f}",
                'Test MAE': f"{result['test_mae']:.2f}"
            })
        
        return pd.DataFrame(summary)
    
    def predict_future_hours(self, station_name, hours_ahead=24, model_name='Random Forest'):
        """
        Predict bike availability for future hours
        """
        if station_name not in self.models:
            print(f"No trained model found for station: {station_name}")
            return None
        
        model = self.models[station_name][model_name]
        
        # Get latest timestamp
        station_data = self.df[self.df['sna'] == station_name]
        last_time = station_data['timestamp'].max()
        
        # Create future time points
        future_times = pd.date_range(start=last_time, periods=hours_ahead+1, freq='H')[1:]
        
        # Prepare features
        future_df = pd.DataFrame({
            'timestamp': future_times
        })
        
        future_df['hour'] = future_df['timestamp'].dt.hour
        future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['is_morning_rush'] = future_df['hour'].isin([7, 8, 9]).astype(int)
        future_df['is_evening_rush'] = future_df['hour'].isin([17, 18, 19]).astype(int)
        future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
        future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        
        # Calculate time index
        first_time = station_data['timestamp'].min()
        future_df['time_index'] = (future_df['timestamp'] - first_time).dt.total_seconds() / 3600
        
        feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_morning_rush', 
                       'is_evening_rush', 'hour_sin', 'hour_cos', 'day_sin', 
                       'day_cos', 'time_index']
        
        X_future = future_df[feature_cols].values
        predictions = model.predict(X_future)
        
        future_df['predicted_bikes'] = predictions
        
        return future_df[['timestamp', 'hour', 'predicted_bikes']]


if __name__ == "__main__":
    from data_collector import BikeDataCollector
    
    collector = BikeDataCollector()
    df = collector.load_all_data()
    
    if df is None or len(df) == 0:
        print("No data available. Please collect data first.")
    else:
        regressor = BikeRegressionModels(df)
        
        # Get a popular station
        popular_station = df.groupby('sna')['Quantity'].mean().nlargest(1).index[0]
        
        print(f"Training regression models for: {popular_station}")
        results = regressor.train_time_regression(popular_station)
        
        if results:
            print("\nModel Performance Summary:")
            print(regressor.get_model_summary(popular_station))
            
            print("\nGenerating visualizations...")
            regressor.plot_model_comparison(popular_station)
            regressor.analyze_feature_importance(popular_station)

