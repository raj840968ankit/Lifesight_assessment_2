import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class MarketingDataProcessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
        self.target_scaler = RobustScaler()
        
    def load_and_clean_data(self, data):
        """Load and clean the marketing data"""
        df = data.copy()
        
        # Handle date column - more flexible approach
        date_col = None
        possible_date_cols = ['date', 'Date', 'DATE', 'week', 'Week', 'period', 'time']
        
        # Check if any date column exists
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        # If no date column found, create one or use index
        if date_col is None:
            if df.index.name in possible_date_cols or pd.api.types.is_datetime64_any_dtype(df.index):
                # Use existing datetime index
                df = df.reset_index()
                date_col = df.columns[0]
            else:
                # Create a date column
                print("No date column found. Creating weekly dates starting from 2022-01-01")
                df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='W')
                date_col = 'date'
        
        # Convert to datetime if not already
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # If conversion fails, create sequential dates
            print(f"Could not convert {date_col} to datetime. Creating sequential weekly dates.")
            df[date_col] = pd.date_range(start='2022-01-01', periods=len(df), freq='W')
        
        # Set as index
        df = df.set_index(date_col)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure all numeric columns are float
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Remove any non-numeric columns except the target
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Removing non-numeric columns: {list(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)
        
        return df
    
    def create_adstock_features(self, df, media_cols, adstock_rates=[0.3, 0.5, 0.7]):
        """Create adstock (carryover) features for media channels"""
        adstock_df = df.copy()
        
        for col in media_cols:
            if col in df.columns:
                for rate in adstock_rates:
                    adstock_col = f"{col}_adstock_{int(rate*100)}"
                    adstock_values = []
                    adstock_val = 0
                    
                    for val in df[col]:
                        adstock_val = val + rate * adstock_val
                        adstock_values.append(adstock_val)
                    
                    adstock_df[adstock_col] = adstock_values
        
        return adstock_df
    
    def create_saturation_features(self, df, media_cols, saturation_params=[0.5, 1.0, 2.0]):
        """Create saturation curves for media channels"""
        saturation_df = df.copy()
        
        for col in media_cols:
            if col in df.columns:
                for param in saturation_params:
                    sat_col = f"{col}_sat_{int(param*100)}"
                    # Hill saturation: x^param / (x^param + 1)
                    x_norm = df[col] / (df[col].max() + 1e-8)
                    saturation_df[sat_col] = (x_norm ** param) / (x_norm ** param + 1)
        
        return saturation_df
    
    def create_time_features(self, df):
        """Create time-based features for seasonality"""
        time_df = df.copy()
        
        # Week of year
        time_df['week_of_year'] = df.index.isocalendar().week
        time_df['week_sin'] = np.sin(2 * np.pi * time_df['week_of_year'] / 52)
        time_df['week_cos'] = np.cos(2 * np.pi * time_df['week_of_year'] / 52)
        
        # Month
        time_df['month'] = df.index.month
        time_df['month_sin'] = np.sin(2 * np.pi * time_df['month'] / 12)
        time_df['month_cos'] = np.cos(2 * np.pi * time_df['month'] / 12)
        
        # Trend
        time_df['trend'] = np.arange(len(df))
        time_df['trend_sq'] = time_df['trend'] ** 2
        
        # Drop intermediate columns
        time_df = time_df.drop(['week_of_year', 'month'], axis=1)
        
        return time_df
    
    def create_lag_features(self, df, cols, lags=[1, 2, 4]):
        """Create lagged features"""
        lag_df = df.copy()
        
        for col in cols:
            if col in df.columns:
                for lag in lags:
                    lag_col = f"{col}_lag_{lag}"
                    lag_df[lag_col] = df[col].shift(lag)
        
        return lag_df
    
    def handle_zero_spend_periods(self, df, media_cols):
        """Handle zero spend periods with indicators"""
        zero_df = df.copy()
        
        for col in media_cols:
            if col in df.columns:
                zero_col = f"{col}_zero_flag"
                zero_df[zero_col] = (df[col] == 0).astype(int)
        
        return zero_df
    
    def prepare_features(self, df, target_col='revenue'):
        """Main feature engineering pipeline"""
        # Identify target column - more flexible
        target_candidates = ['revenue', 'Revenue', 'REVENUE', 'sales', 'Sales', 'SALES']
        actual_target = None
        
        for candidate in target_candidates:
            if candidate in df.columns:
                actual_target = candidate
                break
        
        if actual_target is None:
            # If no standard target found, use the last column or ask user
            print("No standard target column found. Available columns:", list(df.columns))
            if len(df.columns) > 0:
                actual_target = df.columns[-1]  # Use last column as target
                print(f"Using '{actual_target}' as target variable")
            else:
                raise ValueError("No columns found in dataset")
        else:
            target_col = actual_target
        
        # Identify media columns (assume they contain 'spend' or common media terms)
        media_terms = ['google', 'facebook', 'tiktok', 'snapchat', 'spend', 'impressions', 'clicks', 'cost']
        media_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in media_terms) and col != target_col:
                media_cols.append(col)
        
        print(f"Target column: {target_col}")
        print(f"Identified media columns: {media_cols}")
        
        # Apply transformations
        df_processed = self.create_time_features(df)
        
        if media_cols:  # Only create media features if media columns exist
            df_processed = self.create_adstock_features(df_processed, media_cols)
            df_processed = self.create_saturation_features(df_processed, media_cols)
            df_processed = self.create_lag_features(df_processed, media_cols + [target_col])
            df_processed = self.handle_zero_spend_periods(df_processed, media_cols)
        else:
            # Create basic lag features for all numeric columns
            numeric_cols = [col for col in df.columns if col != target_col]
            df_processed = self.create_lag_features(df_processed, numeric_cols[:5])  # Limit to first 5 columns
        
        # Drop rows with NaN (from lagging)
        df_processed = df_processed.dropna()
        
        if len(df_processed) == 0:
            raise ValueError("No data remaining after preprocessing. Check your dataset.")
        
        return df_processed, media_cols
    
    def split_features_target(self, df, target_col='revenue'):
        """Split features and target"""
        # Find actual target column
        target_candidates = ['revenue', 'Revenue', 'REVENUE', 'sales', 'Sales', 'SALES']
        actual_target = None
        
        for candidate in target_candidates:
            if candidate in df.columns:
                actual_target = candidate
                break
        
        if actual_target is None:
            actual_target = df.columns[-1]  # Use last column
            print(f"Using '{actual_target}' as target variable")
        
        y = df[actual_target].values
        X = df.drop(columns=[actual_target])
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, y_train=None, y_test=None):
        """Scale features and target"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if y_train is not None:
            y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        else:
            y_train_scaled = None
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            if y_test is not None:
                y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
            else:
                y_test_scaled = None
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        
        return X_train_scaled, y_train_scaled
    
    def get_time_series_splits(self, X, n_splits=5):
        """Get time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))