import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class MarketingModelBuilder:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.cv_results = {}
        
    def define_models(self):
        """Define candidate models with hyperparameters"""
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [5, 10]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        return models
    
    def train_models(self, X_train, y_train, cv_splits=None):
        """Train all models with time-series cross-validation"""
        models = self.define_models()
        
        if cv_splits is None:
            tscv = TimeSeriesSplit(n_splits=3)
            cv_splits = tscv
        
        best_score = -np.inf
        
        for name, model_config in models.items():
            print(f"Training {name}...")
            
            # Grid search with time series CV
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.models[name] = grid_search.best_estimator_
            self.cv_results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_results': grid_search.cv_results_
            }
            
            # Track best model
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        
        print(f"Best model: {self.best_model_name} with CV score: {best_score:.4f}")
        
        # Calculate feature importance for best model
        self.calculate_feature_importance(X_train)
        
        return self.models
    
    def calculate_feature_importance(self, X_train):
        """Calculate feature importance for the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            self.feature_importance = np.abs(self.best_model.coef_)
        else:
            self.feature_importance = None
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'predictions': y_pred
            }
        
        return results
    
    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        return self.best_model.predict(X)
    
    def get_feature_importance_df(self, feature_names):
        """Get feature importance as DataFrame"""
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_residuals(self, X_test, y_test):
        """Analyze model residuals"""
        if self.best_model is None:
            return None
        
        y_pred = self.best_model.predict(X_test)
        residuals = y_test - y_pred
        
        analysis = {
            'residuals': residuals,
            'predictions': y_pred,
            'actual': y_test,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'residual_autocorr': np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        }
        
        return analysis

class MediationModelBuilder:
    """Two-stage model for handling Google as mediator"""
    
    def __init__(self):
        self.stage1_model = None  # Predict Google spend
        self.stage2_model = None  # Predict revenue
        self.direct_model = None  # Direct effects model
        
    def fit_mediation_model(self, X, y, google_col_idx, feature_names):
        """
        Fit two-stage mediation model
        Stage 1: Social media -> Google spend
        Stage 2: Google spend + other variables -> Revenue
        """
        # Identify social media columns
        social_cols = []
        for i, name in enumerate(feature_names):
            if any(term in name.lower() for term in ['facebook', 'tiktok', 'snapchat', 'social']):
                social_cols.append(i)
        
        # Stage 1: Predict Google spend from social media
        if social_cols:
            X_social = X[:, social_cols]
            y_google = X[:, google_col_idx]
            
            self.stage1_model = Ridge(alpha=1.0)
            self.stage1_model.fit(X_social, y_google)
            
            # Get predicted Google spend
            google_predicted = self.stage1_model.predict(X_social)
        else:
            google_predicted = X[:, google_col_idx]
        
        # Stage 2: Predict revenue using predicted Google + other variables
        X_stage2 = X.copy()
        X_stage2[:, google_col_idx] = google_predicted  # Replace actual with predicted Google
        
        self.stage2_model = Ridge(alpha=1.0)
        self.stage2_model.fit(X_stage2, y)
        
        # Direct effects model (without mediation)
        X_direct = np.delete(X, google_col_idx, axis=1)
        self.direct_model = Ridge(alpha=1.0)
        self.direct_model.fit(X_direct, y)
        
        return self
    
    def predict_mediation(self, X, google_col_idx):
        """Make predictions with mediation model"""
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Mediation model not fitted")
        
        # Stage 1 prediction
        social_cols = [i for i in range(X.shape[1]) if i != google_col_idx]
        if len(social_cols) > 0:
            X_social = X[:, social_cols[:len(self.stage1_model.coef_)]]  # Match training features
            google_predicted = self.stage1_model.predict(X_social)
        else:
            google_predicted = X[:, google_col_idx]
        
        # Stage 2 prediction
        X_stage2 = X.copy()
        X_stage2[:, google_col_idx] = google_predicted
        
        return self.stage2_model.predict(X_stage2)
    
    def calculate_mediation_effects(self, X, y, google_col_idx, feature_names):
        """Calculate direct and indirect effects"""
        # Total effect (without controlling for Google)
        X_no_google = np.delete(X, google_col_idx, axis=1)
        total_model = Ridge(alpha=1.0)
        total_model.fit(X_no_google, y)
        
        # Direct effect (controlling for Google)
        direct_pred = self.direct_model.predict(X_no_google)
        
        # Indirect effect (through Google)
        mediated_pred = self.predict_mediation(X, google_col_idx)
        
        effects = {
            'total_effect': total_model.coef_,
            'direct_effect': self.direct_model.coef_,
            'indirect_effect_magnitude': np.mean(np.abs(mediated_pred - direct_pred)),
            'mediation_ratio': np.mean(np.abs(mediated_pred - direct_pred)) / (np.std(y) + 1e-8)
        }
        
        return effects