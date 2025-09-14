import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

class CausalAnalyzer:
    """Causal analysis for marketing mix modeling"""
    
    def __init__(self):
        self.mediation_results = {}
        self.elasticities = {}
        self.attribution = {}
        
    def calculate_elasticities(self, model, X, y, feature_names, baseline_percentile=50):
        """Calculate price and media elasticities"""
        elasticities = {}
        
        # Get baseline predictions
        baseline_pred = model.predict(X)
        baseline_mean = np.mean(baseline_pred)
        
        for i, feature in enumerate(feature_names):
            # Calculate elasticity by perturbing feature
            X_perturbed = X.copy()
            
            # Get baseline value (median)
            baseline_value = np.percentile(X[:, i], baseline_percentile)
            
            if baseline_value > 0:
                # Increase by 1%
                X_perturbed[:, i] = X[:, i] * 1.01
                perturbed_pred = model.predict(X_perturbed)
                perturbed_mean = np.mean(perturbed_pred)
                
                # Elasticity = % change in revenue / % change in feature
                elasticity = ((perturbed_mean - baseline_mean) / baseline_mean) / 0.01
                elasticities[feature] = elasticity
            else:
                elasticities[feature] = 0
        
        self.elasticities = elasticities
        return elasticities
    
    def perform_mediation_analysis(self, X, y, feature_names, mediator_col='google'):
        """Perform formal mediation analysis"""
        # Find mediator column index
        mediator_idx = None
        for i, name in enumerate(feature_names):
            if mediator_col.lower() in name.lower():
                mediator_idx = i
                break
        
        if mediator_idx is None:
            print(f"Mediator column '{mediator_col}' not found")
            return None
        
        # Find treatment columns (social media)
        treatment_indices = []
        for i, name in enumerate(feature_names):
            if any(term in name.lower() for term in ['facebook', 'tiktok', 'snapchat', 'social']):
                treatment_indices.append(i)
        
        mediation_results = {}
        
        for treatment_idx in treatment_indices:
            treatment_name = feature_names[treatment_idx]
            mediator_name = feature_names[mediator_idx]
            
            # Step 1: Treatment -> Outcome (total effect)
            model_total = Ridge(alpha=1.0)
            X_total = X[:, [treatment_idx]]
            model_total.fit(X_total, y)
            total_effect = model_total.coef_[0]
            
            # Step 2: Treatment -> Mediator
            model_tm = Ridge(alpha=1.0)
            model_tm.fit(X_total, X[:, mediator_idx])
            a_path = model_tm.coef_[0]
            
            # Step 3: Treatment + Mediator -> Outcome (direct effect)
            model_direct = Ridge(alpha=1.0)
            X_direct = X[:, [treatment_idx, mediator_idx]]
            model_direct.fit(X_direct, y)
            direct_effect = model_direct.coef_[0]
            b_path = model_direct.coef_[1]
            
            # Calculate indirect effect
            indirect_effect = a_path * b_path
            
            # Proportion mediated
            if abs(total_effect) > 1e-8:
                prop_mediated = indirect_effect / total_effect
            else:
                prop_mediated = 0
            
            mediation_results[treatment_name] = {
                'total_effect': total_effect,
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'a_path': a_path,
                'b_path': b_path,
                'proportion_mediated': prop_mediated,
                'mediator': mediator_name
            }
        
        self.mediation_results = mediation_results
        return mediation_results
    
    def calculate_attribution(self, model, X, y, feature_names):
        """Calculate marketing attribution using Shapley-like approach"""
        baseline_pred = np.mean(model.predict(np.zeros((1, X.shape[1]))))
        full_pred = np.mean(model.predict(X))
        
        total_contribution = full_pred - baseline_pred
        
        attributions = {}
        
        for i, feature in enumerate(feature_names):
            # Create version with only this feature active
            X_single = np.zeros_like(X)
            X_single[:, i] = X[:, i]
            single_pred = np.mean(model.predict(X_single))
            
            # Attribution as contribution above baseline
            attribution = (single_pred - baseline_pred) / (total_contribution + 1e-8)
            attributions[feature] = attribution
        
        # Normalize to sum to 1
        total_attr = sum(abs(v) for v in attributions.values())
        if total_attr > 0:
            attributions = {k: v/total_attr for k, v in attributions.items()}
        
        self.attribution = attributions
        return attributions
    
    def analyze_saturation_curves(self, model, X, feature_names, feature_idx):
        """Analyze saturation curves for media channels"""
        feature_name = feature_names[feature_idx]
        
        # Get range of feature values
        feature_values = X[:, feature_idx]
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        
        # Create test points
        test_points = np.linspace(0, max_val * 1.5, 50)
        
        # Create test matrix
        X_test = np.median(X, axis=0).reshape(1, -1)
        X_test = np.repeat(X_test, len(test_points), axis=0)
        
        responses = []
        for i, point in enumerate(test_points):
            X_test[i, feature_idx] = point
            pred = model.predict(X_test[i:i+1])
            responses.append(pred[0])
        
        # Calculate marginal returns
        marginal_returns = np.diff(responses) / np.diff(test_points)
        
        return {
            'spend_levels': test_points,
            'responses': responses,
            'marginal_returns': np.concatenate([[marginal_returns[0]], marginal_returns]),
            'feature_name': feature_name
        }
    
    def detect_multicollinearity(self, X, feature_names, threshold=0.8):
        """Detect multicollinearity issues"""
        corr_matrix = np.corrcoef(X.T)
        
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr_matrix[i, j]
                    })
        
        return high_corr_pairs
    
    def calculate_confidence_intervals(self, model, X, y, feature_names, alpha=0.05):
        """Calculate confidence intervals for coefficients (for linear models)"""
        if not hasattr(model, 'coef_'):
            return None
        
        # Calculate residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # Calculate standard errors (simplified)
        n, p = X.shape
        
        try:
            # For Ridge regression, approximate standard errors
            XtX_inv = np.linalg.inv(X.T @ X + model.alpha * np.eye(p))
            var_coef = mse * np.diag(XtX_inv)
            se_coef = np.sqrt(var_coef)
            
            # t-statistic
            t_stat = stats.t.ppf(1 - alpha/2, n - p)
            
            ci_lower = model.coef_ - t_stat * se_coef
            ci_upper = model.coef_ + t_stat * se_coef
            
            ci_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_,
                'std_error': se_coef,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
            
            return ci_df
            
        except np.linalg.LinAlgError:
            print("Could not calculate confidence intervals due to singular matrix")
            return None
    
    def generate_insights(self):
        """Generate actionable insights from analysis"""
        insights = []
        
        # Elasticity insights
        if self.elasticities:
            sorted_elasticities = sorted(self.elasticities.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
            
            insights.append("## Elasticity Analysis")
            insights.append("Top drivers by elasticity:")
            for feature, elasticity in sorted_elasticities[:5]:
                if abs(elasticity) > 0.01:
                    direction = "increases" if elasticity > 0 else "decreases"
                    insights.append(f"- {feature}: 1% increase {direction} revenue by {abs(elasticity):.2%}")
        
        # Mediation insights
        if self.mediation_results:
            insights.append("\n## Mediation Analysis")
            for treatment, results in self.mediation_results.items():
                prop_med = results['proportion_mediated']
                if abs(prop_med) > 0.1:
                    insights.append(f"- {treatment}: {prop_med:.1%} of effect is mediated through {results['mediator']} this means {treatment} ads are also causing users to search on Google,which then leads to a sale")
        
        # Attribution insights
        if self.attribution:
            insights.append("\n## Attribution Analysis")
            sorted_attr = sorted(self.attribution.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
            insights.append("Revenue attribution by channel:")
            for feature, attr in sorted_attr[:5]:
                if abs(attr) > 0.05:
                    insights.append(f"- {feature}: {attr:.1%} of incremental revenue")
        
        return "\n".join(insights)