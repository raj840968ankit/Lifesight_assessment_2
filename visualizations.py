import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MarketingVisualizer:
    """Visualization tools for marketing mix modeling"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_time_series(self, df, target_col='revenue', media_cols=None):
        """Plot time series of revenue and media channels"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Revenue Over Time', 'Media Spend Over Time'],
            vertical_spacing=0.1
        )
        
        # Revenue plot
        fig.add_trace(
            go.Scatter(x=df.index, y=df[target_col], name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Media channels plot
        if media_cols:
            for col in media_cols[:5]:  # Limit to 5 channels for readability
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[col], name=col, opacity=0.7),
                        row=2, col=1
                    )
        
        fig.update_layout(height=600, title_text="Time Series Analysis")
        return fig
    
    def plot_correlation_heatmap(self, df, figsize=(12, 10)):
        """Plot correlation heatmap"""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot feature importance"""
        if importance_df is None:
            return None
        
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def plot_residuals(self, residual_analysis):
        """Plot residual analysis"""
        if residual_analysis is None:
            return None
        
        residuals = residual_analysis['residuals']
        predictions = residual_analysis['predictions']
        actual = residual_analysis['actual']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residuals vs Fitted', 'Q-Q Plot', 
                'Actual vs Predicted', 'Residuals Over Time'
            ]
        )
        
        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers', name='Residuals'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Q-Q Plot (simplified)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q'),
            row=1, col=2
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=actual, y=predictions, mode='markers', name='Predictions'),
            row=2, col=1
        )
        # Perfect prediction line
        min_val, max_val = min(actual.min(), predictions.min()), max(actual.max(), predictions.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Fit', line=dict(dash='dash')),
            row=2, col=1
        )
        
        # Residuals over time
        fig.add_trace(
            go.Scatter(y=residuals, mode='lines+markers', name='Residuals Time'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Residual Analysis")
        return fig
    
    def plot_elasticities(self, elasticities, top_n=10):
        """Plot elasticities"""
        if not elasticities:
            return None
        
        # Sort by absolute elasticity
        sorted_elasticities = sorted(elasticities.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        features = [item[0] for item in sorted_elasticities[:top_n]]
        values = [item[1] for item in sorted_elasticities[:top_n]]
        
        colors = ['red' if v < 0 else 'green' for v in values]
        
        fig = go.Figure(data=[
            go.Bar(x=values, y=features, orientation='h', 
                   marker_color=colors, name='Elasticity')
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Elasticities',
            xaxis_title='Elasticity (%)',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_saturation_curve(self, saturation_data):
        """Plot saturation curve for a media channel"""
        if not saturation_data:
            return None
        
        spend_levels = saturation_data['spend_levels']
        responses = saturation_data['responses']
        marginal_returns = saturation_data['marginal_returns']
        feature_name = saturation_data['feature_name']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Response Curve', 'Marginal Returns']
        )
        
        # Response curve
        fig.add_trace(
            go.Scatter(x=spend_levels, y=responses, mode='lines', name='Response'),
            row=1, col=1
        )
        
        # Marginal returns
        fig.add_trace(
            go.Scatter(x=spend_levels, y=marginal_returns, mode='lines', name='Marginal Returns'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f'Saturation Analysis: {feature_name}',
            height=400
        )
        
        return fig
    
    def plot_mediation_analysis(self, mediation_results):
        """Plot mediation analysis results"""
        if not mediation_results:
            return None
        
        treatments = list(mediation_results.keys())
        direct_effects = [mediation_results[t]['direct_effect'] for t in treatments]
        indirect_effects = [mediation_results[t]['indirect_effect'] for t in treatments]
        total_effects = [mediation_results[t]['total_effect'] for t in treatments]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Direct Effect',
            x=treatments,
            y=direct_effects,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Indirect Effect (Mediated)',
            x=treatments,
            y=indirect_effects,
            marker_color='orange'
        ))
        
        fig.add_trace(go.Scatter(
            name='Total Effect',
            x=treatments,
            y=total_effects,
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond')
        ))
        
        fig.update_layout(
            title='Mediation Analysis: Direct vs Indirect Effects',
            barmode='stack',
            xaxis_title='Treatment Variables',
            yaxis_title='Effect Size'
        )
        
        return fig
    
    def plot_attribution(self, attribution, top_n=10):
        """Plot marketing attribution"""
        if not attribution:
            return None
        
        # Sort by absolute attribution
        sorted_attr = sorted(attribution.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
        
        features = [item[0] for item in sorted_attr[:top_n]]
        values = [item[1] * 100 for item in sorted_attr[:top_n]]  # Convert to percentage
        
        fig = px.pie(
            values=values,
            names=features,
            title=f'Revenue Attribution (Top {top_n})'
        )
        
        return fig
    
    def plot_model_comparison(self, model_results):
        """Plot model comparison metrics"""
        models = list(model_results.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[m.upper() for m in metrics]
        )
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric.upper()),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=400
        )
        
        return fig
    
    def create_executive_summary_plot(self, elasticities, attribution, mediation_results):
        """Create executive summary visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Top Elasticities', 'Revenue Attribution',
                'Mediation Effects', 'Key Metrics'
            ],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Top elasticities
        if elasticities:
            sorted_elast = sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            features = [item[0] for item in sorted_elast]
            values = [item[1] for item in sorted_elast]
            
            fig.add_trace(
                go.Bar(x=features, y=values, name='Elasticity'),
                row=1, col=1
            )
        
        # Attribution pie chart
        if attribution:
            sorted_attr = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            attr_features = [item[0] for item in sorted_attr]
            attr_values = [abs(item[1]) for item in sorted_attr]
            
            fig.add_trace(
                go.Pie(labels=attr_features, values=attr_values, name="Attribution"),
                row=1, col=2
            )
        
        # Mediation effects
        if mediation_results:
            treatments = list(mediation_results.keys())
            prop_mediated = [mediation_results[t]['proportion_mediated'] for t in treatments]
            
            fig.add_trace(
                go.Bar(x=treatments, y=prop_mediated, name='Proportion Mediated'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Executive Summary Dashboard")
        return fig

# Additional utility functions
from scipy import stats

def plot_confidence_intervals(ci_df, top_n=15):
    """Plot confidence intervals for coefficients"""
    if ci_df is None:
        return None
    
    # Sort by absolute coefficient value
    ci_df_sorted = ci_df.reindex(ci_df['coefficient'].abs().sort_values(ascending=False).index)
    top_features = ci_df_sorted.head(top_n)
    
    fig = go.Figure()
    
    # Add coefficient points
    fig.add_trace(go.Scatter(
        x=top_features['coefficient'],
        y=top_features['feature'],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Coefficient'
    ))
    
    # Add confidence intervals
    for i, row in top_features.iterrows():
        fig.add_shape(
            type="line",
            x0=row['ci_lower'], x1=row['ci_upper'],
            y0=row['feature'], y1=row['feature'],
            line=dict(color="gray", width=2)
        )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Coefficient Confidence Intervals",
        xaxis_title="Coefficient Value",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig