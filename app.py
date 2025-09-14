import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processor import MarketingDataProcessor
from model_builder import MarketingModelBuilder, MediationModelBuilder
from causal_analysis import CausalAnalyzer
from visualizations import MarketingVisualizer

# Page configuration
st.set_page_config(
    page_title="Marketing Mix Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Create sample marketing data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='W')
    n_weeks = len(dates)
    
    # Create realistic marketing data
    data = {
        'date': dates,
        'google_spend': np.random.lognormal(8, 0.5, n_weeks) * 1000,
        'facebook_spend': np.random.lognormal(7.5, 0.6, n_weeks) * 1000,
        'tiktok_spend': np.random.lognormal(7, 0.7, n_weeks) * 1000,
        'snapchat_spend': np.random.lognormal(6.5, 0.8, n_weeks) * 1000,
        'email_send': np.random.poisson(50000, n_weeks),
        'sms_send': np.random.poisson(10000, n_weeks),
        'avg_price': 50 + np.random.normal(0, 5, n_weeks),
        'followers': 100000 + np.cumsum(np.random.normal(1000, 500, n_weeks)),
        'promotions': np.random.binomial(1, 0.3, n_weeks),
    }
    
    # Create revenue with realistic relationships
    revenue = (
        0.3 * np.log1p(data['google_spend']) * 1000 +
        0.2 * np.log1p(data['facebook_spend']) * 800 +
        0.15 * np.log1p(data['tiktok_spend']) * 600 +
        0.1 * np.log1p(data['snapchat_spend']) * 400 +
        0.05 * data['email_send'] +
        0.1 * data['sms_send'] +
        -500 * (data['avg_price'] - 50) +
        0.01 * data['followers'] +
        5000 * data['promotions'] +
        np.random.normal(0, 5000, n_weeks) +
        50000  # Base revenue
    )
    
    data['revenue'] = np.maximum(revenue, 0)  # Ensure non-negative revenue
    
    return pd.DataFrame(data)

def main():
    st.markdown('<h1 class="main-header">📊 Marketing Mix Modeling Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard implements a comprehensive Marketing Mix Model (MMM) with causal analysis, 
    specifically handling Google spend as a mediator between social media channels and revenue.
    """)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Data upload section
    st.sidebar.markdown("### Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your marketing data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with weekly marketing data including spend, revenue, and other variables"
    )
    
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    # Load data with better error handling
    df = None
    error_message = None
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data uploaded successfully!")
            
            # Show data info
            st.sidebar.info(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
            
            # Display column names for debugging
            with st.sidebar.expander("📋 Column Names"):
                st.write("Available columns:")
                for i, col in enumerate(df.columns):
                    st.write(f"{i+1}. {col}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
            error_message = str(e)
            df = None
            
    elif use_sample_data:
        df = load_sample_data()
        st.sidebar.info("📊 Using sample data for demonstration")
    else:
        st.sidebar.warning("⚠️ Please upload data or use sample data")
    
    if df is not None:
        try:
            # Initialize processor with error handling
            processor = MarketingDataProcessor()
            visualizer = MarketingVisualizer()
            
            # Show raw data info first
            st.markdown('<h2 class="section-header">📈 Raw Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                # Try to find a revenue-like column
                revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
                if revenue_cols:
                    st.metric("Revenue Column", revenue_cols[0])
                else:
                    st.metric("Potential Target", df.columns[-1])
            with col4:
                # Check for date columns
                date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'week', 'period'])]
                if date_cols:
                    st.metric("Date Column", date_cols[0])
                else:
                    st.metric("Date Column", "None found")
            
            # Show data preview
            with st.expander("📋 Raw Data Preview"):
                st.dataframe(df.head(10))
                
                # Show data types
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtypes_df)
            
            # Process data with detailed error handling
            st.markdown('<h2 class="section-header">🔄 Data Processing</h2>', unsafe_allow_html=True)
            
            with st.spinner("🔄 Processing data..."):
                try:
                    # Clean data
                    df_clean = processor.load_and_clean_data(df)
                    st.success("✅ Data cleaning completed!")
                    
                    # Show processed data info
                    st.info(f"📊 After cleaning: {len(df_clean)} rows, {len(df_clean.columns)} columns")
                    
                    # Feature engineering
                    df_processed, media_cols = processor.prepare_features(df_clean)
                    st.success(f"✅ Feature engineering completed! Created {len(df_processed.columns)} features.")
                    
                    # Show identified columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📺 Media Columns Found")
                        if media_cols:
                            for col in media_cols:
                                st.write(f"• {col}")
                        else:
                            st.write("No media columns identified")
                    
                    with col2:
                        st.subheader("🎯 Target Variable")
                        # Find target column
                        target_candidates = ['revenue', 'Revenue', 'REVENUE', 'sales', 'Sales', 'SALES']
                        target_col = None
                        for candidate in target_candidates:
                            if candidate in df_processed.columns:
                                target_col = candidate
                                break
                        if target_col is None:
                            target_col = df_processed.columns[-1]
                        st.write(f"• {target_col}")
                    
                    # Split features and target
                    X, y = processor.split_features_target(df_processed, target_col)
                    st.success(f"✅ Data split completed! {X.shape[0]} samples, {X.shape[1]} features")
                    
                except Exception as e:
                    st.error(f"❌ Error during data processing: {str(e)}")
                    st.error("Please check your data format and column names.")
                    
                    # Show detailed error info
                    with st.expander("🔍 Debug Information"):
                        st.write("**Error Details:**")
                        st.code(str(e))
                        st.write("**Available Columns:**")
                        st.write(list(df.columns))
                        st.write("**Data Types:**")
                        st.write(df.dtypes)
                    return
            
            # Visualization section
            st.markdown('<h2 class="section-header">📊 Data Visualization</h2>', unsafe_allow_html=True)
            
            # Time series plot
            try:
                fig_ts = visualizer.plot_time_series(df_clean, target_col=target_col, media_cols=media_cols)
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create time series plot: {str(e)}")
            
            # Correlation analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔗 Correlation Matrix")
                try:
                    # Select key variables for correlation
                    key_vars = [target_col] + media_cols + ['avg_price', 'promotions'] if 'avg_price' in df_clean.columns else [target_col] + media_cols
                    key_vars = [col for col in key_vars if col in df_clean.columns][:10]  # Limit to 10 for readability
                    
                    if len(key_vars) > 1:
                        corr_fig = px.imshow(
                            df_clean[key_vars].corr(),
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlations"
                        )
                        st.plotly_chart(corr_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create correlation matrix: {str(e)}")
            
            with col2:
                st.subheader("📈 Target Distribution")
                try:
                    hist_fig = px.histogram(
                        df_clean, 
                        x=target_col,
                        title=f"{target_col} Distribution",
                        nbins=30
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create distribution plot: {str(e)}")
            
            # Model training section
            st.markdown('<h2 class="section-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
            
            # Model configuration
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            if st.button("🚀 Train Models"):
                with st.spinner("🔄 Training models..."):
                    try:
                        # Train-test split (time-aware)
                        split_idx = int(len(X) * (1 - test_size))
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Scale features
                        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = processor.scale_features(
                            X_train, X_test, y_train, y_test
                        )
                        
                        # Get CV splits
                        cv_splits = processor.get_time_series_splits(X_train_scaled, cv_folds)
                        
                        # Train models
                        model_builder = MarketingModelBuilder()
                        models = model_builder.train_models(X_train_scaled, y_train_scaled, cv_splits)
                        
                        # Evaluate models
                        results = model_builder.evaluate_models(X_test_scaled, y_test_scaled)
                        
                        # Store in session state
                        st.session_state['models'] = models
                        st.session_state['model_builder'] = model_builder
                        st.session_state['results'] = results
                        st.session_state['processor'] = processor
                        st.session_state['X_test'] = X_test_scaled
                        st.session_state['y_test'] = y_test_scaled
                        st.session_state['feature_names'] = processor.feature_names
                        st.session_state['target_col'] = target_col
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        st.subheader("📊 Model Performance")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            model: {
                                'RMSE': results[model]['rmse'],
                                'MAE': results[model]['mae'],
                                'R²': results[model]['r2']
                            }
                            for model in results.keys()
                        }).T
                        
                        st.dataframe(results_df.round(4))
                        
                        # Best model info
                        best_model_name = model_builder.best_model_name
                        st.info(f"🏆 Best Model: {best_model_name}")
                        
                        # Feature importance
                        importance_df = model_builder.get_feature_importance_df(processor.feature_names)
                        if importance_df is not None:
                            fig_importance = visualizer.plot_feature_importance(importance_df)
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"❌ Error during model training: {str(e)}")
                        with st.expander("🔍 Debug Information"):
                            st.code(str(e))
            
            # Causal Analysis Section
            if 'model_builder' in st.session_state:
                st.markdown('<h2 class="section-header">🔍 Causal Analysis</h2>', unsafe_allow_html=True)
                
                if st.button("🧠 Perform Causal Analysis"):
                    with st.spinner("🔄 Performing causal analysis..."):
                        try:
                            analyzer = CausalAnalyzer()
                            
                            # Calculate elasticities
                            elasticities = analyzer.calculate_elasticities(
                                st.session_state['model_builder'].best_model,
                                st.session_state['X_test'],
                                st.session_state['y_test'],
                                st.session_state['feature_names']
                            )
                            
                            # Perform mediation analysis
                            mediation_results = analyzer.perform_mediation_analysis(
                                st.session_state['X_test'],
                                st.session_state['y_test'],
                                st.session_state['feature_names']
                            )
                            
                            # Calculate attribution
                            attribution = analyzer.calculate_attribution(
                                st.session_state['model_builder'].best_model,
                                st.session_state['X_test'],
                                st.session_state['y_test'],
                                st.session_state['feature_names']
                            )
                            
                            # Store results
                            st.session_state['analyzer'] = analyzer
                            st.session_state['elasticities'] = elasticities
                            st.session_state['mediation_results'] = mediation_results
                            st.session_state['attribution'] = attribution
                            
                            st.success("✅ Causal analysis completed!")
                            
                            # Display elasticities
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📈 Elasticities")
                                if elasticities:
                                    fig_elast = visualizer.plot_elasticities(elasticities)
                                    st.plotly_chart(fig_elast, use_container_width=True)
                            
                            with col2:
                                st.subheader("🎯 Attribution")
                                if attribution:
                                    fig_attr = visualizer.plot_attribution(attribution)
                                    st.plotly_chart(fig_attr, use_container_width=True)
                            
                            # Mediation analysis
                            if mediation_results:
                                st.subheader("🔗 Mediation Analysis")
                                fig_mediation = visualizer.plot_mediation_analysis(mediation_results)
                                st.plotly_chart(fig_mediation, use_container_width=True)
                                
                                # Mediation summary
                                st.subheader("📋 Mediation Summary")
                                for treatment, results in mediation_results.items():
                                    with st.expander(f"📊 {treatment}"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Effect", f"{results['total_effect']:.4f}")
                                        with col2:
                                            st.metric("Direct Effect", f"{results['direct_effect']:.4f}")
                                        with col3:
                                            st.metric("Proportion Mediated", f"{results['proportion_mediated']:.2%}")
                                            
                        except Exception as e:
                            st.error(f"❌ Error during causal analysis: {str(e)}")
                            with st.expander("🔍 Debug Information"):
                                st.code(str(e))
            
            # Insights and Recommendations
            if 'analyzer' in st.session_state:
                st.markdown('<h2 class="section-header">💡 Insights & Recommendations</h2>', unsafe_allow_html=True)
                
                try:
                    # Generate insights
                    insights = st.session_state['analyzer'].generate_insights()
                    st.markdown(insights)
                    
                    # Additional recommendations
                    st.subheader("🎯 Strategic Recommendations")
                    
                    recommendations = [
                        "**Budget Allocation**: Focus spend on channels with highest positive elasticity",
                        "**Mediation Strategy**: Consider the indirect effects through Google when planning social media campaigns",
                        "**Testing Framework**: Implement geo-based testing to validate causal relationships",
                        "**Monitoring**: Set up alerts for when spend approaches saturation points",
                        "**Attribution**: Use these results to inform attribution models and budget planning"
                    ]
                    
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                        
                except Exception as e:
                    st.warning(f"Could not generate insights: {str(e)}")
            
            # Export Results
            st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
            
            if st.button("📊 Generate Report"):
                try:
                    # Create comprehensive report
                    best_r2 = "N/A"
                    if 'results' in st.session_state and 'model_builder' in st.session_state:
                        best_model_name = st.session_state['model_builder'].best_model_name
                        best_r2 = st.session_state['results'][best_model_name]['r2']
                    
                    report = f"""
# Marketing Mix Modeling Report

## Executive Summary
- **Model Performance**: Best model achieved R² of {best_r2}
- **Data Period**: {len(df)} weeks of data
- **Features**: {len(processor.feature_names) if 'processor' in locals() else 'N/A'} engineered features

## Key Findings
{st.session_state.get('analyzer', type('obj', (object,), {'generate_insights': lambda: 'Analysis not completed'})).generate_insights() if 'analyzer' in st.session_state else 'Analysis not completed'}

## Model Specifications
- **Best Model**: {getattr(st.session_state.get('model_builder', None), 'best_model_name', 'Not trained')}
- **Cross-Validation**: Time-series aware validation

## Recommendations
1. Optimize budget allocation based on elasticity analysis
2. Consider mediation effects in campaign planning
3. Implement continuous monitoring and model updates
4. Validate findings through controlled experiments
"""
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name="mmm_report.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("🔍 Full Error Details"):
                st.code(str(e))
    
    elif error_message:
        st.error(f"❌ Cannot proceed due to data loading error: {error_message}")
        st.info("💡 Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()