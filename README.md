# Market-Mix-Modelling
Marketing Mix Modeling (MMM) Assessment Overview This project implements a comprehensive Marketing Mix Model (MMM) with causal analysis, specifically designed to handle Google spend as a mediator between social media channels and revenue. The solution addresses the assessment requirements for modeling marketing data with mediation assumptions.

Key Features

Data Preparation Seasonality Handling: Automatic detection and modeling of weekly/monthly patterns using sine/cosine transformations Trend Analysis: Linear and quadratic trend components Zero-Spend Periods: Special handling with indicator variables and adstock decay Feature Engineering: Adstock (carryover) effects with multiple decay rates Saturation curves using Hill transformation Lag features for temporal dependencies Time-based seasonal features
Modeling Approach Multiple Algorithms: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting Hyperparameter Optimization: Grid search with time-series cross-validation Regularization: Built-in feature selection and overfitting prevention Time-Series Validation: Proper temporal splits to avoid look-ahead bias
Causal Framework Mediation Analysis: Formal two-stage approach for Google-as-mediator assumption Direct vs Indirect Effects: Decomposition of total effects through mediation paths Causal DAG Consideration: Feature engineering respects causal structure Back-door Path Analysis: Identification of potential confounders
Diagnostics & Validation Out-of-Sample Performance: Time-series aware train/test splits Stability Checks: Rolling window validation and coefficient stability Residual Analysis: Comprehensive diagnostic plots and tests Multicollinearity Detection: VIF analysis and correlation monitoring Sensitivity Analysis: Robustness to price and promotion changes
Insights & Recommendations Elasticity Analysis: Price and media elasticities with confidence intervals Attribution Modeling: Revenue attribution across channels Saturation Curves: Diminishing returns analysis for media channels Strategic Recommendations: Actionable insights for marketing teams Technical Implementation Architecture ├── app.py # Main Streamlit dashboard ├── data_processor.py # Data preprocessing and feature engineering ├── model_builder.py # ML models and mediation analysis ├── causal_analysis.py # Causal inference and elasticity calculations ├── visualizations.py # Interactive plots and diagnostics └── requirements.txt # Dependencies Key Classes MarketingDataProcessor: Handles data cleaning, feature engineering, and scaling MarketingModelBuilder: Implements multiple ML algorithms with time-series CV MediationModelBuilder: Two-stage modeling for causal mediation CausalAnalyzer: Elasticity, attribution, and mediation analysis MarketingVisualizer: Interactive visualizations and diagnostic plots Installation & Usage Prerequisites pip install -r requirements.txt Running the Application streamlit run app.py Data Format The application expects weekly marketing data with columns:
date: Week ending date revenue: Target variable google_spend, facebook_spend, etc.: Media spend variables avg_price: Average product price promotions: Promotional indicator email_send, sms_send: Direct response metrics followers: Social media followers Methodology

Mediation Analysis Approach The model implements a formal mediation framework:
Stage 1: Social Media → Google Spend

Google_spend = α₀ + α₁ × Facebook_spend + α₂ × TikTok_spend + α₃ × Snapchat_spend + ε₁ Stage 2: All Variables → Revenue

Revenue = β₀ + β₁ × Google_spend_predicted + β₂ × Other_variables + ε₂ Effects Decomposition:

Total Effect: Direct impact without controlling for Google Direct Effect: Impact controlling for Google spend Indirect Effect: Impact mediated through Google spend 2. Feature Engineering Adstock Transformation:

Adstock_t = Spend_t + λ × Adstock_{t-1} Where λ ∈ [0.3, 0.5, 0.7] represents carryover rates.

Saturation Transformation:

Saturation = x^α / (x^α + 1) Where α controls the saturation curve shape.

Time-Series Cross-Validation Uses TimeSeriesSplit to ensure:
No look-ahead bias Temporal ordering preservation Realistic out-of-sample evaluation Model Evaluation Criteria

Technical Rigor ✅ Time-series aware cross-validation Robust preprocessing with scaling and transformation Proper handling of zero-spend periods Well-reasoned hyperparameter selection
Causal Awareness ✅ Explicit mediation modeling with two-stage approach DAG-consistent feature engineering Mediation analysis with effect decomposition Back-door path consideration
Interpretability ✅ Clear elasticity calculations and confidence intervals Feature importance rankings Saturation curve analysis Interactive visualizations
Product Thinking ✅ Actionable recommendations for marketing teams Decision boundary identification Trade-off analysis (price vs. demand, search vs. social) Strategic budget allocation guidance
Reproducibility ✅ Clean, documented code Deterministic results with random seeds Environment specifications Comprehensive README Key Insights Generated Elasticity Analysis: Quantifies revenue response to 1% changes in each variable Mediation Effects: Measures how much of social media impact flows through Google Attribution: Revenue contribution by channel Saturation Points: Identifies diminishing returns thresholds Optimal Allocation: Budget recommendations based on marginal returns Limitations & Future Enhancements Current Limitations Assumes linear mediation relationships Limited to weekly aggregation Simplified competitive effects modeling Future Enhancements Bayesian hierarchical modeling for uncertainty quantification Geographic variation analysis Competitive response modeling Real-time model updating A/B test integration for causal validation Business Impact This MMM provides marketing teams with:
Data-Driven Budget Allocation: Optimize spend across channels Causal Understanding: Distinguish correlation from causation Performance Monitoring: Track efficiency and saturation Strategic Planning: Long-term media mix optimization ROI Measurement: Accurate attribution and incrementality Contact & Support For questions or issues:

Check the interactive dashboard help sections Review diagnostic plots for model quality Examine confidence intervals for statistical significance Validate findings through controlled experiments This implementation satisfies all assessment criteria while providing a production-ready marketing analytics solution.
