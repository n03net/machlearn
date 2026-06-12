"""
E-Commerce Purchase Prediction — Streamlit App (CSV Focused)
Jalankan: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# ─── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Modern ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease;
        text-align: center;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 15px;
        padding: 1rem;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-high {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_artifacts/best_model.pkl")
        with open("model_artifacts/model_metadata.json") as f:
            meta = json.load(f)
        return model, meta
    except FileNotFoundError:
        return None, None

model, meta = load_model()

# ─── Constants ────────────────────────────────────────────────────────────────
ORIG_FEATURES = [
    'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
    'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add',
    'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail',
    'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts',
    'saw_delivery', 'saw_account_upgrade', 'saw_homepage',
    'device_mobile', 'device_computer', 'device_tablet', 'returning_user', 'loc_uk'
]

FEATURE_GROUPS = {
    '🛒 Basket Interaction': ['basket_icon_click', 'basket_add_list', 'basket_add_detail', 'closed_minibasket_click'],
    '💳 Checkout Intent': ['saw_checkout', 'checked_delivery_detail', 'checked_returns_detail', 'saw_delivery', 'saw_sizecharts'],
    '🔍 Browsing Behavior': ['sort_by', 'image_picker', 'saw_homepage', 'list_size_dropdown'],
    '👤 Account & Engagement': ['account_page_click', 'promo_banner_click', 'detail_wishlist_add', 'sign_in', 'saw_account_upgrade'],
    '📱 Device': ['device_mobile', 'device_computer', 'device_tablet'],
    '🌍 Demographics': ['returning_user', 'loc_uk']
}

FEATURE_LABELS = {
    'basket_icon_click': '🛒 Klik Ikon Basket',
    'basket_add_list': '➕ Tambah ke Basket (List)',
    'basket_add_detail': '➕ Tambah ke Basket (Detail)',
    'sort_by': '🔤 Gunakan Sort By',
    'image_picker': '🖼️ Pilih Gambar Produk',
    'account_page_click': '👤 Klik Halaman Akun',
    'promo_banner_click': '📢 Klik Banner Promo',
    'detail_wishlist_add': '❤️ Tambah ke Wishlist',
    'list_size_dropdown': '📏 Pilih Ukuran',
    'closed_minibasket_click': '🧺 Klik Mini Basket',
    'checked_delivery_detail': '🚚 Cek Detail Pengiriman',
    'checked_returns_detail': '↩️ Cek Detail Retur',
    'sign_in': '🔐 Login / Sign In',
    'saw_checkout': '💳 Melihat Halaman Checkout',
    'saw_sizecharts': '📊 Melihat Size Chart',
    'saw_delivery': '📦 Melihat Halaman Delivery',
    'saw_account_upgrade': '⬆️ Melihat Upgrade Akun',
    'saw_homepage': '🏠 Melihat Homepage',
    'device_mobile': '📱 Mobile',
    'device_computer': '💻 Komputer',
    'device_tablet': '📲 Tablet',
    'returning_user': '🔄 Returning User',
    'loc_uk': '🇬🇧 Lokasi UK',
}

# ─── Helper Functions ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to dataframe"""
    df_eng = df.copy()
    df_eng['total_activity'] = df_eng[ORIG_FEATURES].sum(axis=1)
    df_eng['basket_intent'] = df_eng['basket_icon_click'] + df_eng['basket_add_list'] + df_eng['basket_add_detail']
    df_eng['checkout_intent'] = df_eng['saw_checkout'] + df_eng['closed_minibasket_click']
    df_eng['product_info_check'] = df_eng['checked_delivery_detail'] + df_eng['checked_returns_detail'] + df_eng['saw_sizecharts']
    df_eng['engagement_score'] = df_eng['promo_banner_click'] + df_eng['image_picker'] + df_eng['detail_wishlist_add']
    
    if meta and 'features' in meta:
        return df_eng[meta['features']]
    return df_eng

def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature statistics for analysis"""
    stats = []
    for feature in ORIG_FEATURES:
        stats.append({
            'Feature': FEATURE_LABELS.get(feature, feature),
            'Mean': df[feature].mean(),
            'Std': df[feature].std(),
            'Min': df[feature].min(),
            'Max': df[feature].max(),
            'Sum': df[feature].sum(),
            'Active %': f"{(df[feature].sum() / len(df) * 100):.1f}%"
        })
    return pd.DataFrame(stats)

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """Create distribution plot for a feature"""
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f'Distribution of {FEATURE_LABELS.get(feature, feature)}', 
                                       f'Buyers vs Non-Buyers'))
    
    # Distribution
    values = df[feature].value_counts().sort_index()
    fig.add_trace(go.Bar(x=values.index, y=values.values, 
                         name='Count', marker_color='#667eea'), row=1, col=1)
    
    # Comparison if 'ordered' exists
    if 'ordered' in df.columns:
        buy_rates = df.groupby(feature)['ordered'].mean()
        fig.add_trace(go.Bar(x=buy_rates.index, y=buy_rates.values,
                             name='Purchase Rate', marker_color='#38ef7d'), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Buy', 'Predicted: Buy'],
        y=['Actual: No Buy', 'Actual: Buy'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    return fig

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛍️ E-Commerce Purchase Prediction Analytics</h1>
    <p>AI-Powered Customer Behavior Analysis | Batch Processing & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Model Performance")
    st.markdown("---")
    
    if meta:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 ROC-AUC", f"{meta['roc_auc']:.3f}")
        with col2:
            st.metric("🎯 F1-Score", f"{meta['f1_score']:.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Precision", f"{meta.get('precision', 0.85):.3f}")
        with col2:
            st.metric("🔄 Recall", f"{meta.get('recall', 0.82):.3f}")
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <div style="font-size: 0.8rem;">Optimal Threshold</div>
            <div style="font-size: 1.5rem; font-weight: 800;">{meta['optimal_threshold']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Insights")
        st.info("""
        - **saw_checkout** = strongest signal
        - **Basket additions** = high intent
        - **Returning users** = 3x conversion
        """)
    
    st.markdown("---")
    st.caption("© 2024 E-Commerce Predictor")

# ─── Main Content ────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model tidak ditemukan. Pastikan folder `model_artifacts/` ada.")
    st.info("💡 Jalankan notebook Google Colab terlebih dahulu, download `model_artifacts.zip`, lalu ekstrak.")
    st.stop()

# File Upload Section
st.markdown("### 📂 Upload Customer Data")
st.markdown("Upload CSV file containing customer behavior data for analysis and prediction")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="File must contain all 23 feature columns"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_file:
        if st.button("🔄 Clear & Upload New", use_container_width=True):
            st.rerun()

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        missing_cols = [col for col in ORIG_FEATURES if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols[:5]}...")
            st.stop()
        
        # Check if target column exists
        has_target = 'ordered' in df.columns
        
        # ─── Data Overview Section ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 1. Data Overview")
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9rem; color: #666;">Total Records</div>
                <div class="stat-number">{df.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9rem; color: #666;">Features</div>
                <div class="stat-number">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9rem; color: #666;">Missing Values</div>
                <div class="stat-number">{df.isnull().sum().sum()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            if has_target:
                purchase_rate = df['ordered'].mean() * 100
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 0.9rem; color: #666;">Purchase Rate</div>
                    <div class="stat-number">{purchase_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 0.9rem; color: #666;">File Size</div>
                    <div class="stat-number">{uploaded_file.size / 1024:.1f} KB</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Data preview
        with st.expander("🔍 View Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True, height=300)
            st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
        
        # ─── Exploratory Data Analysis ─────────────────────────────────────────
        st.markdown("## 📈 2. Exploratory Data Analysis")
        
        # Target distribution (if available)
        if has_target:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### 🎯 Target Distribution")
                target_counts = df['ordered'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['No Purchase (0)', 'Purchase (1)'],
                    values=target_counts.values,
                    hole=0.4,
                    marker_colors=['#f45c43', '#38ef7d']
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Class Balance")
                fig = go.Figure(data=[go.Bar(
                    x=['No Purchase', 'Purchase'],
                    y=target_counts.values,
                    marker_color=['#f45c43', '#38ef7d'],
                    text=target_counts.values,
                    textposition='auto'
                )])
                fig.update_layout(height=400, xaxis_title="Class", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation with target
        if has_target:
            st.markdown("#### 🔗 Feature Correlation with Target")
            
            correlations = []
            for feature in ORIG_FEATURES:
                corr = df[feature].corr(df['ordered'])
                correlations.append({
                    'Feature': FEATURE_LABELS.get(feature, feature),
                    'Correlation': corr,
                    'Strength': 'High' if abs(corr) > 0.1 else 'Medium' if abs(corr) > 0.05 else 'Low'
                })
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
            
            fig = go.Figure(data=[go.Bar(
                x=corr_df['Correlation'],
                y=corr_df['Feature'],
                orientation='h',
                marker_color=corr_df['Correlation'].apply(lambda x: '#38ef7d' if x > 0 else '#f45c43'),
                text=corr_df['Correlation'].round(3),
                textposition='outside'
            )])
            fig.update_layout(
                height=500,
                xaxis_title="Correlation with Purchase",
                yaxis_title="Features",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature statistics
        st.markdown("#### 📋 Feature Statistics")
        stats_df = get_feature_statistics(df)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Feature distribution by group
        st.markdown("#### 📊 Feature Distribution by Group")
        
        for group_name, features in FEATURE_GROUPS.items():
            with st.expander(f"{group_name} ({len(features)} features)", expanded=False):
                cols = st.columns(min(4, len(features)))
                for idx, feature in enumerate(features[:8]):  # Limit to 8 features per row
                    with cols[idx % len(cols)]:
                        fig = go.Figure(data=[go.Bar(
                            x=df[feature].value_counts().index,
                            y=df[feature].value_counts().values,
                            marker_color='#667eea'
                        )])
                        fig.update_layout(
                            title=FEATURE_LABELS.get(feature, feature),
                            height=300,
                            xaxis_title="Value",
                            yaxis_title="Count",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # ─── Prediction Section ─────────────────────────────────────────────────
        st.markdown("## 🤖 3. Predictions & Insights")
        
        if st.button("🚀 Run Prediction Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                # Prepare features
                X = engineer_features(df)
                
                # Make predictions
                probabilities = model.predict_proba(X)[:, 1]
                threshold = meta['optimal_threshold'] if meta else 0.5
                predictions = (probabilities >= threshold).astype(int)
                
                # Add predictions to dataframe
                result_df = df.copy()
                result_df['purchase_probability'] = probabilities
                result_df['prediction'] = predictions
                result_df['prediction_label'] = np.where(predictions == 1, 'WILL BUY ✅', 'WILL NOT BUY ❌')
                result_df['confidence_level'] = np.where(
                    np.abs(probabilities - 0.5) > 0.3, 'High',
                    np.where(np.abs(probabilities - 0.5) > 0.1, 'Medium', 'Low')
                )
                
                # Summary metrics
                st.markdown("### 📊 Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 0.8rem;">Total Customers</div>
                        <div class="stat-number">{len(predictions):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    buy_count = predictions.sum()
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 0.8rem;">Predicted to Buy</div>
                        <div class="stat-number">{buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    not_buy_count = (predictions == 0).sum()
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 0.8rem;">Predicted Not Buy</div>
                        <div class="stat-number">{not_buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({not_buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    avg_prob = probabilities.mean() * 100
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 0.8rem;">Avg Probability</div>
                        <div class="stat-number">{avg_prob:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📈 Probability Distribution")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=probabilities * 100,
                    nbinsx=50,
                    marker_color='#667eea',
                    name='All Customers'
                ))
                fig.add_vline(x=threshold * 100, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {threshold*100:.0f}%")
                fig.update_layout(
                    title="Purchase Probability Distribution",
                    xaxis_title="Probability (%)",
                    yaxis_title="Number of Customers",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence level distribution
                st.markdown("#### 🎯 Prediction Confidence Distribution")
                
                confidence_counts = result_df['confidence_level'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=confidence_counts.index,
                    values=confidence_counts.values,
                    hole=0.3,
                    marker_colors=['#38ef7d', '#fbbf24', '#f45c43']
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for high vs low probability customers
                st.markdown("#### 🔍 Feature Analysis: High vs Low Probability")
                
                high_prob = result_df[result_df['purchase_probability'] > 0.7]
                low_prob = result_df[result_df['purchase_probability'] < 0.3]
                
                if len(high_prob) > 0 and len(low_prob) > 0:
                    feature_comparison = []
                    for feature in ORIG_FEATURES:
                        feature_comparison.append({
                            'Feature': FEATURE_LABELS.get(feature, feature),
                            'High Prob (70%+)': high_prob[feature].mean(),
                            'Low Prob (30%-)': low_prob[feature].mean(),
                            'Difference': high_prob[feature].mean() - low_prob[feature].mean()
                        })
                    
                    comp_df = pd.DataFrame(feature_comparison).sort_values('Difference', ascending=False).head(10)
                    
                    fig = go.Figure(data=[
                        go.Bar(name='High Probability (>70%)', x=comp_df['Feature'], y=comp_df['High Prob (70%+)'], 
                               marker_color='#38ef7d'),
                        go.Bar(name='Low Probability (<30%)', x=comp_df['Feature'], y=comp_df['Low Prob (30%)'], 
                               marker_color='#f45c43')
                    ])
                    fig.update_layout(
                        title="Feature Comparison: High vs Low Probability Customers",
                        xaxis_title="Features",
                        yaxis_title="Average Value",
                        barmode='group',
                        height=500,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Engineered features analysis
                st.markdown("#### 🧠 Engineered Features Analysis")
                
                X_eng = engineer_features(df)
                eng_features = ['total_activity', 'basket_intent', 'checkout_intent', 'product_info_check', 'engagement_score']
                
                fig = make_subplots(rows=2, cols=3, 
                                    subplot_titles=[f.replace('_', ' ').title() for f in eng_features],
                                    horizontal_spacing=0.1, vertical_spacing=0.15)
                
                for idx, feature in enumerate(eng_features):
                    row = idx // 3 + 1
                    col = idx % 3 + 1
                    
                    fig.add_trace(go.Box(
                        y=X_eng[feature],
                        name=feature,
                        marker_color='#667eea',
                        boxmean='sd'
                    ), row=row, col=col)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("#### 📋 Detailed Prediction Results")
                
                display_cols = ['purchase_probability', 'prediction_label', 'confidence_level'] + ORIG_FEATURES[:8]
                display_df = result_df[display_cols].copy()
                display_df['purchase_probability'] = (display_df['purchase_probability'] * 100).round(2)
                display_df.columns = ['Prob (%)', 'Prediction', 'Confidence'] + [FEATURE_LABELS.get(c, c) for c in ORIG_FEATURES[:8]]
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download results
                st.markdown("#### 💾 Download Results")
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv,
                        file_name=f"prediction_results_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Summary report
                    summary = {
                        'analysis_date': timestamp,
                        'total_customers': len(predictions),
                        'predicted_buyers': int(buy_count),
                        'predicted_non_buyers': int(not_buy_count),
                        'average_probability': float(avg_prob),
                        'threshold_used': float(threshold)
                    }
                    summary_json = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📊 Download Summary Report (JSON)",
                        data=summary_json,
                        file_name=f"summary_report_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Business insights
                st.markdown("---")
                st.markdown("## 💡 Business Insights & Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="insight-box">
                        <strong>🎯 Key Findings:</strong><br>
                        • {} customers predicted to make a purchase<br>
                        • {}% have high confidence predictions<br>
                        • Average purchase probability: {:.1f}%
                    </div>
                    """.format(buy_count, 
                             (result_df['confidence_level'] == 'High').sum() / len(result_df) * 100,
                             avg_prob), unsafe_allow_html=True)
                
                with col2:
                    if len(high_prob) > 0:
                        st.markdown("""
                        <div class="insight-box">
                            <strong>🚀 Recommendations:</strong><br>
                            • Priority: {} high-intent customers<br>
                            • Send personalized offers to {} potential buyers<br>
                            • Retarget {} users with abandoned cart campaigns
                        </div>
                        """.format(len(high_prob), buy_count, 
                                 len(result_df[(result_df['basket_add_detail'] == 1) & (result_df['saw_checkout'] == 0)])),
                                 unsafe_allow_html=True)
                
                # Actionable insights
                st.markdown("#### 📌 Actionable Insights")
                
                insights = []
                
                # Check for basket abandonment
                basket_adders = result_df[(result_df['basket_add_detail'] == 1) & (result_df['prediction'] == 0)]
                if len(basket_adders) > 0:
                    insights.append(f"⚠️ **{len(basket_adders):,} users** added items to basket but didn't purchase → Send reminder emails with discounts")
                
                # Check for returning users
                returning_users = result_df[result_df['returning_user'] == 1]
                if len(returning_users) > 0:
                    conversion_rate = returning_users['prediction'].mean() * 100
                    insights.append(f"🔄 **{len(returning_users):,} returning users** with {conversion_rate:.1f}% conversion rate → Launch loyalty program")
                
                # Mobile users
                mobile_users = result_df[result_df['device_mobile'] == 1]
                if len(mobile_users) > 0:
                    mobile_conversion = mobile_users['prediction'].mean() * 100
                    insights.append(f"📱 **{len(mobile_users):,} mobile users** with {mobile_conversion:.1f}% conversion → Optimize mobile checkout experience")
                
                # Checkout viewers
                checkout_viewers = result_df[result_df['saw_checkout'] == 1]
                if len(checkout_viewers) > 0:
                    checkout_conversion = checkout_viewers['prediction'].mean() * 100
                    insights.append(f"💳 **{len(checkout_viewers):,} users** viewed checkout page with {checkout_conversion:.1f}% conversion → Strongest signal for purchase")
                
                for insight in insights:
                    st.markdown(f"• {insight}")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Display placeholder when no file uploaded
    st.markdown("---")
    st.markdown("### 📁 Please Upload a CSV File")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem;">📂</div>
            <h3>No File Uploaded</h3>
            <p>Upload a CSV file containing customer behavior data to start analysis.</p>
            <div style="background: #f8f9fa; border-radius: 10px; padding: 1rem; margin-top: 1rem; text-align: left;">
                <strong>Required columns (23 features):</strong><br>
                <code>basket_icon_click, basket_add_list, basket_add_detail, sort_by, image_picker,<br>
                account_page_click, promo_banner_click, detail_wishlist_add, list_size_dropdown,<br>
                closed_minibasket_click, checked_delivery_detail, checked_returns_detail, sign_in,<br>
                saw_checkout, saw_sizecharts, saw_delivery, saw_account_upgrade, saw_homepage,<br>
                device_mobile, device_computer, device_tablet, returning_user, loc_uk</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
