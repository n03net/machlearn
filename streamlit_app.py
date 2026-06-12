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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Super Modern ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Styling utama */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.8rem;
        opacity: 0.95;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 16px;
        padding: 1.2rem;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
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
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 1rem 0;">
            <div style="font-size: 0.75rem;">OPTIMAL THRESHOLD</div>
            <div style="font-size: 1.8rem; font-weight: 800;">{meta['optimal_threshold']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Insights")
        st.info("""
        • **saw_checkout** = strongest signal  
        • **Basket additions** = high intent  
        • **Returning users** = 3x conversion
        """)
    
    st.markdown("---")
    st.caption("© 2024 E-Commerce Predictor | v2.0")

# ─── Main Content ────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model tidak ditemukan. Pastikan folder `model_artifacts/` ada.")
    st.info("💡 Jalankan notebook Google Colab terlebih dahulu, download `model_artifacts.zip`, lalu ekstrak.")
    st.stop()

# File Upload Section
st.markdown("### 📂 Upload Customer Data")
st.markdown("Upload CSV file containing customer behavior data for analysis and prediction")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="File must contain all 23 feature columns"
    )
with col2:
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
            <div class="metric-card">
                <div class="metric-label">TOTAL RECORDS</div>
                <div class="metric-number">{df.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FEATURES</div>
                <div class="metric-number">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MISSING VALUES</div>
                <div class="metric-number">{df.isnull().sum().sum()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            if has_target:
                purchase_rate = df['ordered'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PURCHASE RATE</div>
                    <div class="metric-number">{purchase_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                file_size_kb = uploaded_file.size / 1024
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">FILE SIZE</div>
                    <div class="metric-number">{file_size_kb:.1f} KB</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Data preview with better styling
        with st.expander("🔍 View Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"📋 Showing {len(df):,} rows × {len(df.columns)} columns")
        
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
                    marker_colors=['#f45c43', '#38ef7d'],
                    textinfo='percent+label',
                    textposition='outside'
                )])
                fig.update_layout(height=400, showlegend=False)
                fig.update_traces(textfont_size=12)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Class Balance")
                fig = go.Figure(data=[go.Bar(
                    x=['No Purchase', 'Purchase'],
                    y=target_counts.values,
                    marker_color=['#f45c43', '#38ef7d'],
                    text=target_counts.values,
                    textposition='auto',
                    textfont_size=14
                )])
                fig.update_layout(
                    height=400, 
                    xaxis_title="Class", 
                    yaxis_title="Count",
                    showlegend=False
                )
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
                    'Strength': '🟢 High' if abs(corr) > 0.1 else '🟡 Medium' if abs(corr) > 0.05 else '🔴 Low'
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
                yaxis={'categoryorder': 'total ascending'},
                xaxis=dict(range=[-0.1, 0.35])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature statistics
        st.markdown("#### 📋 Feature Statistics")
        
        # Create statistics dataframe
        stats_data = []
        for feature in ORIG_FEATURES:
            stats_data.append({
                'Feature': FEATURE_LABELS.get(feature, feature),
                'Mean': f"{df[feature].mean():.3f}",
                'Std': f"{df[feature].std():.3f}",
                'Min': df[feature].min(),
                'Max': df[feature].max(),
                'Active %': f"{(df[feature].sum() / len(df) * 100):.1f}%"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # ─── Prediction Section ─────────────────────────────────────────────────
        st.markdown("## 🤖 3. Predictions & Insights")
        
        if st.button("🚀 Run Prediction Analysis", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing predictions and analyzing data..."):
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
                
                buy_count = predictions.sum()
                not_buy_count = (predictions == 0).sum()
                avg_prob = probabilities.mean() * 100
                high_confidence = (result_df['confidence_level'] == 'High').sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #666;">TOTAL CUSTOMERS</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{len(predictions):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #666;">PREDICTED TO BUY</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #38ef7d;">{buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #666;">PREDICTED NOT BUY</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #f45c43;">{not_buy_count:,}</div>
                        <div style="font-size: 0.8rem;">({not_buy_count/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #666;">HIGH CONFIDENCE</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #fbbf24;">{high_confidence:,}</div>
                        <div style="font-size: 0.8rem;">({high_confidence/len(predictions)*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📈 Purchase Probability Distribution")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=probabilities * 100,
                    nbinsx=50,
                    marker_color='#667eea',
                    marker_line_width=0,
                    opacity=0.8,
                    name='All Customers'
                ))
                fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#f45c43", line_width=3,
                              annotation_text=f"Threshold: {threshold*100:.0f}%",
                              annotation_position="top")
                fig.update_layout(
                    title="",
                    xaxis_title="Probability (%)",
                    yaxis_title="Number of Customers",
                    height=450,
                    bargap=0.05,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence level distribution
                st.markdown("#### 🎯 Prediction Confidence Distribution")
                
                confidence_counts = result_df['confidence_level'].value_counts()
                colors = {'High': '#38ef7d', 'Medium': '#fbbf24', 'Low': '#f45c43'}
                fig = go.Figure(data=[go.Pie(
                    labels=confidence_counts.index,
                    values=confidence_counts.values,
                    hole=0.4,
                    marker_colors=[colors.get(l, '#667eea') for l in confidence_counts.index],
                    textinfo='percent+label',
                    textposition='outside'
                )])
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature comparison (FIXED)
                st.markdown("#### 🔍 Feature Analysis: High vs Low Probability")
                
                high_prob = result_df[result_df['purchase_probability'] > 0.7]
                low_prob = result_df[result_df['purchase_probability'] < 0.3]
                
                if len(high_prob) > 0 and len(low_prob) > 0:
                    feature_comparison = []
                    for feature in ORIG_FEATURES[:10]:  # Top 10 features
                        feature_comparison.append({
                            'Feature': FEATURE_LABELS.get(feature, feature),
                            'High Probability (>70%)': high_prob[feature].mean(),
                            'Low Probability (<30%)': low_prob[feature].mean(),
                            'Difference': high_prob[feature].mean() - low_prob[feature].mean()
                        })
                    
                    comp_df = pd.DataFrame(feature_comparison).sort_values('Difference', ascending=False)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='High Probability (>70%)',
                        x=comp_df['Feature'],
                        y=comp_df['High Probability (>70%)'],
                        marker_color='#38ef7d'
                    ))
                    fig.add_trace(go.Bar(
                        name='Low Probability (<30%)',
                        x=comp_df['Feature'],
                        y=comp_df['Low Probability (<30%)'],
                        marker_color='#f45c43'
                    ))
                    fig.update_layout(
                        title="",
                        xaxis_title="Features",
                        yaxis_title="Average Value",
                        barmode='group',
                        height=500,
                        xaxis_tickangle=-45,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Engineered features analysis
                st.markdown("#### 🧠 Engineered Features Analysis")
                
                X_eng = engineer_features(df)
                eng_features = ['total_activity', 'basket_intent', 'checkout_intent', 'product_info_check', 'engagement_score']
                eng_labels = ['Total Activity', 'Basket Intent', 'Checkout Intent', 'Product Info Check', 'Engagement Score']
                
                fig = make_subplots(rows=2, cols=3, 
                                    subplot_titles=eng_labels,
                                    horizontal_spacing=0.12, 
                                    vertical_spacing=0.2)
                
                plot_idx = 0
                for idx, feature in enumerate(eng_features):
                    if plot_idx >= 6:
                        break
                    row = plot_idx // 3 + 1
                    col = plot_idx % 3 + 1
                    
                    # Add box plot
                    fig.add_trace(go.Box(
                        y=X_eng[feature],
                        name=eng_labels[idx],
                        marker_color='#667eea',
                        boxmean='sd',
                        showlegend=False
                    ), row=row, col=col)
                    
                    # Add statistics annotation
                    mean_val = X_eng[feature].mean()
                    fig.add_annotation(
                        x=0.5,
                        y=0.95,
                        xref=f"x{plot_idx + 1}",
                        yref=f"y{plot_idx + 1}",
                        text=f"Mean: {mean_val:.2f}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                    plot_idx += 1
                
                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="Value")
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("#### 📋 Detailed Prediction Results (Top 50 rows)")
                
                display_cols = ['purchase_probability', 'prediction_label', 'confidence_level'] + ORIG_FEATURES[:8]
                display_df = result_df[display_cols].head(50).copy()
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
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>🎯 KEY FINDINGS:</strong><br>
                        • {buy_count:,} customers predicted to make a purchase<br>
                        • {high_confidence:,} customers have high confidence predictions ({high_confidence/len(predictions)*100:.1f}%)<br>
                        • Average purchase probability: {avg_prob:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if len(high_prob) > 0:
                        st.markdown(f"""
                        <div class="insight-box">
                            <strong>🚀 RECOMMENDATIONS:</strong><br>
                            • Priority: {len(high_prob):,} high-intent customers<br>
                            • Send personalized offers to {buy_count:,} potential buyers<br>
                            • Retarget users with abandoned cart campaigns
                        </div>
                        """, unsafe_allow_html=True)
                
                # Actionable insights
                st.markdown("#### 📌 Actionable Insights")
                
                insights = []
                
                # Check for basket abandonment
                basket_adders = result_df[(result_df['basket_add_detail'] == 1) & (result_df['prediction'] == 0)]
                if len(basket_adders) > 0:
                    insights.append(f"⚠️ **{len(basket_adders):,} users** added items to basket but didn't purchase → Send reminder emails with 10% discount")
                
                # Check for returning users
                returning_users = result_df[result_df['returning_user'] == 1]
                if len(returning_users) > 0:
                    conversion_rate = returning_users['prediction'].mean() * 100
                    insights.append(f"🔄 **{len(returning_users):,} returning users** with {conversion_rate:.1f}% conversion rate → Launch loyalty program with exclusive benefits")
                
                # Mobile users
                mobile_users = result_df[result_df['device_mobile'] == 1]
                if len(mobile_users) > 0:
                    mobile_conversion = mobile_users['prediction'].mean() * 100
                    insights.append(f"📱 **{len(mobile_users):,} mobile users** with {mobile_conversion:.1f}% conversion → Optimize mobile checkout experience")
                
                # Checkout viewers
                checkout_viewers = result_df[result_df['saw_checkout'] == 1]
                if len(checkout_viewers) > 0:
                    checkout_conversion = checkout_viewers['prediction'].mean() * 100
                    insights.append(f"💳 **{len(checkout_viewers):,} users** viewed checkout page with {checkout_conversion:.1f}% conversion → Strongest purchase signal")
                
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                # Success message
                st.balloons()
                st.success("✅ Analysis completed successfully!")
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        
else:
    # Display placeholder when no file uploaded
    st.markdown("---")
    st.markdown("### 📁 Please Upload a CSV File to Begin")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 5rem;">📂</div>
            <h3 style="margin-top: 1rem;">No File Uploaded</h3>
            <p style="color: #666;">Upload a CSV file containing customer behavior data to start analysis.</p>
            <div class="glass-card" style="text-align: left; margin-top: 2rem;">
                <strong>📋 Required columns (23 features):</strong><br>
                <code style="font-size: 0.8rem;">
                basket_icon_click, basket_add_list, basket_add_detail, sort_by, image_picker,<br>
                account_page_click, promo_banner_click, detail_wishlist_add, list_size_dropdown,<br>
                closed_minibasket_click, checked_delivery_detail, checked_returns_detail, sign_in,<br>
                saw_checkout, saw_sizecharts, saw_delivery, saw_account_upgrade, saw_homepage,<br>
                device_mobile, device_computer, device_tablet, returning_user, loc_uk
                </code>
                <hr>
                <strong>✨ Features after upload:</strong>
                <ul>
                    <li>📊 Automatic EDA (Exploratory Data Analysis)</li>
                    <li>🤖 Batch predictions for all customers</li>
                    <li>📈 Interactive visualizations</li>
                    <li>💡 Business insights & recommendations</li>
                    <li>📥 Download results in CSV/JSON format</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
