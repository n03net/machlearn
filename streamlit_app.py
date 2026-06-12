"""
E-Commerce Purchase Prediction — Streamlit App (Premium UI)
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

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shop Predict",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.9rem; }
[data-testid="stSidebar"] hr { border-color: #334155; }

/* ── Top header bar ── */
.top-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.top-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.top-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
.top-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.purple::after { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.metric-card.green::after  { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.blue::after   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.metric-card.amber::after  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.metric-label { font-size: 0.75rem; font-weight: 600; color: #94a3b8;
                text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; line-height: 1; }
.metric-sub   { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }
.metric-icon  { position: absolute; right: 1.25rem; top: 1.25rem;
                font-size: 1.6rem; opacity: 0.15; }

/* ── Section title ── */
.section-title {
    font-size: 1rem; font-weight: 600; color: #1e293b;
    margin: 1.5rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }

/* ── Feature toggle grid ── */
.feat-group-title {
    font-size: 0.72rem; font-weight: 700; color: #7c3aed;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1rem 0 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid #ede9fe;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px; padding: 1.75rem;
    text-align: center; margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.result-buy {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
}
.result-nobuy {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 2px solid #f43f5e;
}
.result-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-buy   .result-label { color: #065f46; }
.result-nobuy .result-label { color: #9f1239; }
.result-prob  { font-size: 3rem; font-weight: 800; line-height: 1; }
.result-buy   .result-prob  { color: #10b981; }
.result-nobuy .result-prob  { color: #f43f5e; }
.result-sub   { font-size: 0.85rem; margin-top: 0.4rem; color: #64748b; }

/* ── Pill badges ── */
.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-red    { background: #ffe4e6; color: #9f1239; }
.badge-yellow { background: #fef3c7; color: #78350f; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-purple { background: #ede9fe; color: #4c1d95; }

/* ── Active features list ── */
.feat-chip {
    display: inline-block; background: #f1f5f9; border: 1px solid #e2e8f0;
    color: #475569; border-radius: 8px; padding: 0.25rem 0.6rem;
    font-size: 0.78rem; margin: 0.2rem;
}
.feat-chip.active { background: #ede9fe; border-color: #c4b5fd; color: #4c1d95; }

/* ── Insight box ── */
.insight-box {
    background: #fafafa; border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1; border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem; margin: 0.75rem 0; font-size: 0.88rem; color: #334155;
}

/* ── Progress bar ── */
.progress-wrap { background: #f1f5f9; border-radius: 999px; height: 8px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }

/* ── Streamlit widget overrides ── */
.stCheckbox > label { font-size: 0.88rem !important; color: #334155 !important; }
div[data-testid="stCheckbox"] { margin-bottom: 0.1rem; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important; font-size: 0.95rem !important;
    transition: all 0.2s !important; width: 100%;
}
.stButton > button:hover { opacity: 0.92; transform: translateY(-1px); }
.stSelectbox > div > div { border-radius: 10px !important; border-color: #e2e8f0 !important; }
div[data-testid="stFileUploader"] { border: 2px dashed #c4b5fd !important;
    border-radius: 12px !important; background: #faf7ff !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
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

# ─── Constants ───────────────────────────────────────────────────────────────
ORIG_FEATURES = [
    'basket_icon_click','basket_add_list','basket_add_detail','sort_by',
    'image_picker','account_page_click','promo_banner_click','detail_wishlist_add',
    'list_size_dropdown','closed_minibasket_click','checked_delivery_detail',
    'checked_returns_detail','sign_in','saw_checkout','saw_sizecharts',
    'saw_delivery','saw_account_upgrade','saw_homepage',
    'device_mobile','device_computer','device_tablet','returning_user','loc_uk'
]

FEATURE_GROUPS = {
    "🛒  Interaksi Keranjang": {
        'basket_icon_click':   'Klik ikon keranjang',
        'basket_add_list':     'Tambah ke keranjang (list)',
        'basket_add_detail':   'Tambah ke keranjang (detail)',
        'closed_minibasket_click': 'Klik mini basket',
    },
    "💳  Niat Checkout": {
        'saw_checkout':            'Melihat halaman checkout',
        'checked_delivery_detail': 'Cek detail pengiriman',
        'checked_returns_detail':  'Cek kebijakan retur',
        'saw_delivery':            'Melihat halaman delivery',
    },
    "🔍  Browsing & Engagement": {
        'sort_by':             'Menggunakan sort/filter',
        'image_picker':        'Memilih foto produk',
        'saw_sizecharts':      'Melihat size chart',
        'promo_banner_click':  'Klik banner promo',
        'detail_wishlist_add': 'Tambah ke wishlist',
        'list_size_dropdown':  'Pilih ukuran (dropdown)',
    },
    "👤  Akun & Sesi": {
        'sign_in':           'Login / sign in',
        'account_page_click':'Klik halaman akun',
        'saw_account_upgrade':'Melihat upgrade akun',
        'saw_homepage':      'Melihat homepage',
        'returning_user':    'Returning user',
    },
    "📱  Perangkat & Lokasi": {
        'device_mobile':   'Mobile',
        'device_computer': 'Komputer / desktop',
        'device_tablet':   'Tablet',
        'loc_uk':          'Lokasi UK',
    },
}

def feat_label(key):
    for g in FEATURE_GROUPS.values():
        if key in g:
            return g[key]
    return key

def engineer_features(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    df['total_activity']     = df[ORIG_FEATURES].sum(axis=1)
    df['basket_intent']      = df['basket_icon_click'] + df['basket_add_list'] + df['basket_add_detail']
    df['checkout_intent']    = df['saw_checkout'] + df['closed_minibasket_click']
    df['product_info_check'] = df['checked_delivery_detail'] + df['checked_returns_detail'] + df['saw_sizecharts']
    df['engagement_score']   = df['promo_banner_click'] + df['image_picker'] + df['detail_wishlist_add']
    if meta:
        df = df[meta['features']]
    return df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 0 1rem; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛍️</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9;">ShopPredict AI</div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;">E-Commerce ML Predictor</div>
    </div>
    <hr style="border-color:#334155; margin: 0 0 1rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["🎯  Prediksi Manual", "📊  Batch Prediksi CSV", "📈  Dashboard Model", "ℹ️  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#334155; margin: 1rem 0;'>", unsafe_allow_html=True)

    if meta:
        st.markdown("""
        <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
             text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">
            Model Aktif
        </div>
        """, unsafe_allow_html=True)

        model_colors = {"LightGBM": "#a855f7", "XGBoost": "#3b82f6",
                        "Random Forest": "#10b981", "Logistic Regression": "#f59e0b"}
        color = model_colors.get(meta['model_name'], "#6366f1")

        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.06); border: 1px solid #334155;
             border-left: 3px solid {color}; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #e2e8f0;">{meta['model_name']}</div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for col, (label, key, fmt) in zip(
            [cols[0], cols[1], cols[0], cols[1]],
            [("ROC-AUC", "roc_auc", ".3f"), ("PR-AUC", "pr_auc", ".3f"),
             ("F1-Score", "f1_score", ".3f"), ("Threshold", "optimal_threshold", ".2f")]
        ):
            with col:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 8px;
                     padding: 0.6rem; text-align: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.65rem; color: #64748b; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9;">
                        {format(meta[key], fmt)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3);
             border-radius: 8px; padding: 0.75rem; font-size: 0.8rem; color: #fca5a5;">
            ⚠️ Model belum terdeteksi.<br>Jalankan notebook Colab terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)

# ─── Helper: gauge ────────────────────────────────────────────────────────────
def make_gauge(prob: float, threshold: float) -> go.Figure:
    pct   = prob * 100
    color = "#10b981" if prob >= threshold else "#f43f5e"
    bg    = "#ecfdf5" if prob >= threshold else "#fff1f2"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Inter"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#e2e8f0",
                     "tickvals": [0, 25, 50, 75, 100], "ticktext": ["0", "25", "50", "75", "100"],
                     "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": bg,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#fef2f2"},
                {"range": [30, 60], "color": "#fefce8"},
                {"range": [60, 100],"color": "#f0fdf4"},
            ],
            "threshold": {
                "line": {"color": "#6366f1", "width": 3},
                "thickness": 0.85,
                "value": threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


# ─── Helper: bar chart fitur aktif ───────────────────────────────────────────
def make_feature_bar(user_input: dict) -> go.Figure:
    eng = {
        "Basket Intent":       user_input['basket_icon_click'] + user_input['basket_add_list'] + user_input['basket_add_detail'],
        "Checkout Intent":     user_input['saw_checkout'] + user_input['closed_minibasket_click'],
        "Product Info Check":  user_input['checked_delivery_detail'] + user_input['checked_returns_detail'] + user_input['saw_sizecharts'],
        "Engagement Score":    user_input['promo_banner_click'] + user_input['image_picker'] + user_input['detail_wishlist_add'],
        "Account Activity":    user_input['sign_in'] + user_input['account_page_click'] + user_input['returning_user'],
    }
    labels = list(eng.keys())
    vals   = list(eng.values())
    maxv   = max(vals) if max(vals) > 0 else 1
    colors = ["#6366f1" if v == max(vals) else "#c4b5fd" for v in vals]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=vals, textposition="outside",
        textfont={"size": 11, "color": "#4c1d95"},
    ))
    fig.update_layout(
        height=200, margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"visible": False, "range": [0, maxv + 1]},
        xaxis={"tickfont": {"size": 11, "color": "#64748b"}},
        showlegend=False, bargap=0.35,
        font={"family": "Inter"}
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prediksi Manual
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯  Prediksi Manual":

    st.markdown("""
    <div class="top-header">
        <h1>🎯 Prediksi Pembelian Real-Time</h1>
        <p>Pilih aktivitas user, lalu klik Prediksi untuk melihat kemungkinan pembelian.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ **Model tidak ditemukan.** Jalankan notebook Colab terlebih dahulu dan ekstrak `model_artifacts.zip` ke folder yang sama.")
        st.stop()

    # ── Preset ──
    col_pre, col_tip = st.columns([2, 1])
    with col_pre:
        preset = st.selectbox(
            "Coba dengan preset user:",
            ["— Pilih preset —", "🟢 User Aktif (kemungkinan beli tinggi)",
             "🟡 User Sedang (ragu-ragu)", "🔴 User Pasif (kemungkinan tidak beli)"],
            label_visibility="visible"
        )

    preset_aktif  = ['basket_icon_click','basket_add_list','basket_add_detail','image_picker',
                     'promo_banner_click','detail_wishlist_add','list_size_dropdown',
                     'closed_minibasket_click','checked_delivery_detail','sign_in',
                     'saw_checkout','saw_sizecharts','saw_delivery','saw_homepage',
                     'device_mobile','returning_user','loc_uk']
    preset_sedang = ['basket_icon_click','sort_by','saw_homepage','checked_delivery_detail',
                     'image_picker','sign_in','device_computer','loc_uk']
    preset_pasif  = ['saw_homepage','device_computer','loc_uk']

    if "🟢" in preset:
        defaults = {f: (1 if f in preset_aktif else 0) for f in ORIG_FEATURES}
    elif "🟡" in preset:
        defaults = {f: (1 if f in preset_sedang else 0) for f in ORIG_FEATURES}
    elif "🔴" in preset:
        defaults = {f: (1 if f in preset_pasif else 0) for f in ORIG_FEATURES}
    else:
        defaults = {f: 0 for f in ORIG_FEATURES}

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Feature grid ──
    left_col, right_col = st.columns([3, 2], gap="large")

    user_input = {}
    with left_col:
        for grp_name, feats in FEATURE_GROUPS.items():
            st.markdown(f'<div class="feat-group-title">{grp_name}</div>', unsafe_allow_html=True)
            cols_feat = st.columns(2)
            for i, (feat_key, feat_label_txt) in enumerate(feats.items()):
                with cols_feat[i % 2]:
                    user_input[feat_key] = int(
                        st.checkbox(feat_label_txt, value=bool(defaults.get(feat_key, 0)), key=feat_key)
                    )

    # ── Predict button & live results ──
    with right_col:
        st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)

        n_active = sum(user_input.values())
        st.markdown(f"""
        <div style="background: #faf7ff; border: 1px solid #ddd6fe; border-radius: 12px;
             padding: 1rem 1.25rem; margin-bottom: 1rem; text-align: center;">
            <div style="font-size: 0.75rem; color: #7c3aed; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.08em;">Aktivitas Terpilih</div>
            <div style="font-size: 2.5rem; font-weight: 800; color: #4c1d95;">{n_active}</div>
            <div style="font-size: 0.8rem; color: #6d28d9;">dari {len(ORIG_FEATURES)} total fitur</div>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🚀  Prediksi Sekarang", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner("Menganalisis perilaku user..."):
                X_inp  = engineer_features(user_input)
                prob   = model.predict_proba(X_inp)[0][1]
                thr    = meta['optimal_threshold'] if meta else 0.5
                pred   = int(prob >= thr)

            # ── Result card ──
            if pred == 1:
                st.markdown(f"""
                <div class="result-card result-buy">
                    <div class="result-label">✅ AKAN MEMBELI</div>
                    <div class="result-prob">{prob*100:.1f}%</div>
                    <div class="result-sub">probabilitas pembelian</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-nobuy">
                    <div class="result-label">❌ TIDAK AKAN BELI</div>
                    <div class="result-prob">{prob*100:.1f}%</div>
                    <div class="result-sub">probabilitas pembelian</div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(make_gauge(prob, meta['optimal_threshold'] if meta else 0.5),
                            use_container_width=True, config={"displayModeBar": False})

            # Badge confidence
            if abs(prob - 0.5) > 0.35:
                badge_cls, conf_txt = "badge-green", "Confidence Tinggi"
            elif abs(prob - 0.5) > 0.15:
                badge_cls, conf_txt = "badge-yellow", "Confidence Sedang"
            else:
                badge_cls, conf_txt = "badge-red", "Confidence Rendah"

            thr_badge = "badge-blue" if pred == 1 else "badge-red"
            st.markdown(f"""
            <div style="text-align: center; margin-top: -0.5rem; margin-bottom: 1rem;">
                <span class="badge {badge_cls}">{conf_txt}</span>&nbsp;
                <span class="badge {thr_badge}">Threshold: {thr:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Feature activity bar ──
            st.markdown('<div class="section-title">Skor Aktivitas User</div>', unsafe_allow_html=True)
            st.plotly_chart(make_feature_bar(user_input), use_container_width=True,
                            config={"displayModeBar": False})

            # ── Insight ──
            basket_score  = user_input['basket_icon_click'] + user_input['basket_add_list'] + user_input['basket_add_detail']
            checkout_seen = user_input['saw_checkout']
            is_return     = user_input['returning_user']

            insight = []
            if checkout_seen:
                insight.append("💳 User sudah melihat halaman checkout — sinyal beli sangat kuat.")
            if basket_score >= 2:
                insight.append("🛒 Interaksi basket tinggi — pertimbangkan notifikasi diskon.")
            if is_return and pred == 0:
                insight.append("🔄 Returning user tapi belum beli — cocok untuk retargeting email.")
            if not checkout_seen and basket_score >= 1 and pred == 0:
                insight.append("⚡ Tambahkan ke basket tapi tidak checkout — kirim reminder cart abandonment.")

            if insight:
                st.markdown('<div class="section-title">💡 Rekomendasi Aksi</div>', unsafe_allow_html=True)
                for ins in insight:
                    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

        else:
            # Placeholder sebelum prediksi
            st.markdown("""
            <div style="background: #f8fafc; border: 2px dashed #e2e8f0; border-radius: 14px;
                 padding: 2.5rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🤖</div>
                <div style="font-size: 1rem; font-weight: 600; color: #475569;">Siap Memprediksi</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem;">
                    Pilih aktivitas user di sebelah kiri,<br>lalu klik tombol Prediksi.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Batch Prediksi CSV":

    st.markdown("""
    <div class="top-header">
        <h1>📊 Batch Prediksi dari CSV</h1>
        <p>Upload file CSV berisi data banyak user — prediksi semua sekaligus dan download hasilnya.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model tidak ditemukan.")
        st.stop()

    uploaded = st.file_uploader(
        "Upload file CSV (format sama seperti training_sample.csv)",
        type=["csv"],
        help="Pastikan CSV memiliki kolom yang sama dengan data training."
    )

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil dimuat: **{df_up.shape[0]:,} baris**, **{df_up.shape[1]} kolom**")

            with st.expander("👀 Preview data (5 baris pertama)"):
                st.dataframe(df_up.head(), use_container_width=True)

            missing_cols = [c for c in ORIG_FEATURES if c not in df_up.columns]
            if missing_cols:
                st.error(f"Kolom tidak ditemukan: `{missing_cols}`")
                st.stop()

            if st.button("🚀  Jalankan Prediksi Batch", type="primary"):
                with st.spinner("Memproses prediksi..."):
                    X_b = df_up[ORIG_FEATURES].copy()
                    X_b['total_activity']     = X_b.sum(axis=1)
                    X_b['basket_intent']      = X_b['basket_icon_click'] + X_b['basket_add_list'] + X_b['basket_add_detail']
                    X_b['checkout_intent']    = X_b['saw_checkout'] + X_b['closed_minibasket_click']
                    X_b['product_info_check'] = X_b['checked_delivery_detail'] + X_b['checked_returns_detail'] + X_b['saw_sizecharts']
                    X_b['engagement_score']   = X_b['promo_banner_click'] + X_b['image_picker'] + X_b['detail_wishlist_add']
                    if meta:
                        X_b = X_b[meta['features']]

                    thr   = meta['optimal_threshold'] if meta else 0.5
                    probs = model.predict_proba(X_b)[:, 1]
                    preds = (probs >= thr).astype(int)

                result_df = df_up.copy()
                result_df['Probabilitas (%)'] = (probs * 100).round(2)
                result_df['Prediksi']         = np.where(preds == 1, 'BELI', 'TIDAK BELI')
                result_df['Confidence']       = pd.cut(
                    np.abs(probs - 0.5),
                    bins=[0, 0.1, 0.25, 1.0],
                    labels=['Rendah', 'Sedang', 'Tinggi']
                )

                n_beli   = preds.sum()
                n_tidak  = (preds == 0).sum()
                rate     = preds.mean() * 100

                # ── Summary metrics ──
                st.markdown('<div class="section-title">Ringkasan Hasil</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"""<div class="metric-card blue">
                    <div class="metric-icon">👥</div>
                    <div class="metric-label">Total User</div>
                    <div class="metric-value">{len(preds):,}</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-card green">
                    <div class="metric-icon">✅</div>
                    <div class="metric-label">Diprediksi Beli</div>
                    <div class="metric-value">{n_beli:,}</div>
                    <div class="metric-sub">{rate:.1f}% dari total</div>
                </div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-card amber">
                    <div class="metric-icon">❌</div>
                    <div class="metric-label">Tidak Beli</div>
                    <div class="metric-value">{n_tidak:,}</div>
                </div>""", unsafe_allow_html=True)
                c4.markdown(f"""<div class="metric-card purple">
                    <div class="metric-icon">📊</div>
                    <div class="metric-label">Rata-rata Probabilitas</div>
                    <div class="metric-value">{probs.mean()*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                # ── Charts ──
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                ch1, ch2 = st.columns(2)

                with ch1:
                    fig_pie = go.Figure(go.Pie(
                        labels=['Tidak Beli', 'Beli'],
                        values=[n_tidak, n_beli],
                        hole=0.55,
                        marker_colors=['#f43f5e', '#10b981'],
                        textinfo='label+percent',
                        textfont={"size": 13, "family": "Inter"},
                    ))
                    fig_pie.update_layout(
                        title={"text": "Distribusi Prediksi", "font": {"size": 14, "family": "Inter"}},
                        height=280, margin=dict(t=40, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                        annotations=[dict(text=f"<b>{rate:.1f}%</b><br>Beli", x=0.5, y=0.5,
                                         font_size=14, showarrow=False, font_family="Inter")]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                with ch2:
                    fig_hist = go.Figure(go.Histogram(
                        x=probs * 100, nbinsx=40,
                        marker_color='#6366f1', opacity=0.85,
                    ))
                    fig_hist.add_vline(x=thr * 100, line_dash="dash", line_color="#f43f5e",
                                       annotation_text=f"Threshold {thr:.2f}",
                                       annotation_font_color="#f43f5e")
                    fig_hist.update_layout(
                        title={"text": "Distribusi Probabilitas", "font": {"size": 14, "family": "Inter"}},
                        xaxis_title="Probabilitas Beli (%)",
                        yaxis_title="Jumlah User",
                        height=280, margin=dict(t=40, b=40, l=40, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        bargap=0.05, font={"family": "Inter"},
                        xaxis={"gridcolor": "#f1f5f9"}, yaxis={"gridcolor": "#f1f5f9"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                # ── Table hasil ──
                st.markdown('<div class="section-title">Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
                display_cols = ['Probabilitas (%)', 'Prediksi', 'Confidence']
                if 'UserID' in result_df.columns:
                    display_cols = ['UserID'] + display_cols
                st.dataframe(
                    result_df[display_cols].head(100).style
                    .map(lambda v: 'color: #065f46; font-weight: 600' if v == 'BELI'
                        else ('color: #9f1239; font-weight: 600' if v == 'TIDAK BELI' else ''),
                        subset=['Prediksi'])
                    .format({'Probabilitas (%)': '{:.2f}'}),
                    use_container_width=True, height=320
                )
                if len(result_df) > 100:
                    st.caption(f"Menampilkan 100 dari {len(result_df):,} baris. Download untuk data lengkap.")

                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️  Download Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error memproses file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dashboard Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Dashboard Model":

    st.markdown("""
    <div class="top-header">
        <h1>📈 Dashboard Performa Model</h1>
        <p>Informasi lengkap tentang model ML yang digunakan untuk prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

    if meta is None:
        st.warning("Model belum tersedia. Jalankan notebook Colab terlebih dahulu.")
        st.stop()

    # ── Top metrics ──
    c1, c2, c3, c4 = st.columns(4)
    metrics_data = [
        (c1, "ROC-AUC", f"{meta['roc_auc']:.4f}", "Kemampuan diskriminasi", "purple"),
        (c2, "PR-AUC",  f"{meta['pr_auc']:.4f}",  "Presisi-recall area",   "blue"),
        (c3, "F1-Score", f"{meta['f1_score']:.4f}", "Harmonic mean P & R",  "green"),
        (c4, "Threshold", f"{meta['optimal_threshold']:.2f}", "Cutoff optimal",   "amber"),
    ]
    for col, label, val, sub, color in metrics_data:
        with col:
            col.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Model comparison bars ──
    st.markdown('<div class="section-title">Perbandingan Model (ilustratif)</div>', unsafe_allow_html=True)

    model_scores = {
        "Logistic Regression": {"ROC-AUC": 0.841, "F1": 0.312, "PR-AUC": 0.381},
        "Random Forest":       {"ROC-AUC": 0.903, "F1": 0.418, "PR-AUC": 0.502},
        "XGBoost":             {"ROC-AUC": 0.922, "F1": 0.451, "PR-AUC": 0.543},
        "LightGBM":            {"ROC-AUC": meta['roc_auc'], "F1": meta['f1_score'], "PR-AUC": meta['pr_auc']},
    }
    df_cmp = pd.DataFrame(model_scores).T.reset_index()
    df_cmp.columns = ['Model', 'ROC-AUC', 'F1', 'PR-AUC']

    metric_sel = st.radio("Pilih metrik:", ["ROC-AUC", "F1", "PR-AUC"], horizontal=True)
    colors_bar = ["#6366f1" if m == meta['model_name'] else "#c4b5fd" for m in df_cmp['Model']]

    fig_bar = go.Figure(go.Bar(
        x=df_cmp['Model'], y=df_cmp[metric_sel],
        marker_color=colors_bar,
        text=df_cmp[metric_sel].round(3),
        textposition="outside",
        textfont={"size": 12, "color": "#4c1d95"},
    ))
    fig_bar.add_hline(y=df_cmp[metric_sel].max(), line_dash="dot",
                       line_color="#10b981", opacity=0.5)
    fig_bar.update_layout(
        height=300, margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"gridcolor": "#f1f5f9", "range": [0, df_cmp[metric_sel].max() + 0.1]},
        xaxis={"tickfont": {"size": 12}}, bargap=0.35,
        font={"family": "Inter"},
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"⭐ Bar ungu = model terbaik ({meta['model_name']})")

    # ── Feature importance ──
    st.markdown('<div class="section-title">Fitur Terpenting (berdasarkan korelasi dengan pembelian)</div>', unsafe_allow_html=True)

    feat_importance = {
        "saw_checkout":            0.312,
        "basket_add_detail":       0.287,
        "basket_add_list":         0.265,
        "basket_icon_click":       0.241,
        "closed_minibasket_click": 0.198,
        "checked_delivery_detail": 0.181,
        "sign_in":                 0.162,
        "returning_user":          0.143,
        "detail_wishlist_add":     0.128,
        "saw_sizecharts":          0.117,
    }
    df_fi = pd.DataFrame(list(feat_importance.items()), columns=['Fitur', 'Importance'])
    df_fi = df_fi.sort_values('Importance')

    fig_fi = go.Figure(go.Bar(
        x=df_fi['Importance'], y=df_fi['Fitur'], orientation='h',
        marker=dict(
            color=df_fi['Importance'],
            colorscale=[[0, '#ede9fe'], [1, '#4c1d95']],
            showscale=False
        ),
        text=df_fi['Importance'].round(3), textposition='outside',
        textfont={"size": 11}
    ))
    fig_fi.update_layout(
        height=340, margin=dict(t=10, b=30, l=180, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False}, yaxis={"tickfont": {"size": 11}},
        font={"family": "Inter"},
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # ── Dataset info ──
    st.markdown('<div class="section-title">Informasi Dataset</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        | Parameter | Detail |
        |---|---|
        | Total user sessions | 455,401 |
        | Fitur asli | 23 |
        | Fitur engineered | 5 |
        | Rasio beli | 4.2% |
        | Missing values | 0 |
        """)
    with col_d2:
        st.markdown("""
        | Teknik | Keterangan |
        |---|---|
        | Imbalance handling | SMOTE + class_weight |
        | Validation | Stratified 80/20 split |
        | Threshold | Optimized F1-score |
        | Explainability | SHAP TreeExplainer |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Tentang
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  Tentang":

    st.markdown("""
    <div class="top-header">
        <h1>ℹ️ Tentang ShopPredict AI</h1>
        <p>Dokumentasi lengkap proyek Machine Learning untuk prediksi pembelian e-commerce.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ### 🎯 Tujuan Proyek
        Memprediksi apakah seorang user akan melakukan **pembelian** pada sesi kunjungan
        mereka di platform e-commerce, berdasarkan **23 fitur perilaku browsing** yang
        direkam secara real-time.

        ### 🤖 Alur Machine Learning
        1. **Data Cleaning** — tidak ada missing values, semua fitur binary (0/1)
        2. **Feature Engineering** — 5 fitur turunan dari kombinasi fitur asli
        3. **Handling Class Imbalance** — SMOTE oversampling (hanya 4.2% user beli)
        4. **Model Training** — 4 algoritma dibandingkan
        5. **Threshold Optimization** — dicari threshold terbaik untuk F1-score
        6. **Explainability** — SHAP values untuk interpretasi model

        ### 💡 Fitur Engineered
        | Fitur Turunan | Rumus |
        |---|---|
        | `total_activity` | sum semua 23 fitur |
        | `basket_intent` | icon_click + add_list + add_detail |
        | `checkout_intent` | saw_checkout + minibasket_click |
        | `product_info_check` | delivery + returns + sizechart |
        | `engagement_score` | promo + image_picker + wishlist |

    with col_b:
        st.markdown("""
        ### 📦 Tech Stack
        """)
        tech = [
            ("🐍", "Python 3.10+",     "Bahasa utama"),
            ("🧪", "Scikit-learn",     "Preprocessing & baseline"),
            ("⚡", "LightGBM / XGBoost", "Model terbaik"),
            ("⚖️", "imbalanced-learn", "SMOTE oversampling"),
            ("🔬", "SHAP",             "Explainability"),
            ("📊", "Plotly",           "Visualisasi interaktif"),
            ("🌐", "Streamlit",        "Web app deployment"),
        ]
        for icon, name, desc in tech:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.75rem;
                 background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                 padding: 0.65rem 1rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.3rem;">{icon}</div>
                <div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">{name}</div>
                    <div style="font-size: 0.75rem; color: #64748b;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        ### 📌 Insight Bisnis Utama
        """)
        insights = [
            ("💳", "saw_checkout → sinyal beli paling kuat"),
            ("🛒", "Add to basket + tidak checkout = target retargeting"),
            ("🔄", "Returning user = prioritas loyalty program"),
            ("📱", "Optimalkan UX checkout di mobile"),
        ]
        for emoji, text in insights:
            st.markdown(f"""
            <div class="insight-box">{emoji} {text}</div>
            """, unsafe_allow_html=True)
