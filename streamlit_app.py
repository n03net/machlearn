"""
E-Commerce Purchase Prediction — Streamlit App
Jalankan: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# ─── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS kustom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 20px; text-align: center;
    color: white; margin-bottom: 12px;
}
.big-metric h1 { font-size: 2.5rem; margin: 0; }
.big-metric p  { font-size: 1rem; margin: 0; opacity: 0.85; }
.pred-beli     { background: #00b894; border-radius: 10px; padding: 18px;
                  color: white; text-align: center; font-size: 1.4rem; font-weight: bold; }
.pred-tidak    { background: #d63031; border-radius: 10px; padding: 18px;
                  color: white; text-align: center; font-size: 1.4rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model    = joblib.load("model_artifacts/best_model.pkl")
        with open("model_artifacts/model_metadata.json") as f:
            meta = json.load(f)
        return model, meta
    except FileNotFoundError:
        return None, None

model, meta = load_model()

# ─── Helper: feature engineering ─────────────────────────────────────────────
ORIG_FEATURES = [
    'basket_icon_click','basket_add_list','basket_add_detail','sort_by',
    'image_picker','account_page_click','promo_banner_click','detail_wishlist_add',
    'list_size_dropdown','closed_minibasket_click','checked_delivery_detail',
    'checked_returns_detail','sign_in','saw_checkout','saw_sizecharts',
    'saw_delivery','saw_account_upgrade','saw_homepage',
    'device_mobile','device_computer','device_tablet','returning_user','loc_uk'
]

FEATURE_LABELS = {
    'basket_icon_click'      : '🛒 Klik Ikon Basket',
    'basket_add_list'        : '➕ Tambah ke Basket (List)',
    'basket_add_detail'      : '➕ Tambah ke Basket (Detail)',
    'sort_by'                : '🔤 Gunakan Sort By',
    'image_picker'           : '🖼️ Pilih Gambar Produk',
    'account_page_click'     : '👤 Klik Halaman Akun',
    'promo_banner_click'     : '📢 Klik Banner Promo',
    'detail_wishlist_add'    : '❤️ Tambah ke Wishlist',
    'list_size_dropdown'     : '📏 Pilih Ukuran (Dropdown)',
    'closed_minibasket_click': '🧺 Klik Mini Basket',
    'checked_delivery_detail': '🚚 Cek Detail Pengiriman',
    'checked_returns_detail' : '↩️ Cek Detail Retur',
    'sign_in'                : '🔐 Login / Sign In',
    'saw_checkout'           : '💳 Melihat Halaman Checkout',
    'saw_sizecharts'         : '📊 Melihat Size Chart',
    'saw_delivery'           : '📦 Melihat Halaman Delivery',
    'saw_account_upgrade'    : '⬆️ Melihat Upgrade Akun',
    'saw_homepage'           : '🏠 Melihat Homepage',
    'device_mobile'          : '📱 Perangkat: Mobile',
    'device_computer'        : '💻 Perangkat: Komputer',
    'device_tablet'          : '📲 Perangkat: Tablet',
    'returning_user'         : '🔄 Returning User',
    'loc_uk'                 : '🇬🇧 Lokasi: UK',
}

def engineer_features(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    df['total_activity']     = df[ORIG_FEATURES].sum(axis=1)
    df['basket_intent']      = df['basket_icon_click'] + df['basket_add_list'] + df['basket_add_detail']
    df['checkout_intent']    = df['saw_checkout'] + df['closed_minibasket_click']
    df['product_info_check'] = df['checked_delivery_detail'] + df['checked_returns_detail'] + df['saw_sizecharts']
    df['engagement_score']   = df['promo_banner_click'] + df['image_picker'] + df['detail_wishlist_add']
    # Pastikan urutan kolom sesuai model
    if meta:
        df = df[meta['features']]
    return df

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("E-Commerce\nPurchase Predictor")
    st.markdown("---")

    if meta:
        st.markdown("### 📌 Info Model")
        st.success(f"**{meta['model_name']}**")
        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC", f"{meta['roc_auc']:.3f}")
        col2.metric("F1-Score", f"{meta['f1_score']:.3f}")
        st.metric("Threshold Optimal", f"{meta['optimal_threshold']:.2f}")
        st.markdown("---")

    page = st.radio("📂 Navigasi",
                    ["🎯 Prediksi Manual", "📊 Batch Prediksi (CSV)", "ℹ️ Tentang Model"])

# ─── Helper: gauge chart ──────────────────────────────────────────────────────
def gauge_chart(prob: float) -> go.Figure:
    color = "#00b894" if prob >= 0.5 else "#d63031"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilitas Beli (%)", 'font': {'size': 18}},
        delta={'reference': 50, 'increasing': {'color': "#00b894"}, 'decreasing': {'color': "#d63031"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#ffe0de'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#d4edda'},
            ],
            'threshold': {'line': {'color': "navy", 'width': 3}, 'thickness': 0.8,
                          'value': (meta['optimal_threshold'] * 100) if meta else 50}
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prediksi Manual
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Prediksi Manual":
    st.title("🎯 Prediksi Pembelian — Input Manual")
    st.markdown("Pilih perilaku user, lalu klik **Prediksi** untuk melihat hasilnya.")

    if model is None:
        st.error("⚠️ Model tidak ditemukan. Pastikan folder `model_artifacts/` ada di direktori yang sama.")
        st.info("Jalankan notebook Google Colab terlebih dahulu, download `model_artifacts.zip`, lalu ekstrak di folder yang sama dengan `streamlit_app.py`.")
        st.stop()

    st.markdown("---")

    # Preset user
    st.markdown("#### 📋 Preset Contoh User")
    preset = st.selectbox("Pilih preset atau isi manual di bawah:",
                          ["— Isi Manual —", "User Aktif (kemungkinan beli)", "User Pasif (kemungkinan tidak beli)"])

    defaults = {f: 0 for f in ORIG_FEATURES}
    if preset == "User Aktif (kemungkinan beli)":
        aktif = ['basket_icon_click','basket_add_list','basket_add_detail','image_picker',
                 'promo_banner_click','detail_wishlist_add','list_size_dropdown',
                 'closed_minibasket_click','checked_delivery_detail','sign_in',
                 'saw_checkout','saw_sizecharts','saw_delivery','saw_homepage',
                 'device_mobile','returning_user','loc_uk']
        defaults = {f: (1 if f in aktif else 0) for f in ORIG_FEATURES}
    elif preset == "User Pasif (kemungkinan tidak beli)":
        defaults = {f: (1 if f in ['saw_homepage','device_computer','loc_uk'] else 0)
                    for f in ORIG_FEATURES}

    st.markdown("---")
    st.markdown("#### ✅ Pilih Aktivitas User (centang = YA / 1)")

    # Kelompokkan fitur
    groups = {
        "🛒 Interaksi Basket": ['basket_icon_click','basket_add_list','basket_add_detail','closed_minibasket_click'],
        "🔍 Perilaku Browsing": ['sort_by','image_picker','saw_homepage','saw_sizecharts'],
        "💳 Intent Checkout":  ['saw_checkout','checked_delivery_detail','checked_returns_detail','saw_delivery'],
        "👤 Akun & Engagement":['account_page_click','promo_banner_click','detail_wishlist_add',
                                 'sign_in','saw_account_upgrade','list_size_dropdown'],
        "📱 Perangkat":        ['device_mobile','device_computer','device_tablet'],
        "🌍 Demografi":        ['returning_user','loc_uk'],
    }

    user_input = {}
    cols_outer = st.columns(2)
    grp_items  = list(groups.items())
    for i, (grp_name, feats) in enumerate(grp_items):
        with cols_outer[i % 2]:
            st.markdown(f"**{grp_name}**")
            for feat in feats:
                default_val = bool(defaults.get(feat, 0))
                user_input[feat] = int(
                    st.checkbox(FEATURE_LABELS.get(feat, feat), value=default_val, key=feat)
                )

    st.markdown("---")
    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
        X_input = engineer_features(user_input)
        prob    = model.predict_proba(X_input)[0][1]
        thr     = meta['optimal_threshold'] if meta else 0.5
        pred    = int(prob >= thr)

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with c2:
            st.markdown("#### 📌 Hasil Prediksi")
            if pred == 1:
                st.markdown('<div class="pred-beli">✅ USER AKAN BELI</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-tidak">❌ USER TIDAK AKAN BELI</div>', unsafe_allow_html=True)

            st.markdown(f"**Probabilitas:** `{prob*100:.1f}%`")
            st.markdown(f"**Threshold:** `{thr:.2f}`")
            confidence = "🟢 Tinggi" if abs(prob-0.5) > 0.3 else "🟡 Sedang" if abs(prob-0.5) > 0.1 else "🔴 Rendah"
            st.markdown(f"**Confidence:** {confidence}")

        with c3:
            st.markdown("#### 🔑 Fitur Aktif")
            active = [FEATURE_LABELS.get(k, k) for k, v in user_input.items() if v == 1]
            if active:
                for a in active:
                    st.write(f"• {a}")
            else:
                st.info("Tidak ada fitur aktif")

        # Engineered features breakdown
        with st.expander("🔧 Detail Fitur Turunan"):
            eng_df = pd.DataFrame({
                'Fitur Turunan': ['Total Activity','Basket Intent','Checkout Intent','Product Info Check','Engagement Score'],
                'Nilai': [
                    sum(user_input.values()),
                    user_input['basket_icon_click'] + user_input['basket_add_list'] + user_input['basket_add_detail'],
                    user_input['saw_checkout'] + user_input['closed_minibasket_click'],
                    user_input['checked_delivery_detail'] + user_input['checked_returns_detail'] + user_input['saw_sizecharts'],
                    user_input['promo_banner_click'] + user_input['image_picker'] + user_input['detail_wishlist_add'],
                ]
            })
            st.dataframe(eng_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Prediksi CSV
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Prediksi (CSV)":
    st.title("📊 Batch Prediksi dari File CSV")
    st.markdown("Upload file CSV dengan kolom yang sama seperti data training.")

    if model is None:
        st.error("⚠️ Model tidak ditemukan.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil dimuat: **{df_up.shape[0]:,}** baris, **{df_up.shape[1]}** kolom")
            st.dataframe(df_up.head(), use_container_width=True)

            missing_cols = [c for c in ORIG_FEATURES if c not in df_up.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di CSV: {missing_cols}")
            else:
                if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
                    with st.spinner("Memproses prediksi..."):
                        X_batch = df_up[ORIG_FEATURES].copy()
                        X_batch['total_activity']     = X_batch[ORIG_FEATURES].sum(axis=1)
                        X_batch['basket_intent']      = X_batch['basket_icon_click'] + X_batch['basket_add_list'] + X_batch['basket_add_detail']
                        X_batch['checkout_intent']    = X_batch['saw_checkout'] + X_batch['closed_minibasket_click']
                        X_batch['product_info_check'] = X_batch['checked_delivery_detail'] + X_batch['checked_returns_detail'] + X_batch['saw_sizecharts']
                        X_batch['engagement_score']   = X_batch['promo_banner_click'] + X_batch['image_picker'] + X_batch['detail_wishlist_add']

                        if meta:
                            X_batch = X_batch[meta['features']]

                        thr          = meta['optimal_threshold'] if meta else 0.5
                        probs        = model.predict_proba(X_batch)[:, 1]
                        preds        = (probs >= thr).astype(int)

                        result_df = df_up.copy()
                        result_df['prob_beli']  = (probs * 100).round(2)
                        result_df['prediksi']   = np.where(preds == 1, 'BELI', 'TIDAK BELI')

                    st.success(f"✅ Prediksi selesai! **{preds.sum():,}** dari **{len(preds):,}** user diprediksi BELI")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total User", f"{len(preds):,}")
                    c2.metric("Diprediksi Beli", f"{preds.sum():,}", f"{preds.mean()*100:.1f}%")
                    c3.metric("Diprediksi Tidak Beli", f"{(preds==0).sum():,}", f"{(preds==0).mean()*100:.1f}%")

                    # Distribusi probabilitas
                    fig = px.histogram(result_df, x='prob_beli', color='prediksi',
                                       barmode='overlay', nbins=50,
                                       color_discrete_map={'BELI': '#00b894', 'TIDAK BELI': '#d63031'},
                                       title='Distribusi Probabilitas Prediksi')
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(result_df[['prob_beli','prediksi'] + ORIG_FEATURES[:5]].head(50),
                                 use_container_width=True)

                    csv_out = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Hasil Prediksi", csv_out,
                                       "hasil_prediksi.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Tentang Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Model":
    st.title("ℹ️ Tentang Model")

    st.markdown("""
    ### 🎯 Tujuan
    Memprediksi apakah seorang user akan melakukan **pembelian** pada platform e-commerce
    berdasarkan **perilaku browsing** mereka dalam satu sesi.

    ### 📊 Dataset
    | Parameter       | Detail              |
    |-----------------|---------------------|
    | Jumlah baris    | 455,401 sesi user   |
    | Jumlah fitur    | 23 fitur perilaku   |
    | Target          | `ordered` (0/1)     |
    | Class imbalance | ~4.2% melakukan pembelian |

    ### 🤖 Model yang Dibandingkan
    | Model              | Keterangan                                    |
    |--------------------|-----------------------------------------------|
    | Logistic Regression| Baseline linear + SMOTE oversampling          |
    | Random Forest      | Ensemble pohon keputusan + SMOTE              |
    | XGBoost            | Gradient boosting dengan scale_pos_weight     |
    | **LightGBM** ⭐    | Gradient boosting cepat, biasanya terbaik     |

    ### 🔧 Feature Engineering
    Selain 23 fitur asli, ditambahkan 5 fitur turunan:
    - **total_activity** — total semua aktivitas user
    - **basket_intent** — niat memasukkan ke keranjang
    - **checkout_intent** — niat menyelesaikan pembayaran
    - **product_info_check** — memeriksa info produk detail
    - **engagement_score** — keterlibatan dengan konten promosi

    ### 📈 Teknik Mengatasi Class Imbalance
    - **SMOTE** (Synthetic Minority Over-sampling Technique) untuk model linear/forest
    - **scale_pos_weight** untuk XGBoost
    - **class_weight='balanced'** untuk model lainnya

    ### 💡 Insight Bisnis
    1. `saw_checkout` adalah sinyal terkuat — user yang melihat halaman checkout sangat mungkin beli
    2. `basket_add_detail` & `basket_add_list` — interaksi basket menunjukkan niat beli tinggi
    3. `returning_user` — user yang kembali memiliki konversi lebih tinggi
    4. User yang add to basket tapi tidak checkout = target retargeting ideal

    ### 🚀 Cara Deploy
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Pastikan folder model_artifacts/ tersedia
    # (download dari Google Colab setelah training)

    # 3. Jalankan Streamlit
    streamlit run streamlit_app.py
    ```
    """)

    if meta:
        st.markdown("### 📌 Info Model yang Sedang Aktif")
        st.json(meta)
