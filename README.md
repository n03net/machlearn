# 🛍️ E-Commerce Purchase Prediction

Machine Learning untuk memprediksi apakah seorang user akan melakukan pembelian pada platform e-commerce berdasarkan perilaku browsing mereka.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📁 Struktur Proyek

```
├── ecommerce_purchase_prediction.ipynb   # Notebook Google Colab (training)
├── streamlit_app.py                      # Aplikasi Streamlit (deployment)
├── requirements.txt                      # Dependensi Python
├── model_artifacts/                      # Folder model (generate dari Colab)
│   ├── best_model.pkl                    # Model ML tersimpan
│   └── model_metadata.json              # Metadata & threshold
└── README.md
```

---

## 🚀 Cara Penggunaan

### Langkah 1: Training Model di Google Colab

1. Buka `ecommerce_purchase_prediction.ipynb` di [Google Colab](https://colab.research.google.com)
2. Upload file `training_sample.csv` ketika diminta
3. Jalankan semua cell (`Runtime > Run All`)
4. Download `model_artifacts.zip` yang otomatis ter-generate
5. Ekstrak zip tersebut ke folder proyek ini

### Langkah 2: Setup Lokal / GitHub

```bash
# Clone repo
git clone https://github.com/username/ecommerce-purchase-prediction.git
cd ecommerce-purchase-prediction

# Install dependencies
pip install -r requirements.txt

# Pastikan folder model_artifacts/ ada (dari step 1)

# Jalankan Streamlit
streamlit run streamlit_app.py
```

### Langkah 3: Deploy ke Streamlit Cloud

1. Push semua file ke GitHub (termasuk `model_artifacts/`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect ke GitHub repo Anda
4. Pilih `streamlit_app.py` sebagai main file
5. Deploy!

---

## 📊 Dataset

| Parameter       | Detail                    |
|-----------------|---------------------------|
| Jumlah baris    | 455,401 sesi user         |
| Fitur           | 23 fitur perilaku browsing |
| Target          | `ordered` (0 = tidak beli, 1 = beli) |
| Class imbalance | ~4.2% melakukan pembelian  |

### Fitur yang Digunakan

| Fitur                  | Deskripsi                        |
|------------------------|----------------------------------|
| basket_icon_click      | Klik ikon keranjang belanja      |
| basket_add_list        | Tambah produk dari halaman list  |
| basket_add_detail      | Tambah produk dari halaman detail|
| saw_checkout           | Melihat halaman checkout         |
| sign_in                | Melakukan login                  |
| returning_user         | User yang pernah berkunjung      |
| device_mobile/computer | Jenis perangkat yang digunakan   |
| ... dan 16 fitur lainnya |                                |

---

## 🤖 Model

Empat model dibandingkan:
- **Logistic Regression** (baseline)
- **Random Forest**
- **XGBoost**
- **LightGBM** ⭐ (biasanya terbaik)

### Teknik Mengatasi Class Imbalance
- SMOTE oversampling
- `scale_pos_weight` (XGBoost)
- `class_weight='balanced'`

### Feature Engineering
5 fitur turunan ditambahkan:
- `total_activity` — total semua interaksi
- `basket_intent` — niat memasukkan ke keranjang
- `checkout_intent` — niat menyelesaikan checkout
- `product_info_check` — memeriksa info produk
- `engagement_score` — engagement dengan promosi

---

## 💡 Insight Bisnis

1. **`saw_checkout`** adalah sinyal terkuat — prioritaskan notifikasi/diskon
2. **Add to basket tapi tidak checkout** = target retargeting terbaik
3. **Returning user** memiliki konversi lebih tinggi — investasikan di loyalty program
4. Optimalkan UX checkout di **mobile** device

---

## 🛠️ Tech Stack

- **Python** 3.10+
- **Scikit-learn** — preprocessing & model baseline
- **XGBoost / LightGBM** — gradient boosting
- **imbalanced-learn** — SMOTE
- **SHAP** — model explainability
- **Streamlit** — web app deployment
- **Plotly** — interactive visualization

---

## 📧 Kontak

Dibuat dengan ❤️ untuk deployment di Streamlit Cloud.
