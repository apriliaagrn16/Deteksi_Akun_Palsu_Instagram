# ==== IMPORT LIBRARY ====
import os
import re
import joblib
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

RAPIDAPI_KEY = st.secrets["PILOTERR_API_KEY"]

# ==== KONFIGURASI STREAMLIT ====
st.set_page_config(page_title="Prediksi Akun Palsu", page_icon="🔍", layout="wide")

# ==== SIDEBAR ====
with st.sidebar:
    page = option_menu("Predict Fake Account Instagram", ["Home", 'Prediction'], 
        icons=['house', 'gear'], menu_icon="eye", default_index=0)

# ==== LOAD MODEL & FITUR ====
def download_file_from_github(raw_url, local_filename):
    if not os.path.exists(local_filename):
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Gagal download {raw_url}: {response.status_code}")

# Ganti dengan user/repo kamu
base_url = "https://raw.githubusercontent.com/apriliaagrn16/dataset-model-fake-instagram/main/"

# Daftar file yang ingin didownload
file_list = ["random_forest_model.pkl", "features.pkl"]

# Download semua file
for file_name in file_list:
    download_file_from_github(base_url + file_name, file_name)

# Load model dan fitur
model = joblib.load("random_forest_model.pkl")
features = joblib.load("features.pkl")


# ==== PAGE: HOME ====
if page == "Home":
    st.title("🏠 Halaman Utama")

    st.subheader("🧠 Tentang Model")
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest Classifier** untuk memprediksi akun Instagram palsu.
    Model dilatih berdasarkan fitur numerik seperti:
    - Rasio angka pada username dan nama
    - Jumlah kata pada nama lengkap
    - Panjang deskripsi
    - Jumlah followers, following, postingan
    - Adanya URL, status privat, dan foto profil
    """)

    st.subheader("🔍 Confusion Matrix (Hasil dari Training)")
    cm = np.array([[708, 42], [6, 744]])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("🎯 Akurasi Model")
    st.markdown("Model memiliki akurasi sebesar **96,8%**")

    st.subheader("💡 Feature Importance")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax_fi)
    plt.title("Feature Importance dari Model Random Forest")
    st.pyplot(fig_fi)

# ==== PAGE: DETECTION ====
elif page == "Prediction":
    st.title("🧪 Prediksi Akun Instagram Palsu")
    tab1, tab2 = st.tabs(["🔗 URL IG", "📤 Upload CSV"])

    # === Tab 1: URL Instagram ===
    with tab1:
        st.markdown("### Masukkan Link Akun Instagram")
        ig_url = st.text_input("Contoh: https://www.instagram.com/username/")
        jumlah_post = st.number_input("Jumlah Postingan", min_value=0, max_value=10000, value=0, step=1)
        jumlah_mutual = st.number_input("Jumlah Mutual Friends", min_value=0, max_value=10000, value=0, step=1)
        jumlah_threads = st.number_input("Jumlah Threads", min_value=0, max_value=10000, value=0, step=1)

        if st.button("Ambil Data & Prediksi"):
            try:
                username = ig_url.strip().split("/")[-2]
                url = "https://piloterr.com/api/v2/instagram/user/info"
                headers = {
                    "x-api-key": RAPIDAPI_KEY,
                    "Content-Type": "application/json"
                }
                params = {"query": username}
                response = requests.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    raise Exception(f"API Error: {response.status_code}, {response.text}")
                data_user = response.json()
                st.subheader("📦 Isi Response JSON")
                st.json(data_user)


                data_instagram = {
                "Followers": int(data_user.get("followers", 0)),
                "Following": int(data_user.get("following", 0)),
                "Following/Followers": round(data_user.get("following", 0) / max(1, data_user.get("followers", 1)), 2),
                "Posts": jumlah_post,
                "Posts/Followers": round(jumlah_post / max(1, data_user.get("followers", 1)), 2),
                "Bio": len(data_user.get("description", "")),  # pakai panjang string
                "Profile Picture": 1 if data_user.get("avatar") else 0,
                "External Link": 1 if data_user.get("website") else 0,
                "Mutual Friends": jumlah_mutual,
                "Threads": jumlah_threads
            }

                df_link = pd.DataFrame([[data_instagram[feat] for feat in features]], columns=features)
                pred_link = model.predict(df_link)[0]

                if pred_link == 0:
                    st.success(f"✅ Akun **{username}** diprediksi sebagai: **Akun Asli**")
                else:
                    st.error(f"⚠️ Akun **{username}** diprediksi sebagai: **Akun Palsu**")

                st.markdown("#### 📊 Data Fitur dari Akun:")
                st.json(data_instagram)

            except Exception as e:
                st.error(f"Gagal mengambil data dari akun: {e}")

    # === Tab 2: Upload CSV ===
    with tab2:
        st.markdown("### Upload file CSV")
        st.markdown("File CSV harus meliputi data berikut:")
        st.image(Image.open("gambar/data.png"), width=300)

        uploaded_file = st.file_uploader("Upload file CSV:", type=["csv"])
        
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah")
            st.dataframe(df_input)

            if st.button("🔍 Lakukan Prediksi"):
                try:
                    df_pred = df_input.copy()
                    pred = model.predict(df_pred[features])
                    df_pred["predict"] = pred
                    st.success("✅ Prediksi berhasil dilakukan")
                    st.dataframe(df_pred)

                    csv = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")