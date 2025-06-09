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
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
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
    cm = np.array([[55, 5], [1, 57]])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Real (0)", "Fake (1)"],
                yticklabels=["Real (0)", "Fake (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("🎯 Akurasi Model")
    st.markdown("Model memiliki akurasi sebesar **95%**")

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
    tab1, tab2, = st.tabs(["📤 Upload CSV", "🔗 URL IG"])

    # === Tab 1: Upload CSV ===
    with tab1:
        st.markdown("### Upload file CSV")
        st.markdown("File CSV harus meliputi data berikut:")
        st.image(Image.open("gambar/data.png"), width=300)

        uploaded_file = st.file_uploader("Upload file CSV:", type=["csv"])
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            try:
                df_pred = df_input.copy()
                df_pred_scaled = scaler.transform(df_pred[features])
                pred = model.predict(df_pred_scaled)
                df_pred["predict"] = pred
                st.success("✅ Prediksi berhasil dilakukan")
                st.dataframe(df_pred)

                csv = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

    # === Tab 3: URL Instagram (RapidAPI - Fast Reliable Scraper) ===
    with tab2:
        st.markdown("### Masukkan Link Akun Instagram")
        ig_url = st.text_input("Contoh: https://www.instagram.com/username/")
        # User input manual jumlah postingan
        jumlah_post = st.number_input("Jumlah Postingan", min_value=0, max_value=10000, value=0, step=1)

        if st.button("Ambil Data & Prediksi"):
            try:
                username = ig_url.strip().split("/")[-2]

                # Gunakan API Piloterr
                url = "https://piloterr.com/api/v2/instagram/user/info"
                headers = {
                    "x-api-key": RAPIDAPI_KEY,  # pastikan RAPIDAPI_KEY berisi key dari Piloterr
                    "Content-Type": "application/json"
                }
                params = {"query": username}
                response = requests.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    raise Exception(f"API Error: {response.status_code}, {response.text}")
                data_user = response.json()

                # Mapping sesuai fitur
                data_instagram = {
                    "fullname words": len(data_user.get("name", "").split()),
                    "nums/length fullname": round(len(re.findall(r'\d', data_user.get("name", ""))) / max(1, len(data_user.get("name", ""))), 2),
                    "nums/length username": round(len(re.findall(r'\d', data_user.get("username", ""))) / max(1, len(data_user.get("username", ""))), 2),
                    "profile pic": 1 if data_user.get("avatar") else 0,
                    "name==username": 1 if data_user.get("name", "").lower() == data_user.get("username", "").lower() else 0,
                    "description length": len(data_user.get("description", "")),
                    "external URL": 1 if data_user.get("website") else 0,
                    "private": int(data_user.get("private", False)),
                    "#posts": jumlah_post,
                    "#followers": int(data_user.get("followers", 0)),
                    "#follows": int(data_user.get("following", 0)),
                }


                # Prediksi & tampilkan hasil
                df_link = pd.DataFrame([[data_instagram[feat] for feat in features]], columns=features)
                df_link_scaled = scaler.transform(df_link)
                pred_link = model.predict(df_link_scaled)[0]

                if pred_link == 0:
                    st.success(f"✅ Akun **{username}** diprediksi sebagai: **Akun Asli**")
                else:
                    st.error(f"⚠️ Akun **{username}** diprediksi sebagai: **Akun Palsu**")

                st.markdown("#### 📊 Data Fitur dari Akun:")
                st.json(data_instagram)

            except Exception as e:
                st.error(f"Gagal mengambil data dari akun: {e}")
