import streamlit as st
import pandas as pd
import joblib
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Deteksi Akun Palsu", page_icon="🔍", layout="wide")
with st.sidebar:
    page = option_menu("Detec Fake Account Instagram", ["Home", 'Detection'], 
        icons=['house', 'gear'], menu_icon="eye", default_index=0)
    page

# Load model dan data
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

feature_defaults = {
    "profile pic": (0, 1, 1, 1),
    "name==username": (0, 1, 1, 0),
    "description length": (0, 500, 10, 100),
    "external URL": (0, 1, 1, 1),
    "private": (0, 1, 1, 0),
    "#posts": (0, 1000, 10, 50),
    "#followers": (0, 100000, 100, 1000),
    "#follows": (0, 10000, 10, 300),
}

if page == "Home":
    st.title("🏠 Halaman Utama")
   
    st.subheader("🧠 Tentang Model")
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest Classifier** untuk mendeteksi akun Instagram palsu.
    Model dilatih berdasarkan fitur numerik seperti:
    - Rasio angka pada username dan nama
    - Jumlah kata pada nama lengkap
    - Panjang deskripsi
    - Jumlah followers, following, postingan
    - Adanya URL, status privat, dan foto profil
    """)

    
    st.subheader("🔍 Confusion Matrix (Hasil dari Training)")
    st.markdown("Matriks ini menunjukkan jumlah prediksi yang benar dan salah untuk masing-masing kelas.")
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
    st.markdown(f"Model memiliki akurasi sebesar **95%**")

    st.subheader("💡 Feature Importance")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax_fi)
    plt.title("Feature Importance dari Model Random Forest")
    plt.xlabel("Tingkat Kepentingan")
    plt.ylabel("Fitur")
    st.pyplot(fig_fi)

elif page == "Detection":
    st.title("🧪 Deteksi Akun Instagram Palsu")

    tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Input Manual"])

    with tab1:
        st.markdown("### Upload file CSV")
        st.markdown("File CSV harus meliputi data berikut:")
        img = Image.open("gambar/data.png")
        st.image(img, width=300)

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

    with tab2:
        st.markdown("### Masukkan Data Akun Manual")

        username = st.text_input("Username")
        nama_lengkap = st.text_input("Nama Lengkap (digunakan untuk hitung otomatis rasio angka dan jumlah kata)")

        jumlah_kata = len(nama_lengkap.split())
        jumlah_angka_nama = len(re.findall(r'\d', nama_lengkap))
        panjang_nama = len(nama_lengkap)
        rasio_angka_nama = round(jumlah_angka_nama / panjang_nama, 2) if panjang_nama > 0 else 0.0

        jumlah_angka_username = len(re.findall(r'\d', username))
        panjang_username = len(username)
        rasio_angka_username = round(jumlah_angka_username / panjang_username, 2) if panjang_username > 0 else 0.0

        st.write(f"📌 Nama Lengkap → Jumlah Kata: {jumlah_kata}, Rasio Angka: {rasio_angka_nama}")
        st.write(f"📌 Username → Panjang: {panjang_username}, Rasio Angka: {rasio_angka_username}")

        manual_input = {
            "fullname words": jumlah_kata,
            "nums/length fullname": rasio_angka_nama,
            "nums/length username": rasio_angka_username,
        }

        for col in features:
            if col in manual_input:
                st.number_input(col, value=manual_input[col], disabled=True)
            else:
                min_val, max_val, step, default = feature_defaults.get(col, (0.0, 1.0, 0.1, 0.0))
                manual_input[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                    value=default,
                    key=col
                )
                if col == "profile pic":
                    st.caption("Masukkan **1** jika akun memiliki foto profil, dan **0** jika tidak ada.")
                elif col == "external URL":
                    st.caption("Masukkan **1** jika akun mencantumkan tautan di bio, dan **0** jika tidak ada.")
                elif col == "name==username":
                    st.caption("Masukkan **1** jika nama sama dengan username, dan **0** jika berbeda.")
                elif col == "private":
                    st.caption("Masukkan **1** jika akun bersifat privat, dan **0** jika akun publik.")

        if st.button("Prediksi Manual"):
            try:
                df_manual = pd.DataFrame([[manual_input[feat] for feat in features]], columns=features)
                df_manual_scaled = scaler.transform(df_manual)
                pred_manual = model.predict(df_manual_scaled)[0]
                if pred_manual == 0:
                    st.success("✅ Hasil Prediksi: **Akun Asli**")
                else:
                    st.error("⚠️ Hasil Prediksi: **Akun Palsu**")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
