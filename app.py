import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# === Load model ===
@st.cache_resource
def load_model():
    model_path = './outputs/best_rf_smote.pkl'  # Pastikan path benar
    return joblib.load(model_path)

# Memanggil fungsi untuk memuat model
model = load_model()


# === Header dan deskripsi utama ===
st.title("🏥 Prediksi Kebutuhan ICU untuk Pasien Prioritas 1 IGD yang Menjalani Operasi Emergensi")
st.subheader("Alat Prediksi untuk Membantu Keputusan Alokasi Sumber Daya ICU di Rumah Sakit")

st.write(
    "Website ini dikembangkan untuk membantu memprediksi kebutuhan perawatan ICU atau HCU pada pasien. "
    "Model prediksi yang digunakan didasarkan pada data profil pasien, termasuk kondisi klinis dan tanda-tanda vital seperti tekanan darah dan laju pernapasan. "
    "Alat ini dirancang untuk memberikan informasi yang berguna bagi dokter dalam pengambilan keputusan terkait alokasi sumber daya ICU."
)

st.markdown("---")

# === Input data pasien ===
st.subheader("🩺 Masukkan Data Pasien")

# Pilihan nilai kesadaran (1–6 dengan label klinis)
kesadaran_label = {
    "1 - Composmentis": 1,
    "2 - Somnolen": 2,
    "3 - Sopor (termasuk kondisi sebelum sedasi)": 3,
    "4 - SemiKoma": 4,
    "5 - Koma": 5,
    "6 - Apatis": 6
}
selected_state = st.selectbox("🧠 Tingkat Kesadaran Pasien:", list(kesadaran_label.keys()))
nilai_kesadaran = kesadaran_label[selected_state]

# Input MAP dan RR
map_value = st.number_input("MAP (Mean Arterial Pressure, mmHg)", min_value=10.0, max_value=200.0, value=80.0)
rr_value = st.number_input("Respiratory Rate (x/menit)", min_value=5.0, max_value=60.0, value=20.0)

st.caption("Nilai kesadaran mengikuti skala klinis: 1=Composmentis hingga 6=Apatis")

# === Hitung fitur model ===
kesadaran_MAP_ratio = nilai_kesadaran / (map_value + 1)
kesadaran_RR_ratio = nilai_kesadaran / (rr_value + 1)

input_data = pd.DataFrame([{
    'kesadaran_MAP_ratio': kesadaran_MAP_ratio,
    'kesadaran_RR_ratio': kesadaran_RR_ratio
}])

# === Prediksi ===
if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.markdown("### 🏥 Hasil Prediksi Tingkat Perawatan")

    if pred == 1:
        st.error(
            f"🚨 **Pasien diprediksi membutuhkan perawatan di ICU** "
        )
        st.caption(
            "Penilaian atau pengambilan keputusan dikembalikan kepada klinisi sebagai dokter penanggung jawab pasien selaku pengambil keputusan medis."
        )
    else:
        st.info(
            f"🩺 **Pasien diprediksi membutuhkan perawatan di HCU (non-ICU)** "
        )
        st.caption(
            "Penilaian atau pengambilan keputusan dikembalikan kepada klinisi sebagai dokter penanggung jawab pasien selaku pengambil keputusan medis."
        )

# === Penjelasan tambahan untuk laporan TA ===
st.markdown("---")
st.subheader("📊 Penjelasan Singkat Model")

st.write(
    """
    Model ini berfokus pada dua **variabel turunan yang paling signifikan** hasil analisis *feature importance*, yaitu:
    - **Rasio kesadaran terhadap MAP**, yang merepresentasikan keseimbangan antara tingkat kesadaran dan tekanan perfusi otak.
    - **Rasio kesadaran terhadap RR**, yang menggambarkan hubungan antara status neurologis dan fungsi respirasi.

    Kedua variabel ini terbukti paling informatif dalam membedakan pasien yang memerlukan **perawatan intensif (ICU)** dibandingkan yang cukup di **HCU**.
    Pendekatan ini menyoroti pentingnya kombinasi antara fungsi kesadaran dan status fisiologis dasar sebagai indikator risiko klinis akut.
    """
)

st.markdown("---")

# === Footer Informasi Akademik ===
st.caption("Model: Random Forest (SMOTE-balanced, 2-feature simplified version, berdasarkan variabel turunan klinis signifikan)")
st.markdown(
    """
    ---
    **Website prediksi ini dibuat untuk memenuhi Tugas Akhir Program Sarjana**  
    oleh **Rachel Jelita Putri Wendyansah**  
    Tahun: **2025**  
    Fakultas Kedokteran Universitas Brawijaya – RSSA Malang
    """
)

