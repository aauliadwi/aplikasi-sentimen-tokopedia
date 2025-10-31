import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.set_page_config(
    page_title="Sistem Klasifikasi Ulasan Tokopedia",
    page_icon="📊",
    layout="centered"
)

# Pra-pemrosesan Data
def clean_text_ml(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|\bRT\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_light(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|\bRT\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Memuat artefak models
@st.cache_resource(show_spinner=True)
def load_artifacts():
    mdir = Path("models")
    if not mdir.exists():
        raise FileNotFoundError("Folder 'models' tidak ditemukan.")

    tfidf = joblib.load(mdir / "tfidf.pkl")
    svd = joblib.load(mdir / "svd.pkl")
    nb = joblib.load(mdir / "nb.pkl")
    lgbm = joblib.load(mdir / "lgbm.pkl")

    with open(mdir / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    meta = joblib.load(mdir / "gru_meta.pkl")
    MAXLEN = int(meta.get("MAXLEN", 150))
    BEST_THR = float(meta.get("best_thr", 0.5))

    sm_dir = mdir / "gru_savedmodel"
    if not sm_dir.exists():
        raise FileNotFoundError(
            "Model GRU SavedModel tidak ditemukan di 'models/gru_savedmodel/'."
        )
    sm = tf.saved_model.load(str(sm_dir))
    sig = sm.signatures.get("serving_default", None)
    if sig is None:
        sig = list(sm.signatures.values())[0]

    return tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, sig


# Prediksi per metode
def predict_nb_lgbm(text: str, tfidf, svd, nb, lgbm):
    clean = clean_text_ml(text)
    X = tfidf.transform([clean])
    X_svd = svd.transform(X)
    nb_label = int(nb.predict(X)[0])
    lgb_label = int(lgbm.predict(X_svd)[0])
    nb_proba = float(nb.predict_proba(X)[0][1])
    lgb_proba = float(lgbm.predict_proba(X_svd)[0][1])
    return (nb_label, nb_proba), (lgb_label, lgb_proba)

def predict_gru(text: str, tok, MAXLEN: int, BEST_THR: float, gru_sig):
    clean = clean_light(text)
    seq = tok.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    try:
        inp = tf.convert_to_tensor(pad, dtype=tf.int32)
        out = gru_sig(inp)
    except Exception:
        inp = tf.convert_to_tensor(pad, dtype=tf.float32)
        out = gru_sig(inp)
    y = list(out.values())[0]
    proba = float(tf.reshape(y, [-1])[0].numpy())
    label = 1 if proba >= BEST_THR else 0
    return label, proba

# Tampilan Hasil
def render_row(model_name: str, proba: float, label: int):
    try:
        proba = float(proba)
    except Exception:
        proba = 0.0
    if np.isnan(proba):
        proba = 0.0
    frac = max(0.0, min(proba, 1.0))
    pct = int(round(frac * 100))

    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        st.write(f"**{model_name}**")
    with col2:
        st.progress(frac)
        st.caption(f"{pct}%")
    with col3:
        text = "Positif" if int(label) == 1 else "Negatif"
        color = "#22c55e" if int(label) == 1 else "#ef4444"
        st.markdown(
            f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;"
            f"background:{color};color:white;font-weight:600;text-align:center;'>{text}</div>",
            unsafe_allow_html=True,
        )

# Tampilan Aplikasi

# Sidebar
st.sidebar.markdown("### 📊 Selamat Datang!")
page = st.sidebar.radio(
    label="",
    options=["Beranda", "Analisis", "Tabel Analisis", "Tentang"],
    index=0,
    label_visibility="collapsed",
)

# Memuat artifacts
if "artifacts" not in st.session_state:
    try:
        st.session_state.artifacts = load_artifacts()
    except Exception as e:
        st.error("Gagal memuat artefak model. Pastikan folder `models/` lengkap.")
        st.code(str(e))
        st.stop()

tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, GRU_SIG = st.session_state.artifacts

# Halaman Beranda
if page == "Beranda":
    st.markdown("## Perancangan Sistem Klasifikasi Ulasan pada Aplikasi Tokopedia Menggunakan Metode Naïve Bayes, LightGBM, dan GRU")
    st.markdown(
        """
        Aplikasi ini dirancang untuk menganalisis sentimen dari ulasan pengguna Tokopedia, apakah bersifat positif atau negatif.
        - **Naive Bayes**: Metode sederhana yang menghitung seberapa besar kemungkinan suatu ulasan bersifat positif atau negatif berdasarkan kata yang digunakan.  
        
        - **LightGBM**: Sistem pembelajaran mesin yang menggabungkan banyak keputusan kecil untuk mencapai hasil prediksi yang lebih baik.  
        
        - **GRU**: Jaringan saraf yang mampu mengenali makna dan alur kalimat, sehingga memberikan pemahaman konteks yang lebih mendalam.

        Hasil analisis ditampilkan dalam bentuk prediksi sentimen, grafik akurasi, dan perbandingan performa dari ketiga model.

        Gunakan menu di kiri untuk mencoba analisis atau melihat perbandingan performa.
        """
    )

# Halaman Analisis
elif page == "Analisis":
    text_input = st.text_area("📝 Masukkan ulasan:", height=130, placeholder="Tulis ulasan di sini...")
    analyze = st.button("Analisis Sentimen")

    if analyze:
        if not text_input.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            try:
                (nb_label, nb_proba), (lgb_label, lgb_proba) = predict_nb_lgbm(
                    text_input, tfidf, svd, nb, lgbm
                )
                gru_label, gru_proba = predict_gru(
                    text_input, tok, MAXLEN, BEST_THR, GRU_SIG
                )
            except Exception as e:
                st.error("Terjadi error saat prediksi.")
                st.code(str(e))
                st.stop()

            st.subheader("Hasil Prediksi")
            render_row("Naive Bayes", nb_proba, nb_label)
            render_row("LightGBM", lgb_proba, lgb_label)
            render_row("GRU", gru_proba, gru_label)


# Halaman Tabel Analisis
elif page == "Tabel Analisis":
    st.subheader("📈 Grafik Performa Model (F1-Score)")
    df_ringkas = pd.DataFrame(
        [
            {"Model": "GRU", "Accuracy": 0.8852, "Precision": 0.8252, "Recall": 0.7984, "F1-Score": 0.8116},
            {"Model": "Naive Bayes", "Accuracy": 0.8841, "Precision": 0.8646, "Recall": 0.7420, "F1-Score": 0.7986},
            {"Model": "LightGBM", "Accuracy": 0.8585, "Precision": 0.7880, "Recall": 0.7429, "F1-Score": 0.7648},
        ]
    )
    df_ringkas_sorted = df_ringkas.sort_values(by="F1-Score", ascending=False, ignore_index=True)
    _df_plot = df_ringkas_sorted[["Model", "F1-Score"]].copy()
    st.bar_chart(data=_df_plot.set_index("Model"))

    st.subheader("📊 Tabel Perbandingan dengan F1-Score")
    st.dataframe(df_ringkas_sorted, use_container_width=True)

    st.subheader("⚙️ Tabel Efisiensi Komputasi")
    df_efisiensi = pd.DataFrame(
        [
            {"Model": "GRU", "Waktu Training (s)": 902.36, "Waktu Inferensi (ms)": 13.91},
            {"Model": "Naive Bayes", "Waktu Training (s)": 0.018,  "Waktu Inferensi (ms)": 0.002},
            {"Model": "LightGBM", "Waktu Training (s)": 44.37,   "Waktu Inferensi (ms)": 0.104},
        ]
    )
    df_efisiensi_sorted = df_efisiensi.sort_values(
        by=["Waktu Inferensi (ms)", "Waktu Training (s)"], ascending=[True, True], ignore_index=True
    )
    st.dataframe(df_efisiensi_sorted, use_container_width=True)

    combined = pd.merge(df_ringkas_sorted, df_efisiensi_sorted, on="Model", how="left")
    csv_all = combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Hasil Analisis (CSV)",
        csv_all,
        file_name="hasil_analisis.csv",
        mime="text/csv",
    )

# Halaman Tentang
elif page == "Tentang":
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini dibuat sebagai bagian dari tugas akhir dengan tujuan membangun sistem yang dapat menganalisis dan mengklasifikasikan sentimen ulasan pengguna Tokopedia secara otomatis.

        Panduan penggunaan lengkap tersedia dalam bentuk Manual Book yang dapat diunduh melalui tombol di bawah.
        """
    )
    pdf_path = Path("assets/Manual_Book.pdf")  # sesuaikan path & nama file
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Unduh Buku Manual (PDF)",
                data=f.read(),
                file_name="Manual_Book.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("File manual belum ditemukan di `assets/Manual_Book.pdf`.")
