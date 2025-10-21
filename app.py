# app.py
import json
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

# ==============
# Util: Cleaning
# ==============
def clean_text_ml(s: str) -> str:
    """Pembersihan untuk model klasik (TF-IDF/LGBM/NB)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|\bRT\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_light(s: str) -> str:
    """Pembersihan ringan untuk GRU (samakan dengan yang dipakai saat training)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|\bRT\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ==================
# Load All Artefacts
# ==================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    mdir = Path("models")
    if not mdir.exists():
        raise FileNotFoundError("Folder 'models' tidak ditemukan.")

    # Klasik
    tfidf = joblib.load(mdir / "tfidf.pkl")
    svd = joblib.load(mdir / "svd.pkl")
    nb = joblib.load(mdir / "nb.pkl")
    lgbm = joblib.load(mdir / "lgbm.pkl")

    # Tokenizer dari JSON string (bukan pickle)
    with open(mdir / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # Meta GRU
    meta = joblib.load(mdir / "gru_meta.pkl")
    MAXLEN = int(meta.get("MAXLEN", 150))
    BEST_THR = float(meta.get("best_thr", 0.5))

    # ==== GRU: SavedModel only ====
    sm_dir = mdir / "gru_savedmodel"
    if not sm_dir.exists():
        raise FileNotFoundError(
            "Model GRU SavedModel tidak ditemukan di 'models/gru_savedmodel/'. "
            "Simpan model dalam format SavedModel atau sesuaikan loader."
        )

    sm = tf.saved_model.load(str(sm_dir))
    sig = sm.signatures.get("serving_default", None)
    if sig is None:
        # fallback: ambil signature pertama yang tersedia
        sig = list(sm.signatures.values())[0]

    return tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, sig


# ======================
# Prediksi per kategori
# ======================
def predict_nb_lgbm(text: str, tfidf, svd, nb, lgbm):
    """Kembalikan (label, proba) untuk NB & LGBM."""
    clean = clean_text_ml(text)
    X = tfidf.transform([clean])
    X_svd = svd.transform(X)

    nb_label = int(nb.predict(X)[0])
    lgb_label = int(lgbm.predict(X_svd)[0])

    nb_proba = float(nb.predict_proba(X)[0][1])  # proba kelas POSITIF
    lgb_proba = float(lgbm.predict_proba(X_svd)[0][1])

    return (nb_label, nb_proba), (lgb_label, lgb_proba)


def predict_gru(text: str, tok, MAXLEN: int, BEST_THR: float, gru_sig):
    """Kembalikan (label, proba) untuk GRU via SavedModel signature.

    Catatan: input default Embedding adalah int32. Kita pakai int32 dulu,
    lalu fallback ke float32 jika signature model ternyata meminta float32.
    """
    clean = clean_light(text)
    seq = tok.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    # Coba sebagai int32 terlebih dahulu
    try:
        inp = tf.convert_to_tensor(pad, dtype=tf.int32)
        out = gru_sig(inp)
    except Exception:
        # Fallback: beberapa signature custom mungkin minta float32
        inp = tf.convert_to_tensor(pad, dtype=tf.float32)
        out = gru_sig(inp)

    # Ambil tensor keluaran (first value di dict)
    y = list(out.values())[0]
    proba = float(tf.reshape(y, [-1])[0].numpy())

    label = 1 if proba >= BEST_THR else 0
    return label, proba


# ===== Util tampilan hasil satu baris (progress + badge) =====
def render_row(model_name: str, proba: float, label: int):
    """Render satu baris hasil prediksi dengan progress bar dan badge label."""
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
            (
                "<div style='display:inline-block;padding:6px 10px;border-radius:8px;"
                f"background:{color};color:white;font-weight:600;text-align:center;'>{text}</div>"
            ),
            unsafe_allow_html=True,
        )


# =========
# UI (App)
# =========
st.set_page_config(page_title="Klasifikasi Sentimen Tokopedia", page_icon="📊", layout="centered")

st.title("📊 Klasifikasi Sentimen Ulasan Tokopedia")
st.caption("Perbandingan hasil dari Naive Bayes, LightGBM, dan GRU.")

# =====================
# Sidebar & Navigation
# =====================
st.sidebar.header("📂 Menu")
page = st.sidebar.radio(
    "Navigasi",
    ["About", "Analisis", "Hasil Analisis"],
    index=1,
)


# =====================
# Halaman: About
# =====================
if page == "About":
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini membandingkan **tiga metode klasifikasi sentimen** untuk ulasan Tokopedia:
        - **Naive Bayes** (fitur TF‑IDF),
        - **LightGBM** (TF‑IDF + SVD/LSA),
        - **GRU** (tokenisasi & padding).

        **Cara pakai:**
        1. Buka tab **Analisis** untuk memasukkan teks ulasan.
        2. Tekan **Analisis Sentimen** untuk melihat prediksi tiap model.
        3. Buka **Hasil Analisis** untuk melihat tabel perbandingan metrik, efisiensi, dan grafik akurasi.
        4. Hasil analisis dapat di unduh untuk laporan
        """
    )
    # with st.expander("Detail Teknis"):
    #     st.write(
    #         f"Tokenizer MAXLEN: `{MAXLEN}` • Ambang GRU (best_thr): `{BEST_THR:.2f}` • Model GRU: SavedModel signature"
    #     )

# Load artefak
try:
    tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, GRU_SIG = load_artifacts()
except Exception as e:
    st.error("Gagal memuat artefak.")
    st.code(str(e))
    st.stop()

# =====================
# Halaman: Analisis
# =====================
if page == "Analisis":
    # Input
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

            st.subheader("🔍 Hasil Prediksi")
            render_row("Naive Bayes", nb_proba, nb_label)
            render_row("LightGBM", lgb_proba, lgb_label)
            render_row("GRU", gru_proba, gru_label)


# =====================
# Halaman: Hasil Analisis — Berdasarkan F1-Score
# =====================
if page == "Hasil Analisis":
    st.subheader("📈 Grafik Performa Model (F1-Score)")

    # Data contoh (silakan ganti dengan hasil evaluasi kamu)
    df_ringkas = pd.DataFrame(
        [
            {"Model": "GRU", "Accuracy": 0.8814, "Precision": 0.8094, "Recall": 0.8072, "F1-Score": 0.8083},
            {"Model": "Naive Bayes", "Accuracy": 0.8842, "Precision": 0.8646, "Recall": 0.7421, "F1-Score": 0.7987},
            {"Model": "LightGBM", "Accuracy": 0.8556, "Precision": 0.7832, "Recall": 0.7377, "F1-Score": 0.7597},
        ]
    )

    # Urutkan dari terbaik ke terendah berdasarkan F1-Score
    df_ringkas_sorted = df_ringkas.sort_values(by="F1-Score", ascending=False, ignore_index=True)

    # Grafik bar F1-Score (urut sesuai F1 terbaik)
    _df_plot = df_ringkas_sorted[["Model", "F1-Score"]].copy()
    st.bar_chart(data=_df_plot.set_index("Model"))

    st.subheader("📊 Tabel Perbandingan dengan F1-Score")
    st.dataframe(df_ringkas_sorted, use_container_width=True)

    # Tabel Efisiensi Komputasi
    st.subheader("⚙️ Tabel Efisiensi Komputasi")
    df_efisiensi = pd.DataFrame(
        [
            {"Model": "GRU", "Waktu Training (s)": 326.45, "Waktu Inferensi (ms)": 4.78},
            {"Model": "Naive Bayes", "Waktu Training (s)": 2.51, "Waktu Inferensi (ms)": 0.35},
            {"Model": "LightGBM", "Waktu Training (s)": 8.79, "Waktu Inferensi (ms)": 0.92},
        ]
    )

    # Definisi efisiensi: inferensi cepat lebih baik, lalu training cepat lebih baik
    df_efisiensi_sorted = df_efisiensi.sort_values(
        by=["Waktu Inferensi (ms)", "Waktu Training (s)"], ascending=[True, True], ignore_index=True
    )

    st.dataframe(df_efisiensi_sorted, use_container_width=True)

    # Gabungkan dua tabel menjadi satu CSV
    combined = pd.merge(df_ringkas_sorted, df_efisiensi_sorted, on="Model", how="left")
    csv_all = combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Analisis (CSV)",
        csv_all,
        file_name="hasil_analisis.csv",
        mime="text/csv",
    )