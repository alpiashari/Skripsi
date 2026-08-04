import streamlit as st
import pandas as pd
from PIL import Image
import os


# CONFIG

st.set_page_config(
    page_title="KNN Traffic Severity Prediction",
    layout="wide"
)


# LOAD DATASET

@st.cache_data
def load_data():
    df_asli = pd.read_csv("dataset/RTA Dataset.csv", sep=";")
    df_indonesia = pd.read_csv("dataset/RTA_Dataset_Indonesia.csv", sep=";")
    df_final = pd.read_csv("dataset/Data_final.csv", sep=";")

    return df_asli, df_indonesia, df_final

df_asli, df_indonesia, df_final = load_data()

def preprocessing():
    df_sebelum_null = pd.read_excel("Preprocessing/Sebelum_Null.xlsx", )
    df_sesudah_null = pd.read_excel("Preprocessing/Sesudah_Null.xlsx")
    df_transformasi = pd.read_csv("Preprocessing/Transformasi.csv", sep=";")

    return df_sebelum_null, df_sesudah_null, df_transformasi

df_sebelum_null, df_sesudah_null, df_transformasi = preprocessing()

def hasil_pengujian():
    df_kfold_non_smote = pd.read_excel("pengujian/Kfold_non_smote.xlsx")
    df_kfold_smote = pd.read_excel("pengujian/Kfold_smote.xlsx")
    df_k_non_smote = pd.read_excel("pengujian/k_non_smote.xlsx")
    df_k_smote = pd.read_excel("pengujian/k_smote.xlsx")
    df_jarak_non_smote = pd.read_excel("pengujian/metric_non_smote.xlsx")
    df_jarak_smote = pd.read_excel("pengujian/metric_smote.xlsx")
    df_bobot_non_smote = pd.read_excel("pengujian/weight_non_smote.xlsx")
    df_bobot_smote = pd.read_excel("pengujian/weight_smote.xlsx")

    return df_kfold_non_smote, df_kfold_smote, df_k_non_smote, df_k_smote, df_jarak_non_smote, df_jarak_smote, df_bobot_non_smote, df_bobot_smote

df_kfold_non_smote, df_kfold_smote, df_k_non_smote, df_k_smote, df_jarak_non_smote, df_jarak_smote, df_bobot_non_smote, df_bobot_smote = hasil_pengujian()

# SIDEBAR

st.sidebar.title("KNN Traffic Severity")
menu = st.sidebar.radio(
    "Menu",
    ["Dataset", "Preprocessing", "EDA", "Hasil Pengujian", "Evaluasi Model", "Tentang"]
)

# HALAMAN DATASET

if menu == "Dataset":
    st.title("📊 Dataset")

    tab1, tab2, tab3 = st.tabs(["Dataset Asli", "Dataset Indonesia", "Data Final"])

    with tab1:
        st.dataframe(df_asli, width="stretch")

    with tab2:
        st.dataframe(df_indonesia, width="stretch")

    with tab3:
        st.dataframe(df_final, width="stretch")


# HALAMAN Preprocessing

elif menu == "Preprocessing":
    st.title("🔄 Preprocessing")

    col1, col2 = st.columns(2)

    #Kolom keSatu Penanganan Nilai Null
    with col1:
        tab1, tab2 = st.tabs(["Sebelum Null", "Sesudah Null"])

        with tab1:
            st.dataframe(df_sebelum_null, width="stretch")

        with tab2:
            st.dataframe(df_sesudah_null, width="stretch")

    # Kolom keDua Penanganan Outlier
    tab1, tab2 = st.tabs(["Sebelum Penanganan Outlier", "Sesudah Penanganan Outlier"])

    with tab1:
        st.image("images/Sebelum_Penanganan_Outlier.png", width=630)

    with tab2:
        st.image(
            "images/Sesudah_Penanganan_Outlier.png", width=630)

    #Kolom keTiga Transformasi
    col_bawah1, col_bawah2 = st.columns(2)

    with col_bawah1:
        tab1, tab2 = st.tabs(["Sebelum Transformasi", "Sesudah Transformasi"])

        with tab1:
            st.dataframe(df_asli, width="stretch")

        with tab2:
            st.dataframe(df_transformasi, width="stretch")

    # Kolom keEmpat Normalisasi
    col_bawah1, col_bawah2 = st.columns(2)

    with col_bawah1:
        tab1, tab2 = st.tabs(["Sebelum Normalisasi", "Sesudah Normalisasi"])

        with tab1:
            st.dataframe(df_transformasi, width="stretch")

        with tab2:
            st.dataframe(df_final, width="stretch")

    
    
# HALAMAN EDA

elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Kelas Target")
        st.image(
            "images/Distribusi Accident.png",
            width="stretch"
        )

    with col2:
        st.subheader("Distribusi Keseluruhan")
        st.image(
            "images/Distribusi Keseluruhan.png",
            width="stretch"
        )

    # Baris kedua
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Korelasi")
        st.image(
            "images/Distribusi Korelasi.png",
            width="stretch"
        )

    with col4:
        st.subheader("Grafik Batang")
        st.image(
            "images/Distribusi_Accident_Bar.png",
            width="stretch"
        )


# Halaman Hasil Pengujian

elif menu == "Hasil Pengujian":
    st.title("📊 Hasil Pengujian")

    col1, col2 = st.columns(2)

    #Kolom keSatu Pengujian
    with col1:
        tab1, tab2 = st.tabs(["Kfold Non Smote", "Kfoold Smote"])

        with tab1:
            st.dataframe(df_kfold_non_smote, width="stretch")
            st.caption("Best Kfold Non Smote : 3")

        with tab2:
            st.dataframe(df_kfold_smote, width="stretch")
            st.caption("Best Kfold Smote : 5")

    #Kolom keDua Pengujian
    col_bawah1, col_bawah2 = st.columns(2)

    with col_bawah1:
        tab1, tab2 = st.tabs(["K Non Smote", "K Smote"])

        with tab1:
            st.dataframe(df_k_non_smote, width="stretch")
            st.caption("Best K Non Smote : 5")

        with tab2:
            st.dataframe(df_k_smote, width="stretch")
            st.caption("Best K Smote : 3")

    #Kolom keTiga Pengujian
    col_bawah1, col_bawah2 = st.columns(2)

    with col_bawah1:
        tab1, tab2 = st.tabs(["Metric Non Smote", "Metric Smote"])

        with tab1:
            st.dataframe(df_jarak_non_smote, width="stretch")
            st.caption("Best Metric Non Smote : Euclidean")

        with tab2:
            st.dataframe(df_jarak_smote, width="stretch")
            st.caption("Best Metric Smote : Manhattan")

        #Kolom keEmpat Pengujian
    col_bawah1, col_bawah2 = st.columns(2)

    with col_bawah1:
        tab1, tab2 = st.tabs(["Weight Non Smote", "Weight Smote"])

        with tab1:
            st.dataframe(df_bobot_non_smote, width="stretch")
            st.caption("Best Weight Non Smote : Uniform")

        with tab2:
            st.dataframe(df_bobot_smote, width="stretch")
            st.caption("Best Weight Smote : Distance")


# HALAMAN EVALUASI MODEL

elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model")


    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        base_dir = os.path.dirname(__file__)
        img_path_dummy = os.path.join(base_dir, "images", "Confusion_matrix_dummy.png")
        img_path_knn_manual_non_smote = os.path.join(base_dir, "images", "Confusion_matrix_knn_manual_non_smote.png")
        img_path_knn_manual_smote = os.path.join(base_dir, "images", "Confusion_matrix_knn_manual_smote.png")
        st.image(img_path_dummy, caption="Dummy Classifier", width="stretch")
        st.image(img_path_knn_manual_non_smote, caption="Confusion Matrix KNN Manual (Non SMOTE)", width="stretch")
        st.image(img_path_knn_manual_smote, caption="Confusion Matrix KNN Manual (SMOTE)", width="stretch")

    with col2:
        st.image("images/Confusion_matrix_grid_non_smote.png", caption="Confusion Matrix Grid Search CV (Non SMOTE)", width="stretch")
        st.image("images/Confusion_matrix_grid_smote.png", caption="Confusion Matrix Grid Search CV (SMOTE)", width="stretch")


    # CLASSIFICATION REPORT

    st.subheader("Dummy Classifier")

    data_knn_manual_non_smote = {
        "Kelas": ["(0) Slight Injury", "(1) Serious Injury", "(2) Fatal Injury"],
        "Precision": [0.0, 0.0, 0.84686],
        "Recall": [0.0, 0.0, 1.00000],
        "F1-Score": [0.0, 0.0, 0.91708],
    }

    df_knn_manual_non_smote_report = pd.DataFrame(data_knn_manual_non_smote)
    st.dataframe(df_knn_manual_non_smote_report, width="stretch")
    
    accuracy = 0.84686

    st.subheader("Accuracy")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")

    st.subheader("Classification Report KNN Manual (Non SMOTE)")

    data_knn_manual_non_smote = {
        "Kelas": ["(0) Slight Injury", "(1) Serious Injury", "(2) Fatal Injury"],
        "Precision": [0.25000, 0.16667, 0.84747],
        "Recall": [0.06250, 0.02395, 0.97727],
        "F1-Score": [0.10000, 0.04188, 0.90776],
    }

    df_knn_manual_non_smote_report = pd.DataFrame(data_knn_manual_non_smote)
    st.dataframe(df_knn_manual_non_smote_report, width="stretch")
    
    accuracy = 0.83180

    st.subheader("Accuracy")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")
    
    
    st.subheader("Classification Report KNN Manual (SMOTE)")

    data_knn_manual_smote = {
        "Kelas": ["(0) Slight Injury", "(1) Serious Injury", "(2) Fatal Injury"],
        "Precision": [0.07895, 0.13169, 0.84573],
        "Recall": [0.18750, 0.19162, 0.76383],
        "F1-Score": [0.11111, 0.15610, 0.80270],
    }
    
    df_knn_manual_smote_report= pd.DataFrame(data_knn_manual_smote)
    st.dataframe(df_knn_manual_smote_report, width="stretch")
    
    accuracy = 0.67615

    st.subheader("Accuracy")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")
    
    st.subheader("Classification Report KNN Grid Search CV (Non SMOTE)")

    data_grid_non_smote = {
        "Kelas": ["(0) Slight Injury", "(1) Serious Injury", "(2) Fatal Injury"],
        "Precision": [0.27778, 0.25000, 0.85426],
        "Recall": [0.09804, 0.08797, 0.95405],
        "F1-Score": [0.14493, 0.13015, 0.90140],
    }

    df_grid_non_smote = pd.DataFrame(data_grid_non_smote)
    st.dataframe(df_grid_non_smote, width="stretch")
    
    accuracy = 0.82190

    st.subheader("Accuracy")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")
    
    st.subheader("Classification Report KNN Grid Search CV (SMOTE)")
    
    data_grid_smote= {
        "Kelas": ["(0) Slight Injury", "(1) Serious Injury", "(2) Fatal Injury"],
        "Precision": [0.10526, 0.17255, 0.86079],
        "Recall": [0.25806, 0.26347, 0.76680],
        "F1-Score": [0.14953, 0.20853, 0.81108],
    }

    df_grid_smote = pd.DataFrame(data_grid_smote)
    st.dataframe(df_grid_smote, width="stretch")
    
    accuracy = 0.68983

    st.subheader("Accuracy")
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")


    # ===========================
    # PERBANDINGAN
    # ===========================

    st.subheader("Perbandingan Performa Seluruh Model")
    
    data_seluruh_model= {
        "Metode": ["Dummy Baseline", "Manual KNN Non-SMOTE", "Manual KNN SMOTE", "GridSearchCV Non-SMOTE", "GridSearchCV SMOTE"],
        "Accuracy": [0.846862, 0.831799, 0.676151, 0.821904, 0.689828],
        "Precision": [0.28229, 0.42138, 0.35212, 0.46068, 0.37953],
        "Recall": [0.33333, 0.35457, 0.38098, 0.38002, 0.42945],
        "F1-Score": [0.30569, 0.34988, 0.35664, 0.39216, 0.38971],
    }

    df_seluruh_model = pd.DataFrame(data_seluruh_model)
    # st.dataframe(df_seluruh_model, width="stretch")

    st.dataframe(
    df_seluruh_model.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}"
    }),width="stretch")

    st.subheader("Perbandingan DUMMY, SMOTE, dan Non-SMOTE")

    st.image("images/perbandingan_hasil_semua_model.png", width="stretch")

# ===============================
# HALAMAN TENTANG
# ===============================
else:
    st.title("📌 Tentang")
    st.write("""
    Aplikasi ini digunakan untuk memprediksi tingkat keparahan kecelakaan lalu lintas
    menggunakan algoritma K-Nearest Neighbor (KNN).
    
    Dataset diambil dari Kaggle:
    Road Traffic Severity Classification.
    """)

    # python -m streamlit run tampilan\tampilan.py
