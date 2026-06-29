# ============================================================
# BBC NEWS ARTICLE CATEGORIZER
# SAIA 2163 NLP FINAL PROJECT
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np         # <--- BENCANA MAUT FIXED!
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re                  # <--- UNTUK CLEANING PUNCTUATION


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BBC News Categorizer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# CUSTOM UI DESIGN
# ============================================================

st.markdown(
"""
<style>
/* Background */
.stApp { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); }

/* Title */
h1 { color:#667eea; font-weight:800; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #667eea, #764ba2); }
[data-testid="stSidebar"] * { color:white; }

/* Buttons */
.stButton > button {
    background:linear-gradient(90deg, #667eea, #764ba2);
    color:white; border-radius:20px; height:45px; font-weight:bold; border:none;
}
.stButton > button:hover { background:linear-gradient(90deg, #764ba2, #667eea); }

/* Metric Cards */
[data-testid="metric-container"] {
    background:white; padding:20px; border-radius:15px; box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

/* Text area */
textarea { border-radius:15px !important; border:2px solid #667eea !important; }
</style>
""",
unsafe_allow_html=True
)


# ============================================================
# LOAD MACHINE LEARNING MODELS & DATA
# ============================================================

@st.cache_resource
def load_models():
    nb_model = joblib.load("models/nb_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    tfidf = joblib.load("models/tfidf_ngram.pkl")
    return nb_model, svm_model, tfidf

nb_model, svm_model, tfidf = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("data/bbc-text.csv")

df = load_data()


# ============================================================
# SIDEBAR MENU (UNIFIED)
# ============================================================

st.sidebar.markdown(
"""
# 📰 SAIA 2163
## Article Categorizer
"""
)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / About",          # <--- HOME PAGE MASUK RUBRIK!
        "🔍 Text Analyzer",
        "📊 Data Explorer",
        "☁️ Word Cloud",
        "📈 Model Performance",
        "ℹ️ Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
"""
### 👨‍💻 Group Budu:
1. **MUHAMMAD AMIR REDZUAN** *(A24AI0112)*
2. **AMIR ASYRAAF** *(A24AI0016)*
3. **NAZMI BIN SIDEK** *(A24AI0069)*

---
"""
)

st.title("📰 BBC News Article Categorizer")


# ============================================================
# PAGE 0: HOME / ABOUT (NEW)
# ============================================================

if page == "🏠 Home / About":
    
    st.markdown(
    """
    <div style="background: linear-gradient(90deg,#667eea,#764ba2); padding:25px; border-radius:20px; color:white;">
        <h2 style="color:white;">🚀 Intelligent News Classification System</h2>
        <p style="color:white;font-size:18px;">
            An automated Natural Language Processing (NLP) web application designed to categorize standard English news articles into five distinct domains: <b>Business, Entertainment, Politics, Sport, and Tech.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(
        """
        ### 📌 The Problem We Solve
        In the era of mass information overload, manual editorial sorting of daily news is slow, expensive, and prone to human cognitive bias. **Group Budu** engineered this Machine Learning pipeline to ingest raw, unstructured article text and instantly route it to the correct publishing desk.
        """
        )
        
    with col_b:
        st.markdown(
        """
        ### 📖 How to Use This App
        1. Navigate to **🔍 Text Analyzer** using the left sidebar.
        2. Copy any standard news article (or paragraph) from the internet.
        3. Paste the text into the input box and click **Analyze Article**.
        4. Inspect the predicted category, confidence score, and trigger keywords!
        """
        )


# ============================================================
# PAGE 1: TEXT ANALYZER
# ============================================================

elif page == "🔍 Text Analyzer":

    st.header("🔍 News Article Classification")
    
    chosen_model = st.selectbox(
        "Select Classification Model:",
        ["⭐ Multinomial Naive Bayes (Highest Accuracy)", "Support Vector Machine (SVM)"]
    )

    text = st.text_area(
        "Enter news article:",
        height=250,
        placeholder="Paste article text here..."
    )

    if st.button("🚀 Analyze Article"):
        if text.strip():
            
            # Sanitasi input (buang titik koma supaya TF-IDF tak pening)
            cleaned_input = re.sub(r'[^\w\s]', '', text.lower()).strip()
            
            # Kalau lepas buang simbol tinggal kosong, kita tendang dia keluar awal-awal
            if not cleaned_input:
                st.warning("⚠️ No valid English words detected after clearing punctuation.")
                st.stop() # Matikan eksekusi kod ke bawah serta-merta!
                
            vector = tfidf.transform([cleaned_input])
            
            if vector.nnz == 0:
                st.warning("⚠️ No recognizable English news vocabulary detected. Please paste a valid news snippet.")
                st.stop()

            if "Naive Bayes" in chosen_model:
                prediction = nb_model.predict(vector)[0]
                confidence = nb_model.predict_proba(vector).max() * 100
            else:
                prediction = svm_model.predict(vector)[0]
                try:
                    confidence = svm_model.predict_proba(vector).max() * 100
                except AttributeError:
                    # Trik Senior Dev: Tukar decision_function kepada pseudo-probability pakai Softmax
                    scores = svm_model.decision_function(vector)[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    confidence = probs.max() * 100

            st.markdown(
            f"""
            <div style="background:linear-gradient(90deg,#43e97b,#38f9d7); padding:20px; border-radius:20px; text-align:center;">
                <h2 style="color:white;">🎯 Prediction Result</h2>
                <h1 style="color:white;">{prediction.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Safeguard progress bar dari error float melebihi 1.0
            st.progress(min(int(confidence) / 100.0, 1.0))
            st.info(f"Confidence Score: {confidence:.2f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔑 Top Influencing Keywords")
            
            feature_names = tfidf.get_feature_names_out()
            sorted_nz_indices = vector.indices[np.argsort(vector.data)[::-1]]
            top_5_words = [feature_names[i] for i in sorted_nz_indices[:5]]
            
            badges_html = " ".join([
                f'<span style="background: linear-gradient(90deg,#667eea,#764ba2); color:white; padding:5px 15px; border-radius:15px; font-size:14px; font-weight:bold; margin-right:5px; display:inline-block;">#{word}</span>' 
                for word in top_5_words
            ])
            st.markdown(badges_html, unsafe_allow_html=True)

        else:
            st.warning("Please enter article text to analyze.")


# ============================================================
# PAGE 2
# DATA EXPLORER
# ============================================================


elif page == "📊 Data Explorer":


    st.header(
        "📊 Dataset Explorer"
    )



    col1,col2,col3 = st.columns(3)



    with col1:

        st.metric(
            "Total Articles",
            len(df)
        )



    with col2:

        st.metric(
            "Categories",
            df["category"].nunique()
        )



    with col3:

        st.metric(
            "Average Length",
            round(
                df["text"]
                .str.len()
                .mean()
            )
        )



    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df.head(10),
        width="stretch"
    )



    st.subheader(
        "Category Distribution"
    )


    fig,ax = plt.subplots()


    sns.countplot(
        data=df,
        x="category",
        ax=ax
    )


    plt.xticks(
        rotation=45
    )


    st.pyplot(fig)



# ============================================================
# PAGE 3: WORD CLOUD
# ============================================================

elif page == "☁️ Word Cloud":

    st.header("☁️ BBC News Word Cloud")

    if os.path.exists("images/wordcloud.png"):
        st.image("images/wordcloud.png", caption="BBC News Global Vocabulary", width="stretch")
    else:
        st.error("Word cloud image not found in /images folder.")


# ============================================================
# PAGE 4
# MODEL PERFORMANCE
# ============================================================


elif page == "📈 Model Performance":



    st.header(
        "📈 Model Performance"
    )



    st.subheader(
        "Model Accuracy Comparison"
    )



    if os.path.exists(
        "images/model_comparison.png"
    ):


        st.image(
            "images/model_comparison.png",
            width="stretch"
        )



    st.divider()



    st.subheader(
        "Confusion Matrix"
    )



    col1,col2 = st.columns(2)



    with col1:


        st.markdown(
            "### ⭐ Naive Bayes"
        )


        if os.path.exists(
            "images/confusion_matrix_nb.png"
        ):


            st.image(
                "images/confusion_matrix_nb.png",
                width="stretch"
            )



    with col2:


        st.markdown(
            "### SVM"
        )


        if os.path.exists(
            "images/confusion_matrix_svm.png"
        ):


            st.image(
                "images/confusion_matrix_svm.png",
                width="stretch"
            )



# ============================================================
# PAGE 5: MODEL INFORMATION
# ============================================================

elif page == "ℹ️ Model Information":

    st.header("ℹ️ NLP Pipeline")

    st.markdown(
"""
<div style="background:white; padding:25px; border-radius:20px; box-shadow: 0px 5px 15px rgba(0,0,0,0.08);">
    <h2>🧠 System Workflow</h2>
    <p><b>1. Dataset</b><br>BBC News Standard Text Corpus (2,225 samples)</p>
    <p><b>2. Preprocessing</b><br>Lowercasing, RegEx Punctuation Removal, Tokenization, Stopword Elimination & Lemmatization</p>
    <p><b>3. Feature Extraction</b><br>TF-IDF (Term Frequency-Inverse Document Frequency) + N-grams</p>
    <p><b>4. Machine Learning Classifiers</b><br>Multinomial Naive Bayes & Support Vector Machine (Linear Kernel)</p>
    <p><b>5. Final Production Model</b><br>⭐ <b>Multinomial Naive Bayes</b> (Selected for optimal validation accuracy)</p>
</div>
""",
unsafe_allow_html=True
)
