# ============================================================
# BBC News Article Categorizer
# SAIA 2163 NLP Final Project
# ============================================================


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os



# ============================================================
# Page Configuration
# ============================================================


st.set_page_config(
    page_title="BBC News Categorizer",
    page_icon="📰",
    layout="wide"
)



# ============================================================
# Custom Sidebar Style
# ============================================================


st.markdown(
"""
<style>

[data-testid="stSidebar"] {

    min-width: 260px;
    max-width: 260px;

}

</style>
""",
unsafe_allow_html=True
)



# ============================================================
# Load Models
# ============================================================


@st.cache_resource
def load_models():


    nb_model = joblib.load(
        "models/nb_model.pkl"
    )


    svm_model = joblib.load(
        "models/svm_model.pkl"
    )


    tfidf = joblib.load(
        "models/tfidf_ngram.pkl"
    )


    return nb_model, svm_model, tfidf



nb_model, svm_model, tfidf = load_models()



# ============================================================
# Load Dataset
# ============================================================


@st.cache_data
def load_data():


    df = pd.read_csv(
        "data/bbc-text.csv"
    )


    return df



df = load_data()



# ============================================================
# Sidebar Navigation
# ============================================================


st.sidebar.title(
    "📰 BBC News NLP"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Text Analyzer",
        "📊 Data Explorer",
        "☁️ Word Cloud",
        "📈 Model Performance",
        "🤖 Model Information"
    ]
)



st.sidebar.divider()


st.sidebar.info(
"""
SAIA 2163

Natural Language Processing
Final Project

Model:
⭐ Multinomial Naive Bayes
"""
)



# ============================================================
# Main Title
# ============================================================


st.title(
    "📰 BBC News Article Categorizer"
)


st.write(
"""
An NLP application that automatically classifies BBC news articles
into different categories using TF-IDF + N-gram feature extraction
and Machine Learning algorithms.
"""
)



# ============================================================
# PAGE 1 : TEXT ANALYZER
# ============================================================


if page == "🔍 Text Analyzer":


    st.header(
        "🔍 News Article Classification"
    )


    text = st.text_area(
        "Enter news article:",
        height=250,
        placeholder="Paste BBC news article here..."
    )


    if st.button(
        "🚀 Analyze Article"
    ):


        if text.strip():


            vector = tfidf.transform(
                [text]
            )


            prediction = nb_model.predict(
                vector
            )[0]


            st.success(
                f"Predicted Category: {prediction.upper()}"
            )



            if hasattr(
                nb_model,
                "predict_proba"
            ):


                confidence = (
                    nb_model
                    .predict_proba(vector)
                    .max()
                    *100
                )


                st.progress(
                    confidence / 100
                )


                st.info(
                    f"Confidence Score: {confidence:.2f}%"
                )



        else:


            st.warning(
                "Please enter article text."
            )



# ============================================================
# PAGE 2 : DATA EXPLORER
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

        avg_length = round(
            df["text"]
            .str.len()
            .mean()
        )


        st.metric(
            "Average Text Length",
            avg_length
        )



    st.subheader(
        "Dataset Sample"
    )


    st.dataframe(
        df.head(10),
        width="stretch"
    )



    st.subheader(
        "Category Distribution"
    )


    fig,ax = plt.subplots(
        figsize=(8,4)
    )


    sns.countplot(
        data=df,
        x="category",
        ax=ax
    )


    ax.set_xlabel(
        "Category"
    )


    ax.set_ylabel(
        "Number of Articles"
    )


    plt.xticks(
        rotation=45
    )


    st.pyplot(fig)



# ============================================================
# PAGE 3 : WORD CLOUD
# ============================================================


elif page == "☁️ Word Cloud":


    st.header(
        "☁️ BBC News Word Cloud"
    )


    image_path = (
        "images/wordcloud.png"
    )


    if os.path.exists(image_path):


        st.image(
            image_path,
            caption="Most Frequent Words in BBC News Dataset",
            width="stretch"
        )


    else:


        st.error(
            "wordcloud.png not found"
        )



# ============================================================
# PAGE 4 : MODEL PERFORMANCE
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


    else:


        st.warning(
            "Model comparison image not found"
        )



    st.divider()



    st.subheader(
        "Confusion Matrix Comparison"
    )


    col1,col2 = st.columns(2)



    with col1:


        st.write(
            "### Naive Bayes"
        )


        if os.path.exists(
            "images/confusion_matrix_nb.png"
        ):


            st.image(
                "images/confusion_matrix_nb.png",
                caption="Naive Bayes Confusion Matrix",
                width="stretch"
            )


        else:


            st.error(
                "Naive Bayes confusion matrix not found"
            )



    with col2:


        st.write(
            "### SVM"
        )


        if os.path.exists(
            "images/confusion_matrix_svm.png"
        ):


            st.image(
                "images/confusion_matrix_svm.png",
                caption="SVM Confusion Matrix",
                width="stretch"
            )


        else:


            st.error(
                "SVM confusion matrix not found"
            )



# ============================================================
# PAGE 5 : MODEL INFORMATION
# ============================================================


elif page == "🤖 Model Information":


    st.header(
        "🤖 Model Information"
    )


    st.markdown(
"""
## Feature Extraction

### TF-IDF + N-gram

- TF-IDF measures the importance of words
- N-gram captures word combinations and context


---

## Machine Learning Models


### ⭐ Multinomial Naive Bayes

Selected final model because it achieved
the highest accuracy.


### Support Vector Machine (SVM)

Used as comparison model.


---

## Final Prediction Model

⭐ Naive Bayes

The system uses Naive Bayes to classify
BBC news articles into categories.
"""
    )
