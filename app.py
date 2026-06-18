# ============================================================
# BBC News Article Categorizer
# SAIA 2163 NLP Final Project
# ============================================================


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="BBC News Categorizer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        "models/svm_model.pkl"
    )

    nb_model = joblib.load(
        "models/nb_model.pkl"
    )

    tfidf = joblib.load(
        "models/tfidf_ngram.pkl"
    )

    return svm_model, nb_model, tfidf



svm_model, nb_model, tfidf = load_models()



# ============================================================
# Load Dataset
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/bbc_news.csv"
    )

    return df



df = load_data()



# ============================================================
# Title
# ============================================================


st.title(
    "📰 BBC News Article Categorizer"
)


st.write(
"""
NLP application that classifies BBC news articles into categories
using TF-IDF + N-gram feature extraction and Machine Learning models.
"""
)



# ============================================================
# Navigation Tabs
# ============================================================


tab1, tab2, tab3, tab4, tab5 = st.tabs(
[
"🔍 Text Analyzer",
"📊 Data Explorer",
"☁️ Word Cloud",
"📈 Model Performance",
"🤖 Model Information"
]
)



# ============================================================
# TAB 1: TEXT ANALYZER
# ============================================================


with tab1:


    st.header(
        "Enter News Article"
    )


    text = st.text_area(
        "News content:",
        height=200
    )


    if st.button(
        "Predict Category"
    ):


        if text.strip():


            vector = tfidf.transform(
                [text]
            )


            prediction = svm_model.predict(
                vector
            )


            st.success(
                f"Prediction: {prediction[0]}"
            )


            if hasattr(
                svm_model,
                "predict_proba"
            ):


                confidence = (
                    svm_model
                    .predict_proba(vector)
                    .max()
                    *100
                )


                st.info(
                    f"Confidence: {confidence:.2f}%"
                )

        else:

            st.warning(
                "Please enter text"
            )



# ============================================================
# TAB 2: DATA EXPLORER
# ============================================================


with tab2:


    st.header(
        "Dataset Explorer"
    )


    st.subheader(
        "Sample Data"
    )


    st.dataframe(
        df.head(10)
    )


    st.subheader(
        "Dataset Statistics"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Articles",
            len(df)
        )


    with col2:

        st.metric(
            "Categories",
            df["label"].nunique()
        )


    with col3:

        st.metric(
            "Average Text Length",
            round(
                df["text"].str.len().mean()
            )
        )



    st.subheader(
        "Category Distribution"
    )


    fig, ax = plt.subplots()


    df["label"].value_counts().plot(
        kind="bar",
        ax=ax
    )


    ax.set_xlabel(
        "Category"
    )

    ax.set_ylabel(
        "Number of Articles"
    )


    st.pyplot(fig)



# ============================================================
# TAB 3: WORD CLOUD
# ============================================================


with tab3:


    st.header(
        "Word Cloud"
    )


    all_text = " ".join(
        df["text"].astype(str)
    )


    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(all_text)



    fig, ax = plt.subplots(
        figsize=(10,5)
    )


    ax.imshow(
        wordcloud
    )


    ax.axis(
        "off"
    )


    st.pyplot(fig)



# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================


with tab4:


    st.header(
        "Model Comparison"
    )


    # Example values
    # Replace with your actual scores


    results = pd.DataFrame(
    {
        "Model":
        [
        "Naive Bayes",
        "SVM"
        ],

        "Accuracy":
        [
        0.984,
        0.982
        ]
    })


    st.dataframe(
        results
    )


    fig, ax = plt.subplots()


    sns.barplot(
        data=results,
        x="Model",
        y="Accuracy",
        ax=ax
    )


    ax.set_ylim(
        0,
        1
    )


    st.pyplot(fig)



    st.subheader(
        "Confusion Matrix"
    )


    try:

        st.image(
            "images/confusion_matrix.png"
        )


    except:

        st.warning(
            "Upload confusion matrix image"
        )



# ============================================================
# TAB 5: MODEL INFORMATION
# ============================================================


with tab5:


    st.header(
        "Model Details"
    )


    st.write(
    """
    Feature Extraction:
    
    ✔ TF-IDF
    
    ✔ N-gram
    
    
    
    Machine Learning Models:
    
    ✔ Multinomial Naive Bayes
    
    ✔ Support Vector Machine
    
    
    
    Best Model:
    
    SVM
    """
    )