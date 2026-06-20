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
# Sidebar
# ============================================================

with st.sidebar:

    st.title(
        "📰 BBC News NLP"
    )

    st.write(
    """
    **Natural Language Processing Application**

    This application classifies BBC news articles
    into different categories.

    NLP Techniques:

    ✔ Text Preprocessing  
    ✔ TF-IDF  
    ✔ N-gram Feature Extraction  
    ✔ Naive Bayes Classification  
    ✔ SVM Comparison
    """
    )


    st.divider()


    st.info(
        """
        SAIA 2163

        NLP Final Project
        """
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
# Title
# ============================================================


st.title(
    "📰 BBC News Article Categorizer"
)


st.write(
"""
An NLP application that automatically classifies BBC news articles
into categories using TF-IDF + N-gram feature extraction and
Machine Learning algorithms.
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
# TAB 1 : TEXT ANALYZER
# ============================================================


with tab1:


    st.header(
        "🔍 News Article Classification"
    )


    text = st.text_area(
        "Enter News Article:",
        height=250,
        placeholder="Paste article content here..."
    )


    if st.button(
        "🚀 Analyze Article"
    ):


        if text.strip():


            # Convert text into TF-IDF + N-gram features

            vector = tfidf.transform(
                [text]
            )


            # Final model = Naive Bayes

            prediction = nb_model.predict(
                vector
            )[0]


            st.success(
                f"Predicted Category: {prediction.upper()}"
            )



            # Confidence score

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
                "Please enter news article text."
            )



# ============================================================
# TAB 2 : DATA EXPLORER
# ============================================================


with tab2:


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
        use_container_width=True
    )



    st.subheader(
        "News Category Distribution"
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
# TAB 3 : WORD CLOUD
# ============================================================


with tab3:


    st.header(
        "☁️ Word Cloud"
    )


    wordcloud_path = (
        "images/wordcloud.png"
    )


    try:

        st.image(
            wordcloud_path,
            caption="BBC News Word Cloud",
            width="stretch"
        )


    except:

        st.error(
            "Word cloud image not found. Please upload images/wordcloud.png"
        )
# ============================================================
# TAB 4 : MODEL PERFORMANCE
# ============================================================


with tab4:


    st.header(
        "📈 Model Performance Comparison"
    )



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
        ],


        "Status":
        [
            "⭐ Selected Model",
            "Comparison Model"
        ]

    })



    st.dataframe(
        results
    )



    # Accuracy chart


    fig,ax = plt.subplots()


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


    ax.set_title(
        "Accuracy Comparison"
    )


    st.pyplot(fig)



    # Confusion Matrix


    st.subheader(
        "Confusion Matrix Comparison"
    )


    col1,col2 = st.columns(2)



    with col1:


        st.markdown(
            "### Naive Bayes"
        )


        st.image(
            "images/confusion_matrix_nb.png",
            caption="Naive Bayes Confusion Matrix",
            width="stretch"
        )



    with col2:


        st.markdown(
            "### SVM"
        )


        st.image(
            "images/confusion_matrix_svm.png",
            caption="SVM Confusion Matrix",
            width="stretch"
        )



# ============================================================
# TAB 5 : MODEL INFORMATION
# ============================================================


with tab5:


    st.header(
        "🤖 Model Information"
    )


    st.write(
    """

## Feature Extraction

✔ TF-IDF Vectorization

✔ N-gram Feature Extraction


## Machine Learning Models

### 1. Multinomial Naive Bayes ⭐

Selected as final model because it achieved
the highest accuracy.


### 2. Support Vector Machine (SVM)

Used as comparison model.


## Final Model

⭐ Naive Bayes

The Naive Bayes classifier is used for the final
BBC News category prediction.

"""
    )
