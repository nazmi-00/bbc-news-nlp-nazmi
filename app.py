# ============================================================
# BBC NEWS ARTICLE CATEGORIZER
# SAIA 2163 NLP FINAL PROJECT
# ============================================================


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os



# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="BBC News Categorizer",
    page_icon="📰",
    layout="wide"
)



# ============================================================
# CUSTOM COLORFUL UI
# ============================================================


st.markdown(
"""
<style>


/* Background */

.stApp {

background:
linear-gradient(
135deg,
#f5f7fa,
#c3cfe2
);

}



/* Title */

h1 {

background:
linear-gradient(
90deg,
#667eea,
#764ba2
);


-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

font-size:45px !important;

font-weight:800;

}



/* Sidebar */

[data-testid="stSidebar"] {


background:

linear-gradient(
180deg,
#667eea,
#764ba2
);


}



[data-testid="stSidebar"] * {

color:white;

}



/* Buttons */


.stButton button {


background:

linear-gradient(
90deg,
#667eea,
#764ba2
);


color:white;

border-radius:20px;

height:45px;

font-weight:bold;

border:none;


}



.stButton button:hover {


background:

linear-gradient(
90deg,
#764ba2,
#667eea
);


}



/* Cards */


[data-testid="metric-container"] {


background:white;

padding:20px;

border-radius:15px;

box-shadow:

0px 5px 15px rgba(0,0,0,0.15);


}



/* Text box */

textarea {


border-radius:15px !important;

border:2px solid #667eea !important;


}


</style>

""",
unsafe_allow_html=True
)




# ============================================================
# LOAD MODELS
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
# LOAD DATASET
# ============================================================


@st.cache_data
def load_data():


    df = pd.read_csv(
        "data/bbc-text.csv"
    )


    return df




df = load_data()




# ============================================================
# SIDEBAR MENU
# ============================================================


st.sidebar.markdown(
"""
# 📰 BBC NLP
## Article Categorizer
"""
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


Final Model:

⭐ Multinomial Naive Bayes

Feature:

TF-IDF + N-gram

"""

)




# ============================================================
# HEADER
# ============================================================


st.title(
"📰 BBC News Article Categorizer"
)



st.markdown(

"""

<div style="

background:linear-gradient(
90deg,
#667eea,
#764ba2
);

padding:25px;

border-radius:20px;

color:white;

">


<h2 style="color:white;">

🚀 Intelligent News Classification System

</h2>


<p>

Automatically classify BBC news articles using
Natural Language Processing and Machine Learning.

</p>


</div>

""",

unsafe_allow_html=True

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

        placeholder=
        "Paste your news article here..."

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




            st.markdown(

            f"""

            <div style="

            background:linear-gradient(
            90deg,
            #43e97b,
            #38f9d7
            );

            padding:20px;

            border-radius:20px;

            text-align:center;

            ">


            <h2>

            🎯 Prediction

            </h2>


            <h1>

            {prediction.upper()}

            </h1>


            </div>

            """,

            unsafe_allow_html=True

            )




            confidence = (

            nb_model

            .predict_proba(vector)

            .max()

            *100

            )



            st.progress(

                confidence/100

            )


            st.info(

            f"Confidence Score: {confidence:.2f}%"

            )



        else:


            st.warning(

            "Please enter article text"

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


        st.metric(

        "Average Length",

        round(
        df["text"]
        .str.len()
        .mean()
        )

        )




    st.subheader(

    "Sample Dataset"

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
# PAGE 3 : WORD CLOUD
# ============================================================


elif page == "☁️ Word Cloud":



    st.header(

    "☁️ BBC News Word Cloud"

    )



    if os.path.exists(

    "images/wordcloud.png"

    ):


        st.image(

        "images/wordcloud.png",

        width="stretch",

        caption="BBC News Word Cloud"

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



    col1,col2 = st.columns(2)



    with col1:



        st.subheader(

        "Model Comparison"

        )


        if os.path.exists(

        "images/model_comparison.png"

        ):


            st.image(

            "images/model_comparison.png",

            width="stretch"

            )




    with col2:



        st.subheader(

        "Naive Bayes Confusion Matrix"

        )


        if os.path.exists(

        "images/confusion_matrix_nb.png"

        ):


            st.image(

            "images/confusion_matrix_nb.png",

            width="stretch"

            )





    st.divider()



    st.subheader(

    "SVM Confusion Matrix"

    )


    if os.path.exists(

    "images/confusion_matrix_svm.png"

    ):


        st.image(

        "images/confusion_matrix_svm.png",

        width="stretch"

        )





# ============================================================
# PAGE 5 : MODEL INFORMATION
# ============================================================


elif page == "🤖 Model Information":



    st.header(

    "🤖 NLP Pipeline Information"

    )



    st.markdown(

    """

<div style="

background:white;

padding:25px;

border-radius:20px;

">


<h2>
🧠 NLP Workflow
</h2>


<b>
1. Data Collection
</b>

<br>

BBC News labelled dataset


<br><br>


<b>
2. Text Preprocessing
</b>

<br>

Cleaning, tokenization,
stopword removal


<br><br>


<b>
3. Feature Extraction
</b>

<br>

TF-IDF + N-gram


<br><br>


<b>
4. Machine Learning Models
</b>

<br>

Multinomial Naive Bayes
<br>
Support Vector Machine


<br><br>


<b>
5. Final Model
</b>

<br>

⭐ Naive Bayes
(selected based on highest accuracy)


</div>


""",

unsafe_allow_html=True

)
