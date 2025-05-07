import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
def set_bg_image(image_url):
  st.markdown(f"""<style>.stApp {{background-image: url("{image_url}");background-size: cover;background-position: center;background-repeat: no-repeat;}}
        </style>""",unsafe_allow_html=True)
set_bg_image("https://dezyre.gumlet.io/images/blog/nlp-projects-ideas-/image_23691283681737376453775.png?w=376&dpr=2.6")

spell = SpellChecker()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
  text = emoji.demojize(text)
  text = text.lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text)
  text = re.sub(r"@\w+|\#", "", text)
  text = re.sub(r"\d+", "", text)
  text = re.sub(r"[^\w\s]", "", text)
  text = re.sub(r"\s+", " ", text).strip()

  tokens = word_tokenize(text)
  tokens = [spell.correction(word) for word in tokens]
  tokens = [word for word in tokens if word not in stop_words]
  tokens = [lemmatizer.lemmatize(word) for word in tokens]

  return " ".join(tokens)

model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Sentiment Analysis Dashboard")
st.sidebar.title("Prediction")
mode = st.sidebar.radio("Choose Mode", ["📊 Dataset Analysis", "Predict Single Review"])

if mode == "📊 Dataset Analysis":
  uploaded_file = st.sidebar.file_uploader("Upload Your File", type=["csv", "xlsx"])

  if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
      df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
      df = pd.read_excel(uploaded_file)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    missing_dates = df[df['date'].isnull()]
    if not missing_dates.empty:
      st.warning(f"There are {len(missing_dates)} missing or invalid dates in the dataset.")

    df['date'] = df['date'].dt.date
    st.write("### Data")
    st.dataframe(df)

    question = st.selectbox("Select Analysis", [
        "1. What is the overall sentiment of user reviews?",
        "2. How does sentiment vary by rating?",
        "3. Which keywords or phrases are most associated with each sentiment class?",
        "4. How has sentiment changed over time?",
        "5. Do verified users tend to leave more positive or negative reviews?",
        "6. Are longer reviews more likely to be negative or positive?",
        "7. Which locations show the most positive or negative sentiment?",
        "8. Is there a difference in sentiment across platforms (Web vs Mobile)?",
        "9. Which ChatGPT versions are associated with higher or lower sentiment?",
        "10. What are the most common negative feedback themes?"
    ])

    if question == "1. What is the overall sentiment of user reviews?":
      sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
      st.subheader("Overall Sentiment Distribution")
      st.plotly_chart(px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Proportions"))

    elif question == "2. How does sentiment vary by rating?":
      st.subheader("Sentiment by Rating")
      st.plotly_chart(px.histogram(df, x="rating", color="sentiment", barmode="group"))

    elif question == "3. Which keywords or phrases are most associated with each sentiment class?":
      sentiment_filter = st.selectbox("Choose Sentiment", ["positive", "neutral", "negative"])
      text = " ".join(df[df["sentiment"] == sentiment_filter]["merged_review"])
      wc = WordCloud(width=800, height=400, background_color="white").generate(text)
      fig, ax = plt.subplots()
      ax.imshow(wc, interpolation="bilinear")
      ax.axis("off")
      st.pyplot(fig)

    elif question == "4. How has sentiment changed over time?":
      df['year_month'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('M')
      time_df = df.groupby('year_month')['sentiment'].value_counts(normalize=True).unstack().fillna(0) * 100
      st.subheader("Sentiment Trends Over Time")
      st.plotly_chart(px.line(time_df, x=time_df.index.astype(str), y=["positive", "neutral", "negative"],
                              title="Sentiment Over Time", labels={"year_month": "Date", "value": "Sentiment %"}))

    elif question == "5. Do verified users tend to leave more positive or negative reviews?":
      st.subheader("Sentiment by Verified Purchase")
      st.plotly_chart(px.histogram(df, x="verified_purchase", color="sentiment", barmode="group"))

    elif question == "6. Are longer reviews more likely to be negative or positive?":
      st.subheader("Sentiment vs Review Length")
      st.plotly_chart(px.histogram(df, x="review_length", y="sentiment", color="sentiment"))

    elif question == "7. Which locations show the most positive or negative sentiment?":
      st.subheader("Sentiment by Location")
      st.plotly_chart(px.histogram(df, x="location", color="sentiment", barmode="group"))

    elif question == "8. Is there a difference in sentiment across platforms (Web vs Mobile)?":
      st.subheader("Sentiment by Platform")
      st.plotly_chart(px.histogram(df, x="platform", color="sentiment", barmode="group"))

    elif question == "9. Which ChatGPT versions are associated with higher or lower sentiment?":
      st.subheader("Sentiment by ChatGPT Version")
      st.plotly_chart(px.histogram(df, x="version", color="sentiment", barmode="group"))

    elif question == "10. What are the most common negative feedback themes?":
      neg_reviews = df[df['sentiment'] == 'negative']['merged_review']
      text = " ".join(neg_reviews)
      wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
      st.subheader("Common Negative Feedback Themes")
      plt.figure(figsize=(10, 5))
      plt.imshow(wordcloud, interpolation="bilinear")
      plt.axis('off')
      st.pyplot(plt)
  else:
      st.info("Please upload a cleaned file.")


elif mode == "Predict Single Review":
  st.title("Single Review Sentiment Prediction")
  user_input = st.text_area("Enter your review here:")
  if st.button("Predict"):
    if user_input.strip():
      preprocessed_text = preprocess_text(user_input)
      text_vector = vectorizer.transform([preprocessed_text])
      prediction = model.predict(text_vector)
      sentiment = label_encoder.inverse_transform(prediction)[0]
      st.subheader("Predicted Sentiment")
      st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
      st.warning("Please enter a review before clicking Predict.")
