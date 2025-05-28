# 🤖 AI Echo: Your Smartest Conversational Partner

An intelligent NLP-powered app that classifies ChatGPT user reviews into **Positive, Neutral, or Negative** sentiments. Built to provide deep insights into user satisfaction, feedback trends, and common themes — all wrapped in an interactive Streamlit interface for real-time exploration and prediction.

<p align="center">
  <img src="https://img.shields.io/badge/Python-✓-blue" />
  <img src="https://img.shields.io/badge/NLP-✓-purple" />
  <img src="https://img.shields.io/badge/Streamlit-✓-red" />
  <img src="https://img.shields.io/badge/Machine_Learning-✓-brightgreen" />
  <img src="https://img.shields.io/badge/Deep_Learning-✓-brightgreen" />
</p>

---

## 📌 Project Highlights

- 🧹 Cleaned and preprocessed user reviews by combining `title` + `review` → `merged_review`.
- 🧼 Text preprocessing included demojization, spell correction, stopword removal, and lemmatization.
- 🧠 Generated sentiment labels using **VADER** as Positive, Neutral, or Negative.
- 📊 Performed **intensive EDA** to analyze user behavior, sentiment, and trends.
- 🤖 Built **ML models** (TF-IDF + classifiers) and **DL models** (LSTM, BiLSTM).
- ✅ Machine learning models performed better due to short, non-sensible review formats.
- 🚀 Deployed with **Streamlit**, served online via **ngrok** for external access.

---

## 📈 EDA Questions Answered

1. 📊 **What is the distribution of review ratings?**  
   → Visualize how reviews are spread across star ratings (1 to 5).

2. 👍👎 **How many reviews were marked as helpful (above a threshold)?**  
   → Determine how many users found each review useful.

3. 🧭 **What are the most common keywords in positive vs. negative reviews?**  
   → Use word clouds and frequency plots per sentiment.

4. 📆 **How has the average rating changed over time?**  
   → Analyze trends by month/year to detect changes in user satisfaction.

5. 🌍 **How do ratings vary by user location?**  
   → Visualize sentiment by region to detect geographic differences.

6. 🖥️📱 **Which platform (Web vs Mobile) gets better reviews?**  
   → Compare experience across platforms.

7. ✅❌ **Are verified users more satisfied than non-verified ones?**  
   → See if verified users leave better/worse reviews.

8. 🔠 **What’s the average length of reviews per rating category?**  
   → Do longer reviews correspond to more negative/positive feedback?

9. 💬 **What are the most mentioned words in 1-star reviews?**  
   → Extract and analyze common pain points.

10. 🧪📱 **Which ChatGPT version received the highest average rating?**  
    → Evaluate impact of each version on user sentiment.

---

## 💬 Streamlit App Features

In the deployed Streamlit app, users can interactively explore:

1. 🤔 **What is the overall sentiment of user reviews?**  
   → Classify each review as Positive, Neutral, or Negative and show proportions.

2. 📊 **How does sentiment vary by rating?**  
   → Discover if ratings align with the actual sentiment of the text.

3. 🧠 **Which keywords or phrases are most associated with each sentiment class?**  
   → Use word clouds or bar charts to explore keywords per sentiment.

4. 📈 **How has sentiment changed over time?**  
   → Spot spikes in satisfaction/dissatisfaction using time-based plots.

5. ✅❌ **Do verified users tend to leave more positive or negative reviews?**  
   → Visualize verified vs non-verified sentiment.

6. 🔠 **Are longer reviews more likely to be negative or positive?**  
   → Correlate review length with sentiment.

7. 🌍 **Which locations show the most positive or negative sentiment?**  
   → Map sentiment by region.

8. 🖥️📱 **Is there a difference in sentiment across platforms (Web vs Mobile)?**  
   → Compare UX across devices.

9. 🧪📱 **Which ChatGPT versions are associated with higher/lower sentiment?**  
   → Determine sentiment by version release.

10. 🚫 **What are the most common negative feedback themes?**  
    → Use topic modeling to summarize complaints.

---

## 🧠 Models Used

- **TF-IDF + Machine Learning Classifiers**:
  - Logistic Regression
  - SVM
  - Random Forest
  - Decision tree
  - Gaussian NB
  - Multinomial NB etc

- **Deep Learning**:
  - LSTM
  - GloVe Embeddings
  - Attention Layer (optional)
    
> Each model evaluated with accuracy, F1-score, precision, recall, and confusion matrix.  
> Best performing model integrated into the Streamlit app for live prediction.

---

## 🖥️ Tech Stack

| Tool             | Purpose                             |
|------------------|-----------------------------------|
| Python           | Core programming language          |
| pandas, numpy    | Data manipulation                  |
| NLTK, spaCy      | Text preprocessing & NLP           |
| VADER            | Sentiment labeling                 |
| scikit-learn     | ML algorithms & evaluation          |
| TensorFlow/Keras | Deep learning models                |
| matplotlib, seaborn, wordcloud | Visualization            |
| Streamlit        | Web app UI                        |
| ngrok            | Public tunneling for hosted app    |
| joblib           | Model serialization & loading     |

---
## 🧾 Conclusion

This project, **AI Echo: Your Smartest Conversational Partner**, provided a comprehensive analysis of user reviews about ChatGPT using both traditional machine learning and deep learning approaches. Through meticulous preprocessing, VADER-based sentiment labeling, and extensive exploratory data analysis, we uncovered valuable insights into user satisfaction trends, platform differences, and key feedback themes.

While deep learning models like BiLSTM were implemented, the non-contextual and often noisy nature of the reviews made classical machine learning models (especially Logistic Regression with TF-IDF features etc.) more effective in sentiment classification. This outcome highlights the importance of model selection based on data quality and task complexity.

The final solution was deployed using Streamlit, enabling users to explore review sentiment, identify key concerns, and gain an interactive understanding of public feedback. The application empowers product teams to track changes in user sentiment, improve feature development, and address user pain points more effectively.

---

✅ **Key Takeaways:**
- Text preprocessing and class balancing are crucial for sentiment modeling.
- Simple ML models can outperform DL models when data is noisy or lacks structure.
- Streamlit is effective for rapid, interactive data app deployment.
- User sentiment varies across time, platforms, versions, and review length — valuable for business strategy.

---

## 🧑‍💻 Author

Built with ❤️ by **[PREETHI S]**

---

## 🏷️ Tags

`#NLP` `#SentimentAnalysis` `#ChatGPT` `#MachineLearning` `#DeepLearning` `#Streamlit` `#Python` `#DataScience` `#TextClassification` `#VADER` `#LSTM` `#TFIDF` `#WordEmbeddings`
