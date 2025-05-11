# Exposing-the-truth-with-advanced-fake-news-detection-powered-by-natural-language-processing

## 📌 Project Overview

Fake news has emerged as a major challenge in the digital age, threatening public trust, influencing elections, and spreading misinformation at scale. This project aims to build an automated system that can **classify news as real or fake** using **Natural Language Processing (NLP)** and **Machine Learning**.

By leveraging cleaned textual data and state-of-the-art NLP techniques like **TF-IDF**, the project delivers a reliable classification model to expose misinformation.

---

## 🎯 Objectives

- Detect fake news using NLP and supervised learning.
- Clean, preprocess, and vectorize news article text.
- Train and evaluate models for high classification accuracy.
- Visualize model performance and top features.
- Support public awareness efforts by identifying fake content.

---

## 🧠 Problem Statement

Manual fact-checking is too slow to match the speed at which misinformation spreads. This project addresses the need for a **scalable, automated solution** to detect fake news in real time using NLP.

---

## 🗂️ Dataset Description

- **Name:** Cleaned Fake News Dataset
- **Source:** Derived from Kaggle’s Fake News Challenge dataset
- **Data Type:** Unstructured text
- **Target Variable:** `label` (0 = Fake, 1 = Real)
- **Features Used:** `text` (main article content)

---

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Environment:** Google Colab, Jupyter Notebook
- **Libraries:**
  - Data Handling: `pandas`, `numpy`
  - NLP: `nltk`, `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - Modeling: `scikit-learn`

---

## 🛠️ Project Structure
├── data/
|
│ └── cleaned_fakenews.csv
|
├── notebooks/
|
│
|└── FakeNewsDetection.ipynb
├
|── models/
│ └── tfidf_model.pkl (optional)
├
|── README.md
├
|── LICENSE
|
|── requirements.txt

## 📈 Methodology

### 1. Data Preprocessing
- Removed null values and duplicate articles.
- Converted all text to lowercase.
- Removed punctuation, numbers, and stop words.
- Tokenized and cleaned the text.

### 2. Feature Extraction
- Used **TF-IDF Vectorizer** to convert text to numerical vectors.
- Tuned parameters like `max_df=0.7` to filter overly common terms.
- Explored both unigrams and bigrams for contextual richness.

### 3. Model Building
- Implemented **Logistic Regression** and **Passive Aggressive Classifier**.
- Used an 80/20 Train-Test split for evaluation.
- Measured performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC AUC Score

### 4. Visualization & Interpretation
- **Word Clouds** for visual comparison of fake vs. real articles.
- **Confusion Matrix** and **ROC Curve** to interpret classifier performance.
- Analyzed top predictive words using model coefficients.

---

## ✅ Results

| Model                    | Accuracy | Precision | Recall | F1 Score | AUC   |
|--------------------------|----------|-----------|--------|----------|-------|
| Logistic Regression      | 93.6%    | 93.8%     | 93.2%  | 93.5%    | 0.97  |
| Passive Aggressive Classifier | **94.5%** | **94.7%** | **94.2%** | **94.4%** | **0.98** |

- Most indicative fake words: `click`, `shocking`, `breaking`, `alert`
- Most indicative real words: `report`, `official`, `statement`, `analysis`

---

## 👥 Team Members and Contributions

| Name               | Role & Responsibilities                                   |
|--------------------|-----------------------------------------------------------|
| **M Sanjai Pravin** | Data Cleaning, Preprocessing, Feature Encoding            |
| **V Sanjay Kumar**  | EDA, Statistical Analysis, Visualization                  |
| **K Sathya Sri**    | Feature Engineering, Model Development                    |
| **S Sathya Priya**  | Hyperparameter Tuning, Model Evaluation, Report Writing   |

---

## 🔗 Repository Link

GitHub Repo: [https://github.com/sanj2006/fake-news-detection](https://github.com/sanj2006/fake-news-detection)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)
- Scikit-learn, NLTK, Matplotlib, Seaborn
- HuggingFace (optional for future deep learning extensions)

---

## 🚀 Future Enhancements

- Integration with BERT or RoBERTa for contextual classification
- Real-time news stream classification using Flask API
- Deployment as a web app using Streamlit or Gradio


