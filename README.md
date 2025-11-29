# Sentiment-analysis-amazon-Products-Review
Sentiment analysis model on amazon Products Reviews
Data : From the Multi-Domain Sentiment Dataset (version 2.0): 25 product reviews on amazon products : ([unprocessed.tar.gz])
www.cs.jhu.edu/~mdredze/datasets/sentiment/

Link to download the data:
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz (1.5 G)

 Sentiment Analysis on Amazon Product Reviews

A Natural Language Processing (NLP) project that performs sentiment classification on Amazon product reviews using TF-IDF vectorization and Logistic Regression.
The project includes full preprocessing, model training, evaluation, and visualization inside a Jupyter Notebook.

📌 Project Overview

This project focuses on classifying Amazon product reviews into positive or negative sentiment.
Using the Multi-Domain Sentiment Dataset (version 2.0), the model learns the patterns in customer opinions and predicts the sentiment of unseen product reviews.

The workflow includes:

Dataset loading

Text preprocessing

TF-IDF feature extraction

Building a Logistic Regression classifier

Evaluating accuracy and performance

Visualizing model results

This is a great introduction to machine learning and NLP using classical ML techniques.

🎯 Objectives

Preprocess and clean raw review text

Convert textual data into numerical vectors using TF-IDF

Train a Logistic Regression classifier

Analyze accuracy, confusion matrix, and classification report

Visualize insights (top words, feature weights, etc.)

📊 Dataset Information

Dataset Name: Multi-Domain Sentiment Dataset (Version 2.0)
Source: Johns Hopkins University Center for Language and Speech Processing
Contains:

Amazon product reviews across multiple domains

Labeled as positive and negative

Over 25 categories (books, electronics, kitchen, etc.)

Includes unprocessed review text

📥 Dataset Download

Link:
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz

(Size: ~1.5 GB)

Note: Only a subset of the dataset is used in this project to keep training fast and lightweight.

🛠 Tech Stack
Component	Technology
Language	Python
ML/NLP Libraries	scikit-learn, NLTK
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
Notebook	Jupyter Notebook
🚀 Features
✔️ Text Preprocessing

Lowercasing

Removing punctuation

Stopword removal

Tokenization

Lemmatization

✔️ TF-IDF Vectorization

Converts text into weighted numeric vectors

Uses unigrams & bigrams

Removes high-frequency irrelevant words

✔️ Logistic Regression Model

Simple & effective linear classifier

Works well for sparse text data

Fast training even on large datasets

✔️ Model Evaluation

Includes:

Accuracy score

Confusion matrix heatmap

Precision, Recall, F1-score

ROC curve (optional)

✔️ Visual Insights

Word frequency plots

Class distribution

Most informative words

📁 Project Structure
📦 Sentiment-analysis-amazon-Products-Review
│── notebook/
│     └── sentiment_analysis_amazon.ipynb
│── data/
│     └── amazon_reviews_subset.csv
│── images/
│     ├── confusion_matrix.png
│     ├── wordcloud_positive.png
│     └── wordcloud_negative.png
│── requirements.txt
│── README.md
└── LICENSE

🔧 How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/Sentiment-analysis-amazon-Products-Review.git
cd Sentiment-analysis-amazon-Products-Review

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Launch Jupyter Notebook
jupyter notebook

5. Open & Run

Open:

sentiment_analysis_amazon.ipynb


Run all cells sequentially.

📘 Results Summary

Achieved high classification accuracy using Logistic Regression

TF-IDF representation effectively captured review semantics

Positive and negative reviews were well-separated in vector space

Common words like “great”, “excellent” influenced positive sentiment

Negative reviews commonly used “broken”, “bad”, “poor”, etc.

💡 Why Logistic Regression?

Lightweight and fast

Performs well on high-dimensional sparse data

Works great with TF-IDF vectors

Easy to interpret feature weights

📌 Future Enhancements

Try models like SVM, Random Forest, or Naïve Bayes

Use pretrained embeddings (Word2Vec, GloVe, BERT)

Deploy the model as a Flask or FastAPI app

Build a simple UI to classify new reviews

📜 License

This project is licensed under the MIT License.






<img width="1859" height="867" alt="image" src="https://github.com/user-attachments/assets/4dcfbf9f-27e9-4a95-a881-c3331082cd2f" />
