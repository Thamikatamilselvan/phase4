# AI phase wise project submission
#fake news detection using nlp
Data source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
referance: google.com
# Data Preprocessing
python
Copy code
import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("fake_news_data.csv")

# Preprocess text data
def preprocess_text(text):
     # Tokenize and remove stopwords
     tokens = nltk.word_tokenize(text)
     tokens = [word for word in tokens if word.lower() not in stopwords.words("english")]
     return " ".join(tokens) 
     
data["text"] = data["text"].apply(preprocess_text)

# Split the dataset into training and testing sets 
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features 
tfidf_vectorizer = TfidfVectorizer(max_features=5000
) X_train_tfidf = tfidf_vectorizer.fit_transform(X_train
) X_test_tfidf = tfidf_vectorizer.transform(X_test)
Model Training
python
Copy code
from sklearn.naive_bayes import MultinomialNB # Train a Naive Bayes classifier model = MultinomialNB() model.fit(X_train_tfidf, y_train)

Model Evaluation
python
Copy code
from sklearn.metrics import accuracy_score, classification_report
# Make predictions 
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Real News", "Fake News"])

print(f"Accuracy: {accuracy}")
This project is a fake news detection system that uses Natural Language Processing (NLP) techniques to classify news articles as either real or fake. It leverages a pre-trained model and a dataset to achieve this.
Table of Contents
1.	Prerequisites
2.	Installation
3.	Getting Started
4.	Usage
5.	Contributing
6.	License
Prerequisites
Before you can run the code, make sure you have the following dependencies installed:
•	Python 3.x
•	Required Python packages (list these, e.g., pandas, scikit-learn, tensorflow, etc.)
•	A dataset of labeled news articles (provide a link to the dataset or instructions on how to obtain it)
•	(If applicable) Pre-trained NLP model (provide details on how to obtain and load it)
Installation
Clone the repository:

git clone https://github.com/thamikatamilselvan/fake-news-detection.git
Change into the project directory:
cd fake-news-detection
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
Install the required Python packages:
pip install -r requirements.txt
Getting Started
To get started with the fake news detection code, follow these steps:
1.	Ensure you have met all the prerequisites mentioned in the Prerequisites section.
2.	Prepare your dataset:
•	Place your labeled news articles dataset in a directory, e.g., data/.
•	Update the dataset path in the configuration file (if applicable).
3.	(If applicable) Load the pre-trained NLP model:
•	Download the pre-trained NLP model and place it in the appropriate directory.
•	Update the model path in the configuration file.
Usage
To run the code for fake news detection, use the following commands:
1.	Data preprocessing (if needed):
python train_model.py
Train the fake news detection model:
python train_model.py
Make predictions on new data:
python predict.py --text "Your news article goes here."
Evaluate the model (if applicable):
python evaluate_model.py
Contributing
If you'd like to contribute to this project, please follow these guidelines:
1.	Fork the repository.
2.	Create a new branch for your feature or bug fix:
git checkout -b feature-name
Make your changes and commit them:
git commit -m "Your commit message"
Push your changes to your fork:
git push origin feature-name
Open a pull request on the original repository, explaining the changes you made and why they should be merged.

print(report)
This is a basic example of building a fake news detection system using NLP. Depending on your dataset and specific requirements, you might want to use more advanced NLP techniques, different algorithms, and hyperparameter tuning to improve the model's performance. Additionally, you should ensure that you have a sizable and representative dataset for robust results.

