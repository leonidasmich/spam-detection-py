import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

# Load Dataset (Assuming 'spam_ham_dataset.csv' is in the working directory)
df = pd.read_csv('spam_ham_dataset.csv')

# Preprocess Text (Combined and Improved Steps)
def preprocess_text(text):
    """Preprocesses a given email text for spam detection.

    Args:
        text (str): The email text to preprocess.

    Returns:
        str: The preprocessed text.
    """

    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing whitespace

    # Remove punctuation (consider keeping some like exclamation marks for sentiment analysis)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization (consider regular expressions for advanced splitting logic)
    text = text.split()

    # Stop word removal and stemming (consider stemming vs. lemmatization)
    stopwords_set = set(stopwords.words('english'))
    text = [PorterStemmer().stem(word) for word in text if word not in stopwords_set]

    return ' '.join(text)

df['text'] = df['text'].apply(preprocess_text)

# Feature Extraction
vectorizer = CountVectorizer(max_features=2000)  # Limit features to reduce model complexity
x = vectorizer.fit_transform(df['text']).toarray()
y = df.label_num

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Set random state for reproducibility

# Model Training with Hyperparameter Tuning (GridSearch/RandomizedSearch)
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(n_jobs=-1)  # Leverage all cores

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Evaluate Model Performance
best_model = grid_search.best_estimator_
predictions = best_model.predict(x_test)
accuracy = best_model.score(x_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Classify New Email (Assuming 'df.text.values[10]' contains the email text)
email_to_classify = df.text.values[10]
email_processed = preprocess_text(email_to_classify.lower())
email_corpus = [email_processed]

x_email = vectorizer.transform(email_corpus)  # Use the same vectorizer that trained the model
prediction = best_model.predict(x_email)[0]

if prediction == 1:
    print("This email is classified as spam.")
else:
    print("This email is classified as ham (not spam).")
