# Spam Email Detection with Python

This repository implements a simple spam email detection system using machine learning. It utilizes pre-processing techniques, feature extraction, and a Random Forest classifier to distinguish between spam and non-spam (ham) emails.

## Libraries:

- **pandas** (`pd`): For data manipulation (loading and cleaning the CSV dataset).
- **nltk**: For natural language processing tasks like stop word removal and stemming.
- **scikit-learn** (`sklearn`): Machine learning library used for feature extraction (`CountVectorizer`) and classification (`RandomForestClassifier`).

## Instructions:

1. **Clone this repository:**
    ```bash
    git clone https://github.com/leonidasmich/spam-detection-py.git
    ```
    Use code with caution.

2. **Install dependencies:**
    Navigate to the project directory and run the following command to install all the required libraries listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Use code with caution.

3. **Run the script:**
    Execute the main script:
    ```bash
    python main.py
    ```
    Use code with caution.

## Code Breakdown (`main.py`):

The code is organized within the `main.py` script. Here's a breakdown of the key steps:

### Data Loading:
- Loads the spam dataset (assuming it's named `spam_ham_dataset.csv` and in the same directory) using `pandas.read_csv`.

### Text Preprocessing:
- Defines a function `preprocess_text` to perform the following tasks:
  - Converts text to lowercase.
  - Removes leading/trailing whitespace.
  - Removes punctuation (consider keeping some like exclamation marks for sentiment analysis).
  - Tokenizes the text (splits into words).
  - Removes stop words and performs stemming on remaining words.
- Applies the `preprocess_text` function to the text column of the DataFrame.

### Feature Extraction:
- Creates a `CountVectorizer` object, setting `max_features` to limit the number of features (reduce model complexity).
- Fits the vectorizer on the preprocessed text and transforms it into a numerical feature matrix.

### Train-Test Split:
- Splits the data into training and testing sets using `train_test_split` (0.8 for training, 0.2 for testing).
- Sets a random state (e.g., `random_state=42`) for reproducibility.

### Model Training:
- Creates a `RandomForestClassifier` object, leveraging all CPU cores with `n_jobs=-1`.
- Performs hyperparameter tuning using `GridSearchCV` to optimize the model's performance.
- Trains the model on the training data.

### Evaluation:
- Evaluates the model's performance on the testing data using the `score` method (accuracy).

### Classifying New Email:
- Extracts a sample email from the dataset for demonstration.
- Preprocesses the sample email using the `preprocess_text` function.
- Transforms the preprocessed email into a feature vector using the same `CountVectorizer` that was used for training.
- Predicts the spam/ham classification for the email using the trained model.
