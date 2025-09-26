import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# --- Preprocessing and Feature Engineering ---

# 1. Create the target variable
# Fill NA values in 'Misconception' with the string 'NA'
train_df['Misconception'] = train_df['Misconception'].fillna('NA')
train_df['Target'] = train_df['Category'].astype(str) + ':' + train_df['Misconception'].astype(str)

# 2. Combine text features for a single input
train_df['Full_Text'] = train_df['QuestionText'].astype(str) + ' ' + \
                        train_df['MC_Answer'].astype(str) + ' ' + \
                        train_df['StudentExplanation'].astype(str)

test_df['Full_Text'] = test_df['QuestionText'].astype(str) + ' ' + \
                       test_df['MC_Answer'].astype(str) + ' ' + \
                       test_df['StudentExplanation'].astype(str)

# --- TF-IDF Vectorization ---
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to the top 5000 for efficiency
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    stop_words='english'
)

X_train = tfidf_vectorizer.fit_transform(train_df['Full_Text'])
X_test = tfidf_vectorizer.transform(test_df['Full_Text'])

# --- Target Variable Binarization ---
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['Target'].apply(lambda x: [x]))


# --- Model Training ---
# Using OneVsRestClassifier with Logistic Regression for multi-label classification
model = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42), n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Prediction ---
y_pred_proba = model.predict_proba(X_test)
print("Prediction complete.")

# --- Submission File Generation ---
# Get the top 3 predictions for each row
top_3_preds = []
for i in range(len(y_pred_proba)):
    top_3_indices = np.argsort(y_pred_proba[i])[-3:][::-1]
    top_3_labels = mlb.classes_[top_3_indices]
    top_3_preds.append(' '.join(top_3_labels))

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': top_3_preds
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully: submission.csv")