import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load Data
train = pd.read_csv("mental_health_training.csv")
test = pd.read_csv("mental_health_test.csv")

# 2. Create the Vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# 3. Fit & Transform TRAIN (Learn the words)
train_matrix = tfidf.fit_transform(train['text'])

# 4. Transform TEST (Reuse the words learned above)
test_matrix = tfidf.transform(test['text'])

# 5. Print Results
print("Vocabulary Size:", len(tfidf.get_feature_names_out()))
print("Train Shape:", train_matrix.shape)
print("Test Shape: ", test_matrix.shape)