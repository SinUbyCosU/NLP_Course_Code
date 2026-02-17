#TF-IDF Vocabulary
#This reads both files and prints the unique words (vocabulary) for each.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train_df=pd.read_csv("mental_health_train.csv")
test_df=pd.read_csv("mental_health_test.csv")

tfidf=TfidfVectorizer(stop_words="english")
print("Train Vocabulary:")
tfidf.fit(train_df['text'])
print(tfidf.get_feature_names_out())
print("Training Matrix Shape:", tfidf.transform(train_df['text']).shape)

print("Training corpus")
tfidf_test=TfidfVectorizer(stop_words='english')
tfidf_test.fit(test_df['text'])
print("Test matrix shape:", tfidf_test.transform(test_df['text']).shape)