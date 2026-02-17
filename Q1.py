#TF-IDF Vocabulary
#This reads both files and prints the unique words (vocabulary) for each.

import pandas as pd
from sklearn.feature-extraction.text import TfidfVectorizer

train_df=pd.read_csv("mental_health_train.csv")
test_df=pd.read_csv("mental_health_test.csv")

tfidf=TfidfVectorizer(stop_words="english")
print("Train Vocabulary:")
