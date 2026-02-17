#TF-IDF Vocabulary
#This reads both files and prints the unique words (vocabulary) for each.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_text_series(df, dataset_name):
	for column in df.columns:
		if column.lower() == "text":
			return df[column]
	raise KeyError(f"No 'Text' column found in {dataset_name} CSV")


train_df=pd.read_csv("mental_health_training.csv")
test_df=pd.read_csv("mental_health_test.csv")

train_text=get_text_series(train_df, "training")
test_text=get_text_series(test_df, "test")

tfidf=TfidfVectorizer(stop_words="english")
print("Train Vocabulary:")
tfidf.fit(train_text)
print(tfidf.get_feature_names_out())
print("Training Matrix Shape:", tfidf.transform(train_text).shape)

print("Training corpus")
tfidf_test=TfidfVectorizer(stop_words='english')
tfidf_test.fit(test_text)
print("Test matrix shape:", tfidf_test.transform(test_text).shape)