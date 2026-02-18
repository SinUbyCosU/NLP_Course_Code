import pandas as pd
import nltk
from nltk.corpus import wordnet,stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from collections import Counter

train_df=pd.read_csv("mental_health_training.csv")
test_df=pd.read_csv("mental_health_test.csv")

print("--- Task 1: TF-IDF ---")
vectorizer=TfIdfVectorizer(stop_words='english')    
X_train=vectorizer.fit_transform(train_df['text'])
X_test=vectorizer.transform(test_df['text'])
print("Vocabulary:", vectorizer.get_feature_names_out()[:10])
print(X_train.shape)
print(X_test.shape)