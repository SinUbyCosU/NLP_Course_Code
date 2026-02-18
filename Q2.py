import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter
import nltk

# Load Data
train_df = pd.read_csv("mental_health_training.csv")
test_df = pd.read_csv("mental_health_test.csv")

# Combine text for whole corpus
corpus = pd.concat([train_df['text'], test_df['text']])

# Preprocess and Tokenize
stop_words = set(stopwords.words("english"))
all_tokens = []
sentences = []

for text in corpus:
    words = word_tokenize(str(text).lower())
    clean_words = [w for w in words if w.isalnum() and w not in stop_words]
    all_tokens.extend(clean_words)
    sentences.append(clean_words)

# Find Highest Frequency Word
word_counts = Counter(all_tokens)
top_word = word_counts.most_common(1)[0][0]
print(f"Highest frequency word: {top_word}")

# Print WordNet Synsets
print("\nSynsets:")
synsets = wordnet.synsets(top_word)
for syn in synsets:
    print(f"{syn.name()}: {syn.definition()}")

# Train Word2Vec and Print Embeddings
model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=1)

print("\nWord2Vec Embeddings (Length 10):")
print(model.wv[top_word])