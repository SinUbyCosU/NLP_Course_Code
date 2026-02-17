import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter

# 1. Load both datasets (to get the "whole corpus")
train_df = pd.read_csv("mental_health_training.csv")
test_df = pd.read_csv("mental_health_test.csv")

# Combine them into one list of text
full_corpus = pd.concat([train_df['text'], test_df['text']])

# 2. Preprocessing (Tokenize and remove stopwords)
stop_words = set(stopwords.words("english"))
all_tokens = []           # For counting frequency
tokenized_sentences = []  # For training Word2Vec

for text in full_corpus:
    # Tokenize and lowercase
    words = word_tokenize(str(text).lower())
    # Keep only real words (no punctuation) that are not stopwords
    clean_words = [w for w in words if w.isalnum() and w not in stop_words]
    
    all_tokens.extend(clean_words)
    tokenized_sentences.append(clean_words)

# 3. Find the Highest Frequency Word
# Counter counts every word instantly
word_counts = Counter(all_tokens)
most_freq_word = word_counts.most_common(1)[0][0]

print(f"Highest frequency word in whole corpus: '{most_freq_word}'")

# 4. Print WordNet Synsets (Meanings)
print(f"\n--- WordNet Synsets for '{most_freq_word}' ---")
synsets = wordnet.synsets(most_freq_word)
if synsets:
    # Print the definition of the first few meanings
    for syn in synsets[:3]: 
        print(f"- {syn.name()}: {syn.definition()}")
else:
    print("No synsets found.")

# 5. Train Word2Vec and Print Embeddings
# vector_size=10 (as requested), min_count=1 (so it doesn't ignore words)
model = Word2Vec(tokenized_sentences, vector_size=10, window=5, min_count=1, workers=1)

print(f"\n--- Word2Vec Embeddings (Length 10) ---")
print(model.wv[most_freq_word])

print("\nGensim OK")