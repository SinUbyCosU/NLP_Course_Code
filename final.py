import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from collections import Counter

# Download NLTK data 
nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')

# Load Data
train_df = pd.read_csv("mental_health_training.csv")
test_df = pd.read_csv("mental_health_test.csv")

# --- TASK 1: TF-IDF Vocabulary ---
print("--- Task 1: TF-IDF ---")
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

print("Vocabulary:", vectorizer.get_feature_names_out()[:10]) # Printing first 10
print("Train Matrix Shape:", X_train.shape)
print("Test Matrix Shape:", X_test.shape)

# --- TASK 2: WordNet & Word2Vec ---
print("\n--- Task 2: WordNet & Word2Vec ---")
# Combine text to find most frequent word
full_text = pd.concat([train_df['text'], test_df['text']])
stop_words = set(stopwords.words("english"))
all_tokens = []
sentences = []

for text in full_text:
    tokens = word_tokenize(str(text).lower())
    clean = [w for w in tokens if w.isalnum() and w not in stop_words]
    all_tokens.extend(clean)
    sentences.append(clean)

# Find Highest Frequency Word
top_word = Counter(all_tokens).most_common(1)[0][0]
print(f"Highest Frequency Word: '{top_word}'")

# Print Synsets
synsets = wordnet.synsets(top_word)
if synsets:
    print(f"Definition: {synsets[0].definition()}")

# Train Word2Vec and Print Embedding
w2v_model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=1)
print("Embedding (first 10):", w2v_model.wv[top_word])

# --- TASK 3: Doc2Vec Classification ---
print("\n--- Task 3: Doc2Vec Classification ---")
# Tag documents
train_tagged = [TaggedDocument(words=word_tokenize(str(t).lower()), tags=[str(i)]) for i, t in enumerate(train_df['text'])]
test_tokens = [word_tokenize(str(t).lower()) for t in test_df['text']]

# Train Doc2Vec
d2v_model = Doc2Vec(vector_size=50, window=2, min_count=1, workers=1, epochs=30)
d2v_model.build_vocab(train_tagged)
d2v_model.train(train_tagged, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

# Prepare Vectors
X_train_vec = [d2v_model.infer_vector(doc.words) for doc in train_tagged]
X_test_vec = [d2v_model.infer_vector(tokens) for tokens in test_tokens]

# Train Classifier & Predict
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, train_df['label'])
y_pred = clf.predict(X_test_vec)

print("F1 Score:", f1_score(test_df['label'], y_pred, average='weighted'))