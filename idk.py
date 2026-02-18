import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from collections import Counter

# 1. SETUP
nltk.download(['punkt', 'wordnet'], quiet=True)
train = pd.read_csv("mental_health_training.csv")
test = pd.read_csv("mental_health_test.csv")

# 2. TASK 1: TF-IDF (Directly print first 5 keys)
print("Train Vocab:", list(TfidfVectorizer().fit(train['Text']).vocabulary_)[:5])
print("Test Vocab:",  list(TfidfVectorizer().fit(test['Text']).vocabulary_)[:5])

# 3. TASK 2: WORDNET & WORD2VEC
# Tokenize all training text in one line
tokens = [word_tokenize(t.lower()) for t in train['Text']]

# Flatten list to find top word (sum(list, []) connects lists together)
top_word = Counter(sum(tokens, [])).most_common(1)[0][0] 

print("Top Word:", top_word)
print("Synsets:", wordnet.synsets(top_word))

# Train Word2Vec
w2v = Word2Vec(tokens, vector_size=10, min_count=1)
print("Embedding:", w2v.wv[top_word])

# 4. TASK 3: DOC2VEC & LABELS
# Prepare Tagged Docs
tagged = [TaggedDocument(d, [i]) for i, d in enumerate(tokens)]
d2v = Doc2Vec(tagged, vector_size=50, epochs=20, min_count=1)

# Infer Vectors
x_train = [d2v.infer_vector(d) for d in tokens]
x_test  = [d2v.infer_vector(word_tokenize(t.lower())) for t in test['Text']]

# Predict and Save
clf = LogisticRegression().fit(x_train, train['Label'])
pd.DataFrame(clf.predict(x_test)).to_csv("labels.txt", index=False, header=False)