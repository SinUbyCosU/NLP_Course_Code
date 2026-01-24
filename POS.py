import nltk
from nltk.corpus import brown

# Download required NLTK data
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)

text= nltk.word_tokenize("she sells seashells on the seashore")
print("\n POS tags:", nltk.pos_tag(text))
