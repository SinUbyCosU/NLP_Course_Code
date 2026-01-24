# Import the Natural Language Toolkit library for text processing
import nltk
# Import the Brown corpus (though not used in this code)
from nltk.corpus import brown

# Download the POS tagger model - needed to identify parts of speech like noun, verb, etc.
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# Download the punkt tokenizer - needed to split sentences into individual words
nltk.download('punkt', quiet=True)

# Break the sentence into a list of individual words (tokenization)
text= nltk.word_tokenize("she sells seashells on the seashore")
# Tag each word with its part of speech (PRP=pronoun, VBZ=verb, NN=noun, etc.) and print the result
print("\n POS tags:", nltk.pos_tag(text))

# ambigious words in browns news corpus
