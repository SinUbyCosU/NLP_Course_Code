# Import the Natural Language Toolkit library for text processing
import nltk
# Import the Brown corpus (though not used in this code)
from nltk.corpus import brown

# Download the POS tagger model - needed to identify parts of speech like noun, verb, etc.
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# Download the punkt tokenizer - needed to split sentences into individual words
nltk.download('punkt', quiet=True)
# Download the Brown corpus - a collection of text samples for linguistic research
nltk.download('brown', quiet=True)

# Break the sentence into a list of individual words (tokenization)
text= nltk.word_tokenize("she sells seashells on the seashore")
# Tag each word with its part of speech (PRP=pronoun, VBZ=verb, NN=noun, etc.) and print the result
print("\n POS tags:", nltk.pos_tag(text))

# ambigious words in browns news corpus 
# Get all words with their POS tags from the Brown corpus news section
brown_news_tagged= brown.tagged_words(categories='news')
# Create a frequency distribution that tracks which tags each word can have
data=nltk.ConditionalFreqDist((word.lower(),tag) for (word,tag) in brown_news_tagged)
# Counter to track how many ambiguous words we've found
count=0
# Loop through each unique word in the corpus
for word in data.conditions():
    # Get all the different tags this word can have
    tags=data[word].keys()
    # If a word has more than one tag, it's ambiguous (can be used as different parts of speech)
    if len(tags)>1:
        # Print the word and all its possible tags
        print(f"\nWord: {word} \t Tags: {list(tags)}")
        count+=1
        # Stop after showing 10 examples
        if count==10:
            break
# Calculate what percentage of all words are ambiguous
pr=round(count/len(data.conditions())*100,2)
# Display the percentage result
print(f"\nPercentage of ambigious words in brown news corpus: {pr} %")