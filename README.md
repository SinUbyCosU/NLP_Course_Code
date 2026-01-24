# 🧠 Playing With WordNet

> A hands-on exploration of Natural Language Processing using NLTK and WordNet

This repository documents my journey learning NLP fundamentals through practical implementations. Each script demonstrates core concepts with real examples and detailed explanations.

## 📚 What's Inside

### Day 1: Synsets Basics ([Word_Net_1.py](Word_Net_1.py))
**Exploring WordNet's lexical database**

Understanding how words are organized by meaning in WordNet:
- Synsets (synonym sets) - grouping words with similar meanings
- Accessing definitions, examples, and word relationships
- Working with lemmas and parts of speech

```python
import nltk
from nltk.corpus import wordnet as wn

# Get all meanings of a word
synsets = wn.synsets('bank')
for syn in synsets:
    print(syn.definition())
```

**Key Takeaway:** Words aren't just strings - they're organized by semantic relationships.

---

### Day 2: POS Tagging ([POS.py](POS.py))
**Understanding grammatical structure through Part-of-Speech tagging**

Learning how to automatically identify the grammatical role of each word:
- Implementing NLTK's POS tagger
- Analyzing the Brown corpus for ambiguous words
- Understanding why context matters in language

```python
text = nltk.word_tokenize("she sells seashells on the seashore")
tagged = nltk.pos_tag(text)
# Output: [('she', 'PRP'), ('sells', 'VBZ'), ('seashells', 'NNS'), ...]
```

**Key Insight:** Words like "primary" can be a noun OR adjective depending on context - POS tagging helps resolve this ambiguity.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install nltk
```

### Download Required NLTK Data
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('brown')
```

### Run the Examples
```bash
python Word_Net_1.py
python POS.py
```

## 🎯 Learning Objectives

- [x] Understand WordNet's structure and synsets
- [x] Implement POS tagging for text analysis
- [x] Analyze corpus data for linguistic patterns
- [ ] Word sense disambiguation
- [ ] Semantic similarity metrics
- [ ] Named entity recognition

## 📖 Resources

- [NLTK Documentation](https://www.nltk.org/)
- [WordNet Official](https://wordnet.princeton.edu/)
- [NLTK WordNet Interface](https://www.nltk.org/howto/wordnet.html)
- [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus)

## 💡 Why This Matters

Natural Language Processing is fundamental to:
- Search engines understanding queries
- Virtual assistants comprehending commands
- Sentiment analysis in social media
- Machine translation systems
- Text summarization tools

Understanding these basics opens doors to advanced NLP applications like transformers, BERT, and GPT models.

---

**Last Updated:** January 25, 2026
