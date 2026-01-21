from nltk.corpus import wordnet as wn
syn=wn.synsets("computer")[0]
print("Synset name:", syn.name())
print('sysnsets abstract term:',syn.hypernyms())
print("Synsets specific term;", syn.hypernyms()[0].hyponyms())
print("Definitions:", syn.definition())
print("Root hypernyms:", syn.root_hypernyms())