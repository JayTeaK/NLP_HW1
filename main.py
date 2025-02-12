#The tokenize part of the NLP 
from collections import Counter
import re

def load_data(filepath):
    # read data from file line by line, as each line corresponds to a review
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # preprocess the reviews
    return preprocess_text(lines)

# to deal with the non-alphabetical characters and case sensitivity
#TODO: implement the rest of the preprocessing
def preprocess_text(lines):
    processed_lines = []
    rating_pattern = r'\b(one|two|three|four|five|[1-5])[-\s]?stars?\b'
    for line in lines:
        # turn everything to lower case
        line = line.lower()

        # replace ratings with tokens
        line = re.sub(rating_pattern, '<RATING>', line)

        # replace dates with tokens

        # replace price with tokens

        # replace distances with tokens

        # remove all non-alphabetical characters

        processed_lines.append(line.split())
    
    return processed_lines

def count_words(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    return word_counts

# unigram model
#TODO: double check probability, add smoothing
class UnigramModel:
    def __init__(self):
        self.unigram_counts = {}
        self.total_words = 0

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.total_words = sum(self.word_counts.values())

    def probability(self, word):
        return self.unigram_counts[word] / self.total_words if self.total_words > 0 else 0

# bigram model
#TODO: finish this
class BigramModel:
    def __init__(self):
        self.bigram_counts = {}
        self.unigram_counts = {}
        self.total_bigrams = 0

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.bigram_counts = count_bigrams(corpus)
        self.total_bigrams = sum(self.bigram_counts.values())

    def probability(self, word1, word2):
        bigram = (word1, word2)
        return self.bigram_counts[bigram] / self.unigram_counts[word1] if self.unigram_counts[word1] > 0 else 0

train_data = load_data("A1_DATASET/train.txt")
word_counts = count_words(train_data)

#Test to see if it works 
print(word_counts.most_common(10))