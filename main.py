# the tokenize part of the NLP 
from collections import Counter
from collections import defaultdict
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
    date_pattern = r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b'
    price_pattern = r'\$\d+(\.\d{1,2})?'
    distances_pattern = r'\b\d+(\.\d+)?\s?(miles|mile|km|kilometers|meters|m)\b'
    for line in lines:
        # turn everything to lower case
        line = line.lower()

        # replace ratings with tokens
        line = re.sub(rating_pattern, '<RATING>', line)

        # replace dates with tokens
        line = re.sub(date_pattern, '<DATE>', line)

        # replace price with tokens
        line = re.sub(price_pattern, '<PRICE>', line)

        # replace distances with tokens
        line = re.sub(distances_pattern, '<DISTANCE>', line)

        # remove all non-alphabetical characters
        line = re.sub(r'[^a-z\s]', '', line)

        processed_lines.append(line.split())
    
    return processed_lines

def count_words(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    return word_counts

def count_bigrams(corpus):
    bigram_count = defaultdict(int)
    for i in corpus:
        for j in range(len(i) - 1):
            bigram = (i[j], i[j + 1])
            bigram_count[bigram] += 1
    return bigram_count

# unigram model
#TODO: double check probability, add smoothing
class UnigramModel:
    def __init__(self):
        self.unigram_counts = {}
        self.total_words = 0

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.total_words = sum(self.unigram_counts.values())

    def probability(self, word):
        return self.unigram_counts[word] / self.total_words if self.total_words > 0 else 0

# bigram model
#TODO: finish this
class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.total_bigrams = 0

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.bigram_counts = count_bigrams(corpus)
        self.total_bigrams = sum(self.bigram_counts.values())

    def probability(self, word1, word2):
        bigram = (word1, word2)
        return self.bigram_counts[bigram] / self.unigram_counts[word1] if self.unigram_counts[word1] > 0 else 0

# laplace smoothing unigram
class LaplaceUnigramModel:
    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.total_words = 0
        self.vocab_size = 0

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.total_words = sum(self.unigram_counts.values())
        self.vocab_size = len(self.unigram_counts)

    def probability(self, word):
        return (self.unigram_counts[word] + 1) / (self.total_words + self.vocab_size)

# load training data
train_data = load_data("A1_DATASET/train.txt")
word_counts = count_words(train_data)

#initializing unigram and bigram
uniModel = UnigramModel()
biModel = BigramModel()
laplaceUniModel = LaplaceUnigramModel()
uniModel.train(train_data)
biModel.train(train_data)
laplaceUniModel.train(train_data)

# unigram model test
print("Unigram probability of 'the':", uniModel.probability("the"))
print("Unigram total words:", uniModel.total_words)

# bigram model test
print("Bigram probability of 'the':", biModel.probability('the', 'hotel'))
print("Bigram total bigrams:", biModel.total_bigrams)

# laplace smoothing unigram model test
print("Laplace unigram probability of 'the':", laplaceUniModel.probability("the"))
print("Laplace total words:", laplaceUniModel.total_words)

#print(uniModel.unigram_counts)
#print(biModel.bigram_counts)
print(laplaceUniModel.unigram_counts)
#print(sorted(biModel.bigram_counts.items(), key=lambda x: x[1], reverse=True)[:5])
