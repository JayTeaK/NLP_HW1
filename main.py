# the tokenize part of the NLP 
from collections import Counter
from collections import defaultdict
import math
import re

def load_data(filepath, vocab=None, min_freq=1):
    # read data from file line by line, as each line corresponds to a review
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # preprocess the reviews
    return preprocess_text(lines, vocab, min_freq)

# to deal with the non-alphabetical characters and case sensitivity
def preprocess_text(lines, vocab=None, min_freq=1):
    processed_lines = []
    word_counts = Counter()

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

        words = line.split()
        word_counts.update(words)
        processed_lines.append(words)

    if vocab is None:
        vocab = {word for word, count in word_counts.items() if count >= min_freq}
        vocab.add('<UNK>')

    processed_lines = [[word if word in vocab else '<UNK>' for word in line] for line in processed_lines]
    
    return processed_lines, vocab

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
    
    def smooth_probability(self, word1, word2, k):
        bigram = (word1, word2)
        return (self.bigram_counts[bigram] + k) / (self.unigram_counts[word1] + k * len(self.unigram_counts)) if word1 in self.unigram_counts else k / (k * len(self.unigram_counts))
        

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

# kneser-ney smoothing bigram
class KneserNeyBigramModel:
    def __init__(self, discount=0.75):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.continuation_counts = defaultdict(int)
        self.total_bigrams = 0
        self.discount = discount

    def train(self, corpus):
        self.unigram_counts = count_words(corpus)
        self.bigram_counts = count_bigrams(corpus)
        self.total_bigrams = sum(self.bigram_counts.values())

        for bigram in self.bigram_counts:
            self.continuation_counts[bigram[1]] += 1
    
    def probability(self, word1, word2):
        bigram = (word1, word2)
        unigram_count = self.unigram_counts[word1]
        bigram_count = self.bigram_counts[bigram]
        continuation_count = self.continuation_counts[word2]
        total_unique_bigrams = len(self.bigram_counts)

        if unigram_count > 0:
            p_continuation = continuation_count / total_unique_bigrams
            p_bigram = max(bigram_count - self.discount, 0) / unigram_count
            lambda_w1 = (self.discount / unigram_count) * len([w for w in self.bigram_counts if w[0] == word1])
            return p_bigram + lambda_w1 * p_continuation
        else:
            return 0

# perplexity for unigram
def unigram_perplexity(model, corpus):
    totalProbabilityLog = 0
    totalWords = 0

    for i in corpus:
        sentenceLogProb = 0
        for j in i:
            sentenceLogProb += math.log2(model.probability(j))
        totalProbabilityLog += sentenceLogProb
        totalWords += len(i)
    return 2 ** -(totalProbabilityLog / totalWords) if totalWords > 0 else float("inf")

# perplexity for bigram
def bigram_perplexity_k(model, corpus, k, epsilon=1e-10):
    totalProbabilityLog = 0
    totalBigrams = 0

    for i in corpus:
        for j in range(len(i) - 1):
            word1 = i[j]
            word2 = i[j + 1]
            totalBigrams += 1
            sentenceLogProb = model.smooth_probability(word1, word2, k)
            if sentenceLogProb > 0:
                totalProbabilityLog += math.log2(sentenceLogProb + epsilon)
            else:
                totalProbabilityLog += math.log2(epsilon)
    return 2 ** -(totalProbabilityLog / totalBigrams) if totalBigrams > 0 else float("inf")

# perplexity for kneser-ney bigram
def bigram_kneser_perplexity(model, corpus, epsilon=1e-10):
    totalProbabilityLog = 0
    totalBigrams = 0

    for i in corpus:
        for j in range(len(i) - 1):
            word1 = i[j]
            word2 = i[j + 1]
            totalBigrams += 1
            sentenceLogProb = model.probability(word1, word2)
            if sentenceLogProb > 0:
                totalProbabilityLog += math.log2(sentenceLogProb + epsilon)
            else:
                totalProbabilityLog += math.log2(epsilon)
    return 2 ** -(totalProbabilityLog / totalBigrams) if totalBigrams > 0 else float("inf")


# load training data
train_data, vocab = load_data("A1_DATASET/train.txt", min_freq=2)
word_counts = count_words(train_data)

# initializing unigram and bigram
uniModel = UnigramModel()
biModel = BigramModel()
laplaceUniModel = LaplaceUnigramModel()
KneserNeyBiModel = KneserNeyBigramModel()
uniModel.train(train_data)
biModel.train(train_data)
laplaceUniModel.train(train_data)
KneserNeyBiModel.train(train_data)

# unigram model probability test
print("---  PROBABILITY AND WORD COUNT RAW  ---")
print("Unigram probability of 'the':", uniModel.probability("the"))
print("Unigram total words:", uniModel.total_words)

# bigram model probability test
print("Bigram probability of 'the', 'hotel':", biModel.probability('the', 'hotel'))
print("Bigram total bigrams:", biModel.total_bigrams)
print("\n")

print("---  PROBABILITY AND WORD COUNT SMOOTHED  ---")
# bigram model probability test with k smoothing
print("Bigram probability of 'the', 'hotel' with k smoothing:", biModel.smooth_probability('the', 'hotel', 0.1))
print("Bigram total bigrams:", biModel.total_bigrams)

# laplace smoothing unigram model probability test
print("Laplace unigram probability of 'the':", laplaceUniModel.probability("the"))
print("Laplace total words:", laplaceUniModel.total_words)

# kneser-ney smoothing bigram model probability test
print("Kneser-Ney bigram probability of 'the', 'hotel':", KneserNeyBiModel.probability('the', 'hotel'))
print("Kneser-Ney total bigrams:", KneserNeyBiModel.total_bigrams)
print("\n")

print("---  PERPLEXITY TEST  ---")
# load validation data
val_data, _ = load_data("A1_DATASET/val.txt", vocab=vocab)

print("\n")
print("-Validation Unsmoothed-")
print("Unigram perplexity of \"val.txt\":", unigram_perplexity(uniModel, val_data))
print("Bigram perplexity of \"val.txt\":", bigram_perplexity_k(biModel, val_data, 0))
print("\n")

# unigram perplexity with val.txt
print("Unigram perplexity of \"val.txt\", laplace smoothed:", unigram_perplexity(laplaceUniModel, val_data))

# bigram perplexity with val.txt
print("Bigram perplexity of \"val.txt\", k = 0.01 smoothed:", bigram_perplexity_k(biModel, val_data, 0.01))
print("Bigram perplexity of \"val.txt\", k = 0.1 smoothed:", bigram_perplexity_k(biModel, val_data, 0.1))
print("Bigram perplexity of \"val.txt\", k = 1 smoothed:", bigram_perplexity_k(biModel, val_data, 1))
print("Bigram perplexity of \"val.txt\", k = 5 smoothed:", bigram_perplexity_k(biModel, val_data, 5))

# kneser-ney Bigram perplexity with val.txt
print("Bigram kneser-ney perplexity of \"val.txt\", :", bigram_kneser_perplexity(KneserNeyBiModel, val_data))
