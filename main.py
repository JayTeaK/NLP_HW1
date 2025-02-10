#The tokenize part of the NLP 
from collections import Counter
import re

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    processed_Lines = []
    for line in lines:
        line = re.sub(r'[^\w\s]', '', line)
        processed_Lines.append(line.strip().split())

    return processed_Lines

def count_words(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    return word_counts

train_data = load_data("A1_DATASET/train.txt")
word_counts = count_words(train_data)

#Test to see if it works 
print(word_counts.most_common(10))