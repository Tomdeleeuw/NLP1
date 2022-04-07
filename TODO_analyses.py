# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

## '\n' moet er uit
## Voor average alle worden tellen


import spacy
import pandas as pd
from collections import Counter
nlp = spacy.load('en_core_web_sm')


def tokenization(doc, show=True):
    sents = doc.sents
    num_sents = len(list(sents))

    frequencies = Counter()
    for sentence in doc.sents:
        words = []
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        frequencies.update(words)

    # Remove line endings
    # frequencies.pop('\n')

    num_tokens = len(doc)                   # amount of tokens is with the punctuation
    num_words = sum(frequencies.values())   # amount of words is the amount of tokens without the punctuation
    num_types = len(frequencies.keys())     # amount of keys is the amount of different words
    avg_words = num_words / num_sents       # number of words (without punctuation) divided by the number of sentences)
    avg_len = sum([len(word)*frequencies[word] for word in frequencies.keys()]) / num_words

    if show is True:
        print("Number of tokens: ", num_tokens, "\n",
              "Number of words: ", num_words, "\n",
              "Number of types: ", num_types, "\n",
              "Average number of words per sentence: ", round(avg_words, 3), "\n",
              "Average word length: ", round(avg_len, 3))

    return num_tokens, num_words, num_types, avg_words, avg_len


def pos(doc):
    tags = {}
    for token in doc:
        print(token.text, token.pos_, token.tag_)
        if token.tag_ not in tags:
            tags[token.tag_] = [token]
        else:
            tags[token.tag_].append(token)
    for tag in tags:
        print(tag, len(tags[tag]))
    print(tags["NN"])


if __name__ == "__main__":
    with open("data/preprocessed/train/sentences.txt") as sent_file:
        dataset = sent_file.read()

    dataset = dataset.replace('\n', '')
    # dataset = dataset.replace('"', '')
    # dataset = dataset.replace("-", '')

    doc = nlp(dataset)
    print(doc)
    # (out) = tokenization(doc)

    pos(doc)









