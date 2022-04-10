# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt


import spacy
import pandas as pd
from collections import Counter
nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe("merge_entities")


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

    return [num_tokens, num_words, num_types, avg_words, avg_len]


def pos(doc):
    tags = {}
    uni = {}
    for token in doc:
        if token.tag_ not in tags:
            tags[token.tag_] = [token]
            uni[token.tag_] = token.pos_
        else:
            tags[token.tag_].append(token)

    # Create DataFrame object with finegrained tag, universal tag and occurrences
    data = pd.DataFrame({"tag": tags.keys(), "pos": [uni[tag] for tag in tags],
                         "occ": [len(tags[tag]) for tag in tags]})

    # Add relative frequency
    data['freq'] = [round(data.loc[i, 'occ'] / sum(data['occ']), 4) for i in data.index]

    # Find most frequent and infrequent words for each tag
    common_words = []
    rare_words = []
    for tag in tags:
        frequencies = Counter()
        tokens = []
        for token in tags[tag]:
            tokens.append(token.text)
        frequencies.update(tokens)

        common_words.append([i[0] for i in frequencies.most_common(3)])
        rare_words.append([frequencies.most_common()[-1][0]])

    # Add common and uncommon words to dataframe
    data['common'] = common_words
    data['rare'] = rare_words

    # Sort on relative frequency
    data.sort_values("freq", ascending=False, inplace=True, ignore_index=True)

    return data

def ngrams(doc, n):
    frequencies = Counter()
    words = []
    posFrequencies = Counter()
    pos = []
    for i, token in enumerate(doc):
        if i < len(doc)-n:
            if not doc[i].is_punct:
                words.append(doc[i:i+n].text)

            string = ""
            for j in range(n):
                string += doc[i+j].pos_
            pos.append(string)

    frequencies.update(words)
    print(frequencies)
    posFrequencies.update(pos)
    print(posFrequencies)


def entities(doc):
    entities = []
    frequencies = Counter()
    for i, sentence in enumerate(doc.sents):
        if i < 5:
            for token in sentence:
                entities.append(token.ent_type_)

    frequencies.update(entities)
    print(frequencies)
    print(ents)


def entities2(doc):
    ents = []
    entities = []
    frequencies = Counter()
    for i, sentence in enumerate(doc.sents):
        if i < 5:
            for ent in sentence.ents:
                entities.append(ent.label_)
                ents.append(ent.text)

    frequencies.update(entities)
    print(frequencies)
    print(ents)


if __name__ == "__main__":
    with open("data/preprocessed/train/sentences.txt", encoding='cp1252', errors='ignore') as sent_file:
        dataset = sent_file.read()

    # Replace line endings by blank space
    dataset = dataset.replace('\n', ' ')

    doc = nlp(dataset)
    tokenization(doc)
    data = pos(doc)
    print(data)
    # ngrams(doc,2)
    # ngrams(doc,3)
    # entities(doc)
    entities2(doc)
