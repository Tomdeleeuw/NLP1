# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt


import spacy
import pandas as pd
from collections import Counter
nlp = spacy.load('en_core_web_sm')


# Task 1
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


# Task 2
def pos(doc):
    tags = {}
    uni = {}
    for token in doc:
        if token.tag_ not in tags:
            tags[token.tag_] = [token.lower_]
            uni[token.tag_] = token.pos_
        else:
            tags[token.tag_].append(token.lower_)

    # Create DataFrame object with finegrained tag, universal tag and occurrences
    data = pd.DataFrame({
        "tag": tags.keys(),
        "pos": [uni[tag] for tag in tags],
        "occ": [len(tags[tag]) for tag in tags]
    })

    # Add relative frequency
    data['freq'] = [round(data.loc[i, 'occ'] / sum(data['occ']), 4) for i in data.index]

    # Find most frequent and infrequent words for each tag
    common_words = []
    rare_words = []
    for tag in tags:
        frequencies = Counter()
        tokens = []
        for token in tags[tag]:
            tokens.append(token)
        frequencies.update(tokens)

        common_words.append([i[0] for i in frequencies.most_common(3)])
        rare_words.append([frequencies.most_common()[-1][0]])

    # Add common and uncommon words to dataframe
    data['common'] = common_words
    data['rare'] = rare_words

    # Sort on relative frequency
    data.sort_values("freq", ascending=False, inplace=True, ignore_index=True)

    return data


# Task 4
def lemmatization(doc):

    lemmas = {}
    for token in doc:
        # Check if lemma is the same as the original word
        if not token.lower_ == token.lemma_:
            if token.lemma_ not in lemmas.keys():
                lemmas[token.lemma_] = [token.lower_]
            else:
                lemmas[token.lemma_].append(token.lower_)

    # Find examples of more than 2 inflections
    lemma_inflections = {}
    for lemma, inflection in lemmas.items():
        if len(set(inflection)) > 2:
            if lemma not in lemma_inflections.keys():
                lemma_inflections[lemma] = set(inflection)
            else:
                lemma_inflections[lemma].add(set(inflection))

    selected_lemma = "run"
    inflections = lemma_inflections["run"]  # run {'runs', 'running', 'ran'}

    # Find sentence with inflection in doc
    inflected_sentences = []
    for sentence in doc.sents:
        for token in sentence:
            if token.lower_ in inflections:
                inflected_sentences.append(sentence)

    return selected_lemma, inflections, inflected_sentences


if __name__ == "__main__":
    with open("data/preprocessed/train/sentences.txt", encoding="utf-8") as sent_file:
        dataset = sent_file.read()

    # Replace line endings by blank space
    dataset = dataset.replace('\n', ' ')

    doc = nlp(dataset)
    task_1 = tokenization(doc)
    task_2 = pos(doc)

    selected_lemma, inflections, inflected_sentences = lemmatization(doc)
    print(selected_lemma, inflections)
    [print(sen) for sen in inflected_sentences]