# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt


import spacy
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe("merge_entities")


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


# Task 3
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


# Task 5
def entities(doc):
    entities = []
    frequencies = Counter()
    for i, sentence in enumerate(doc.sents):
        if i < 5:
            for token in sentence:
                entities.append(token.ent_type_)

    frequencies.update(entities)
    print(frequencies)
    print(entities)


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



# Task 6
# a) The start offset refers to the index of the first letter of the target word.
# The end offset referes to the index of the last letter of the target word.
# If we take for example the following sentence:
# “The tail of Epidexipteryx also bore unusual vertebrae towards the tip which resembled the feather-anchoring pygostyle of modern birds and some oviraptorosaurs .”,
# with the target word “Epidexipteryx”.
# T	h	e		t	a	i	l		o	f
# 0	1	2	3	4	5	6	7	8	9	10	11
#
# E	p	i	i	d	e	x	i	p	t	e	r	y	x		a	l	s	o
# 12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30
# In this example the start offset would be 12 and the end offset 25.

# b) The probabilistic label is calculated by: the number of annotators who marked the word as difficult / the total number of annotators.
# This means that if there were for example 10 native and 10 annotative annotators, we have in total 10 + 10 = 20 annotators.
# And 2 native and 6 non-native annotators marked the sentence as difficult, in total 2 + 6 = 8 annotators.
# The probabilistic label would be 8 / 20 = 0.4. This value 0.4 means that 40 percent of all the annotators marked the sentence as difficult.

# c) For the binary label there is no distinction between native or non-native speakers as this is simply not possible to capture in one binary label.
# If you would like to make a distinction you could make two binary labels, one for native and non-native respectively.
#
# For the probabilistic label there is again no differentiation for native and non-native as it simply describes the fraction of annotators that signal the word as difficult.
# One possibility could be to make a weighted probabilistic label where you would give either the opinion of the native or the non-native annotators more relative importance.


def make_task_7_and_8_data():
    task_7_and_8_data = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t', header=None)
    task_7_and_8_data['strip_tokens'] = task_7_and_8_data[4].str.replace(r'<[^>]*>', '', regex=True).str.strip()
    task_7_and_8_data['strip_tokens'] = task_7_and_8_data['strip_tokens'].str.replace('-', ' ', regex=True).str.strip()
    # task_7_and_8_data['amount_tokens'] = task_7_and_8_data['strip_tokens'].str.split().str.len()
    task_7_and_8_data['amount_tokens'] = [len(nlp(token)) for token in task_7_and_8_data[4]]
    return task_7_and_8_data


# Task 7
def extract_basic_statistic():
    task_7_data = make_task_7_and_8_data()
    # The binary information
    print("binary information : ")
    print(task_7_data[9].value_counts())

    # The probabilistic information
    print("probabilitic information :")
    print("minimum = ", task_7_data[10].min())
    print("maximum = ", task_7_data[10].max())
    print("median = ", task_7_data[10].median())
    print("mean = ", task_7_data[10].mean().round(3))
    print("standard deviation = ", task_7_data[10].std().round(2))

    # The target information
    print("target word information (False -> <= 1, True -> > 1):")
    task_7_data['strip_tokens'] = task_7_data[4].str.replace(r'<[^>]*>', '', regex=True).str.strip()
    task_7_data['amount_tokens'] = task_7_data['strip_tokens'].str.split().str.len()
    print((task_7_data['amount_tokens'] > 1).value_counts())
    print("maximum number of tokens = ", task_7_data['amount_tokens'].max())

    return None


# Task 8
def linguistic_characteristics():
    # Create dataframe
    task_8_data = make_task_7_and_8_data()

    task_8_data = task_8_data.loc[np.logical_and(task_8_data[9] == 1, task_8_data['amount_tokens'] == 1)]
    task_8_data.reset_index(inplace=True)

    # Add a column for the number of characters in a token
    task_8_data["number_characters"] = task_8_data["strip_tokens"].str.len()

    # Add column for word frequency
    task_8_data["word_frequency"] = pd.Series([word_frequency(word, "en") for word in task_8_data["strip_tokens"]])
    print(len(task_8_data))

    task_8_data["POS"] = [token.pos_ for sent in task_8_data["strip_tokens"] for token in nlp(sent)]

    # Calculate correlations
    len_cor = np.corrcoef(task_8_data['number_characters'], task_8_data[10])[0, 1]
    freq_cor = np.corrcoef(task_8_data['word_frequency'], task_8_data[10])[0, 1]
    print('correlation between word length and complexity: ', len_cor)
    print('correlation between word frequency and complexity: ', freq_cor)

    # Create figures
    fig1 = px.scatter(task_8_data, x='word_frequency', y=10,
                      labels={'word_frequency': 'Word frequency',
                              '10': 'Probabilistic complexity'},
                      title='Probabilistic complexity and word frequency')
    # fig1.show()

    fig2 = px.scatter(task_8_data, x='number_characters', y=10,
                      labels={'number_characters': 'Word length',
                              '10': 'Probabilistic complexity'},
                      title='Probabilistic complexity and word length')
    # fig2.show()

    fig3 = px.scatter(task_8_data, x='POS', y=10,
                      labels={'POS': "POS tag",
                              '10': 'Probabilistic complexity'},
                      title='Probabilistic complexity and POS tags')
    # fig3.show()

    return None



if __name__ == "__main__":
    with open("data/preprocessed/train/sentences.txt", encoding='cp1252', errors='ignore') as sent_file:
        dataset = sent_file.read()

    # Replace line endings by blank space
    dataset = dataset.replace('\n', ' ')

    doc = nlp(dataset)
    # tokenization(doc)
    # data = pos(doc)
    # print(data)
    # ngrams(doc,2)
    # ngrams(doc,3)
    # entities(doc)
    # entities2(doc)

    # selected_lemma, inflections, inflected_sentences = lemmatization(doc)
    # print(selected_lemma, inflections)
    # [print(sen) for sen in inflected_sentences]
    linguistic_characteristics()