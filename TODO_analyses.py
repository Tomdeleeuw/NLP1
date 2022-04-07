# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

## '\n' moet er uit
## Voor average alle worden tellen

import spacy
from collections import Counter
nlp = spacy.load('en_core_web_sm')

if __name__ == "__main__":
    with open("data/preprocessed/train/sentences.txt") as sent_file:
        train_sentences = sent_file.read()

    doc = nlp(train_sentences)
    sentences = doc.sents
    num_sents = len(list(sentences))
    print(num_sents)
    # for sentence in sentences:
    #     print()
    #     print(sentence)
    #     for token in sentence:
    #         print(token.text)

    word_frequencies = Counter()

    for sentence in doc.sents:
        words = []
        for token in sentence:
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        word_frequencies.update(words)

    # print(word_frequencies)

    num_tokens = len(doc) #amount of tokens is with the punctuation
    num_words = sum(word_frequencies.values()) #amount of words is the amount of tokens without the puctuation
    num_types = len(word_frequencies.keys()) #amount of keys is the amount of different words

    print(num_tokens, num_words, num_types)

    average_word = num_words/num_sents #average is the number of words (without puctuation divided by the number of sentences)

    # Average word length
    all_words = list(word_frequencies)
    print(word_frequencies)
    print(list(word_frequencies))

    all_lengths = []
    num_of_strings = len(all_words)

    for item in all_words:
        string_size = len(item)
        all_lengths.append(string_size)
        total_size = sum(all_lengths)
    ave_size = float(total_size) / float(num_of_strings)
    print(ave_size)







