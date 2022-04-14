# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

#from model.data_loader import DataLoader
from collections import Counter
from random import sample
from sklearn import metrics
from itertools import chain
# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    # determine the majority class based on the training data
    labels = []
    for sentence in train_labels:
        train_tokens = sentence.split(" ")
        for token in train_tokens:
            token = token.replace('\n', '')
            labels.append(token)

    majority_class = str(max(Counter(labels).keys()))
    predictions = []
    true_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        tokens = [t.replace('\n', '') for t in tokens]
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
        true_labels.append(tokens)

    # calculate accuracy for the test input
    predictions = list(chain.from_iterable(predictions))
    true_labels = list(chain.from_iterable(true_labels))
    accuracy = metrics.accuracy_score(true_labels,predictions)

    return accuracy, predictions


def random_baseline(train_sentence, train_labels, testinput, testlabels):
    labels = []
    for sentence in train_labels:
        train_tokens = sentence.split(" ")
        for token in train_tokens:
            token = token.replace('\n', '')
            labels.append(token)

    unique_labels = set(labels)

    predictions = []
    true_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        tokens = [t.replace('\n', '') for t in tokens]
        instance_predictions = [sample([*unique_labels],1)[0] for t in tokens]
        predictions.append(instance_predictions)
        true_labels.append(tokens)

    # calculate accuracy for the test input
    accuracy = 0

    for k in range(10):
        predictions = list(chain.from_iterable(predictions))
        true_labels = list(chain.from_iterable(true_labels))
        accuracy += metrics.accuracy_score(true_labels,predictions)
    accuracy /= 10

    return accuracy, predictions


def length_baseline(threshold, sentences, labels):
    true_labels = []
    for sentence in labels:
        tokens = sentence.split(" ")
        tokens = [t.replace('\n', '') for t in tokens]
        true_labels.append(tokens)

    predictions = []
    for sentence in sentences:
        tokens = sentence.split(" ")
        for token in tokens:
            token = token.replace('\n', '')
            if len(token) > threshold:
                predictions.append("C")
            else:
                predictions.append("N")

    true_labels = list(chain.from_iterable(true_labels))
    accuracy = metrics.accuracy_score(true_labels,predictions)

    return accuracy


def frequency_baseline(threshold, sentences, labels):
    true_labels = []
    for sentence in labels:
        tokens = sentence.split(" ")
        tokens = [t.replace('\n', '') for t in tokens]
        true_labels.append(tokens)

    words = []
    for sentence in sentences:
        tokens = sentence.split(" ")
        tokens = [t.replace('\n', '') for t in tokens]
        words.append(tokens)
    words = list(chain.from_iterable(words))
    frequencies = Counter(words)

    predictions = []
    for sentence in sentences:
        tokens = sentence.split(" ")
        for token in tokens:
            token = token.replace('\n', '')
            if frequencies[token] > threshold:
                predictions.append("N")
            else:
                predictions.append("C")

    true_labels = list(chain.from_iterable(true_labels))
    accuracy = metrics.accuracy_score(true_labels,predictions)

    return accuracy


def optimal_threshold(type, dev_sentences, dev_labels):
    if type == "length":
        max_len = 0

        for sentence in dev_sentences:
            tokens = sentence.split(" ")
            tokens = [t.replace('\n', '') for t in tokens]
            if len(max(tokens, key=len)) > max_len:
                max_len = len(max(tokens, key=len))

        accuracy = 0

        for i in range(max_len):
            if length_baseline(i, dev_sentences, dev_labels) > accuracy:
                threshold = i
                accuracy = length_baseline(i, dev_sentences, dev_labels)

    else:
        words = []

        for sentence in dev_sentences:
            tokens = sentence.split(" ")
            for token in tokens:
                token = token.replace('\n', '')
            words.append(token)
        max_freq = max(Counter(words).values())

        accuracy = 0

        for i in range(max_freq):
            if frequency_baseline(i, dev_sentences, dev_labels) > accuracy:
                threshold = i
                accuracy = frequency_baseline(i, dev_sentences, dev_labels)

    return threshold


if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/val"
    test_path = "data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "/sentences.txt", encoding='cp1252', errors='ignore') as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "/labels.txt", encoding='cp1252', errors='ignore') as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "/sentences.txt", encoding='cp1252', errors='ignore') as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "/labels.txt", encoding='cp1252', errors='ignore') as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "/sentences.txt", encoding='cp1252', errors='ignore') as testfile:
        testinput = testfile.readlines()

    with open(test_path + "/labels.txt", encoding='cp1252', errors='ignore') as test_label_file:
        testlabels = test_label_file.readlines()

    accuracy_majority, predictions_majority = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    accuracy_random, predictions_random = random_baseline(train_sentences, train_labels, testinput, testlabels)
    threshold_length = optimal_threshold("length", dev_sentences, dev_labels)
    accuracy_length = length_baseline(threshold_length, testinput, testlabels)
    threshold_frequency = optimal_threshold("frequency", dev_sentences, dev_labels)
    accuracy_frequency = frequency_baseline(threshold_frequency, testinput, testlabels)
    # TODO: output the predictions in a suitable way so that you can evaluate them
    print("Majority baseline: ", accuracy_majority)
    print("Random baseline: ", accuracy_random)
    print("Length baseline: ", threshold_length, accuracy_length)
    print("Frequency baseline: ", threshold_frequency, accuracy_frequency)