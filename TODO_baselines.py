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
import numpy as np
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

    # calculate matrix for further calculations
    m = np.zeros([2,2])
    for i in range(len(predictions)):
        if predictions[i] == "N" and predictions[i] == true_labels[i]:
            m[0,0] += 1
        elif predictions[i] == "N" and predictions[i] != true_labels[i]:
            m[1,0] += 1
        elif predictions[i] == "C" and predictions[i] == true_labels[i]:
            m[1,1] += 1
        else:
            m[0,1] += 1

    return accuracy, m


def random_baseline(train_sentence, train_labels, testinput, testlabels):
    labels = []
    for sentence in train_labels:
        train_tokens = sentence.split(" ")
        for token in train_tokens:
            token = token.replace('\n', '')
            labels.append(token)

    unique_labels = set(labels)

    accuracy = 0
    m = np.zeros([2,2])
    for k in range(10):
        predictions = []
        true_labels = []
        for instance in testlabels:
            tokens = instance.split(" ")
            tokens = [t.replace('\n', '') for t in tokens]
            instance_predictions = [sample([*unique_labels],1)[0] for t in tokens]
            predictions.append(instance_predictions)
            true_labels.append(tokens)

        # calculate accuracy for the test input
        predictions = list(chain.from_iterable(predictions))
        true_labels = list(chain.from_iterable(true_labels))
        accuracy += metrics.accuracy_score(true_labels,predictions)

        for i in range(len(predictions)):
            if predictions[i] == "N" and predictions[i] == true_labels[i]:
                m[0,0] += 1
            elif predictions[i] == "N" and predictions[i] != true_labels[i]:
                m[1,0] += 1
            elif predictions[i] == "C" and predictions[i] == true_labels[i]:
                m[1,1] += 1
            else:
                m[0,1] += 1

    accuracy /= 10

    return accuracy, m


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

    m = np.zeros([2,2])
    for i in range(len(predictions)):
        if predictions[i] == "N" and predictions[i] == true_labels[i]:
            m[0,0] += 1
        elif predictions[i] == "N" and predictions[i] != true_labels[i]:
            m[1,0] += 1
        elif predictions[i] == "C" and predictions[i] == true_labels[i]:
            m[1,1] += 1
        else:
            m[0,1] += 1

    return accuracy, m


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

    m = np.zeros([2,2])
    for i in range(len(predictions)):
        if predictions[i] == "N" and predictions[i] == true_labels[i]:
            m[0,0] += 1
        elif predictions[i] == "N" and predictions[i] != true_labels[i]:
            m[1,0] += 1
        elif predictions[i] == "C" and predictions[i] == true_labels[i]:
            m[1,1] += 1
        else:
            m[0,1] += 1

    return accuracy, m


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
            if length_baseline(i, dev_sentences, dev_labels)[0] > accuracy:
                threshold = i
                accuracy = length_baseline(i, dev_sentences, dev_labels)[0]

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
            if frequency_baseline(i, dev_sentences, dev_labels)[0] > accuracy:
                threshold = i
                accuracy = frequency_baseline(i, dev_sentences, dev_labels)[0]

    return threshold


def statistics(m):
    precision_n = m[0,0] / sum(m[:,0])
    recall_n = m[0,0] / sum(m[0,:])
    f1_n = 2*precision_n*recall_n / (precision_n+recall_n)

    precision_c = m[1,1] / sum(m[:,1])
    recall_c = m[1,1] / sum(m[1,:])
    f1_c = 2*precision_c*recall_c / (precision_c+recall_c)


    return precision_n, recall_n, f1_n, precision_c, recall_c, f1_c


def weighted_average(f1_n, f1_c, testlabels):
    labels = []
    for sentence in testlabels:
        test_tokens = sentence.split(" ")
        for token in test_tokens:
            token = token.replace('\n', '')
            labels.append(token)

    weighted_average = (f1_n*labels.count("N") + f1_c*labels.count("C")) / len(labels)

    return weighted_average


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

    accuracy_majority, majority_m = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    accuracy_random, random_m = random_baseline(train_sentences, train_labels, testinput, testlabels)
    threshold_length = optimal_threshold("length", dev_sentences, dev_labels)
    accuracy_length, length_m = length_baseline(threshold_length, testinput, testlabels)
    threshold_frequency = optimal_threshold("frequency", dev_sentences, dev_labels)
    accuracy_frequency, frequency_m = frequency_baseline(threshold_frequency, testinput, testlabels)

    # TODO: output the predictions in a suitable way so that you can evaluate them
    print("Majority baseline accuracy: ", accuracy_majority)
    print("Random baseline accuracy: ", accuracy_random)
    print("Length baseline accuracy: ", threshold_length, accuracy_length)
    print("Frequency baseline accuracy: ", threshold_frequency, accuracy_frequency)

    print("Majority baseline class N (precision, recall, f1): ", statistics(majority_m)[0:3])
    print("Majority baseline class C (precision, recall, f1): ", statistics(majority_m)[3:6])
    print("Random baseline class N (precision, recall, f1): ", statistics(random_m)[0:3])
    print("Random baseline class C (precision, recall, f1): ", statistics(random_m)[3:6])
    print("Length baseline class N (precision, recall, f1): ", statistics(length_m)[0:3])
    print("Length baseline class C (precision, recall, f1): ", statistics(length_m)[3:6])
    print("Frequency baseline class N (precision, recall, f1): ", statistics(frequency_m)[0:3])
    print("Frequency baseline class C (precision, recall, f1): ", statistics(frequency_m)[3:6])

    print("Majority baseline weighted average: ", weighted_average(statistics(majority_m)[2], statistics(majority_m)[5], testlabels))
    print("Random baseline weighted average: ", weighted_average(statistics(random_m)[2], statistics(random_m)[5], testlabels))
    print("Length baseline weighted average: ", weighted_average(statistics(length_m)[2], statistics(length_m)[5], testlabels))
    print("Frequency baseline weighted average: ", weighted_average(statistics(frequency_m)[2], statistics(frequency_m)[5], testlabels))