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
    # TODO: determine the majority class based on the training data
    labels = []
    for sentence in train_labels:
        train_tokens = sentence.split(" ")
        for token in train_tokens:
            token = token.replace('N\n', 'N')
            labels.append(token)

    majority_class = str(max(Counter(labels).keys()))
    predictions = []
    true_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        tokens = [t.replace('N\n', 'N') for t in tokens]
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
        true_instance = [tokens]
        true_labels.append(tokens)

    # TODO: calculate accuracy for the test input
    predictions = list(chain.from_iterable(predictions))
    true_labels = list(chain.from_iterable(true_labels))
    accuracy = metrics.accuracy_score(true_labels,predictions)

    return accuracy, predictions


def random_baseline(train_sentence, train_labels, testinput, testlabels):
    labels = []
    for sentence in train_labels:
        train_tokens = sentence.split(" ")
        for token in train_tokens:
            token = token.replace('N\n', 'N')
            labels.append(token)

    unique_labels = set(labels)

    predictions = []
    true_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        tokens = [t.replace('N\n', 'N') for t in tokens]
        instance_predictions = [sample([*unique_labels],1)[0] for t in tokens]
        predictions.append(instance_predictions)
        true_instance = [tokens]
        true_labels.append(tokens)

    # TODO: calculate accuracy for the test input
    right_labels = 0
    total_labels = 0
    accuracy = 0
    for k in range(10):
        predictions = list(chain.from_iterable(predictions))
        true_labels = list(chain.from_iterable(true_labels))
        accuracy += metrics.accuracy_score(true_labels,predictions)
    accuracy /= 10

    return accuracy, predictions


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

    with open(train_path + "/labels.txt", encoding='cp1252', errors='ignore') as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "/sentences.txt", encoding='cp1252', errors='ignore') as testfile:
        testinput = testfile.readlines()

    with open(test_path + "/labels.txt", encoding='cp1252', errors='ignore') as test_label_file:
        testlabels = test_label_file.readlines()

    accuracy, predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    accuracy2, predictions2 = random_baseline(train_sentences, train_labels, testinput, testlabels)
    # TODO: output the predictions in a suitable way so that you can evaluate them
    print(accuracy, accuracy2)