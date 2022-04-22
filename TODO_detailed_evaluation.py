import numpy as np
# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

def process_output(model_output):
    m = np.zeros([2,2])
    true_labels = []
    for sentence in model_output:
        tokens = sentence.split("\t")
        tokens = [t.replace('\n', '') for t in tokens]

        if len(tokens) > 1:
            true_labels.append(tokens[1])
            if tokens[1] == tokens[2] == "N":
                m[0,0] += 1
            elif tokens[1] == "N":
                m[0,1] += 1
            elif tokens[1] == tokens[2] == "C":
                m[1,1] += 1
            else:
                m[1,0] += 1

    return m, true_labels


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
    with open("experiments/base_model/model_output.tsv", encoding='cp1252', errors='ignore') as sent_file:
        model_output = sent_file.readlines()

    m, true_labels = process_output(model_output)
    print("LSTM model class N (precision, recall, f1): ", statistics(m)[0:3])
    print("LSTM model class C (precision, recall, f1): ", statistics(m)[3:6])
    print("LSTM model weighted average: ", weighted_average(statistics(m)[2], statistics(m)[5], true_labels))