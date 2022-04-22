import matplotlib.pyplot as plt

LR = {0.0001: 0.7116, 0.001: 0.8602, 0.36: 0.7287, 0.135: 0.8602, 0.05: 0.8385, 0.25: 0.77, 0.5: 0.7346, 0.4: 0.75,
      0.1: 0.84, 0.22: 0.75}

plt.scatter(list(LR.keys()), list(LR.values()))
plt.xlabel('Learning rate')
plt.ylabel('F1 score')
plt.title('F1 score for different learning rates')
plt.show()
