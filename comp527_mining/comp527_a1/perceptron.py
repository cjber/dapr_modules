import numpy as np
import matplotlib.pyplot as plt


def perceptron_train(data, num_iter):

    # assume features all excluding final row
    features = data[:, :-1]
    # labels as final row
    labels = data[:, -1]

    # init weights as zero in correct shape
    w = np.zeros(shape=(1, features.shape[1]+1))
    w = w[0]
    b = 0
    misclassified_ = []  # empty list
    for _ in range(num_iter):
        misclassified = 0
        for x, y in zip(features, labels):
            # add bias as w0
            w[0] = b
            # always on x0
            x = np.insert(x, 0, 1)
            # dot product of weights and features
            a = np.dot(w, x.transpose())
            if y*a <= 0:
                misclassified += 1
                w += (y*x)
                b += y
        misclassified_.append(misclassified)
    return (w, misclassified_, b)
