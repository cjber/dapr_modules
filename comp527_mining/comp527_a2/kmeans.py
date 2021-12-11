import numpy as np
import random
import pandas as pd

random.seed(1)


def read_data(categories):
    data = pd.DataFrame()
    for c in categories:
        d = pd.read_table('./clustering-data/' + c, delimiter=' ', header=None)
        d['category'] = c
        d.rename(columns={0: 'word'}, inplace=True)
        data = data.append(d)
    return data


def normalise(vectors):
    row_sums = vectors.sum(axis=1)
    return vectors / row_sums[:, np.newaxis]


def euclidean_distance(vectors, centroids, K):
    euc_dist = np.array([]).reshape(vectors.shape[0], 0)
    for k in range(K):
        temp_dist = np.sum((vectors - centroids[:, k])**2, axis=1)
        euc_dist = np.c_[euc_dist, temp_dist]
    return euc_dist


def manhattan_distance(vectors, centroids, K):
    man_dist = np.array([]).reshape(vectors.shape[0], 0)
    for k in range(K):
        temp_dist = np.sum((abs(vectors - centroids[:, k])), axis=1)
        man_dist = np.c_[man_dist, temp_dist]
    return man_dist


def cosine_similarity(vectors, centroids, K):
    cos_dist = np.array([]).reshape(vectors.shape[0], 0)
    for k in range(K):
        temp_dist = np.sum((np.cos(vectors - centroids[:, k])), axis=1)
        cos_dist = np.c_[cos_dist, temp_dist]
    return cos_dist


class K_Means:
    def __init__(self,
                 data,
                 k=4,
                 max_iterations=50,
                 distance='euclidean_distance',
                 normalised=False,
                 evaluate=True):
        self.k = k
        self.max_iterations = max_iterations
        self.eval = evaluate
        self.data = data

        self.distance = distance
        self.normalised = normalised

    def fit(self, cols=range(1, 301)):
        self.vectors = self.data[cols].values.astype(float)
        self.vectors = normalise(
            self.vectors) if self.normalised else self.vectors

        m = self.vectors.shape[0]
        n = self.vectors.shape[1]

        centroids = np.array([]).reshape(n, 0)
        for _ in range(self.k):
            rand = random.randint(0, m-1)
            centroids = np.c_[centroids, self.vectors[rand]]

        for _ in range(self.max_iterations):
            if self.distance == 'euclidean_distance':
                euc_dist = euclidean_distance(self.vectors, centroids, self.k)
                c = np.argmin(euc_dist, axis=1)+1
                self.data['cluster'] = c
            elif self.distance == 'manhattan_distance':
                man_dist = manhattan_distance(self.vectors, centroids, self.k)
                c = np.argmin(man_dist, axis=1)+1
                self.data['cluster'] = c
            elif self.distance == 'cosine_distance':
                cos_dist = cosine_similarity(self.vectors, centroids, self.k)
                c = np.argmin(cos_dist, axis=1)+1
                self.data['cluster'] = c

        y = {k+1: np.array([]).reshape(n, 0) for k in range(self.k)}
        for i in range(m):
            y[c[i]] = np.c_[y[c[i]], self.vectors[i]]
        for k in range(self.k):
            y[k+1] = y[k+1].T
        for k in range(self.k):
            centroids[:, k] = np.mean(y[k+1], axis=0)

        if self.eval:
            self.evaluate()
        return self.data

    def evaluate(self):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(self.data)):
            for j in range(i+1, len(self.data)):
                if self.data.iloc[i]['category'] ==\
                        self.data.iloc[j]['category'] and\
                        self.data.iloc[i]['cluster'] ==\
                        self.data.iloc[j]['cluster']:
                    true_positives += 1
                elif self.data.iloc[i]['category'] !=\
                        self.data.iloc[j]['category'] and\
                        self.data.iloc[i]['cluster'] !=\
                        self.data.iloc[j]['cluster']:
                    true_negatives += 1
                elif self.data.iloc[i]['category'] !=\
                        self.data.iloc[j]['category']:
                    false_positives += 1
                else:
                    false_negatives += 1

        self.precision = true_positives / \
            (true_positives + false_positives)
        self.recall = true_positives / \
            (true_positives + false_negatives)
        self.f_score = 2 * self.precision * \
            self.recall / (self.precision + self.recall)
