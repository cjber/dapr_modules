import matplotlib.pyplot as plt
import numpy as np


class Perceptron(object):
    def __init__(self, num_epochs: int, lr: float) -> None:
        self.num_epochs = num_epochs
        self.lr = lr

    def predict(self, input: np.ndarray) -> int:
        activation = np.dot(input, self.weights[1:]) + self.weights[0]
        return 1 if activation > 0 else 0

    def train(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = data
        self.labels = labels
        self.weights = np.random.rand(self.data.shape[1] + 1)

        for epoch in range(self.num_epochs):
            accuracy = 0
            assert len(self.data) == len(self.labels)
            for idx, _ in enumerate(self.data):
                prediction = self.predict(self.data[idx])

                self.weights[1:] += (
                    self.lr * (self.labels[idx] - prediction) * self.data[idx]
                )
                self.weights[0] += self.lr * (self.labels[idx] - prediction)

                accuracy += 1 if prediction == self.labels[idx] else 0
            print(
                f"Epoch {epoch + 1}\n"
                f"Weights updated: {self.weights[1:3]}...\n"
                f"Bias updated: {self.weights[0]}\n"
                f"Accuracy: {(accuracy / len(self.data)) * 100}%\n"
            )

    def plot(self) -> None:
        y_intercept: tuple[int, int] = (-self.weights[0] / self.weights[2], 0)
        m: int = -(self.weights[0] / self.weights[2]) / (
            self.weights[0] / self.weights[1]
        )
        axes = plt.axes(
            xlim=(min(self.data[:, 0]), max(self.data[:, 0])),
            ylim=(min(self.data[:, 1]), max(self.data[:, 1])),
        )
        x_vals = np.array(axes.get_xlim())
        y_vals = y_intercept + m * x_vals
        plt.plot(x_vals, y_vals)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)
        plt.show()
