import numpy as np
from sklearn.datasets import make_blobs


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def error(A, y):
    return (np.mean(np.power(A - y, 2))) / 2


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer_derivative(output):
    return output * (1.0 - output)


class MLP(object):
    def __init__(
        self,
        num_inputs,
        num_epochs: int,
        lr: float,
        hidden_layers=[1],
        num_outputs=2,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.num_epochs = num_epochs
        self.lr = lr

        self.network = []
        self.input_weights = [
            {"weights": np.append(np.random.rand(self.num_inputs), 1)}
        ]
        self.hidden_weights = [
            {"weights": np.append(np.random.rand(i), 1)} for i in self.hidden_layers
        ]
        self.output_weights = [
            {"weights": np.append(np.random.rand(self.hidden_layers[-1]), 1)}
            for _ in range(self.num_outputs)
        ]
        self.network.append(self.input_weights)
        self.network.append(self.hidden_weights)
        self.network.append(self.output_weights)

    def forward(self, input):
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = (
                    np.dot(neuron["weights"][:-1], input) + neuron["weights"][-1]
                )
                neuron["output"] = relu(activation)
                new_inputs.append(neuron["output"])
            input = new_inputs
        return input

    def backward(self, label):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            for j in range(len(layer)):
                if i != len(self.network) - 1:
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron["weights"][j] * neuron["delta"]
                        errors.append(error)
                else:
                    neuron = layer[j]
                    errors.append(label[j] - neuron["output"])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])

    def update_weights(self, point):
        for i in range(len(self.network)):
            inputs = point[:-1]
            if i != 0:
                inputs = [neuron["output"] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron["weights"][j] += self.lr * neuron["delta"] * inputs[j]
                neuron["weights"][-1] += self.lr * neuron["delta"]

    def train(self, train, labels):
        for epoch in range(self.num_epochs):
            sum_error = 0
            for point, label in zip(train, labels):
                if type(label) != []:
                    label = [label]
                outputs = self.forward(point)
                sum_error += sum((label[i] - outputs[i]) ** 2 for i in range(self.num_outputs))
                self.backward(label)
                self.update_weights(point)
            print(f">epoch={epoch}, error={sum_error}")


if __name__ == "__main__":
    X, y = make_blobs(n_samples=10, centers=2, n_features=2, random_state=100)  # type: ignore

    mlp = MLP(num_inputs=2, num_epochs=10, lr=0.001, num_outputs=1)
    mlp.train(train=X, labels=y)
