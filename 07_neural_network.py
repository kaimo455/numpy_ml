import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, inp_size, out_size, hid_sizes=[6, 6]):
        self.inp_size = inp_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.params = self._init_params()

    @staticmethod
    def create_dataset():
        np.random.seed(42)
        num_sample = 400
        num_cls = 2
        num_sample_each_cls = int(num_sample / num_cls)
        dim = 2
        X = np.zeros((num_sample, dim))
        Y = np.zeros((num_sample, 1), dtype=np.int8)
        a = 4
        for cls in range(num_cls):
            ix = range(num_sample_each_cls * cls, num_sample_each_cls * (cls + 1))  # index for different class
            theta = np.linspace(cls * 3.12, (cls + 1) * 3.12, num_sample_each_cls) + np.random.randn(
                num_sample_each_cls) * 0.2  # theta for different class
            radius = a * np.sin(4 * theta) + np.random.randn(num_sample_each_cls) * 0.2  # radius for different class
            X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
            Y[ix] = cls
        return X, Y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _compute_loss(y_true, y_pred):
        """Compute cross entropy loss."""
        num_samples = y_true.shape[0]
        return np.sum(y_true * np.log2(y_pred) + (1 - y_true) * np.log2(1 - y_pred), axis=None) / num_samples * (-1)

    def _init_params(self):
        params = {}
        # X of shape [N, inp_size]
        # W0 of shape [inp_size, h1_size]  -->  X * W0 + b
        # then use loop to create weight_i and bias_i
        pre_size = self.inp_size
        i = 0
        for h_size in self.hid_sizes:
            params[f'W{i}'] = np.random.randn(pre_size, h_size)
            params[f'b{i}'] = np.zeros((1, h_size))
            pre_size = h_size
            i += 1
        # last hidden layer to output layer weight and bias
        params[f'W{i}'] = np.random.randn(pre_size, self.out_size)
        params[f'b{i}'] = np.zeros((1, self.out_size))
        # Test
        print('parameters:')
        for key in params.keys():
            print(key, ': ', params[key].shape)
        #
        return params

    def _forward(self, X):
        """hidden layer use tanh activation function, while output layer use sigmoid activation function."""
        i = 0
        cache = {}
        inp = X.copy()
        # hidden layers
        for _ in range(len(self.hid_sizes)):
            cache[f'Z{i}'] = np.dot(inp, self.params[f'W{i}']) + self.params[f'b{i}']
            cache[f'A{i}'] = np.tanh(cache[f'Z{i}'])
            inp = cache[f'A{i}']
            i += 1
        # output layer
        cache[f'Z{i}'] = np.dot(inp, self.params[f'W{i}']) + self.params[f'b{i}']
        cache[f'A{i}'] = self.sigmoid(cache[f'Z{i}'])
        return cache[f'A{i}'], cache

    def _backward(self, cache, X, Y):
        i = len(self.hid_sizes)
        A_n = cache[f'A{i}']  # output from output layer (after sigmoid activation). [N, 1]
        dZ_n = A_n - Y  # [N, 1]
        num_samples = X.shape[1]
        # back propagation in hidden layers
        # pass in dZ_i, A_(i-1), W_i to calculate dW_i, db_i, dZ_(i-1). Let j = i - 1
        dZ_i = dZ_n
        grads = {}
        for _ in range(len(self.hid_sizes)):
            grads[f'dW{i}'] = np.dot(cache[f'A{i - 1}'].T, dZ_i) / num_samples
            grads[f'db{i}'] = np.sum(dZ_i, axis=0, keepdims=True) / num_samples
            dZ_i = np.dot(dZ_i, self.params[f'W{i}'].T) * (1 - cache[f'A{i - 1}'] ** 2)  # tanh(x)' = 1 - tanh(x)^2
            i -= 1
        grads[f'dW{i}'] = np.dot(X.T, dZ_i) / num_samples
        grads[f'db{i}'] = np.sum(dZ_i, axis=0, keepdims=True) / num_samples

        return grads

    def _update(self, grads, lr):
        i = 0
        for _ in range(len(self.hid_sizes) + 1):
            self.params[f'W{i}'] -= lr * grads[f'dW{i}']
            self.params[f'b{i}'] -= lr * grads[f'db{i}']
            i += 1

    def train(self, X, Y, lr, epochs):
        loss_list = []
        for epoch in range(epochs):
            # forward
            Y_pred, cache = self._forward(X)
            # compute loss
            loss = self._compute_loss(Y, Y_pred)
            loss_list.append(loss)
            # backward
            grads = self._backward(cache, X, Y)
            # update parameters
            self._update(grads, lr)
            if epoch % 100 == 0:
                print(f'Epoch {epoch} loss {loss:.4f}')
        return loss_list

    def predict(self, X):
        Y_pred, _ = self._forward(X)
        return (Y_pred > 0.5).astype(np.int8)


if __name__ == '__main__':
    # create dataset
    X, Y = NeuralNetwork.create_dataset()
    # train model
    nn = NeuralNetwork(2, 1, [10, 10])
    loss_list = nn.train(X, Y, 0.001, 10000)
    # plot loss
    plt.plot(loss_list, label='loss')
    plt.show()
    # plot sample points and decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=plt.cm.Spectral)
    _x_min, _x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    _y_min, _y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(_x_min, _x_max, 0.02), np.arange(_y_min, _y_max, 0.02))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.1)
    plt.show()
