import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch


class NN:

    def __init__(self, weights, biases, afs, dafs=None):
        """ Constructor.

            Args:
                weights: a list of weight matricies
                biases: a list of bias vectors
                afs: a list of vectorized activation functions, one for each layer
                dafs: a list of vectorized activation function derivatives, 
                except for dafs[-1] which should be the derivative of the chosen
                cost function wrt the last layer prior to applying af[-1].
        """
        self.weights = weights
        self.biases = biases
        self.afs = afs
        self.dafs = dafs

    def __call__(self, x):
        """ Evaluates the neural network on input(s) x. 

            Args:
                x: either an input vector (array) or an array of input vectors

            Returns:
                the last layer as a vector (array), or array of such vectors, 
                depending on x. """
        curr = x.T
        for weight, bias, af in zip(self.weights, self.biases, self.afs):
            curr = af(np.matmul(weight, curr) + bias.reshape(-1, 1))

        return curr.T

    def backpropagaton_batch(self, x, y, learning_rate):
        """ Updates the parameters (weights and biases) of the neural network 
            with the average of the updates from each training example according
            to gradient descent. 

            Args:
                x: a 2d array, where x[i] is a training example.
                y: an array of true classes. y[i] corresponds to x[i]
                learning_rate: scales the update
        """

        # from here on all data is stored column-wise, so the i-th column stores
        # whatever value is associated with the i-th training example x[i]

        batch_size = len(x)
        
        # forward propagation
        z_s = []
        a_s = [x.T]
        for weight, bias, af in zip(self.weights, self.biases, self.afs):
            z_s.append(np.matmul(weight, a_s[-1]) + bias.reshape(-1, 1))
            a_s.append(af(z_s[-1]))

        # a_s[-1] now holds a matrix of values, where the i-th column is the 
        # vector of the last layer when the neural network is evaluated on x[i]

        # backward propagation.
        delta_rev = [self.dafs[-1](a_s[-1], y)] 
        for i in range(len(z_s) - 2, -1, -1):
            delta_rev.append(np.matmul(self.weights[i+1].T, delta_rev[-1]) * self.dafs[i](z_s[i]))
        
        # update the parameters with the average update from each training 
        # example, scaled by the learning rate
        scl = learning_rate / batch_size
        for i, delta in enumerate(reversed(delta_rev)):
            self.biases[i] -= scl * delta.sum(axis=1) 

            # a_s[0] stores the input, so a_s[i] corresponds to a_s[i-1] in the notes
            self.weights[i] -= scl * sum(d.reshape(-1, 1) * a for d, a in zip(delta.T, a_s[i].T))

    @staticmethod
    def cross_entropy_loss(final_layer, y):
        """ Computes the cross entropy loss in a manner consistant with the 
            output of these neural networks """
        return -np.log(np.choose(y, final_layer.T)).sum() / len(y)

    @staticmethod
    def d_cross_entropy_loss(final_layer, y):
        """ The derivative of cross entropy loss with respect to z, where
            final_layer = NN.softmax(z)"""
        d = final_layer.copy()
        for i, t in enumerate(y):
            d[t][i] -= 1
        return d

    @staticmethod
    def softmax(x):
        """ A softmax function suitible to use as an activation function for
            this class"""
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0)

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function suitible for use as an activation function for this
            class """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        """ The derivative of the sigmoid function """
        s = NN.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def classify(final_layer):
        """ Use as NN.classify(nn(x)) to get the classes predicted by a neural 
            network nn, given input(s) x"""
        return np.argmax(final_layer, axis=1)

def get_batches(data, batch_size=20):
    """ Provides an iterator over batches of size batch_size of a dataset"""
    xs, ys = data
    num_batches = len(xs) // batch_size

    for i in range(num_batches):
        yield xs[i * batch_size : (i + 1) * batch_size], ys[i * batch_size : (i + 1) * batch_size]


if __name__ == "__main__":

    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_size = 10000
    val_size = 5000

    normalization = 1 / 255

    tmp = trainset.data.reshape((len(trainset), -1)).numpy()
    train_data = normalization * tmp[:train_size] 
    val_data = normalization * tmp[train_size:train_size+val_size]

    tmp = trainset.targets.numpy()
    train_targets = tmp[:train_size]
    val_targets = tmp[train_size:train_size+val_size]

    test_data = normalization * testset.data.reshape((len(testset), -1)).numpy()
    test_targets = testset.targets.numpy()

    in_size = train_data.shape[-1]
    hidden_size = 100
    out_size = len(trainset.targets.unique())

    #initialize weights randomly and biases to zero
    w0 = np.random.uniform(-1, 1, (hidden_size, in_size))
    b0 = np.zeros(hidden_size)

    w1 = np.random.uniform(-1, 1, (out_size, hidden_size))
    b1 = np.zeros(out_size)

    nn = NN([w0, w1], [b0, b1], [NN.sigmoid, NN.softmax], dafs=[NN.d_sigmoid, NN.d_cross_entropy_loss])
    
    total_epochs = 40
    epoch = 0

    train_accuracy = np.empty(total_epochs)
    val_accuracy = np.empty(total_epochs)
    train_loss = np.empty(total_epochs)
    val_loss = np.empty(total_epochs)

    train_output = nn(train_data)
    train_accuracy[epoch] = np.count_nonzero(NN.classify(train_output) == train_targets) / len(train_targets)
    train_loss[epoch] = NN.cross_entropy_loss(train_output, train_targets)

    val_output = nn(val_data)
    val_accuracy[epoch] = np.count_nonzero(NN.classify(val_output) == val_targets) / len(val_targets)
    val_loss[epoch] = NN.cross_entropy_loss(val_output, val_targets)

    print(f"Epoch: {epoch}")
    print(f"Train accuracy: {train_accuracy[epoch] * 100}%")
    print(f"Train loss: {train_loss[epoch]}")
    print(f"Validation accuracy: {val_accuracy[epoch] * 100}%")
    print(f"Validation loss: {val_loss[epoch]}\n")


    for epoch in range(1, total_epochs):
        
        for x, y in get_batches((train_data, train_targets)):
            nn.backpropagaton_batch(x, y, learning_rate=0.005)


        train_output = nn(train_data)
        train_accuracy[epoch] = np.count_nonzero(NN.classify(train_output) == train_targets) / len(train_targets)
        train_loss[epoch] = NN.cross_entropy_loss(train_output, train_targets)

        val_output = nn(val_data)
        val_accuracy[epoch] = np.count_nonzero(NN.classify(val_output) == val_targets) / len(val_targets)
        val_loss[epoch] = NN.cross_entropy_loss(val_output, val_targets)

        print(f"Epoch: {epoch}")
        print(f"Train accuracy: {train_accuracy[epoch] * 100}%")
        print(f"Train loss: {train_loss[epoch]}")
        print(f"Validation accuracy: {val_accuracy[epoch] * 100}%")
        print(f"Validation loss: {val_loss[epoch]}\n")

    print(train_accuracy)
    print(val_accuracy)
    print(train_loss)
    print(val_loss)

