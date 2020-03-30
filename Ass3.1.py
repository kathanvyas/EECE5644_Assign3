import h5py
import numpy as np
from random import randrange
from numpy import exp, array, random, dot
import sklearn.metrics
from fun_lib import min_P_error_classifier
np.random.seed(404)
predict_proba = []
def mat_to_array(filepath,flag,N,name):
    '''
    Function that converts mat file to numpy array.
    filepath: path to the mat file.
    flag: 0-data |  1-label
    n: number of samples
    name: name of mat file ins string format
    Author: Kathan Vyas
    '''
    if flag == 0:
        x_y_numpy_array = np.zeros((N, 2), dtype=float)
    else:
        x_y_numpy_array = np.zeros((N, 1), dtype=float)

    
    with h5py.File(filepath, 'r') as f:
        for idx, element in enumerate(f[name]):
            x_y_numpy_array[idx] = element[:]
    return x_y_numpy_array
def cross_validation_split(dataset, folds=3):
    '''
    Function performs k-fold cross validation
    dataset: numpy array
    folds: number of folds
    Author: Kathan Vyas
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
def expandthree(df,num):
        y = np.zeros((num,3))
        for i in range(num):
            if df[[i]]==1:
                y[i,0]=1
                y[i,1]=0
                y[i,2]=0
            if df[i]==2:
                y[i,0]=0
                y[i,1]=1
                y[i,2]=0
            if df[i]==3:
                y[i,0]=0
                y[i,1]=0
                y[i,2]=1       
        return y
class elu:
    @staticmethod
    def activation(z):
        alpha = 0.1
        z[z < 0] = ((alpha) * (np.exp(z)-1))
        return z
    @staticmethod
    def prime(z):
        alpha = 0.1
        z[z < 0] = (alpha * (np.exp(a)-1)) + alpha
        z[z > 0] = 1
        return z
class Selu:
    @staticmethod
    def activation(z):
        alpha = 1.67
        lamda = 1.050
        z[z < 0] = (lamda * (alpha * (np.exp(z)-1)))
        return z
    @staticmethod
    def prime(z):
        alpha = 1.67
        lamda = 1.050
        z[z < 0] = lamda * (alpha * np.exp(z))
        z[z > 0] = lamda
        return z
class Softmax:
    f = True
    @staticmethod
    def activation(z):
        alpha = 0.2
        z[z < 0] = z[z < 0] * alpha
        return z
    @staticmethod
    def prime(z):
        alpha = 0.2
        # Implementing MAP classifier along with Softmax
        if not f:
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
            scores = [0.0, 1.0, 2.0]
            print(Softmax(scores))
            q = np.array(3)
            q[0] = np.len(x[y==0]) / np.len(x)
            q[1] = np.len(x[y==1]) / np.len(x)
            q[2]= np.len(x[y==2]) / np.len(x)
            for l in range(2):
                z = np.argmax(scores[l]*q[l])
        z[z < 0] = alpha
        z[z > 0] = 1
        return z      
class LRelu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z
    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z
class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))
class MSE:
    def __init__(self, activation_fn=None):
        """
        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)
class Network:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        Example of one hidden layer with
        - 2 inputs
        - 3 hidden nodes
        - 3 outputs
        layers -->    [1,        2,          3]
        ----------------------------------------
        dimensions =  (2,     3,          3)
        activations = (      Relu,      Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]
    def _feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a
    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
            dw = np.dot(a[i - 1].T, delta)
            update_params[i - 1] = (dw, delta)

        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])
    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)
    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_tue: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l])
                self._back_prop(z, a, y_[k:l])

            if (i + 1) % 10 == 0:
                _, a = self._feed_forward(x)
                print("Loss:", self.loss.loss(y_true, a[self.n_layers]))
    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self._feed_forward(x)
        return a[self.n_layers]

if __name__ == "__main__":
    filepath_data_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_100.mat'
    filepath_labels_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_label_100.mat'
    filepath_data_10k = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_10k.mat'
    
    x = mat_to_array(filepath_data_100,0,100,'three_class_100')
    _y_ = mat_to_array(filepath_labels_100,1,100,'three_label_100')
    x_test = mat_to_array(filepath_data_10k,0,10000,'three_class_10k')
    # one hot encoding
    y = expandthree(_y_,100) 

    nn = Network((2, 6, 3), (Sigmoid, Softmax))
    nn.fit(x, y, loss=MSE, epochs=1000, batch_size=1, learning_rate=0.01)

    prediction = nn.predict(x)

    y_true = []
    y_pred = []
    for i in range(len(y)):
        y_pred.append(np.argmax(prediction[i]))
        y_true.append(np.argmax(y[i]))

    print(sklearn.metrics.classification_report(y_true, y_pred))
    
    
    preds = nn.predict(x_test)
    y_pred1 = []
    for i in range(len(x_test)):
        y_pred1.append(np.argmax(preds[i]))
    
    y_test = y_pred1
    
    #print(np.shape(y_test))
    #print(np.unique(y_test))
    
    
    
    
    




