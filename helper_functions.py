import h5py
import numpy as np
from random import randrange
from numpy import exp, array, random, dot


filepath_data_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_class_100.mat'
filepath_labels_100 = 'c:/Users/Kathan/Desktop/AML/Assignment3/datafile/three_label_100.mat'

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

def min_P_error_classifier(sample_size,class_prior0,class_prior1,dataset,orig_label,gmean,gcov):
    
    #As it is min P(error) classifer, we will always take 0/1 loss
    loss = np.array([[0,1], [1,0]])
    size = sample_size
    prior = [class_prior0,class_prior1]
    
    mean = np.zeros((2,4)) 
    mean[:,0] = gmean[:,0] 
    mean[:,1] = gmean[:,1]
    
    cov = np.zeros((2,2,4))
    cov[:,:,0] = gcov[:,:,0]
    cov[:,:,1] = gcov[:,:,1]
    
    # Gamma/ threshold
    gamma = ((loss[1,0]-loss[0,0])/(loss[1,0] - loss[1,1])) * (prior[0]/prior[1])
    orig_labels = orig_label

    
    new_labels = np.zeros((1,size))
    # Calculation for discriminant score and decisions
    cond_pdf_class0_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,0],cov = cov[:,:,0])))
    cond_pdf_class1_log = np.log((multivariate_normal.pdf(dataset.T,mean=mean[:,1],cov = cov[:,:,1])))
    
    discriminant_score = cond_pdf_class1_log - cond_pdf_class0_log


    new_labels[0,:] = (discriminant_score >= np.log(gamma)).astype(int)

    # Code to plot the distribution after Classification
    x00 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 0)]
    x01 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 0 and new_labels[0,i] == 1)]
    x10 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 0)]
    x11 = [i for i in range(new_labels.shape[1]) if (orig_labels[0,i] == 1 and new_labels[0,i] == 1)]
    plt.plot(dataset[0,x00],dataset[1,x00],'.',color ='g')
    plt.plot(dataset[0,x01],dataset[1,x01],'.',color = 'r')
    plt.plot(dataset[0,x11],dataset[1,x11],'+',color ='g')
    plt.plot(dataset[0,x10],dataset[1,x10],'+',color = 'r')
    plt.legend(["class 0 correctly classified",'class 0 wrongly classified','class 1 correctly classified','class 1 wrongly classified'])
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title('Distribution after classification')
    plt.show()
    
    
    c0 = np.argwhere(orig_labels[0,:]==0).shape[0]
    c1 = np.argwhere(orig_labels[0,:]==1).shape[0]
    #print("Class 0:",c0)
    #print("Class 1:",c1)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tpr = 0
    fpr = 0
    min_TPR = 0
    min_FPR = 0
    TPR = []
    FPR = []
    new_labels1 = np.zeros((1,size))
    d_labels1 = np.zeros((1,size))
    r=map(lambda x: x/10.0,range(0,500))
    print(r)
    for i in r:
        gamma1 = i
        #print(gamma)
        new_labels1[0,:] = (discriminant_score >= np.log(gamma1)).astype(int)
        #d_labels1[0,:] = discriminant_score >= np.log(gamma)
        for i in range(new_labels1.shape[1]): 
            #print("innerforloop")
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 1):
               TP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 1):
               FP += 1
            if (orig_labels[0,i] == 0 and new_labels1[0,i] == 0):
               TN += 1
            if (orig_labels[0,i] == 1 and new_labels1[0,i] == 0):
               FN += 1
        tpr = TP / (TP+FN)
        fpr = FP / (FP+TN)
        TPR.append(tpr)
        FPR.append(fpr)
        if gamma1 == 9.00000:
            min_TPR = tpr
            min_FPR = fpr
        

    plt.plot(FPR,TPR,'-',color = 'r')
    plt.plot(min_FPR,min_TPR, 'g*')
    plt.legend(["ROC Curve",'Min P Error'])
    plt.show()
    plt.close()
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


'''
def calculate_loss(X,model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    
    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    
    return model








class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.predict(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def predict(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("Layer 1 (4 neurons, each with 3 inputs):")
        print (self.layer1.synaptic_weights)
        print ("Layer 2 (1 neuron, with 4 inputs):")
        print (self.layer2.synaptic_weights)


'''
