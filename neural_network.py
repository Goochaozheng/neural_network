import numpy as np
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default="breast_cancer")
parser.add_argument("-num_of_layers", type=int, default=2)
parser.add_argument("-num_of_units", type=int, default=4)
parser.add_argument("-max_it", type=int, default=500)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-tolerant", type=float, default=5e-5)

class neural_network:

    def __init__(self, input_size, num_of_layers=1, num_of_units=5, learning_rate=1e-2, task="classification"):
        """
        Neural network with multiple fully connected layer.
        For convenience, the number of unit in each layer is the same.
        Weights of hidden layers are managed as ndarray where hidden_weights[0] is the first layer.
        Gradients of hidden neural unit are also managed as ndarray.
        Weights are initialized with random value.
        """
        # input layer
        self.input_weight = np.random.uniform(-1, 1, (num_of_units, input_size))
        self.input_bias = np.random.uniform(-1, 1, num_of_units)
        self.input_result = np.zeros(num_of_units)

        # hidden layer
        self.hidden_weights = np.random.uniform(-1, 1, (num_of_layers, num_of_units, num_of_units))
        self.hidden_bias = np.random.uniform(-1, 1, (num_of_layers, num_of_units))
        self.hidden_result = np.zeros([num_of_layers, num_of_units])

        # output layer
        self.output_weight = np.random.uniform(-1, 1, num_of_units)
        self.output_bias = np.random.uniform(-1, 1)
        self.final_result = 0     
        
        # gradient of different layers
        self.input_grad = np.zeros(num_of_units)
        self.hidden_grads = np.zeros([num_of_layers, num_of_units])
        self.output_grad = 0
        
        self.learning_rate = learning_rate
        self.task = task
        if not (self.task == "classification" or self.task == "regression"):
            raise Exception("Invalid parameter Task, should be 'classification' or 'regression'")


    def forward(self, x):
        """
        Forward propagation for single input sample,
        Return the output of network.
        """

        # Check input size
        if not len(x) == self.input_weight.shape[-1]:
            raise ValueError("Input size not match, expect input size: %d" % self.hidden_weights.shape[-1])

        self.input = x

        # Compute result for input layer f(x)=sigmoid(wx-b)
        self.input_result = [self.sigmoid(np.dot(weight, self.input) - bias) for weight, bias in zip(self.input_weight, self.input_bias)]

        # Compute result for hidden layers
        self.hidden_result[0] = [self.sigmoid(np.dot(weight, self.input_result) - bias) for weight, bias in zip(self.hidden_weights[0], self.hidden_bias[0])]

        if self.hidden_weights.shape[0] > 1:
            for layer in range(1, self.hidden_weights.shape[0]):
                for unit in range(self.hidden_weights.shape[1]):
                    self.hidden_result[layer, unit] = self.sigmoid(np.dot(self.hidden_weights[layer, unit], self.hidden_result[layer-1]) - self.hidden_bias[layer, unit])

        # Output layer
        if self.task == "classification":
            self.final_result = self.sigmoid(np.dot(self.output_weight, self.hidden_result[-1]) - self.output_bias)
        elif self.task == "regression":
            self.final_result = np.dot(self.output_weight, self.hidden_result[-1]) - self.output_bias

        return self.final_result

    # compute loss
    def loss(self, x, y):
        if self.task == "classification":
            return self.cross_entropy(x, y)
        elif self.task == "regression":
            return self.mse_loss(x, y)

    def mse_loss(self, x, y):
        self.label = y
        return (1/2) * (x-y)**2

    def cross_entropy(self, x, y):
        self.label = y
        return - (y*np.log(x) + (1-y)*np.log(1-x))

    def backward(self):
        """
        Compute the gradient of loss on output scalar(wx+b) of each neural unit
        Update the coefficient of each unit
        """

        # Compute gradient for output layer
        if self.task == "classification":
            self.output_grad = ((1 - self.label) / (1 - self.final_result) - self.label / self.final_result) * (1 - self.final_result) * self.final_result
        else:
            self.output_grad = (self.final_result - self.label)
        # Update output layer
        self.output_weight -= self.learning_rate * self.output_grad * self.hidden_result[-1]
        self.output_bias -= self.learning_rate * self.output_grad * -1

        # Compute gradient for hidden layers
        for unit in range(self.hidden_weights.shape[1]): # Last layer
            self.hidden_grads[-1, unit] = self.hidden_result[-1, unit] * (1 - self.hidden_result[-1, unit]) * self.output_grad * self.output_weight[unit]
        
        for layer in range(self.hidden_weights.shape[0]-2, -1, -1): # Other layers, backward iteration
            for unit in range(self.hidden_weights.shape[1]):
                self.hidden_grads[layer, unit] = self.hidden_result[layer, unit] * (1 - self.hidden_result[layer, unit]) * \
                                                 np.sum([w*g for w,g in zip(self.hidden_weights[layer+1, :, unit], self.hidden_grads[layer+1])])

        # Update hidden layers
        for layer in range(self.hidden_weights.shape[0]):
            for unit in range(self.hidden_weights.shape[1]):
                self.hidden_weights[layer, unit] -= self.learning_rate * self.hidden_grads[layer, unit] * self.hidden_result[layer-1]
                self.hidden_bias[layer, unit] -= self.learning_rate * self.hidden_grads[layer, unit] * -1

        # Compute gradient for input layer
        for unit in range(self.input_weight.shape[0]):
            self.input_grad[unit] = (1 - self.input_result[unit]) * self.input_result[unit] * np.sum([w*g for w,g in zip(self.hidden_weights[0, :, unit], self.hidden_grads[0])])
        # Update input layer
        for unit in range(self.input_weight.shape[0]):
            self.input_weight[unit] -= self.learning_rate * self.input_grad[unit] * self.input
            self.input_bias[unit] -= self.learning_rate * self.input_grad[unit] * -1


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lr_decrease(self):
        self.learning_rate *= 0.95


def fit(net, train_data, train_label, max_iter, tor):
    min_loss = np.inf
    loss_change = []
    loss_mean = np.inf
    early_stop_count = 50
    for e in range(max_iter):
        # iteration over whole dataset
        for i in range(len(train_data)):
            res = net.forward(train_data[i])
            l = net.loss(res, train_label[i])
            net.backward()
        # Check current model loss
        previous_loss = loss_mean
        loss_mean = 0
        for i in range(len(train_data)):
            res = net.forward(train_data[i])
            loss_mean += net.loss(res, train_label[i])
        loss_mean = loss_mean/len(train_data)
        loss_change.append(loss_mean)
        # save best model
        if loss_mean < min_loss: 
            min_loss = loss_mean
            best_model = deepcopy(net)
        if e % 10 == 0:
            print("Epoch: %d, Loss: %.5f" % (e, loss_mean))
        if previous_loss - loss_mean < tor:
            early_stop_count -= 1
        else:
            early_stop_count = 50
        if early_stop_count <= 0:
            print("Early Stop.")
            break
    print("===========================================================")
    print("Training Finish. Best Model Saved. Best Loss: %.5f " % min_loss)
    return best_model


def pred(net, test_data):
    res = []
    for i in range(len(test_data)):
        temp = net.forward(test_data[i])
        res.append(temp)
    return res


def accuracy(predict, ground_truth):
    if not len(predict) == len(ground_truth):
        raise Exception("Size not match")
    count = (predict == ground_truth).astype(int).sum()
    return (count/len(predict))

def rmse(predict, ground_truth):
    if not len(predict) == len(ground_truth):
        raise Exception("Size not match")
    return np.sqrt(((predict-ground_truth)**2).sum()/len(ground_truth))

def main():
    args = parser.parse_args()

    # Load data
    print("Loading dataset: ", args.dataset)
    if args.dataset == "breast_cancer":
        data = np.genfromtxt("data/breast-cancer-wisconsin.data", delimiter=",")
        # Remove nan data samples and ID col
        data = data[~np.isnan(data).any(axis=1)][:,1:]
        np.random.shuffle(data)
        x = data[:,:-1]
        label = data[:, -1] # Map label to 0,1
        unique, y = np.unique(label, return_inverse=True)
    elif args.dataset == "energy_efficiency":
        data = np.genfromtxt("data/ENB2012_data.csv", delimiter=",")
        np.random.shuffle(data)
        raw_x = data[:, :-2].astype(float)
        # Min Max Scale
        x = deepcopy(raw_x)
        for i in range(x.shape[1]):
            col = raw_x[:,i]
            col = (col - col.min()) / (col.max() - col.min())
            x[:, i] = col
        y = data[:, -1].astype(float)
    else:
        raise Exception("Dataset can only be breast_cancer or energy_efficiency")

    # Train test split
    partition = 0.8
    train_size = int(data.shape[0] * 0.8)

    train_data = x[:train_size]
    train_label = y[:train_size]

    test_data = x[train_size:]
    test_label = y[train_size:]

    # Model initialization
    if args.dataset == "breast_cancer":
        net = neural_network(
            input_size=train_data.shape[1], 
            num_of_layers=args.num_of_layers, 
            num_of_units=args.num_of_units, 
            learning_rate=args.lr, 
            task='classification')
    else:
        net = neural_network(
            input_size=train_data.shape[1], 
            num_of_layers=args.num_of_layers, 
            num_of_units=args.num_of_units, 
            learning_rate=args.lr, 
            task='regression')

    # Training
    best_model = fit(net, train_data, train_label, max_iter=args.max_it, tor=args.tolerant)

    # Predict
    res = pred(net, test_data)

    # Validation
    if args.dataset == "breast_cancer":
        res = (np.array(res)>0.5).astype(int)
        print("Accuracy: %.5f" % accuracy(res, test_label))
    else:
        print("RMSE: %.5f" % rmse(res, test_label))

if __name__ == "__main__":
    main()