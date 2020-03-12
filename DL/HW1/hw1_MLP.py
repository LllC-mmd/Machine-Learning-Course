# %%
import numpy as np
import time

## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons

## Hyperparameters
NUM_HIDDEN = 50
LEARNING_RATE = 0.05
BATCH_SIZE = 64
NUM_EPOCH = 40

print("NUM_HIDDEN: ", NUM_HIDDEN)
print("LEARNING_RATE: ", LEARNING_RATE)
print("BATCH_SIZE: ", BATCH_SIZE)
print("NUM_EPOCH: ", NUM_EPOCH)


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
def unpack(w):
    W1 = np.reshape(w[:NUM_INPUT * NUM_HIDDEN], (NUM_INPUT, NUM_HIDDEN))
    w = w[NUM_INPUT * NUM_HIDDEN:]
    b1 = np.reshape(w[:NUM_HIDDEN], NUM_HIDDEN)
    w = w[NUM_HIDDEN:]
    W2 = np.reshape(w[:NUM_HIDDEN * NUM_OUTPUT], (NUM_HIDDEN, NUM_OUTPUT))
    w = w[NUM_HIDDEN * NUM_OUTPUT:]
    b2 = np.reshape(w, NUM_OUTPUT)
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
def pack(W1, b1, W2, b2):
    W1_ = np.reshape(W1, NUM_INPUT * NUM_HIDDEN)
    W2_ = np.reshape(W2, NUM_HIDDEN * NUM_OUTPUT)
    w = np.concatenate((W1_, b1, W2_, b2))
    return w


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("./data/mnist_{}_images.npy".format(which))
    labels = np.load("./data/mnist_{}_labels.npy".format(which))
    return images, labels


## 1. Forward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss.
def fCE(X, Y, w):
    # print(X.shape)
    n_sample = X.shape[0]
    W1, b1, W2, b2 = unpack(w)
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    h1 = np.maximum(z1, 0)
    z2 = np.dot(h1, W2) + b2
    z2 = np.transpose(z2)
    y_hat = np.exp(z2) / np.sum(np.exp(z2), axis=0)
    y_hat = np.transpose(y_hat)
    loss = -1/n_sample * np.sum(Y*np.log(y_hat))
    return z1, h1, y_hat, loss


## 2. Backward Propagation
# Given training images X, associated labels Y, and a 1D vector of combined weights
# and bias terms w, compute and return the gradient of fCE.
def gradCE(X, Y, z1, h1, y_hat, w):
    n_sample = Y.shape[0]
    W1, b1, W2, b2 = unpack(w)

    delta_W_2 = 1/n_sample * np.dot(np.transpose(y_hat-Y), h1)
    delta_W_2 = np.transpose(delta_W_2)

    delta_b_2 = 1/n_sample * np.sum(y_hat-Y, axis=0)

    delta_W_1 = np.dot(y_hat-Y, W2.transpose())*np.sign(z1)
    delta_W_1 = 1/n_sample * np.dot(np.transpose(delta_W_1), X)
    delta_W_1 = np.transpose(delta_W_1)
    delta_b_1 = 1/n_sample * np.sum(np.dot(y_hat-Y, W2.transpose())*np.sign(z1), axis=0)

    delta = pack(delta_W_1, delta_b_1, delta_W_2, delta_b_2)
    return delta


## 3. Parameter Update
# Given training and testing datasets and an initial set of weights/biases,
# train the NN.
def train(trainX, trainY, testX, testY, w):
    num_train = len(trainX)
    num_test = len(testX)
    test_label = np.argmax(testY, axis=1)

    num_batch = int(num_train / BATCH_SIZE)
    id_set = np.array([i for i in range(0, num_train)])

    for epoch in range(0, NUM_EPOCH):
        np.random.shuffle(id_set)
        for b in range(0, num_batch):
            train_id = id_set[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            train_batchX = trainX[train_id, :]
            train_batchY = trainY[train_id, :]
            train_label = np.argmax(train_batchY, axis=1)

            # Train Part
            # ---Forward propagation
            z1_train, h1_train, y_hat_train, loss_train = fCE(train_batchX, train_batchY, w)
            train_hat = np.argmax(y_hat_train, axis=1)
            train_diff = np.abs(train_hat - train_label)
            train_mse = np.sum(np.where(train_diff > 0, 1, 0)) / BATCH_SIZE
            # ---Backward propagation
            delta = gradCE(train_batchX, train_batchY, z1_train, h1_train, y_hat_train, w)
            delta_W_1, delta_b_1, delta_W_2, delta_b_2 = unpack(delta)
            W1, b1, W2, b2 = unpack(w)
            # ---Update the parameters
            W1 = W1 - LEARNING_RATE * delta_W_1
            b1 = b1 - LEARNING_RATE * delta_b_1
            W2 = W2 - LEARNING_RATE * delta_W_2
            b2 = b2 - LEARNING_RATE * delta_b_2
            w = pack(W1, b1, W2, b2)

            # Test Part
            z1_test, h1_test, y_hat_test, loss_test = fCE(testX, testY, w)
            test_hat = np.argmax(y_hat_test, axis=1)
            test_diff = np.abs(test_hat - test_label)
            test_mse = np.sum(np.where(test_diff > 0, 1, 0))/num_test

            # Report the train accuracy and test accuracy
            print("Epoch: %d\tIteration: %d\tTrain Accuracy: %.3f\tTest Accuracy: %.3f"%(epoch, b, 1-train_mse, 1-test_mse))


if __name__ == "__main__":
    np.random.seed(111)
    # Load data
    start_time = time.time()
    trainX, trainY = loadData("train")
    testX, testY = loadData("test")

    print("len(trainX): ", len(trainX))
    print("len(testX): ", len(testX))

    # Initialize weights randomly
    W1 = 2 * (np.random.random(size=(NUM_INPUT, NUM_HIDDEN)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    w = pack(W1, b1, W2, b2)
    print("Shape of w:", w.shape)

    # # Train the network and report the accuracy on the training and test set.
    train(trainX, trainY, testX, testY, w)

# %%


