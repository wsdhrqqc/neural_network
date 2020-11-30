# CS 5033: Machine Learning
# Supervised learning project
# Neural Network
# Author: Precious Jatau

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle


"""creates a layer in a neural network.  dimIn is the input dimension, dimOut is the output dimension. lR is the learning rate"""
class Layer:

    # constructor
    def __init__(self, dimIn, dimOut, lR):
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.lR = lR

    # set imput dimension
    def setInputDimension(self, dimIn):
        self.dimIn = dimIn

    # set output dimension
    def setOutputDimension(self, dimOut):
        self.dimOut = dimOut

    """ __repr__ is complete representation of an object for use (debugging) by other developers. A good rule is that the 
            string returned by repr should be usable for recreating the object"""
    def __repr__(self):
        return "Layer({},{})".format(self.dimIn, self.dimOut)

    """__str__ is readable representation of an object meant for use by the end user"""
    def __str__(self):
        return "{} - {}".format(self.dimIn, self.dimOut)


"""Weight layer has weights for dot product with input image x"""
class WeightLayer(Layer):

    # constructor for weight layer
    def __init__(self, dimIn, dimOut, lR, update_type):
        super().__init__(dimIn, dimOut, lR)
        self.weights = np.random.randn(dimIn+1,dimOut)/np.sqrt(dimIn/2)
        self.weights[0] = 0
        self.update_type = update_type
        self.grad_squared = 0

    # forward pass computes Wx
    def forwardPass(self, x):

        x0 = np.ones((x.shape[0], 1))
        x = np.concatenate((x0, x), axis=1)

        # compute local gradient for back propagation
        self.dw = np.transpose(x)

        return np.dot(x,self.weights)

    # backward pass updates weights and biases
    def backwardPass(self, dl):

        if self.update_type == 'Adagrad':
            self.grad_squared += self.dw**2
            adj_weights = self.dw / (np.sqrt(self.grad_squared) + 1e-7)
            self.weights -= self.lR * np.dot(adj_weights, dl)
        elif self.update_type == 'RMSprop':
            self.grad_squared = 0.9*self.grad_squared + 0.1*self.dw**2
            adj_weights = self.dw / (np.sqrt(self.grad_squared) + 1e-7)
            self.weights -= self.lR * np.dot(adj_weights, dl)
        elif self.update_type == 'SGD':
            self.weights -= self.lR * np.dot(self.dw, dl)



"""Middle weight layer has activation from previous layer as input"""
class MiddleWeightLayer(WeightLayer):

    def forwardPass(self, h):

        h0 = np.ones((h.shape[0], 1))
        h = np.concatenate((h0, h), axis=1)

        # compute local gradient for back propagation
        self.dw = np.transpose(h)
        self.dh = np.transpose(self.weights[range(1,len(self.weights))])

        return np.dot(h,self.weights)

    def backwardPass(self, dl):

        if self.update_type == 'Adagrad':
            self.grad_squared += self.dw**2
            adj_weights = self.dw / (np.sqrt(self.grad_squared) + 1e-7)
            self.weights -= self.lR * np.dot(adj_weights, dl)
        elif self.update_type == 'RMSprop':
            self.grad_squared = 0.9*self.grad_squared + 0.1*self.dw**2
            adj_weights = self.dw / (np.sqrt(self.grad_squared) + 1e-7)
            self.weights -= self.lR * np.dot(adj_weights, dl)
        elif self.update_type == 'SGD':
            self.weights -= self.lR * np.dot(self.dw, dl)


        # return upstream gradient for other channel
        return np.dot(dl,self.dh)

"""Activation layer applies RELU activation function to input.
Possible activations are RELU and Leaky RELU"""
class ActivationLayer(Layer):

    # constructor
    def __init__(self,lR, activation):
        super().__init__(0, 0, lR)
        self.activation = activation

    # compute activations using a RELU
    def forwardPass(self, q):

        if self.activation == 'RELU':
            act = np.maximum(0, q)
            # local gradient
            self.dAct = (act > 0) * 1

        elif self.activation == 'Leaky RELU':
            alpha = 0.01
            act = np.maximum(alpha*q,q)
            mask = q <= alpha*q
            # local gradient
            self.dAct = np.ones(q.shape)
            self.dAct[mask] = alpha

        return act

    # compute upstream gradient
    def backwardPass(self, dl):
        return np.multiply(self.dAct,dl)


"""This layer computes softmax scores and cross entropy loss for the neural network"""
class LossLayer(Layer):

    # constructor
    def __init__(self,lR):
        super().__init__(0, 0, lR)

    # computes cross entropy loss
    # x is output from previous layer (numExamples by numClasses)
    # y is correct label (numExamples by 1). labels should be encoded as 0,1,...
    def forwardPass(self, x, y):

        # calculate softmax scores
        scores = self.softmax(x)[:]
        self.scores = scores[:]

        # calculate cross entropy loss
        numExamples = y.shape[0]
        loglik = -np.log(scores[range(numExamples),y])
        self.loss =  np.sum(loglik)/numExamples

    def backwardPass(self,y):

        # gradient
        numExamples = y.shape[0]
        dLoss = self.scores[:]
        dLoss[range(numExamples), y] -= 1
        dLoss = dLoss / numExamples
        self.dLoss = dLoss[:]

        return self.dLoss


    # Calculates softmax scores for tensor x
    # x is numExamples by numClasses
    def softmax(self, x):
        scores = np.exp(x)
        scores[scores < 1e-10] = 1e-30
        scores[scores > 1e10] = 1e30
        sumScores = scores.sum(axis=1)

        for i in range(0, len(scores)):
            scores[i] = scores[i] / sumScores[i]

        return scores


# TO DO
# check constructors
# update string representation (in progress)

class myNeuralNetwork():

    # neural net constructor
    def __init__(self, lr, numEpochs, dimIn1, dimOut1, dimIn2, dimOut2, dimIn3, dimOut3, batchSize,
                 update_type,activation,saveLoss,toPrint, rotateUpdate = None):
        self.lr = lr
        self.numEpochs = numEpochs
        self.saveLoss = saveLoss

        # initialize layers
        self.w1 = WeightLayer(dimIn1, dimOut1, lr, update_type)
        self.a1 = ActivationLayer(lr,activation)
        self.w2 = MiddleWeightLayer(dimIn2, dimOut2, lr, update_type)
        self.a2 = ActivationLayer(lr,activation)
        self.w3 = MiddleWeightLayer(dimIn3, dimOut3, lr, update_type)
        self.loss_l = LossLayer(lr)
        self.toPrint = toPrint
        self.rotateUpdate = rotateUpdate

        # string representation of neural network
        self.repr_string = "myNeuralNetwork(lr = {},numEpochs={},dimIn1={},dimOut1 = {},dimIn2 = {}," \
        "dimOut2= {},dimIn3 = {}, dimOut3={},batchSize = {}, update_type = {},activation = {}, rotateUpdate = {}, saveLoss = {},toPrint = {})".format(lr, numEpochs,dimIn1,
                                                                                          dimOut1,dimIn2,dimOut2, dimIn3,dimOut3, batchSize,update_type,activation,rotateUpdate, saveLoss,toPrint)

        if saveLoss:
            self.trainLoss = np.zeros((numEpochs,1))
            self.valLoss = np.zeros((numEpochs,1))


    # forward pass
    def forwardPass(self, x,y):
        q1 = self.w1.forwardPass(x)
        h = self.a1.forwardPass(q1)
        q2 = self.w2.forwardPass(h)
        h2 = self.a2.forwardPass(q2)
        q3 = self.w3.forwardPass(h2)
        self.loss_l.forwardPass(q3, y)

    # backward pass
    def backwardPass(self,y):
        dL = self.loss_l.backwardPass(y)
        dL_dh2 = self.w3.backwardPass(dL)
        dAct2 = self.a2.backwardPass(dL_dh2)
        dL_dh1 = self.w2.backwardPass(dAct2)  # updates weights
        dAct = self.a1.backwardPass(dL_dh1)
        self.w1.backwardPass(dAct)

    # set activation
    def setUpdate(self, update):
        self.w1.update_type = update
        self.w2.update_type = update
        self.w3.update_type = update

    # train
    def train(self, x, y, xVal = None, yVal = None):

        if self.saveLoss:
            self.trainLoss = np.zeros((self.numEpochs))
            self.valLoss = np.zeros((self.numEpochs))

        for i in range(self.numEpochs):

            trainLoss = 0
            valLoss = 0

            if self.rotateUpdate is not None:
                self.setUpdate(self.rotateUpdate[i%len(self.rotateUpdate)])

            for indBatch in range(len(x)):
                self.forwardPass(x[indBatch], y[indBatch])
                self.backwardPass(y[indBatch])
                trainLoss += self.loss_l.loss

                if self.saveLoss:
                    if xVal is not None and yVal is not None:
                        self.predict(x=xVal, y=yVal)
                        valLoss += self.loss_l.loss

            trainLoss = trainLoss/len(x)
            valLoss = valLoss/len(x)

            if self.saveLoss:
                self.trainLoss[i] = trainLoss
                self.valLoss[i] = valLoss


            if self.toPrint and i%100 == 0:
                print("epoch ", i, ", training loss: ", trainLoss)

    # tune network
    def tune(self, x , y, x_val, y_val, lr_grid, dimOut1_grid, numEpochs):
        bestModel = 0
        bestModel_loss = 1000

        for lr in lr_grid:
            for currDimOut1 in dimOut1_grid:
                dimOut2_grid = dimOut1_grid[dimOut1_grid < currDimOut1]
                for currDimOut2 in dimOut2_grid:

                    # initialize network
                    NN = myNeuralNetwork(lr=lr, numEpochs=numEpochs, dimIn1=dimIn1, dimOut1=currDimOut1,dimIn2=currDimOut1,
                                         dimOut2=currDimOut2, dimIn3=currDimOut2, dimOut3=dimOut3, batchSize=50,update_type = self.w1.update_type,
                                         activation= self.a1.activation,saveLoss=self.saveLoss,
                                         toPrint=False)


                    NN.train(x, y)
                    NN.predictScores(x_val, y_val)

                    if NN.loss_l.loss < bestModel_loss:
                        bestModel = NN
                        bestModel_loss = bestModel.loss_l.loss

                        if self.toPrint:
                            print("Best model has validation loss: ", bestModel_loss)
        return bestModel

    # predict probabilities
    def predictScores(self,x,y):
        q1 = self.w1.forwardPass(x)
        h = self.a1.forwardPass(q1)
        q2 = self.w2.forwardPass(h)
        h2 = self.a2.forwardPass(q2)
        q3 = self.w3.forwardPass(h2)
        self.loss_l.forwardPass(q3, y)
        return self.loss_l.scores

    # predict classes
    def predict(self, x, y):
        scores = self.predictScores(x,y)
        outClasses = list()
        for i in range(scores.shape[0]):
            outClasses.append(list(scores[i]).index(max(scores[i])))
        return outClasses

   # model string
    def __repr__(self):
        return self.repr_string
