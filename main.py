# import libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import NeuralNetwork_Precious as net
import time

# load data
with open('trainData.pkl', 'rb') as f:
    trainData = pickle.load(f)
with open('valData.pkl', 'rb') as g:
    valData = pickle.load(g)
with open('testData.pkl', 'rb') as h:
    testData = pickle.load(h)

# split predictor and target
x = [item[0] for item in trainData]
y = [item[1] for item in trainData]
xVal, yVal = valData[0], valData[1]
xTest, yTest = testData[0], testData[1]

# set up network architecture
dimIn1 = x[0].shape[1]
dimOut1 = 25
dimIn2 = dimOut1
dimOut2 = 10
dimIn3 = dimOut2
dimOut3 = 10    # number of classes
batchSize = 5
numEpochs = 500

# RMS prop  took  7320.570811986923 seconds for 10 repeats of 500 epochs
# SGD took  3525.5227842330933 for 10 repeats of 500 epochs

update_type = 'RMSprop'
rotateUpdate = ['SGD','RMSprop','Adagrad']
activation = 'Leaky RELU'
numRepeats = 30
numSteps_training = 1

start_time = time.time()

for repeat in range(numRepeats):
    # initialize network

    NN = net.myNeuralNetwork(1e-3, numEpochs, dimIn1, dimOut1, dimIn2, dimOut2, dimIn3,
                             dimOut3, batchSize, update_type, activation, saveLoss=True, toPrint=True, rotateUpdate = rotateUpdate)

    """
    # or load already trained network for fine tuning
    with open('myNN 10 Leaky RELU RMSprop.pkl', 'rb') as f:
        NN = pickle.load(f)
    """


    for step in range(numSteps_training):

        NN.train(x=x, y=y, xVal=xVal, yVal=yVal)


        with open('trainLoss ' + str(repeat) + ' rotate_3 updates' + activation + '.pkl', 'wb') as ff:
            pickle.dump(NN.trainLoss, ff)

        with open('valLoss ' + str(repeat) + ' rotated_3 updates'  + activation + '.pkl', 'wb') as ff:
            pickle.dump(NN.valLoss, ff)


        """
        # for saving the network
        with open('myNN ' + str(step) + ' '+ activation + ' ' + update_type + '.pkl', 'wb') as ff:
            pickle.dump(NN, ff)
        """

time_taken = time.time() - start_time
print("took ", time_taken)



