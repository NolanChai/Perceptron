from numpy import exp, array, random, dot
import numpy as np
import time

class NeuralNetwork():
    def __init__(self):
        random.seed(100)
        self.Weights = random.random((3,1))
    def ReLu(self, x):
        if np.isscalar(x):
            result = np.max((x, 0))
        else:
            zero_aux = np.zeros(x.shape)
            meta_x = np.stack((x, zero_aux), axis = -1)
            result = np.max(meta_x, axis = -1)
        return result

    def ReLuDx(self, x):
        result = 1 * (x > 0)
        return result
  #-----------------------------------------------------------------------------------
  #Custom Training function using a ReLu activation instead.
    def TrainLu(self, Inputs, Outputs, Epochs):
        for Epoch in range(Epochs):
            Output = self.PredictLu(Inputs)
            Error = Outputs - Output
            Gradient = dot(Inputs.T, Error * self.ReLuDx(Output))
            self.Weights += Gradient

  #Custom Prediction function using a ReLu activation instead.
    def PredictLu(self, Inputs):
        return self.ReLu(dot(Inputs, self.Weights))
#-----------------------------------------------------------------------------------
    def Train(self, Inputs, Outputs, Epochs):
        for Epoch in range(Epochs):
            Output = self.Predict(Inputs)
            Error = Outputs - Output
            Gradient = dot(Inputs.T, Error * self.SigmoidDx(Output))
            self.Weights += Gradient

    def Predict(self, Inputs):
        return self.Sigmoid(dot(Inputs, self.Weights))
#-----------------------------------------------------------------------------------

def Wait(seconds):
    time.sleep(seconds)

""" Test Data
if __name__ == '__main__':
	nn = NeuralNetwork()
	print ("Here are our initial weights:")
	print ("-----------------------------")
	print (nn.Weights)
	TrainedI = array([[1,0,1],[1,0,0],[0,1,0],[0,0,1],[0,1,1]])
	TrainedO = array ([[0.4, 0.6, 1, 0.2, 0]]).T
	nn.Train(TrainedI, TrainedO, 10000)
	print ("New Weights after the Training: ")
	print (nn.Weights)
	print ("And here is our prediction: ")
	print (nn.Predict(array([[1, 1, 0]])))
#The above code is a modified version of Milo Harper's 'Simple Neural Network'
#On Github.
#https://github.com/miloharper/simple-neural-network/blob/master/main.py
	Prediction = nn.Predict(array([[1, 0, 1]]))
	if Prediction > 0.5:
		Response = 1
	if Prediction < 0.5:
		Response = 0
	print (Response)
#Testing ReLu Function
	print ("-----------------------------")
	print ("Testing the Relu Function")
	print ("Initial Weights: ")
	random.seed(11)
	print (nn.Weights)
	nn.TrainLu(TrainedI, TrainedO, 10000)
	print ("New Weights after the Training: ")
	print (nn.Weights)
	print ("And here is our prediction: ")
	print (nn.Predict(array([[1, 1, 0]])))
"""
#My modification of the above code:

#Actual Functions:

def Exe():
    print ("Hello and welcome to my program!")
    print ("Essentially, this program will be a simulation of a neuron trying to make a decision.")
    print ("The neuron will be using an activation function to simulate the movement of potassium and sodium ions")
    print ("between its axon's cellular membrane. Much like how a neuron decides to send a signal based on its input,")
    print ("the activation function I chose is a Rectified Linear Unit (ReLu), which means that any number larger than 0 will")
    print ("return 1, and any number below 0 will return 0. Simple as that.")
    print ("The piece of code I modified from Raymundo Cassani @rcassani on Github uses the function of a ReLu to return")
    print ("a number between 0 and 1 from the input.")
    print ("Using this concept, I will train the neuron to remember certain patterns, and use those patterns to make")
    print ("predictions on new patterns.")
    print ("In this case, I'll use matrices to display models ")
    print ("#-----------------------------------------------------------------------------------\n")
    print ("We will be simulating an AI trying to move its way around obstacles, labeled as")
    print ("left, right, and center.")

  

	
	
