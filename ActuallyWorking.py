from numpy import exp, array, random, dot
import numpy as np
import time
Obstacles = 0
ObstacleTypes = [1,0]
left = 0
centerleft = 0.2
center = 0.4
centerright = 0.6
right = 0.8
rightleft = 1
Ob1 = random.random()
Ob2 = random.random()
Ob3 = random.random()
#INDENTATIONS ARE 2

class NeuralNetwork():
  def __init__(self):
    random.seed(1)
    self.Weights = random.random((3,1))
  #Modified from https://raw.githubusercontent.com/rcassani/mlp-example/master/mlp.py
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
      Error = Outputs - 1.5*Output
      Gradient = dot(Inputs.T, Error * self.SigmoidDx(Output))
      self.Weights += Gradient

  def Predict(self, Inputs):
    return self.Sigmoid(dot(Inputs, self.Weights))
#-----------------------------------------------------------------------------------

def Wait(seconds):
  time.sleep(seconds)

#My modification of the above code:

#Actual Functions:
def DisplayPredict(Input):
  nn = NeuralNetwork()
  Var = nn.PredictLu(Input)
  return Var
  print (Var)

def Exe():
  #Hi, this is going to be all the inside stuff going on before the program starts ;)
  TrainedI = array([[1,0,1],[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0.7,1],[1,0.5,1],[1,1,0]])  #These are all the inputs and outputs we'll train our data with
  TrainedO = array ([[0.4, 0.6, 1, 0.2, 0, 0.4, 0.4, 0.8]]).T
  #0.0   means left
  #0.2   means center/left
  #0.4   means center
  #0.6   means center/right
  #0.8   means right
  #1.0   means right/left
  nn = NeuralNetwork()  #This is used to use the class more easily so I can just type nn instead of Neural Network every time
  #I need to use a function from it. Really convenient to have an object here, so it's not restricted by the rules
  #of being an integer, boolean, etc. 
  nn.TrainLu(TrainedI, TrainedO, 100000)
  Display()
  
def GenerateConfidence(Input):
  if Var == 0:
    Output = abs(0 - Input)
  if Var == 0.2:
    Output = abs(0.1 - Input)
  if Var == 0.4:
    Output = abs(0.3 - Input)
  if Var == 0.6:
    Output = abs(0.5 - Input)
  if Var == 0.8:
    Output = abs(0.7 - Input)
  if Var == 1:
    Output = abs(0.9 - Input)
  Percentage = Output * 100
  if Input == 0:
    Percentage = 100
  print ("The AI is",int(Percentage),"% confident in its decision.")
  return (int(Percentage))

def ReversePsychology():
  #This is to help parse our previous data using the following: 
  #0.0   means left
  #0.2   means center/left
  #0.4   means center
  #0.6   means center/right
  #0.8   means right
  #1.0   means right/left
  if Var == 0:
    print ("The AI decides to go right or center.")
  elif Var == 0.2:
    print ("The AI decides to go right.")
  elif Var == 0.4:
    print ("The AI decides to go left or right")
  elif Var == 0.6:
    print ("The AI decides to go left")
  elif Var == 0.8:
    print ("The AI decides to go left or center")
  elif Var == 1:
    print ("The AI decides to go center.")
  
def CheckChance(Chance):
  global Var
  if Chance <= 0.1:
    Var = 0
    print ("The AI is about to go left")
  elif Chance <= 0.3:
    Var = 0.2
    print ("The AI is about to go center or left")
  elif Chance <= 0.5:
    Var = 0.4
    print ("The AI is about to go center")
  elif Chance <= 0.7:
    Var = 0.6
    print ("The AI is about to go center or right")
  elif Chance <= 0.9:
    Var = 0.8
    print ("The AI is about to go right")
  elif Chance > 0.9:
    Var = 1
    print ("The AI is about to go right or left")

def GenerateRandObstacle():
    LearnInput = array([[Ob1, Ob2, Ob3]])
    print ("Here is the obstacle:")
    print (LearnInput)
    Chance = DisplayPredict(LearnInput)
    CheckChance(Chance)
    if GenerateConfidence(Chance) < 50:
      print ("The AI decides not to make this decision.")
    ReversePsychology()
  
def Display():
  GenerateRandObstacle()
  
    
Exe()
