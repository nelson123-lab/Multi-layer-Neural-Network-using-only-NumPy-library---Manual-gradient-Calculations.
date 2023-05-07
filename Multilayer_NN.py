
import numpy as np

"""
The code below pass all test cases of the Assignment_01_tests_single_sample.py
"""
def sigmoid(x):
  # X : The output of each Neurons
  return 1 / (1 + np.exp(-x))

def MSE(predicted, Actual):
  # predicted : The yhat that is obtained after passing sample throught the Neural Network
  # Actual : The Y_test of the data that is being compared with the yhat
  # MSE is the mean of square of (predicted - Actual)
  return np.mean((predicted - Actual)**2)


def Weights_initializer(layers, X_train, seed):
  # layers : The no of layers in the Neural Network
  # X_train : The X_train of the data
  # seed : The seed value of the randamization
  
  Weights = []
  for idx, ele in enumerate(layers):

    # Taking the seed value of the randamization.
    np.random.seed(seed)

    # The intial weights are taken according to the shape of the inputs
    if idx == 0: Weights.append(np.random.randn(layers[idx], X_train.shape[0] +1))

    # The weights in the other layers are taken depending on the previous layers
    else: Weights.append(np.random.randn(layers[idx], layers[idx-1]+1))
  return Weights

def shape_normalizer(X):
  # X : The data to which ones should be added on the top to make the shape match with the weights + bias matrix
  return np.concatenate((np.ones((1, X.shape[1])), X))


def layer_output_finder(layer_weights, X_train_sample, layers):
  # layer_weights : Weights of each layers, it can be updated weights or the final weights of the fully trained neural network.
  # X_train_sample : The training samples one at a time
  # layers : The no of layers in the Neural network when helps in finding the outputs in each layer.

  Outputs_layers = [] # Outputs of each layer in neural network is stored in this list.
  for i in range(len(layers)): 
    # Output of first layer depends on the inputs
    if len(Outputs_layers) == 0: Outputs_layers.append(sigmoid(np.dot(layer_weights[i], X_train_sample)))

    # Outputs of next layer depends on the previous layers
    else: Outputs_layers.append(sigmoid(np.dot(layer_weights[i], shape_normalizer(Outputs_layers[-1]))))
  return Outputs_layers[-1]

def MSE_UWm(Updated_Wm, X_train_sample, Y_train_sample, layers):
  yhat = layer_output_finder(Updated_Wm, X_train_sample, layers)
  return MSE(yhat, Y_train_sample)

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    # This function creates and trains a multi-layer neural Network
    Weights = Weights_initializer(layers, X_train, seed) # LIst containing the weight matrix of each layer.
    
    # List containing the MSE per epoch.
    MSE_Final = [] 

    # Adding the extra ones to the X_train and X_test so that it can be multiplied with the weights + bias.
    X_train_new, X_test_new = shape_normalizer(X_train), shape_normalizer(X_test)

    # Creating a deep copy of the weights matrix so that it won't have the reference to the original weights.
    U_w = [np.copy(i) for i in Weights]
    
    for i in range(epochs):
      # Within each epoch, all the samples are running.
      for i in range(X_train.shape[1]):
        # Taking each sample at a time by iterating through the coulmns
        X_train_sample, Y_train_sample = X_train_new[:,i:i+1], Y_train[:,i:i+1]

        # Changing the entire weights wrt to a single sample.
        for idx1, layer_weight in enumerate(Weights): # Going through each layer
          for idx2, weights in enumerate(layer_weight): # Going through the first row of each weight matrix within a layer.
            for idx3, single_w in enumerate(weights): # Taking each element of the weight matrix
              """
              Derivative are made by changing the weights values from w to w+h first and then
              chaning it to w-2h and finally w-h. Thus the weights will be remain as w itself.
              """
              # Only changing the initial_w to initial_w + h
              Weights[idx1][idx2][idx3] = Weights[idx1][idx2][idx3] + h

              # FInding f(x+h)
              fa = MSE_UWm(Weights, X_train_sample, Y_train_sample, layers)

              # Making the weights to w-h
              Weights[idx1][idx2][idx3] = Weights[idx1][idx2][idx3] - (2*h)
              
              # FInding f(x-h)
              fb = MSE_UWm(Weights, X_train_sample, Y_train_sample, layers)

              # Making the weights back to the original weights
              Weights[idx1][idx2][idx3] = Weights[idx1][idx2][idx3] + h

              #dMSE/dw = f(x+h) - f(x-h)/2h
              dmse_W = (fa - fb)/(2*h)

              # Wnew = Wold - alpha*dmse_W
              Wnew = Weights[idx1][idx2][idx3] - alpha * dmse_W

              # Storing the Wnew to a Temp_w matrix in the same position.
              U_w[idx1][idx2][idx3] = Wnew

        # Updating the new updated weights to the old weights after each sample is trained.
        Weights = [np.copy(i) for i in U_w] 
      
      # Testing starts after training of entrie samples occurs.
      MSE_List = []
      for i in range(X_test.shape[1]):
        # Taking the test samples one at a time.
        X_test_sample, Y_test_sample = X_test_new[:,i:i+1], Y_test[:,i:i+1]
        MSE_List.append(MSE_UWm(U_w, X_test_sample, Y_test_sample, layers))
      MSE_Final.append(np.mean(MSE_List))
  
    # Storing the results to the final weights that is beinng outputed.
    W = U_w

    # err will be the final MSE list that is the error from each epoch.
    err = MSE_Final

    # Finding the final outputs from the final weights that the Neural Network.
    Out = layer_output_finder(W, X_test_new, layers)

    return [W, err, Out]


