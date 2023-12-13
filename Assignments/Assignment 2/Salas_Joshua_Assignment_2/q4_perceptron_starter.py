import pandas as pd
import numpy as np
import math
import sklearn.metrics.pairwise as kernel_lib

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
num_features = 4
start_index = 0 #for the task of classifying versicolor vs virginia, change it to 50
end_index = 100 #for the task of classifying versicolor vs virginia, change it to 150
y = df.iloc[start_index:end_index, num_features].values
y = np.array(y)
y = np.where(y == 'Iris-versicolor', -1, 1)
X = df.iloc[start_index:end_index,:num_features].values

def baseline_perceptron(X,Y):
  max_iter = 100000
  w = np.zeros(X.shape[1])
  b = 0
  i = 0
  correct_count = 0
  while correct_count < X.shape[0] and i < max_iter:
    # exit when number of mistakes = 0
    # if mistake > 0 iterate over entire perceptron
    i = i+1
    for j in range(X.shape[0]):
      #implement baseline perceptron
      #iterate over your entire dataset
  #return i epoch at which you exit or epoch at which your algorithm converges

def kernelized_perceptron(X,Y):
  max_iter = 100000
  K_map = kernel_lib.polynomial_kernel(X,X,degree=25)
  alpha_map = np.zeros(X.shape[0])
  b = 0
  i = 0
  correct_count = 0
  while correct_count < X.shape[0] and i < max_iter:
    # exit when number of mistakes = 0
    # if mistake > 0 iterate over entire perceptron
    i = i+1
    for j in range(X.shape[0]):
      #implement baseline perceptron
      #iterate over your entire dataset
  # return i epoch at which you exit or epoch at which your algorithm converges

epoch_baseline = baseline_perceptron(X,Y)
epoch_kernelized = kernelized_perceptron(X,Y)

print("epoch_baseline: ", epoch_baseline)
print("epoch_kernelized: ", epoch_kernelized)