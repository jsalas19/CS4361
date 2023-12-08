# -*- coding: utf-8 -*-


# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Let's create our training data 12 pairs {x_i, y_i}
# We'll try to fit the straight line model to these data
data = np.array([[0.05,0.22,0.34,0.46,0.68,0.91,1.08,1.18,1.39,1.60,1.65,1.90],
                 [0.37,0.82,1.05,1.00,1.20,1.50,1.30,1.54,1.25,1.68,1.73,1.60]])

# Let's define our model
def model(phi,x):
  y_pred = phi[0]+phi[1] * x + phi[2]*x*x
  return y_pred

# Initialize the parameters to some arbitrary values
phi = np.zeros((3,1))
phi[0] = 0.7
phi[1] = 0.4
phi[2] = 0.3

#define MSE loss:
def compute_loss(data_x, data_y, phi):

  print(data_x)
  error = 0
  for i in range(data_x.shape[0]):
    y_p = model(phi,data_x[i])
    #print('y_p is {} y_true is {}'.format(y_p,data_y[i]))
    error += (y_p - data_y[i])*(y_p - data_y[i])

  loss = (error)/data_x.shape[0]
  print('Loss is {}'.format(loss))
  return loss

loss = compute_loss(data[0,:],data[1,:],phi)
print('Your loss = %3.3f'%(loss))

def change_in_loss(data_x,data_y,gradients,phi,lr=0.01):
  #Implement this function
  new_phi = phi - lr * gradients #implement
  loss = compute_loss(data_x, data_y, phi)
  loss_new = compute_loss(data_x, data_y, new_phi)
  print('Difference between old loss {} and new loss {}'.format(loss,loss_new))





def compute_gradient(data_x, data_y, phi,d=0.005):

    dl_dphi0 = (compute_loss(data_x, data_y, phi + np.array([[d], [0], [0]])) - compute_loss(data_x, data_y, phi)) / d
    dl_dphi1 = (compute_loss(data_x, data_y, phi + np.array([[0], [d], [0]])) - compute_loss(data_x, data_y, phi)) / d
    dl_dphi2 = (compute_loss(data_x, data_y, phi + np.array([[0], [0], [d]])) - compute_loss(data_x, data_y, phi)) / d

    #d = 0.005
    #implement this function
    # Return the gradient
    return np.array([[dl_dphi0],[dl_dphi1],[dl_dphi2]])


# Compute the gradient using your function

gradients = compute_gradient(data[0,:],data[1,:], phi)
change_in_los = change_in_loss(data[0,:],data[1,:],gradients,phi)


