## Version 5 - try to add layer unlimited NN  processing

import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation4, backward_propagation4, forward_propagation5, backward_propagation5, forward_propagation6, backward_propagation6
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
#from testCases import *

import ast
import pandas as pd
import random
import time
import itertools
#import matplotlib.pyplot as plt


from itertools import groupby
import re

from os import listdir
from os.path import isfile, join
from PIL import Image


def get_deep_modificator(df = None):
    #modif = max(df['z'])
    modif = 959
    return modif  # must return 959

def get_files_list(train_images_path, train_masks_path):
    train_image_files = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
    train_mask_files = [f for f in listdir(train_masks_path) if isfile(join(train_masks_path, f))]
    return (train_image_files,train_mask_files)

def get_columns_list(L):
    columns = list(np.array([['W{}'.format(i), 'b{}'.format(i)] for i in range(1,L)]).reshape(1,-1)[0])
    return columns

def write_model_csv(filename, parameters, L):
    model_df = pd.DataFrame(columns = get_columns_list(L))
    model_df = model_df.append([parameters], sort = False)
    model_df.to_csv('model_nn'+'.csv', mode='w', header=True)
    
    model_cost = pd.DataFrame(columns = ['costs'])
    model_cost = model_cost.append([parameters['costs']])
    model_cost.to_csv('model_nn_cost'+'.csv', mode='w', header=True)

    
def continue_parameters(layer_dims, file):
    L = len(layer_dims)     
    file_columns = ['i'] + get_columns_list(L) + ['cost','epoch','meta']

    frame = pd.read_csv(file['filename']+ '.csv', names = file_columns)
    offset = len(frame.index)-1
    model = frame.iloc[offset]
    parameters = {}
    
    for i in range(1,L):
        parameters['W' + str(i)] = np.array(ast.literal_eval(model['W' + str(i)]))
        parameters['b' + str(i)] = np.array(ast.literal_eval(model['b' + str(i)]))

    #parameters['W2'] = np.array(ast.literal_eval(model['W2']))
    #parameters['b2'] = np.array(ast.literal_eval(model['b2']))

    #parameters['W3'] = np.array(ast.literal_eval(model['W3']))
    #parameters['b3'] = np.array(ast.literal_eval(model['b3']))

    #parameters['W4'] = np.array(ast.literal_eval(model['W4']))
    #parameters['b4'] = np.array(ast.literal_eval(model['b4']))

    
    return parameters, offset



def processing_single_file(fname, img_arr, msk_arr, frames_label, deep, add_xy = False):
       
    time_check = [0 for i in range(8)]
    #begin = time.time()
    #time_check[0] = 0
    #tdf = pd.DataFrame(columns = [ 'file_id', 'filename', 'deep', 'surrounds_id', 'px_id', 'pix_x', 'pix_y', 'row', 'col', 'surround', 'len','sault','sault_scale','extended_scaled_surround'])    
    tdf = pd.DataFrame(columns = ['sault_scale','extended_scaled_surround'])    
    #start_t = time.time() 
    arr_iw = np.hstack((img_arr for x in range(25)))  #step 2
    #end_t = time.time()
    #time_check[2] = end_t - start_t 
    
    arr_iw = np.vstack((arr_iw for x in range(25)))
    arr_mw = np.hstack((msk_arr for x in range(25)))
    arr_mw = np.vstack((arr_mw for x in range(25)))
    
    split_k = 101
 
    #start_t = time.time() 
 # step 3
    
    #end_t = time.time()
    #time_check[3] = end_t - start_t 


    palette_i = [np.array_split(arr, split_k, axis=0) for arr in np.array_split(arr_iw, split_k, axis = 1)]     
    frames = [np.array(palette_i[j][i]).reshape(1,-1)[0] for i,j in itertools.product(range(split_k), range(split_k))]  # step 4 
    
  
    #start_t = time.time()     
    #palette_m = [np.array_split(arr, split_k, axis=0) for arr in np.array_split(arr_mw, split_k, axis = 1)]    
    #frames_label = [palette_m[j][i][12][12] for i,j in itertools.product(range(split_k), range(split_k))]

    #j_and_i = [(j*25+12 - ((j*25+12)//101)*101, i*25+12 - ((i*25+12)//101)*101) for i,j in itertools.product(range(split_k), range(split_k))]
    #j_ind = [x[0] for x in j_and_i]
    #i_ind = [x[1] for x in j_and_i]
    #print(j_ind)
    #mm = np.array(msk_arr[i_ind, j_ind]) == np.array(frames_label)
    
    #print(mm[mm == False])
    #print(np.array(frames_label))
    #end_t = time.time()
    #time_check[4] = end_t - start_t    
    # 
    #start_t = time.time() 
    #frames_label_len = len(frames_label)
    frames_label_len = 10201
    
    #end_t = time.time()
    #time_check[4] = end_t - start_t    

    #start_t = time.time() 
    #tdf['file_id'] = [id for x in range(frames_label_len)]  # step 5
    #end_t = time.time()
    #time_check[5] = end_t - start_t 
    
    #tdf['filename'] = [fname for x in range(frames_label_len)]
    tdf['surrounds_id'] = [x for x in range(frames_label_len)]
    tdf['row'] = tdf['surrounds_id']%101
    tdf['col'] = tdf['surrounds_id']//101
    tdf['pix_x'] = 12+tdf['col']*25 - 101*((12 + tdf['col']*25)//101)
    tdf['pix_y'] = 12+tdf['row']*25 - 101*((12 + tdf['row']*25)//101)
    #tdf['px_id'] = tdf['pix_y'] + tdf['pix_x']*101
    #tdf['len'] = tdf['surround'].apply(lambda x: len(x)) 
    #tdf['deep'] = [deep for x in range(frames_label_len)]
    #start_t = time.time()    
    #tdf['sault'] = frames_label
    #tdf['sault_scale'] =  (tdf['sault'] > 0)*1
    tdf['sault_scale'] =  frames_label
    #print(frames_label)
    #end_t = time.time()
    #time_check[5] = end_t - start_t 


    
    
    #start_t = time.time() 
    #tdf['surround'] = frames
    #tdf['extended_scaled_surround'] = [np.append(j, tdf['deep'][i]) for i,j in enumerate(tdf['surround'])] # step 6
    #tdf['extended_scaled_surround'] = [np.append(j, deep) for i,j in enumerate(tdf['surround'])] # step 6   
    
    tdf['extended_scaled_surround'] = [np.append(j, deep) for i,j in enumerate(frames)] # step 6    
    if add_xy == True:
        f1 = np.array([x for x in tdf['extended_scaled_surround'].values])
        f2 = tdf['pix_x'].values.reshape(-1,1)/101
        f3 = tdf['pix_y'].values.reshape(-1,1)/101
        f_res = np.hstack((f1, f2, f3))
        #print(f_res.shape)
        tdf['extended_scaled_surround'] = list(f_res)
        #print(tdf.head())

    
    #end_t = time.time()
    #time_check[6] = end_t - start_t     
    
    #end = time.time()
    #time_check[0] = end - begin
    #print(time_check)
    return tdf 

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        ### END CODE HERE ###
        
    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches :]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        ### END CODE HERE ###
        
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
        ### END CODE HERE ###
        
    return parameters, v

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    ### END CODE HERE ###
    
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = beta1 * v["dW"+str(l+1)] + (1 - beta1)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db"+str(l+1)] + (1 - beta1)*grads["db"+str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l+1)] = beta2 * s["dW"+str(l+1)] + (1-beta2) * grads["dW"+ str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db"+str(l+1)] + (1-beta2) * grads["db"+ str(l+1)]**2
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l+1)] = s["dW"+str(l+1)]/(1-beta2**t)
        s_corrected["db" + str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*(v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon))
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*(v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon))
        ### END CODE HERE ###

    return parameters, v, s

def convert_file_to_frame(file_obj):
    
    
    df = pd.DataFrame(columns = ['id', '2D-img', '2D-msk', '2D-msk-resh'])
    train_images_path = file_obj['train_images_path']
    train_masks_path = file_obj['train_masks_path'] 
    train_image_files = file_obj['train_image_files'] 
    train_mask_files = file_obj['train_mask_files']    
    #train_filename = [x.split('.')[0] for x in train_image_files]
    deep_filename = file_obj['deep_filename']
    deep_frame = pd.read_csv(deep_filename, index_col = 'id')
    split_k = 101
    j_and_i = [(j*25+12 - ((j*25+12)//101)*101, i*25+12 - ((i*25+12)//101)*101) for i,j in itertools.product(range(split_k), range(split_k))]
    j_ind = [x[0] for x in j_and_i]
    i_ind = [x[1] for x in j_and_i]  
    #msk_arr[i_ind, j_ind]
    
    for ind, name in enumerate(train_image_files):
        img2D = np.array(Image.open(train_images_path + '/' + name).convert(mode='L'))/255
        msk2D = np.array(Image.open(train_masks_path + '/' + name).convert(mode='L'))/255
        msk2D_resh = msk2D[i_ind, j_ind]
        #print(msk2D)
        #tdf['sault_scale'] =  (tdf['sault'] > 0)*1
        item = {'id':name, '2D-img': img2D, '2D-msk': msk2D, '2D-msk-resh': msk2D_resh}
        df = df.append([item], sort = False, ignore_index=True)
    df = df.set_index('id')    
    deep_frame['ind'] = [x + '.png' for x in deep_frame.index] 
    deep_frame['fname'] = deep_frame.index
    deep_frame['z1'] = deep_frame['z']/959
    deep_frame = deep_frame.set_index('ind')
    deep = deep_frame[['z1','fname']]
    df = df.merge(deep, left_index = True, right_index = True)
#    deep_frame.set_index('ind')
    return df

def model(file_obj, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True, continue_model = False, add_xy = False):


    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    start_time = time.time()
    iterat_time = time.time()

    train_images_path = file_obj['train_images_path']
    train_masks_path = file_obj['train_masks_path'] 
    train_image_files = file_obj['train_image_files'] 
    train_mask_files = file_obj['train_mask_files']
    filename = file_obj['filename']
    #file_columns = ['W1','b1','W2','b2','W3','b3','cost','epoch','meta']

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    cost = 0
    cost_avg_epoch = 0
    param_to_write = {}
    param_to_write_progress = {}
    offset = 0
    
    start_t = time.time() 
    print('processing image files to DataFrame')
    file_frame = convert_file_to_frame(file_obj)
    end_t = time.time()
    print('takes {} second '.format(end_t - start_t))

  
    if (L-1 == 4):
        print('4th layers NN')
        forward_func = forward_propagation4
        backward_func = backward_propagation4
    if (L-1 == 5):
        print('5th layers NN')
        forward_func = forward_propagation5
        backward_func = backward_propagation5
    if (L-1 == 6):
        print('6th layers NN')
        forward_func = forward_propagation6
        backward_func = backward_propagation6
        
       
        
    # Initialize parameters
    if continue_model == True:
        print('lets continur')
        parameters, offset = continue_parameters(layers_dims, file_obj)
    else:
        parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):
        
        costs_in_epoch = []

        np.random.seed(round(time.time())) 
        fname_list = list(file_frame.index)
        
        permutation = list(np.random.permutation(len(fname_list)))
        shuffled_files = np.array(file_frame.index)[permutation]
        #print(fname_list)
        #print(shuffled_files)

        for ind, name in enumerate(shuffled_files):
            #print('file',name)
            
            img_array = file_frame.loc[name]['2D-img']
            msk_array = file_frame.loc[name]['2D-msk']
            msk_array_resh = file_frame.loc[name]['2D-msk-resh']
            deep = file_frame.loc[name]['z1']
            fname = file_frame.loc[name]['fname']

            #img_file = train_images_path + '/'+ name
            #msk_file = train_masks_path + '/'+ name
            #print('img_file:', img_file)
            
            frame = processing_single_file(fname, img_array, msk_array, msk_array_resh, deep, add_xy)

            X_data_arr = frame['extended_scaled_surround']
            Y_data_arr = frame['sault_scale']
            #print(X_data_arr.tolist())
            #print(np.array(X_data_arr.tolist()))
            minibatch_X  = np.array([x for x in X_data_arr])
            minibatch_X  = minibatch_X.T
            minibatch_Y  = np.array([y for y in Y_data_arr]).reshape(1,-1)

            # Forward propagation
            a_last, caches = forward_func(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a_last, minibatch_Y)
            costs_in_epoch.append(cost)

            # Backward propagation
            grads = backward_func(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
            
            if ind % 5 == 0:
                print('epoch {}; current ind - #{}; cost - {}'.format(i, ind, cost))
                
        cost_avg_epoch = np.average(np.array(costs_in_epoch))
        if (i==0):
            write_mode = 'w'
            write_header = True
            if (continue_model == True):
                write_mode = 'a'
                write_header = False
            meta = '<b> layers: </b>' + str(layers_dims)+'; Total epoches: <b>' + str(num_epochs) + '</b>; Training size: <b>' + str(len(shuffled_files)) + ' </b> learning_rate:<b>' + str(learning_rate) +  ' </b> momentum beta: <b>'+ str(beta) +' </b> <br><br>'
            meta_proc = meta 
        else:  
            write_header = False
            write_mode = 'a'
            meta = 'i: <b>' + str(i) + ' </b>offset: <b>'+  str(offset) + '</b> index: <b>' + str(i + offset) + '</b> avg epoch cost: ' + ' {:}'.format(cost_avg_epoch) +  ' last cost: ' + str(cost) +' end</b> <br><br>'
            meta_proc = 'i: <b>' + str(i) + ' </b>offset: <b>' + str(offset) + '</b> index: <b>' + str(i + offset) +' </b>avg cost: <b>' + ' {:}'.format(cost_avg_epoch) + ' </b>last cost: <b>'+ str(cost) + '</b> time: <b>' + time.strftime('%d.%m.%Y %H:%M:%S', time.localtime()) + '</b><br><br>'

        
        model_file = pd.DataFrame(columns = get_columns_list(L) + ['cost','epoch','meta'])
        param_to_write['meta'] = meta
        param_to_write['epoch'] = i+offset
        param_to_write['cost'] = cost
        for i in range(1,L):
            param_to_write['W'+str(i)] = [list(parameters['W'+str(i)][k]) for k in range(layers_dims[i])]
            param_to_write['b'+str(i)] = [list(parameters['b'+str(i)][k]) for k in range(layers_dims[i])]
#        param_to_write['W2'] = [list(parameters['W2'][k]) for k in range(layers_dims[2])]
#        param_to_write['b2'] = [list(parameters['b2'][k]) for k in range(layers_dims[2])]
#        param_to_write['W3'] = [list(parameters['W3'][k]) for k in range(layers_dims[3])]
#        param_to_write['b3'] = [list(parameters['b3'][k]) for k in range(layers_dims[3])]
#        param_to_write['W4'] = [list(parameters['W4'][k]) for k in range(layers_dims[4])]
#        param_to_write['b4'] = [list(parameters['b4'][k]) for k in range(layers_dims[4])]


        
        model_file = model_file.append([param_to_write]).set_index('epoch')
        model_file.to_csv(filename +'.csv', mode=write_mode, header=write_header)

        model_process = pd.DataFrame(columns = ['epoch','avg cost','total time','iteration time','meta'])
        param_to_write_progress['epoch'] = i+ offset
        param_to_write_progress['avg cost'] = cost_avg_epoch
        param_to_write_progress['total time'] = '{:{width}.{prse}f} sec.'.format(time.time() - start_time, width = 5, prse = 2)
        param_to_write_progress['iteration time'] = '{:{width}.{prse}f} sec.'.format(time.time() - iterat_time, width = 5, prse = 2)
        param_to_write_progress['meta'] = meta_proc

        
        model_process = model_process.append([param_to_write_progress], sort = False).set_index('epoch')
        model_process.to_csv(filename +'_process.csv.html', mode=write_mode, header=write_header)
        print('avg cost', cost_avg_epoch)
        iterat_time = time.time()

        costs.append(cost)
    return parameters, costs, frame


def rle(input_list = []):
    
    l1 = ([[k, (sum(1 for _ in g) )] for k,g in groupby(input_list) if k ==1])
    src1 = [0] + input_list
    l2 = ''.join(str(e) for e in src1)
    l3 = [m.start()+1 for m in re.finditer('01', l2)]
    l4 = [a for a in zip(l1,l3)]
    res = [str(p[1]) + ' ' + str(p[0][1]) for p in l4]
    return str(res).replace('[','').replace(']','').replace(',','').replace("'",'')

def convert_mask(arr):
    im_l = list(arr.T.reshape(1,-1)[0])
    res = rle(im_l)
    return res


def load_parameters(file, L):
    file_columns = ['i'] + get_columns_list(L) + ['cost','epoch','meta']

    frame = pd.read_csv(file['filename']+ '.csv', names = file_columns)
    model = frame.iloc[len(frame.index)-1]
    parameters = {}
    
    for i in range(1,L):
        parameters['W' + str(i)] = ast.literal_eval(model['W' + str(i)])
        parameters['b' + str(i)] = ast.literal_eval(model['b' + str(i)])

    #parameters['W2'] = ast.literal_eval(model['W2'])
    #parameters['b2'] = ast.literal_eval(model['b2'])

    #parameters['W3'] = ast.literal_eval(model['W3'])
    #parameters['b3'] = ast.literal_eval(model['b3'])

    #parameters['W4'] = ast.literal_eval(model['W4'])
    #parameters['b4'] = ast.literal_eval(model['b4'])


    
    return parameters


def convert_resfile_to_frame(file_obj):
    
    
    df = pd.DataFrame(columns = ['id', '2D-img'])
    test_images_path = file_obj['test_images_path']
    test_image_files = file_obj['test_image_files'] 
    #train_filename = [x.split('.')[0] for x in train_image_files]
    deep_filename = file_obj['deep_filename']
    deep_frame = pd.read_csv(deep_filename, index_col = 'id')
    split_k = 101
    #j_and_i = [(j*25+12 - ((j*25+12)//101)*101, i*25+12 - ((i*25+12)//101)*101) for i,j in itertools.product(range(split_k), range(split_k))]
    #j_ind = [x[0] for x in j_and_i]
    #i_ind = [x[1] for x in j_and_i]  
    #msk_arr[i_ind, j_ind]
    
    for ind, name in enumerate(test_image_files):
        img2D = np.array(Image.open(test_images_path + '/' + name).convert(mode='L'))/255
        #print(msk2D)
        #tdf['sault_scale'] =  (tdf['sault'] > 0)*1
        item = {'id':name, '2D-img': img2D}
        df = df.append([item], sort = False, ignore_index=True)
    df = df.set_index('id')    
    deep_frame['ind'] = [x + '.png' for x in deep_frame.index] 
    deep_frame['fname'] = deep_frame.index
    deep_frame['z1'] = deep_frame['z']/959
    deep_frame = deep_frame.set_index('ind')
    deep = deep_frame[['z1','fname']]
    df = df.merge(deep, left_index = True, right_index = True)
#    deep_frame.set_index('ind')
    return df

def get_testfiles_list(test_images_path):
    test_image_files = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
    return test_image_files


def processing_single_testfile(fname, img_arr, deep, add_xy = False):
       
    time_check = [0 for i in range(8)]
    #begin = time.time()
    #time_check[0] = 0
    #tdf = pd.DataFrame(columns = [ 'file_id', 'filename', 'deep', 'surrounds_id', 'px_id', 'pix_x', 'pix_y', 'row', 'col', 'surround', 'len','sault','sault_scale','extended_scaled_surround'])    
    #tdf = pd.DataFrame(columns = ['sault_scale','extended_scaled_surround'])    
    
    tdf = pd.DataFrame(columns = ['surrounds_id', 'px_id', 'pix_x', 'pix_y', 'row', 'col', 'sault_scale','extended_scaled_surround'])    

    
    #start_t = time.time() 
    arr_iw = np.hstack((img_arr for x in range(25)))  #step 2
    #end_t = time.time()
    #time_check[2] = end_t - start_t 
    
    arr_iw = np.vstack((arr_iw for x in range(25)))
    #arr_mw = np.hstack((msk_arr for x in range(25)))
    #arr_mw = np.vstack((arr_mw for x in range(25)))
    
    split_k = 101
 
    #start_t = time.time() 
 # step 3
    
    #end_t = time.time()
    #time_check[3] = end_t - start_t 


    palette_i = [np.array_split(arr, split_k, axis=0) for arr in np.array_split(arr_iw, split_k, axis = 1)]     
    frames = [np.array(palette_i[j][i]).reshape(1,-1)[0] for i,j in itertools.product(range(split_k), range(split_k))]  # step 4 
    
  
    #start_t = time.time()     
    #palette_m = [np.array_split(arr, split_k, axis=0) for arr in np.array_split(arr_mw, split_k, axis = 1)]    
    #frames_label = [palette_m[j][i][12][12] for i,j in itertools.product(range(split_k), range(split_k))]

    #j_and_i = [(j*25+12 - ((j*25+12)//101)*101, i*25+12 - ((i*25+12)//101)*101) for i,j in itertools.product(range(split_k), range(split_k))]
    #j_ind = [x[0] for x in j_and_i]
    #i_ind = [x[1] for x in j_and_i]
    #print(j_ind)
    #mm = np.array(msk_arr[i_ind, j_ind]) == np.array(frames_label)
    
    #print(mm[mm == False])
    #print(np.array(frames_label))
    #end_t = time.time()
    #time_check[4] = end_t - start_t    
    # 
    #start_t = time.time() 
    #frames_label_len = len(frames_label)
    frames_label_len = 10201
    
    #end_t = time.time()
    #time_check[4] = end_t - start_t    

    #start_t = time.time() 
    #tdf['file_id'] = [id for x in range(frames_label_len)]  # step 5
    #end_t = time.time()
    #time_check[5] = end_t - start_t 
    
    #tdf['filename'] = [fname for x in range(frames_label_len)]
    tdf['surrounds_id'] = [x for x in range(frames_label_len)]
    tdf['row'] = tdf['surrounds_id']%101
    tdf['col'] = tdf['surrounds_id']//101
    tdf['pix_x'] = 12+tdf['col']*25 - 101*((12 + tdf['col']*25)//101)
    tdf['pix_y'] = 12+tdf['row']*25 - 101*((12 + tdf['row']*25)//101)
    tdf['px_id'] = tdf['pix_y'] + tdf['pix_x']*101
    #tdf['len'] = tdf['surround'].apply(lambda x: len(x)) 
    #tdf['deep'] = [deep for x in range(frames_label_len)]
    #start_t = time.time()    
    #tdf['sault'] = frames_label
    #tdf['sault_scale'] =  (tdf['sault'] > 0)*1
    
    #tdf['sault_scale'] =  frames_label
    
    #print(frames_label)
    #end_t = time.time()
    #time_check[5] = end_t - start_t 

    
    
    #start_t = time.time() 
    #tdf['surround'] = frames
    #tdf['extended_scaled_surround'] = [np.append(j, tdf['deep'][i]) for i,j in enumerate(tdf['surround'])] # step 6
    #tdf['extended_scaled_surround'] = [np.append(j, deep) for i,j in enumerate(tdf['surround'])] # step 6    
    tdf['extended_scaled_surround'] = [np.append(j, deep) for i,j in enumerate(frames)] # step 6    
    
    #if add_xy == True:
    f1 = np.array([x for x in tdf['extended_scaled_surround'].values])
    f2 = tdf['pix_x'].values.reshape(-1,1)/101
    f3 = tdf['pix_y'].values.reshape(-1,1)/101
    f_res = np.hstack((f1, f2, f3))
    #print(f_res.shape)
    tdf['extended_scaled_surround_xy'] = list(f_res)   
    #end_t = time.time()
    #time_check[6] = end_t - start_t     
    
    #end = time.time()
    #time_check[0] = end - begin
    #print(time_check)
    return tdf 