import numpy as np
import random
import math
from matplotlib import pyplot as plt

#Change this tp choose which graphs to replicate:
create_figure_3 = False
create_figure_4 = False
create_figure_5 = False


#Create random action:
def action():
    x = random.uniform(0,1)
    if (x > 0.5): return 1
    else: return 0

#Create a transition function(state, action), return new state:
def transition_function(current_state,action):
    if(np.array_equal(current_state,np.array([[1,0,0,0,0]]).T) and action == 0): return 0
    elif (np.array_equal(current_state,np.array([[0,0,0,0,1]]).T) and action == 1): return 1
    empty_state = [0,0,0,0,0]
    current_index = np.where(current_state == 1)[0]
    if (action == 1): new_index = current_index + 1
    else: new_index = current_index - 1
    empty_state[new_index.item()] = 1
    new_state = np.array([empty_state]).T
    return new_state

#Create value function:
def value_function(state,weight):
    if (isinstance(state, np.ndarray)): return weight.T.dot(state).item()
    elif(state == 1) or (state == 0): return state

#Experiment 1: Repeated presentations training paradigm
actual_values = np.array([[1/6,2/6,3/6,4/6,5/6]]).T
initial_state = np.array([[0,0,1,0,0]]).T

if create_figure_3:
    alpha = 0.01
    right_side_termination = 0
    left_side_termination = 0
    lambda_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    RMSE_list = [0,0,0,0,0,0,0]
    training_set = 100 #Number of training set for my paper figure 2 and 3, increase to 500 for my paper figure 4
    for i in range(len(lambda_list)):
        w = np.array([[0,0,0,0,0]]).T
        set = 0
        delta = np.array([[0,0,0,0,0]]).T
        RMSE = 0
        while set < training_set:
            random.seed(3) #Data set for replication of Sutton Figure 3 (Corresponding to my paper figure 2)
            #random.seed(456) #Data set for my paper figure 3
            for k in range(10):
                #Start of a sequence:
                sequence = True
                e = np.array([[0,0,0,0,0]]).T
                current_state = initial_state
                while sequence:
                    a = action()
                    next_state = transition_function(current_state,a)
                    if isinstance(next_state, np.ndarray):
                        delta = delta + alpha*(value_function(next_state,w) - value_function(current_state,w))*e
                        e = e*lambda_list[i] + next_state
                        current_state = next_state
                    else:
                        delta = delta + alpha*(value_function(next_state,w) - value_function(current_state,w))*e
                        sequence = False
            w = w + delta
            RMSE = RMSE + math.sqrt(np.sum(np.square(w-actual_values)/5))
            delta = np.array([[0,0,0,0,0]]).T
            set = set + 1
        RMSE_avg = RMSE/(set)
        RMSE_list[i] = RMSE_avg
    #Create figure 3
    plt.plot(lambda_list,RMSE_list)
    plt.xlabel('λ')
    plt.ylabel('Average RMSE')
    plt.show()

#Experiment 2: Examining learning rate when training set is presented once:
if create_figure_4: #Replicating Sutton's Figure 4 (My paper figure 5)
    lambda_list = [0,0.3,0.8,1]
    alpha_list = np.array(list(range(0,65,5)))/100
    RMSE_avg_matrix = np.zeros((len(lambda_list),len(alpha_list)))

    for i in range(len(lambda_list)):
        for k in range(len(alpha_list)):
            random.seed(456)
            set = 0
            delta = np.array([[0,0,0,0,0]]).T
            RMSE = 0
            while set < 100:
                w = np.array([[0.5,0.5,0.5,0.5,0.5]]).T
                for seq in range(10):
                    #Start of a sequence:
                    sequence = True
                    e = np.array([[0,0,0,0,0]]).T
                    current_state = initial_state
                    while sequence:
                        a = action()
                        next_state = transition_function(current_state,a)
                        if isinstance(next_state, np.ndarray):
                            delta = delta + alpha_list[k]*(value_function(next_state,w) - value_function(current_state,w))*e
                            e = e*lambda_list[i] + next_state
                            current_state = next_state
                        else:
                            delta = delta + alpha_list[k]*(value_function(next_state,w) - value_function(current_state,w))*e
                            sequence = False
                    w = w + delta
                    RMSE = RMSE + math.sqrt(np.sum(np.square(w-actual_values)/5))
                    delta = np.array([[0,0,0,0,0]]).T
                set = set + 1
            RMSE_avg = RMSE/(set*10)
            RMSE_avg_matrix[i,k] = RMSE_avg

    colors_fig4 = ['b', 'g', 'r', 'c']

    for i in range(len(lambda_list)):
        plt.plot(alpha_list, RMSE_avg_matrix[i, :], label='λ = ' + str(lambda_list[i]), color=colors_fig4[i])

    plt.xlabel('α')
    plt.ylabel('Average RMSE')
    plt.legend()
    plt.show()

#Figure 5:
if create_figure_5: #Replicating Sutton's Figure 5 (My paper figure 6)
    lambda_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha_list = np.array(list(range(0,65,5)))/100
    RMSE_avg_matrix = np.zeros((len(lambda_list),len(alpha_list)))

    for i in range(len(lambda_list)):
        for k in range(len(alpha_list)):
            random.seed(456)
            set = 0
            delta = np.array([[0,0,0,0,0]]).T
            RMSE = 0
            while set < 100:
                w = np.array([[0.5,0.5,0.5,0.5,0.5]]).T
                for seq in range(10):
                    #Start of a sequence:
                    sequence = True
                    e = np.array([[0,0,0,0,0]]).T
                    current_state = initial_state
                    while sequence:
                        a = action()
                        next_state = transition_function(current_state,a)
                        if isinstance(next_state, np.ndarray):
                            delta = delta + alpha_list[k]*(value_function(next_state,w) - value_function(current_state,w))*e
                            e = e*lambda_list[i] + next_state
                            current_state = next_state
                        else:
                            delta = delta + alpha_list[k]*(value_function(next_state,w) - value_function(current_state,w))*e
                            sequence = False
                    w = w + delta
                    RMSE = RMSE + math.sqrt(np.sum(np.square(w-actual_values)/5))
                    delta = np.array([[0,0,0,0,0]]).T
                set = set + 1
            RMSE_avg = RMSE/(set*10)
            RMSE_avg_matrix[i,k] = RMSE_avg

    #Find best learning rate
    best_error = []
    for i in range(len(lambda_list)):
        best_error.append(np.amin(RMSE_avg_matrix[i]))

    plt.plot(lambda_list,best_error)
    plt.xlabel('λ')
    plt.ylabel('Average RMSE Using Best α')
    plt.show()
