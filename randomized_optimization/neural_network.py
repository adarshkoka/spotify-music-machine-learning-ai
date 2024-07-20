from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import  mlrose_hiive as mlrose
import time

def neural_network_opt_weights(userInputValue):
    
    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    if userInputValue == '1':
        bestTrainingSize = .229
        bestActivation = 'identity'
        bestHiddenLayerSize = 10
    else:
        bestTrainingSize = .1875
        bestActivation = 'identity'
        bestHiddenLayerSize = 10        

    finalSplitList= getSplit(X,Y, .40);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    # finalMlpc = MLPClassifier(max_iter = 1000,
    #                 hidden_layer_sizes=bestHiddenLayerSize, 
    #                 random_state=0, activation = bestActivation, verbose=True)
    # 
    
    print("doing rhc")
    nn_rhc = mlrose.NeuralNetwork(activation=bestActivation, hidden_nodes= [bestHiddenLayerSize],
        algorithm='random_hill_climb', max_attempts=100, max_iters=1000, is_classifier=True, early_stopping=True)
    
    print("doing sa")
    nn_sa = mlrose.NeuralNetwork(activation=bestActivation, hidden_nodes = [bestHiddenLayerSize],
        algorithm='simulated_annealing', max_attempts=100, max_iters=1000, is_classifier=True, early_stopping=True)

    print("doing ga")
    nn_ga = mlrose.NeuralNetwork(activation=bestActivation, hidden_nodes = [bestHiddenLayerSize],
        algorithm='genetic_alg', max_attempts=100, max_iters=1000, is_classifier=True, early_stopping=True)

    neural_net_algs = [nn_rhc,nn_sa,nn_ga]

    acc_list = []
    time_list = []

    for nn_alg in neural_net_algs:
        print('...')

        global GLO_start
        temp = time.time()
        GLO_start = temp

        nn_alg.fit(x_train,y_train)
        y_preds = nn_alg.predict(x_test)

        global GLO_acc
        temp_acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
        GLO_acc = temp_acc

        global GLO_end
        temp2 = time.time()
        GLO_end = temp2

        acc_list.append(GLO_acc)
        time_list.append((GLO_end - GLO_start) * 1000)
        print('Accuracy: ', GLO_acc,"%")

    return [acc_list, time_list]

def neural_network_original(userInputValue):
    
    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    if userInputValue == '1':
        bestTrainingSize = .229
        bestActivation = 'identity'
        bestHiddenLayerSize = 10
    else:
        bestTrainingSize = .1875
        bestActivation = 'identity'
        bestHiddenLayerSize = 10        

    start = time.time()

    finalSplitList= getSplit(X,Y, .40);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    finalMlpc = MLPClassifier(max_iter = 1000,
                    hidden_layer_sizes=bestHiddenLayerSize, 
                    random_state=0, activation = bestActivation, verbose=False)

    finalMlpc.fit(x_train, y_train)
    y_preds = finalMlpc.predict(x_test)
    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Oringal NN Accuracy: ', acc,"%")

    end = time.time()

    return [acc, (end - start) * 1000]