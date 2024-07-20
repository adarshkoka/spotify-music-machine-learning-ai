from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time

def neural_network(userInputValue):
    
    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    # COMMETNED cross validation and learning curve below
    # MANUALLY SETTING BEST HYPERPARAMS to avoid long run time

    # activations = ['logistic', 'tanh', 'relu', 'identity']
    # layers = []
    # for i in range(6, 14):
    #     layers.append((i,))

    # # https://stackoverflow.com/questions/62363657/how-can-i-plot-validation-curves-using-the-results-from-gridsearchcv
    # grid_params = {
    #     'activation' : activations,
    #     'hidden_layer_sizes': layers
    # }
    # gs = GridSearchCV(
    #     MLPClassifier(),
    #     grid_params,
    #     cv=5,
    #     return_train_score=True,
    #     scoring='accuracy'
    # )
    # clf = gs.fit(X, Y)
    # plot_grid_search(clf.cv_results_, layers, activations, 'Hidden Layer Sizes', 'Activations')
    # print(gs.best_params_)

    # # https://www.geeksforgeeks.org/using-learning-curves-ml/
    # sizes, training_scores, testing_scores = learning_curve(MLPClassifier(
    #                     max_iter = 1000,
    #                     hidden_layer_sizes=gs.best_params_.get('hidden_layer_sizes'), 
    #                     random_state=0, activation = gs.best_params_.get('activation')
    #                 ), 
    #                 X, Y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, .50, 10))

    # # # Training Size limit set to .50 of data to avoid Bias 
    # mean_training = np.mean(training_scores, axis=1)
    # mean_testing = np.mean(testing_scores, axis=1)
    # plt.plot(sizes, mean_training, '--', color="b",  label="Training Score")
    # plt.plot(sizes, mean_testing, color="g", label="Cross Validation Score")
    # plt.title("Learning Curve - Neural Network")
    # plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    # plt.tight_layout()
    # plt.show()
    # max_index_acc = mean_testing.argmax()
    # train_sizes=np.linspace(0.01, .50, 10)
    # bestTrainingSize = train_sizes[max_index_acc]

    # print(bestTrainingSize)

    # Fitting Final Neural Network with below Hyperparameters
    # Using most accurate max_depth, and min_samples_leaf based on above validation curves and pruning

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
                    random_state=0, activation = bestActivation, verbose=True)

    finalMlpc.fit(x_train, y_train)
    y_preds = finalMlpc.predict(x_test)
    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy: ', acc,"%")

    end = time.time()

    return [acc, (end - start) * 1000]