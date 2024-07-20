from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
import time

def k_neighbors(userInputValue):

    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    # Validation Curve

    numneighbors = np.arange(100, 601, 50)
    leafSizes = np.arange(1, 30, 1)

    if(userInputValue_DatasetChoice == "1"):

        hyperParam = "leaf_size"
        parameter_range = leafSizes
        bestLeafSize = getValidationCurveForHP("1", KNeighborsClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, K Neighbors, " + hyperParam, hyperParam)

        parameter_range = numneighbors
        hyperParam = "n_neighbors"
        bestNumNeigh = getValidationCurveForHP("1", KNeighborsClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, K Neighbors, " + hyperParam, hyperParam)
    else:

        hyperParam = "leaf_size"
        parameter_range = leafSizes
        bestLeafSize = getValidationCurveForHP("2", KNeighborsClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, K Neighbors, " + hyperParam, hyperParam)

        parameter_range = numneighbors
        hyperParam = "n_neighbors"
        bestNumNeigh = getValidationCurveForHP("2", KNeighborsClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, K Neighbors, " + hyperParam, hyperParam)

    # https://www.geeksforgeeks.org/using-learning-curves-ml/

    sizes, training_scores, testing_scores = learning_curve(KNeighborsClassifier(
                    # need to limit based on training size
                    # n_neighbors =  bestNumNeigh, 
                    leaf_size= bestLeafSize
                    ), 
                    X, Y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, .5, 10))

    # Training Size limit set to .50 of data to avoid Bias 
    mean_training = np.mean(training_scores, axis=1)
    mean_testing = np.mean(testing_scores, axis=1)
    plt.plot(sizes, mean_training, '--', color="b",  label="Training Score")
    plt.plot(sizes, mean_testing, color="g", label="Cross Validation Score")
    plt.title("Learning Curve - K Neighbors")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("Best Hyperparamter Values and Training Size:")
    print(bestNumNeigh)
    print(bestLeafSize)

    # set these based on results
    bestNumNeigh = 400
    bestTrainingSize =.5

    start = time.time()

    # No need to split into training for K neighbors
    finalSplitList= getSplit(X,Y, bestTrainingSize);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    k_neighbors = KNeighborsClassifier(n_neighbors=bestNumNeigh, leaf_size=bestLeafSize)
    k_neighbors.fit(x_train, y_train)

    y_preds = k_neighbors.predict(x_test)

    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy : ', acc,"%")

    end = time.time()

    return [acc, (end - start) * 1000]