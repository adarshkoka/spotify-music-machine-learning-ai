from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import time

def svm(userInputValue):
    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    # c_values = np.arange(1.0, 10.1, 1.0)
    # # kernelFunctions = ['poly', 'rbf', 'sigmoid']
    # kernelFunctions = ['rbf', 'poly'] #deg for poly is 3

    # grid_params = {
    #     'kernel' : kernelFunctions,
    #     'C': c_values
    # }
    # gs = GridSearchCV(
    #     SVC(),
    #     grid_params,
    #     cv=5,
    #     return_train_score=True,
    #     scoring='accuracy'
    # )
    # clf = gs.fit(X, Y)
    # plot_grid_search(clf.cv_results_, c_values, kernelFunctions, 'C Values', 'Kernel Functions')
    # print(gs.best_params_)

    # bestC = gs.best_params_.get('C')
    # bestKernelFunction = gs.best_params_.get('kernel')

    # # https://www.geeksforgeeks.org/using-learning-curves-ml/
    # sizes, training_scores, testing_scores = learning_curve(SVC(
    #                 C =  bestC, 
    #                 kernel= bestKernelFunction
    #                 ), 
    #                 X, Y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, .51, 50))

    # # Training Size limit set to .50 of data to avoid Bias 
    # mean_training = np.mean(training_scores, axis=1)
    # mean_testing = np.mean(testing_scores, axis=1)
    # plt.plot(sizes, mean_training, '--', color="b",  label="Training Score")
    # plt.plot(sizes, mean_testing, color="g", label="Cross Validation Score")
    # plt.title("Learning Curve - SVM")
    # plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    # plt.tight_layout()
    # plt.show()
    # max_index_acc = mean_testing.argmax()
    # train_sizes=np.linspace(0.01, .51, 50)
    # bestTrainingSize = train_sizes[max_index_acc]

    # print("Best Hyperparamter Values and Training Size:")
    # print(bestKernelFunction)
    # print(bestC)
    # print(bestTrainingSize)

    # setting best variables from cross validation and learning curve 
    # manually to avoid long run times
    if userInputValue == '1':
        bestC = 6.0
        bestKernelFunction = 'rbf'
        bestTrainingSize = .5
    else:
        bestC = 1.0
        bestKernelFunction = 'rbf'
        bestTrainingSize = .5

    # Fitting Final SVM with Hyperparameters

    start = time.time()
    finalSplitList= getSplit(X,Y,bestTrainingSize);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    finalSVM = SVC(
        C =  bestC, 
        kernel= bestKernelFunction,
        verbose = True
    )

    finalSVM.fit(x_train, y_train)
    y_preds = finalSVM.predict(x_test)

    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy: ', acc,"%")

    end = time.time()

    return [acc, (end - start) * 1000]

