from sklearn.base import BaseEstimator
from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import time

def decision_tree(userInputValue):
    
    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

    maxDepths = np.arange(2, 17, 1)
    minSamplesLeafs = np.arange(10, 31, 2)


    # grid_params = {
    #     'max_depth' : maxDepths,
    #     'min_samples_leaf': minSamplesLeafs
    # }
    # gs = GridSearchCV(
    #     DecisionTreeClassifier(),
    #     grid_params,
    #     cv=5,
    #     return_train_score=True,
    #     scoring='accuracy'
    # )
    # clf = gs.fit(X, Y)
    # plot_grid_search(clf.cv_results_, maxDepths, minSamplesLeafs, 'Hidden Layer Sizes', 'Activations')
    # print(gs.best_params_)

    # bestMaxDepth = gs.best_params_.get('max_depth')
    # bestMinSamplesLeaf = gs.best_params_.get('min_samples_leaf')


    # https://www.geeksforgeeks.org/validation-curve/
    # Validation Curve max depth

    if(userInputValue_DatasetChoice == "1"):

        parameter_range = np.arange(2, 17, 1)
        hyperParam = "max_depth"
        bestMaxDepth = getValidationCurveForHP("1", DecisionTreeClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, Decision Tree, " + hyperParam, hyperParam)

        hyperParam = "min_samples_leaf"
        parameter_range = np.arange(10, 31, 2)
        bestMinSamplesLeaf = getValidationCurveForHP("1", DecisionTreeClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, Decision Tree, " + hyperParam, hyperParam)
    else:
        parameter_range = np.arange(2, 17, 1)
        hyperParam = "max_depth"
        bestMaxDepth = getValidationCurveForHP("2", DecisionTreeClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, Decision Tree, " + hyperParam, hyperParam)

        hyperParam = "min_samples_leaf"
        parameter_range = np.arange(10, 31, 1)
        bestMinSamplesLeaf = getValidationCurveForHP("2", DecisionTreeClassifier(), X,Y,
            hyperParam, parameter_range, 5, "accuracy",
            "Validation Curve, Decision Tree, " + hyperParam, hyperParam)

    # https://www.geeksforgeeks.org/using-learning-curves-ml/

    sizes, training_scores, testing_scores = learning_curve(DecisionTreeClassifier(
                    random_state = 0, criterion='entropy',
                    max_depth =  bestMaxDepth, 
                    min_samples_leaf= bestMinSamplesLeaf
                    ), 
                    X, Y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, .51, 50))

    # Training Size limit set to .50 of data to avoid Bias 
    mean_training = np.mean(training_scores, axis=1)
    mean_testing = np.mean(testing_scores, axis=1)
    plt.plot(sizes, mean_training, '--', color="b",  label="Training Score")
    plt.plot(sizes, mean_testing, color="g", label="Cross Validation Score")
    plt.title("Learning Curve - Decision Tree")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    max_index_acc = mean_testing.argmax()
    train_sizes=np.linspace(0.01, .51, 50)
    bestTrainingSize = train_sizes[max_index_acc]

    bestAlpha = getPrune(X,Y, bestMaxDepth, bestMinSamplesLeaf, bestTrainingSize)

    print("Best Hyperparamter Values and Training Size:")
    print(bestMaxDepth)
    print(bestMinSamplesLeaf)
    print(bestAlpha)
    print(bestTrainingSize)

    # Fitting Final Decision Tree with below Hyperparameters
    # Using most accurate Alpha, max_depth, and min_samples_leaf based on above validation curves and pruning

    start = time.time()

    finalSplitList= getSplit(X,Y,bestTrainingSize);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    # No Pruning
    finalDecisionTree = DecisionTreeClassifier(
        max_depth = bestMaxDepth,
        min_samples_leaf = bestMinSamplesLeaf,
        random_state = 0, 
        criterion='entropy'
    )

    finalDecisionTree.fit(x_train, y_train)
    y_preds = finalDecisionTree.predict(x_test)

    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy without Pruning: ', acc,"%")

    fig, _ = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(finalDecisionTree,
                feature_names = X.columns, 
                class_names=np.unique(Y).astype('str'),
                filled = True)
    fig.savefig('decision_tree_no_pruning.png')

    # With Pruning
    finalDecisionTreePruned = DecisionTreeClassifier(
        ccp_alpha = bestAlpha,
        max_depth = bestMaxDepth,
        min_samples_leaf = bestMinSamplesLeaf,
        random_state = 0, 
        criterion='entropy'
    )

    finalDecisionTreePruned.fit(x_train, y_train)
    y_preds = finalDecisionTreePruned.predict(x_test)

    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy with Pruning: ', acc,"%")

    end = time.time()

    fig2, _ = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(finalDecisionTreePruned,
                feature_names = X.columns, 
                class_names=np.unique(Y).astype('str'),
                filled = True)
    fig2.savefig('decision_tree_pruned.png')

    return [acc, (end - start) * 1000]