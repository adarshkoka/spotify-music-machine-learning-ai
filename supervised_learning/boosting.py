from algorithm_manager import *
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
import time

def boosting(userInputValue):

    xyList = getDataset(userInputValue);
    X = xyList[0]
    Y = xyList[1]
    userInputValue_DatasetChoice = xyList[2]

# Commented out below because Boosting is taking a long time for Cross validation and Learning curve
# Best Values for hyperparamters and training Size are inputted manually based on curves

    # https://www.geeksforgeeks.org/validation-curve/
    # Validation Curve max depth
    # grid_params = {
    #     'n_estimators' : 
    #     'learning_rate': 
    # }
    # gs = GridSearchCV(
    #     AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')), X,Y,
            # hyperParam, parameter_range, 5, "accuracy",
            # "Validation Curve, Boosted DT, " + hyperParam, hyperParam),
    #     grid_params,
    #     cv=5,
    #     return_train_score=True,
    #     scoring='accuracy'
    # )
    # clf = gs.fit(X, Y)
    # plot_grid_search(clf.cv_results_, n_estimators, learning_rate, 'N Estimators', 'Learning Rate')
    # print(gs.best_params_)

    # if(userInputValue_DatasetChoice == "1"):
    #     max_depth = 6
    #     min_samples_leaf = 28
    #     alpha = 0.002416808801660595
    #     training_size = .39
    #     parameter_range = np.arange(50, 301, 50)
    #     hyperParam = "n_estimators"
    #     DecisionTreeClassifier(ccp_alpha = alpha,max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')
    #     bestNestimators = getValidationCurveForHP("1", AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')), X,Y,
    #         hyperParam, parameter_range, 5, "accuracy",
    #         "Validation Curve, Boosted DT, " + hyperParam, hyperParam)
    #     print("cross validating learning rate...")

    #     hyperParam = "learning_rate"
    #     parameter_range = np.arange(.001, .100, .010)
    #     bestLearningRate = getValidationCurveForHP("1",AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')), X,Y,
    #         hyperParam, parameter_range, 5, "accuracy",
    #         "Validation Curve, Boosted DT, " + hyperParam, hyperParam)
    # else:
    #     max_depth = 7
    #     min_samples_leaf = 17
    #     alpha = 0.0030861651359571085
    #     training_size = .45
    #     parameter_range = np.arange(50, 301, 50)
    #     hyperParam = "n_estimators"
    #     bestNestimators = getValidationCurveForHP("2",AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')), X,Y,
    #         hyperParam, parameter_range, 5, "accuracy",
    #         "Validation Curve, Boosted DT, " + hyperParam, hyperParam)
    #     print("cross validating learning rate...")
    #     hyperParam = "learning_rate"
    #     parameter_range = np.arange(.001, .100, .010)
    #     # max_depth = 7,min_samples_leaf = 17, ccp_alpha=alpha, random_state=0, criterion = 'entropy'
    #     bestLearningRate = getValidationCurveForHP("2", AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy')), X,Y,
    #         hyperParam, parameter_range, 5, "accuracy",
    #         "Validation Curve, Boosted DT, " + hyperParam, hyperParam)

    # # https://www.geeksforgeeks.org/using-learning-curves-ml/

    # sizes, training_scores, testing_scores = learning_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy'),
    #                 n_estimators=bestNestimators,learning_rate=bestLearningRate),
    #                 X, Y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, .51, 50))

    # # Training Size limit set to .50 of data to avoid Bias 
    # mean_training = np.mean(training_scores, axis=1)
    # mean_testing = np.mean(testing_scores, axis=1)
    # plt.plot(sizes, mean_training, '--', color="b",  label="Training Score")
    # plt.plot(sizes, mean_testing, color="g", label="Cross Validation Score")
    # plt.title("Learning Curve - Boosted DT")
    # plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    # plt.tight_layout()
    # plt.show()
    # max_index_acc = mean_testing.argmax()
    # train_sizes=np.linspace(0.01, .51, 50)
    # bestTrainingSize = train_sizes[max_index_acc]
    # bestTrainingSize = training_size

    # bestAlpha = getPrune(X,Y, bestMaxDepth, bestMinSamplesLeaf, bestTrainingSize)

    bestNestimators = 300
    if userInputValue == '1':
        bestLearningRate = .071
        bestTrainingSize = .39
        max_depth = 6
        min_samples_leaf = 28
    else:
        max_depth = 7
        min_samples_leaf = 17
        bestLearningRate = .0209
        bestTrainingSize = .45

    print("Best Hyperparamter Values and Training Size:")
    print(bestNestimators)
    print(bestLearningRate)
    print(bestTrainingSize)

    # Fitting Final Decision Tree with below Hyperparameters
    # Using most accurate Alpha, max_depth, and min_samples_leaf based on above validation curves and pruning

    start = time.time()

    finalSplitList= getSplit(X,Y,bestTrainingSize);
    x_train = finalSplitList[0]
    x_test = finalSplitList[1]
    y_train = finalSplitList[2]
    y_test = finalSplitList[3]

    # Ada Boost
    finalBoostedDecisionTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = 0, criterion='entropy'), 
        n_estimators=bestNestimators, learning_rate=bestLearningRate)

    finalBoostedDecisionTree.fit(x_train, y_train)
    y_preds = finalBoostedDecisionTree.predict(x_test)

    acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
    print('Accuracy: ', acc,"%")

    end = time.time()

    return [acc, (end - start) * 1000]
