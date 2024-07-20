import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

def getDataset(userInputValue):

    if(userInputValue == '1'):
        # https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset
        print("\nPredicting hit or flop for songs...")
        # Specify the Decade you would like to test below - 60s, 70s, 80s, 90s, 00s, or 10s
        df = pd.read_csv('datasets/dataset-of-10s.csv', encoding='latin-1')

        X = df[[
        'mode','liveness', 'valence', 'tempo',
        'duration_ms', 'time_signature',
        'danceability', 'energy', 'key', 'loudness', 
        'speechiness', 'acousticness', 'instrumentalness'
        ]]
        Y = df['Target'] # 1 for Hit song, 0 for Flop
        
        return [X, Y, userInputValue]

    else:
        # https://github.com/kvithana/spotify-audio-features-to-csv
        # above repo was used to generate Dataset of my liked and not liked songs
        print("\nPredicting Adarsh's liked songs...")
        df = pd.read_csv('datasets/liked-disliked-songs-big-dataset.csv', encoding='latin-1')

        # X = df[[
        # 'duration_ms', 'key','mode', 'time_signature', 'acousticness',
        # 'danceability', 'energy', 'instrumentalness', 'liveness',
        # 'loudness', 'speechiness', 'valence', 'tempo'
        # ]]
        
        X = df[[
        'duration_ms', 'key','mode', 'time_signature', 'acousticness',
        'danceability', 'energy', 'instrumentalness', 'liveness',
        'loudness', 'speechiness', 'valence', 'tempo'
        ]]
        Y = df['Target'] # 1 for  Liked song, 0 for Not Liked 

        return [X, Y, userInputValue]

def getSplit(X, Y, bestTrainingSize):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True, train_size=bestTrainingSize)
    return [x_train,x_test,y_train,y_test]  

def getValidationCurveForHP(datasetNum, alg, X,Y, paramName, parameter_range,cvNum, scor, plotTitle, plotParamString):
    train_score, test_score = validation_curve(alg, X, Y,
        param_name = paramName,
        param_range = parameter_range,
        cv = cvNum, scoring = scor)

    mean_train_score = np.mean(train_score, axis = 1)
    mean_test_score = np.mean(test_score, axis = 1)

    # Used for plotting Validation curves
    plt.plot(parameter_range, mean_train_score,
        label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score,
    label = "Cross Validation Score", color = 'g')
    plt.title("Dataset " + datasetNum + " - " + plotTitle)
    plt.xlabel(plotParamString)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

    max_index = mean_test_score.argmax()
    return parameter_range[max_index]

def getPrune(X,Y, bestMaxDepth, bestMinSamplesLeaf, bestTrainingSize):
    # BELOW CODE IS USED FOR PRUNING
    tempTree = DecisionTreeClassifier(
        random_state = 0, 
        max_depth = bestMaxDepth,
        min_samples_leaf=bestMinSamplesLeaf,
        criterion='entropy'
        )
    print("Pruning...")
    # Used to find alpha for Decision tree
    # https://ranvir.xyz/blog/practical-approach-to-tree-pruning-using-sklearn/
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html


    # Using the Best training size from Learning Curve for pruning
    xyList = getSplit(X,Y, bestTrainingSize);
    x_train = xyList[0]
    x_test = xyList[1]
    y_train = xyList[2]
    y_test = xyList[3]

    path = tempTree.cost_complexity_pruning_path(x_train, y_train)
    alphas = path.ccp_alphas

    clfs = []
    # print("Testable alphas list size: ", len(alphas))

    counter = 0
    for i in alphas:
        testTree = DecisionTreeClassifier(
            ccp_alpha = i,
            random_state = 0, 
            max_depth = bestMaxDepth,
            min_samples_leaf=bestMinSamplesLeaf,
            criterion='entropy'
            )
        testTree.fit(x_train, y_train)
        clfs.append(testTree)
        # print("Tested an Alpha indexed at", counter)
        counter = counter+1

    acc_scores = [accuracy_score(y_test, clf.predict(x_test)) for clf in clfs]
    max_acc = max(acc_scores)
    max_index = acc_scores.index(max_acc)
    # print("Max Accuracy Score: ", max(acc_scores))
    bestAlpha = alphas[max_index]
    # print("Most Accurate Alpha: ", bestAlpha)
    plt.figure(figsize=(10,  6))
    plt.grid()
    plt.plot(alphas[:-1], acc_scores[:-1])
    plt.xlabel("Alpha Values")
    plt.ylabel("Accuracy Scores")
    plt.tight_layout()
    plt.show()

    return bestAlpha

# https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    mean_train_score = cv_results['mean_train_score']
    mean_train_score = np.array(mean_train_score).reshape(len(grid_param_2),len(grid_param_1))
    mean_test_score = cv_results['mean_test_score']
    mean_test_score = np.array(mean_test_score).reshape(len(grid_param_2),len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, mean_train_score[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Hyperparamters using GridSearch")
    ax.set_xlabel(name_param_1)
    ax.set_ylabel('Accuracy Score')
    ax.legend(loc="best")
    ax.grid('on')
    plt.show()
