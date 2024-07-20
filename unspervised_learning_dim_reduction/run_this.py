from dataset_manager import *
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import homogeneity_score
import matplotlib.cm as cm
import matplotlib.style as style
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.feature_selection import VarianceThreshold
import  mlrose_hiive as mlrose

def k_means_cluster(df, datasetNum):
    start = time.time()
    distortions = []
    sils = []
    for k in range (2, 16):
        kmeans = KMeans(n_clusters = k)
        labels = kmeans.fit_predict(df)
        distortions.append(kmeans.inertia_)
        sils.append(silhouette_score(df, labels))
    plt.plot(range(2, 16), sils, marker='o')
    plt.xlabel('Number of K Clusters')
    plt.ylabel('Silhouette Score')
    plt.title("Silhouette Score vs. K plot")
    plt.show()
    plt.plot(range(2, 16), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title("Elbow Method plot")
    plt.show()

    bestNumClusters = 2
    if datasetNum == '1':
        bestNumClusters = 5
    else:
        bestNumClusters = 4

    kmeans = KMeans(n_clusters = bestNumClusters)
    end = time.time()

    df_temp = getDataset(num)
    Y = df_temp['Target']

    labels = kmeans.fit_predict(df)
    print("Homogeneity score:")
    print(homogeneity_score(Y, labels))
    print("Completeness score:")
    print(completeness_score(Y, labels))
    print("Silhouette score:")
    print(silhouette_score(df, labels))

    centroids = kmeans.cluster_centers_
    u_labels = np.unique(labels)
        
    for i in u_labels:
        plt.scatter(df[labels == i , 0] , df[labels == i , 1] , label = i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.title("K Means Cluster scatterplot")
    plt.show()
    return df

def gmm(df, datasetNum):

    sils = []
    for k in range (2, 20):
        gmm = GaussianMixture(n_components = k)
        labels = gmm.fit_predict(df)
        sils.append(silhouette_score(df, labels))
    plt.plot(range(2, 20), sils, marker='o')
    plt.xlabel('Number of K Clusters')
    plt.ylabel('Silhouette Score')
    plt.title("Silhouette Score vs. K plot")
    plt.show()

    
    n_components = np.arange(2, 20)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df) for n in n_components]
    gmm_model_comparisons=pd.DataFrame({"n_components" : n_components,
                                  "BIC" : [m.bic(df) for m in models],
                                   "AIC" : [m.aic(df) for m in models]})
    plt.figure(figsize=(8,6))
    plt.title("GMM AIC and BIC Plot")
    sns.lineplot(data=gmm_model_comparisons[["BIC","AIC"]])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.show()

    bestNumClusters = 2
    if datasetNum == '1':
        bestNumClusters = 10
    else:
        bestNumClusters = 13

    gmm = GaussianMixture(bestNumClusters, random_state=30).fit(df)
    labels = gmm.predict(df)
    df_temp = getDataset(num)
    Y = df_temp['Target']
    print(homogeneity_score(Y, labels))
    print(completeness_score(Y, labels))
    print(silhouette_score(df, labels))
    return df

def pca(df):
    pca = PCA(n_components = 2) # only looking at 2 features
    df = pca.fit_transform(df)
    print(df)
    return df

def ica(df):
    ica = FastICA(n_components=2, random_state=0)
    df = ica.fit_transform(df)
    return df

def randProj(df):
    randomProjection = random_projection.GaussianRandomProjection(n_components=2)
    df = randomProjection.fit_transform(df)
    return df

def varianceThreshold(df):
    vt = VarianceThreshold()
    df = vt.fit_transform(df)
    return df

def runNeuralNet(df, Y, datasetNum):
        startTime = time.time()

        if datasetNum == '1':
            bestTrainingSize = .229
            bestActivation = 'identity'
            bestHiddenLayerSize = 10
        else:
            bestTrainingSize = .1875
            bestActivation = 'identity'
            bestHiddenLayerSize = 10

        # df = getDataset(datasetNum)
        # Y = df['Target']
        # X = df.drop('Target', axis=1).values

        x_train, x_test, y_train, y_test = train_test_split(df, Y, random_state=0, shuffle=True, train_size=.4)
        
        nn_ga = mlrose.NeuralNetwork(activation=bestActivation, hidden_nodes = [bestHiddenLayerSize],
            algorithm='genetic_alg', max_attempts=100, max_iters=1000, is_classifier=True, early_stopping=True)

        nn_ga.fit(x_train,y_train)
        y_preds = nn_ga.predict(x_test)

        acc = round(accuracy_score(y_test, y_preds)*100.0, 2)
        endTime = time.time()

        # acc_list.append(GLO_acc)
        # time_list.append((GLO_end - GLO_start) * 1000)
        print('Accuracy: ', acc,"%")
        print('Time: ', (endTime - startTime) * 1000, "ms")

num = '1'

df = getDataset(num)
Y = df['Target']
df = df.drop('Target', axis=1).values

# Comment/Uncomment below lines to run the Experiment Combinations

df = pca(df)
# df = ica(df)

print(df)
# df = randProj(df)
# df = varianceThreshold(df)

# df = k_means_cluster(df, num)
df = gmm(df, num)

runNeuralNet(df, Y, num)