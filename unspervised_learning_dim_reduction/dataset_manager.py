import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def getDataset(userInputValue):

    if(userInputValue == '1'):
        # https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset
        print("\nPredicting hit or flop for songs...")
        df = pd.read_csv('datasets/dataset-of-10s.csv')

        # X = df[[
        # 'mode','liveness', 'valence', 'tempo',
        # 'duration_ms', 'time_signature',
        # 'danceability', 'energy', 'key', 'loudness', 
        # 'speechiness', 'acousticness', 'instrumentalness'
        # ]]
        # Y = df['Target'] # 1 for Hit song, 0 for Flop
        
        return df

    else:
        # https://github.com/kvithana/spotify-audio-features-to-csv
        # above repo was used to generate Dataset of my liked and not liked songs
        print("\nPredicting Adarsh's liked songs...")
        df = pd.read_csv('datasets/liked-disliked-songs-big-dataset.csv')
        
        # X = df[[
        # 'duration_ms', 'key','mode', 'time_signature', 'acousticness',
        # 'danceability', 'energy', 'instrumentalness', 'liveness',
        # 'loudness', 'speechiness', 'valence', 'tempo'
        # ]]
        # Y = df['Target'] # 1 for  Liked song, 0 for Not Liked 

        return df

def getSplit(X, Y, bestTrainingSize):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True, train_size=bestTrainingSize)
    return [x_train,x_test,y_train,y_test]  