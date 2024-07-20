import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import *
from boosting import *
from k_neighbors import *
from neural_network import *
from svm import *

print("\nDataset 1 - Hit or Flop? Music by the Decade (specify decade in algorithm_manager.py, default is 2010s)")
print("Dataset 2 - Adarsh's Liked Songs")
userInputValue = input("\nWhich dataset would you like to test (1,2)?\n")

# NN, Boosting, and SVM validation curves and Learning Curves  are commented out to avoid long run times
# Best hyperparameter values have already been added manually for these (except DT and K neighbor)
# uncomment the code in each file to view graphs similar to Decision Tree and K Neigbhor
acc_dt, time_dt = decision_tree(userInputValue)
acc_nn, time_nn = neural_network(userInputValue)
acc_boost, time_boost = boosting(userInputValue)
acc_svm, time_svm = svm(userInputValue)
acc_k, time_k = k_neighbors(userInputValue)

accs = [acc_dt, acc_nn, acc_boost, acc_svm, acc_k]
tick_label = ['Decision Tree', 'Neural Network', 'Boosted DT', 'SVM', 'K Neighbors']
plt.bar([1,3,5,7,9] , accs, tick_label = tick_label,
        width = 0.4,color =['blue', 'red'])

plt.xlabel('Learning Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy for Learning Algorithms')
plt.show()

times = [time_dt, time_nn, time_boost, time_svm, time_k]
plt.bar([1,3,5,7,9] , times, tick_label = tick_label,
        width = 0.5,color =['blue', 'red'])

plt.xlabel('Learning Algorithm')
plt.ylabel('Time Elapsed (ms)')
plt.title('Wall Clock Time for Learning Algorithms')
plt.show()