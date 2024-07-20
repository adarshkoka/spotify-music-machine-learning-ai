import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neural_network import *

print("\nDataset 1 - Hit or Flop? Music by the Decade (specify decade in algorithm_manager.py, default is 2010s)")
print("Dataset 2 - Adarsh's Liked Songs")
userInputValue = input("\nWhich dataset would you like to test (1,2)?\n")

# NN, Boosting, and SVM validation curves and Learning Curves  are commented out to avoid long run times
# Best hyperparameter values have already been added manually for these (except DT and K neighbor)
# uncomment the code in each file to view graphs similar to Decision Tree and K Neigbhor
original_acc_nn, original_time_nn = neural_network_original(userInputValue)

# print(original_acc_nn)

opt_acc_list, opt_time_list = neural_network_opt_weights(userInputValue)

print("Accuracies")
print(opt_acc_list)
print("Wall Clock Times")
print(opt_time_list)

accs = [original_acc_nn, opt_acc_list[0], opt_acc_list[1], opt_acc_list[2]]
tick_label = ['Original NN', 'RHC NN', 'SA NN', 'GA NN']
plt.bar([1,3,5,7] , accs, tick_label = tick_label,
        width = 0.4,color =['blue', 'red'])

plt.xlabel('Neural Network Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy for NN Algorithms')
plt.show()

times = [original_time_nn, opt_time_list[0], opt_time_list[1], opt_time_list[2]]
plt.bar([1,3,5,7] , times, tick_label = tick_label,
        width = 0.5,color =['blue', 'red'])

plt.xlabel('Neural Network Algorithm')
plt.ylabel('Time Elapsed (ms)')
plt.title('Wall Clock Time for NN Algorithms')
plt.show()