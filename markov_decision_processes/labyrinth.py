import hiive.mdptoolbox as mdptoolbox
import numpy as np
from numpy.core.fromnumeric import mean
import matplotlib.pyplot as plt

print("Labyrinth Problem")
np.random.seed(0)

P = [[[0 for _ in range(8)] for _ in range(8)] for _ in range(2)]

f = open("labyrinth.txt")

for line in f:
    k = line.split(" ")
    a = int(k[0])
    s = int(k[1])
    ns = int(k[2])
    p = float(k[3])
    P[a][s][ns] = p

P = np.array(P)
R = [[[0 for _ in range(8)] for _ in range(8)] for _ in range(2)]

# Escape Reward
R[0][7][7] = 100 
R[1][7][7] = 100

# Fight the Minotaur
R[1][4][7] = 500

# Gifting Hades
R[1][0][0] = -200

# River Styx Burn
R[1][2][2] = -50

# Death "Reward"
R[0][6][6] = -50
R[1][6][6] = -100

R = np.array(R)

# vi.setVerbose()
vi = mdptoolbox.mdp.ValueIteration(P, R, .9, epsilon = .01, max_iter = 1000, run_stat_frequency = 1)
print("")
print("Value Iteration")
run_stats = vi.run()
print("Number of Iterations: ", vi.iter)
print("Time: ", vi.time)
# print(vi.V)
print(vi.policy)

mean_per_iteration = [x['Mean V'] for x in run_stats]
rewards_per_iteration = [x['Reward'] for x in run_stats]
plt.plot(mean_per_iteration)
plt.xlabel('Iteration')
plt.ylabel('Mean Value')
plt.title("Value Iteration Mean Values")
plt.show()
plt.plot(rewards_per_iteration)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title("Value Iteration Reward")
plt.show()

pi = mdptoolbox.mdp.PolicyIteration(P, R, .9, max_iter = 10000, run_stat_frequency = 1)
run_stats = pi.run()

mean_per_iteration = [x['Mean V'] for x in run_stats]
rewards_per_iteration = [x['Reward'] for x in run_stats]
plt.plot(mean_per_iteration)
plt.xlabel('Iteration')
plt.ylabel('Mean Value')
plt.title("Policy Iteration Mean Values")
plt.show()
plt.plot(rewards_per_iteration)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title("Policy Iteration Reward")
plt.show()
print("Number of Iterations: ", pi.iter)
print("Time: ", pi.time)

# Reward vs. Iteration graph

print("")
print("Q Learning")
# cant be less than 10K n_iter -- even tho 4K is better to show curve
ql = mdptoolbox.mdp.QLearning(P,R, .9, run_stat_frequency = 1, n_iter = 10000, epsilon = .01)
run_stats = ql.run()
mean_per_iteration = [x['Mean V'] for x in run_stats]

plt.plot(mean_per_iteration)
plt.xlabel('Episode')
plt.ylabel('Mean Value')
plt.title("Q-Learning Mean Values")
plt.show()