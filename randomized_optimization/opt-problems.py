import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt

#max k color
arr = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]

# four peaks
# arr = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])

#Flip flop
# arr = np.array([0, 1, 0, 1, 1, 1, 1,
# 0, 1, 0, 1, 1, 1, 1, 0, 1, 1 ,1, 0, 0, 0, 0,
# 0, 1, 0, 1, 1, 1, 1,0, 1, 0, 1, 1, 1, 1
# ])

#N-Queens
# arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# fitnessFunction = mlrose.FourPeaks(t_pct=0.15)
fitnessFunction = mlrose.MaxKColor(arr)
# fitnessFunction = mlrose.FlipFlop()
# fitnessFunction = mlrose.Queens()

problem = mlrose.DiscreteOpt(length = len(arr), fitness_fn = fitnessFunction, maximize = False, max_val = 2)

print("doing rand hill climb")
_,_, randhill_curve = mlrose.random_hill_climb(problem,
    max_attempts = 1000, max_iters = 1000, curve = True
    )

print("doing rand sim ann")
_,_, simulat_curve = mlrose.simulated_annealing(problem,
    max_attempts = 1000, max_iters = 1000, curve = True, schedule=mlrose.ExpDecay()
    )

print("doing genetic alg")
_,_, genetic_curve = mlrose.genetic_alg(problem,
    max_attempts = 1000, max_iters = 1000, curve = True, pop_size=200,mutation_prob=.2
    )

print("doing mimic")
_,_, mimic_curve = mlrose.mimic(problem,
    max_attempts = 1000, max_iters = 1000, curve = True
    )

iterations = range(1,1001)
plt.plot(iterations, randhill_curve[:,0], label = "RandHillClimb", color = "brown") # four peaks best
plt.plot(iterations, simulat_curve[:,0], label = "SimAnneal", color = "blue") # Max K Color Best
plt.plot(iterations, genetic_curve[:,0], label = "GeneticAlg", color = "green") # Flip Flop large problem size best
plt.plot(iterations, mimic_curve[:,0], label = "MIMIC", color = "pink")


plt.title('Max K Color')

plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.show()
