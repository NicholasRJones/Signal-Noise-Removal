from Optimization.Functions import signalnoisefunction as snf
from Optimization.Algorithm import classy, optisolve as op
import numpy as np
import matplotlib.pyplot as plt

# Project 3:
# Data for project
n = 500
z = 1 / 4 + 24000000j
# z = 1 / 4 + 9876j
d = snf.data(z, n, 0)
beta = (abs(d).sum() / len(d)) / 100
beta = max(beta, 10 ** (-4))
alpha = 2 ** 5

# Parameter class for project
para = classy.para(0.0001, 0.19, d, [alpha, 2, beta, d[0], d[len(d) - 1]], 0, 0, 0)
# Function class to optimize
pr = classy.funct(snf.noise, 'LBFGS', 'strongwolfe', d, para, 1)

a = op.optimize(pr)
x = np.arange(1, n + 1)
plt.plot(x, d, color = 'red')
plt.plot(x, a.input, color = 'black')
plt.show()


"""""""""
# Reiterative beta method Project 3
for k in range(2):
    para = classy.para(0.0001, 0.19, d, [alpha, 2, beta, d[0], d[len(d) - 1]])
    # Function class to optimize
    pr = classy.funct(snf.noise, 'SR1', 'TRdog', d, para, 1)
    a = optimize(pr)
    alpha = 1 / 2 * ((pr.input - d) ** 2).sum()

x = np.arange(1, n + 1)
plt.plot(x, d, color = 'red')
plt.plot(x, a.input, color = 'black')
plt.show()
"""""""""
