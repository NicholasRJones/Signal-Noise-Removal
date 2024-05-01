"""""""""
Project 3 Function File:
This file contains the function for computing our objective value and gradient to optimize.
This function is the sum of squares of a data fitting problem.
The parameters of this function are:
    - Sum of squares scalar (para 1)
    - Sum of squares power (para 2)
    - Beta norm adjustment (para 3)
    - Left endpoint (para 4)
    - Right endpoint (para 5)
Data is stored using the para class as para.data.
The p input determines which function value you'd like to calculate.
    0 - function value
    1 - gradient
    2 - both
"""""""""


import numpy as np


def noise(x, para, p):
    y = x - para.data
    s = np.diff(x)
    dx = np.insert(x, 0, para.parameter[3])
    dx = np.insert(dx, len(dx), para.parameter[4])
    ds = np.diff(dx)
    if p == 0 or p > 1:
        f = 1 / 2 * (y ** 2).sum() + para.parameter[0] / para.parameter[1] * ((ds ** 2 + para.parameter[2] ** 2) ** (para.parameter[1] / 2)).sum()
        if p == 0:
            return f
    if p > 0:
        s = np.insert(s, len(s), ds[len(ds) - 1])
        ds = np.delete(ds, len(ds) - 1)
        g = y - para.parameter[0] * (s * ((s ** 2 + para.parameter[2] ** 2) ** (para.parameter[1] / 2 - 1)) - ds * (
                    (ds ** 2 + para.parameter[2] ** 2) ** (para.parameter[1] / 2 - 1)))
        if p > 1:
            return f, g
        return g


def data(z, n, p):
    f = []
    g = []
    for k in range(n):
        f.append((1 / (k + 1)) ** z)
        g.append(sum(f))
    f = np.array(f)
    g = np.array(g) - p
    f = f * g
    return abs(f)
