import numpy as np
from gilp.simplex import LP
from gilp.visualize import simplex_visual

A = np.array([[3,1],
              [1,2]])
b = np.array([[9],
              [8],])
c = np.array([[40],
              [50]])

lp = LP(A,b,c)

simplex_visual(lp).show()