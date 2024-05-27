import numpy as np
import matplotlib.pyplot as plt

# Funkcje celu i ograniczenia
def profit_function(x, y):
    return 40*x + 50*y

def constraint1(x):
    return 9 - 3*x

def constraint2(x):
    return (8 - x) / 2

# Tworzenie siatki punktów dla wykresu
x = np.linspace(0, 5, 400)
y1 = constraint1(x)
y2 = constraint2(x)

# Rysowanie ograniczeń
plt.plot(x, y1, label='3x + y ≤ 9')
plt.plot(x, y2, label='x + 2y ≤ 8')

# Rysowanie obszaru dostępnego
y3 = np.minimum(y1, y2)
plt.fill_between(x, 0, y3, where=(y3>=0), color='grey', alpha=0.3, label='Feasible Region')

# Rysowanie punktu optymalnego
x_opt = 2
y_opt = 3
plt.plot(x_opt, y_opt, 'ro', label='Optimal Point (2, 3)')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Linear Programming Graphical Solution')
plt.legend()
plt.grid(True)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()
