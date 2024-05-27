import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Funkcja celu i gradient
def func(x, y):
    return x**2 + 4*(y-4)**2

def grad(x, y):
    return np.array([2*x, 8*(y-4)])

# Funkcja projekcji na obszar dopuszczalny
def project(x, y):
    # Najpierw sprawdzamy, czy (x, y) leży wewnątrz obszaru ograniczeń
    if x + y > 7:
        y = 7 - x
    if -x + 2 * y > 4:
        y = (4 + x) / 2
    
    # Sprawdzamy, czy przesunięcie w stronę (0,4) jest możliwe
    # Dążymy do (0, 4) w ramach obszaru ograniczeń
    target_y = 4
    if y > target_y:
        if -x + 2 * target_y <= 4 and x + target_y <= 7:
            y = target_y

    x = max(x, 0)
    y = max(y, 0)

    return x, y



def gradient_descent_with_momentum(x0, y0, lr, n_iter, gamma=0.9):
    x, y = x0, y0
    vx, vy = 0, 0  # Inicjalizacja prędkości dla x i y
    path = []  # Ścieżka do wizualizacji
    for i in range(n_iter):
        g = grad(x, y)
        vx = gamma * vx + lr * g[0]
        vy = gamma * vy + lr * g[1]
        x -= vx
        y -= vy
        x, y = project(x, y)
        path.append((x, y))
        if i % 100 == 0:  # Informacje co 100 iteracji
            print(f"Iteracja {i}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {func(x, y):.4f}")
    return x, y, path

# Algorytm spadku gradientu z projekcją
def gradient_descent(x0, y0, lr, n_iter):
    x, y = x0, y0
    prev_cost = func(x, y)
    path = [(x, y)]
    
    for i in range(n_iter):
        g = grad(x, y)
        new_x = x - lr * g[0]
        new_y = y - lr * g[1]
        new_x, new_y = project(new_x, new_y)
        new_cost = func(new_x, new_y)

        # Jeśli nowy koszt jest lepszy, aktualizujemy pozycję
        if new_cost < prev_cost:
            x, y = new_x, new_y
            prev_cost = new_cost
        else:
            lr *= 0.9  # Zmniejszamy learning rate
        
        path.append((x, y))

    return x, y, path

def distance_to_center(x, y):
    return np.sqrt((x - 0)**2 + (y - 4)**2)


# Parametry początkowe
x0, y0 = 5, 0.1  # Punkt startowy
learning_rate = 0.01  # Współczynnik uczenia
n_iterations = 100  # Liczba iteracji

# Uruchom algorytm
opt_x, opt_y, path = gradient_descent(x0, y0, learning_rate, n_iterations)
path = np.array(path)

x1 = np.linspace(-10, 8, 400)
y1 = np.linspace(-10, 8, 400)

X, Y = np.meshgrid(x1, y1)
Z = X**2 + 4*(Y - 4)**2  # Funkcja celu

# Animacja
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, levels=[4,8,12,14], colors='blue', linestyles='solid')
ax.clabel(contour, inline=True, fontsize=8)
ax.set_xlim([-4, 7])
ax.set_ylim([0, 7])
plt.plot(7 - np.linspace(0, 7, 100), np.linspace(0, 7, 100), 'r--', label='x + y <= 7')
plt.plot(np.linspace(0, 8, 100), (4 + np.linspace(0, 8, 100)) / 2, 'b--', label='-x + 2y <= 4')
plt.fill_between(np.linspace(0, 7, 100), 0, np.minimum(7 - np.linspace(0, 7, 100), (4 + np.linspace(0, 7, 100)) / 2), color='gray', alpha=0.5)
plt.title('Gradient Descent with Projections Animation')
line, = ax.plot([], [], 'ro-',markersize=5, label='Optimization Path')
plt.legend()
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)  # Ledwo widoczna siatka
ax.plot(0, 4, 'go')
distance_text = ax.text(0.5, 1.05, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    distance_text.set_text('')
    return line, distance_text

def update(frame):
    x, y = path[frame]
    line.set_data(path[:frame, 0], path[:frame, 1])
    r = distance_to_center(x, y)
    distance_text.set_text(f'r = {r:.2f}')
    return line, distance_text

ani = FuncAnimation(fig, update, frames=len(path), init_func=init, interval=10, repeat=True)
print(f"Optymalne wartości: x = {opt_x}, y = {opt_y}")
plt.show()
