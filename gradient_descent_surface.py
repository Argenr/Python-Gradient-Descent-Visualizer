import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generación de los datos
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Función de Coste
def cost_function(m, b, X, y):
    """Calcula el Error Cuadrático Medio."""
    N = len(y)
    predictions = m * X + b
    cost = np.sum((predictions - y) ** 2) / N
    return cost

# Descenso de gradiente
m = -5.0  # Empezar desde un punto malo
b = 10.0
learning_rate = 0.1
n_iterations = 100
N = len(X)

# Almacenar el historial de parámetros y coste
m_history = []
b_history = []
cost_history = []

for i in range(n_iterations):
    # Guardar el estado actual
    m_history.append(m)
    b_history.append(b)
    cost_history.append(cost_function(m, b, X, y))

    # Calcular predicciones y gradientes
    predictions = m * X + b
    error = predictions - y
    gradient_m = (2/N) * np.sum(error * X)
    gradient_b = (2/N) * np.sum(error)

    # Actualizar parámetros
    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b

m_vals = np.linspace(-6, 10, 100)
b_vals = np.linspace(-2, 12, 100)
m_grid, b_grid = np.meshgrid(m_vals, b_vals)

# Calcular el coste Z para cada punto (m, b) en la malla
Z = np.zeros(m_grid.shape)
for i in range(m_grid.shape[0]):
    for j in range(m_grid.shape[1]):
        Z[i, j] = cost_function(m_grid[i, j], b_grid[i, j], X, y)

# Graficar
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(m_grid, b_grid, Z, cmap='viridis', alpha=0.6, edgecolor='none')

ax.plot(m_history, b_history, cost_history, color='r', marker='o', markersize=5, label='Camino del Descenso')

ax.scatter(m_history[0], b_history[0], cost_history[0], color='black', s=100, label='Inicio')

ax.set_title('Descenso de Gradiente sobre la Superficie de Coste', fontsize=16)
ax.set_xlabel('Pendiente (m)', fontsize=12)
ax.set_ylabel('Ordenada (b)', fontsize=12)
ax.set_zlabel('Costo (MSE)', fontsize=12)
ax.view_init(elev=30., azim=120) 
ax.legend()

plt.show()