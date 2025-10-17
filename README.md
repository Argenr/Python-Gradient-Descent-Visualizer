# Visualizador 3D del Descenso de Gradiente en Python

Este proyecto muestra de forma tridimensional cómo el algoritmo de descenso de gradiente se desplaza sobre la superficie de la función de coste (Error Cuadrático Medio - MSE) hasta encontrar el mínimo global.
El objetivo es ofrecer una visualización intuitiva del proceso de optimización en un modelo de regresión lineal simple.
El script genera datos de ejemplo que siguen una relación lineal con ruido y aplica el descenso de gradiente para ajustar los parámetros del modelo (m y b). 

# Características

- Representación 3D de la **superficie de coste** en función de m y b.
- **Trayectoria del descenso de gradiente** mostrada como una línea roja que desciende sobre la superficie.
- Cálculo manual del **Error Cuadrático Medio (MSE)** en cada iteración.
- Parámetros iniciales intencionalmente alejados del óptimo para visualizar mejor el proceso de convergencia.
- Visualización interactiva mediante matplotlib en 3D.


# Conceptos

El modelo busca ajustar una recta de la forma:
```math
\begin{aligned}
y &= mX + b \\
\end{aligned}
```
La función de coste utilizada es el Error Cuadrático Medio (MSE):

```math
\begin{aligned}
J(m, b) &= \frac{1}{N} \sum_{i=1}^{N} (mX_i + b - y_i)^2 \\
\end{aligned}
```

El descenso de gradiente actualiza los parámetros siguiendo:

```math
\begin{aligned}
m &:= m - \eta \frac{2}{N} \sum_{i=1}^{N} (mX_i + b - y_i) X_i \\
b &:= b - \eta \frac{2}{N} \sum_{i=1}^{N} (mX_i + b - y_i)
\end{aligned}
```

Donde:
- η es la tasa de aprendizaje (learning rate)
- N es el número de muestras


# Requisitos
```
pip install matplotlib
```
```
pip install numpy
```

# Ejecución
```
python gradient_descent_surface.py
```
Se abrirá una ventana mostrando una superficie 3D con el recorrido del descenso de gradiente.

# Resultado
![Descenso de Gradiente](Gradient-Descent.png)

- Superficie de coste: representa el error MSE para cada combinación de m y b.
- Línea roja: trayectoria del descenso de gradiente.
- Punto negro: posición inicial de los parámetros.

Al final del recorrido, el camino converge hacia la zona más baja de la superficie.

