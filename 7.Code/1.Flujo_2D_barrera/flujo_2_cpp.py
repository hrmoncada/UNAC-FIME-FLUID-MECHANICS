'''
Ejecutar
$ python flujo_2_cpp.py
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_flow_from_csv(csv_file, obstacle, every=5):
    """
    Read CSV output from C++ solver and plot streamlines, velocity vectors,
    and the square obstacle.
    """
    # Load data
    data = pd.read_csv(csv_file)
    # Extract unique grid coordinates
    x = np.sort(data['x'].unique())
    y = np.sort(data['y'].unique())
    nx, ny = len(x), len(y)

    # Reshape fields to 2D arrays
    psi = data['psi'].values.reshape(nx, ny)
    u = data['u'].values.reshape(nx, ny)
    v = data['v'].values.reshape(nx, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Create plot
    plt.figure(figsize=(10, 5))
    # Streamlines (contours of psi)
    plt.contour(X, Y, psi, levels=20, colors='blue', alpha=0.6)
    # Velocity vectors (downsampled)
    step = every
    plt.quiver(X[::step, ::step], Y[::step, ::step],
               u[::step, ::step], v[::step, ::step],
               alpha=0.5, color='red')
    # Obstacle
    x0, y0, size = obstacle
    rect = Rectangle((x0, y0), size, size,
                     facecolor='gray', edgecolor='black', alpha=0.7)
    plt.gca().add_patch(rect)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Potential flow around square obstacle (C++ solver)')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Obstacle definition must match the one used in C++
    obstacle = (0.8, 0.35, 0.3)
    plot_flow_from_csv("flow_data.csv", obstacle, every=4)
