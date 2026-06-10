'''
Ejecutar
$ python flujo_1.py
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def solve_streamfunction(Lx, Ly, nx, ny, v, obstacle, tol=1e-5, max_iter=20000, omega=1.9):
    """
    Solve for streamfunction psi using Laplace equation (potential flow)
    with Dirichlet/Neumann BCs and a fixed square obstacle.

    Parameters:
    Lx, Ly        : domain dimensions [m]
    nx, ny        : number of grid points in x and y
    v             : inflow speed at left boundary [m/s]
    obstacle      : tuple (x0, y0, size) position of lower-left corner and side length [m]
    tol           : convergence tolerance
    max_iter      : maximum number of SOR iterations
    omega         : over-relaxation factor (1 < omega < 2)

    Returns:
    psi           : streamfunction field (2D array)
    x, y          : meshgrid coordinates
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Grid coordinates
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initialize psi array
    psi = np.zeros((nx, ny))

    # Dirichlet boundary conditions
    # Left boundary: uniform inflow, psi = v * y
    for j in range(ny):
        psi[0, j] = v * y[j]

    # Top boundary: psi = v * Ly (constant)
    psi[:, -1] = v * Ly
    # Bottom boundary: psi = 0
    psi[:, 0] = 0.0

    # Right boundary: Neumann (∂ψ/∂x = 0) initially, will be updated in iterations
    # No need to set here, we impose in the solver loop

    # Square obstacle: set interior to a constant psi value.
    # We choose the value corresponding to the y-coordinate of the obstacle's centre.
    x0, y0, size = obstacle
    x1, y1 = x0 + size, y0 + size
    i0 = max(1, int(x0 / dx) + 1)
    i1 = min(nx-2, int(x1 / dx))
    j0 = max(1, int(y0 / dy) + 1)
    j1 = min(ny-2, int(y1 / dy))
    # constant psi on obstacle = psi at y = y0 + size/2 (centre height) for uniform inflow
    psi_obstacle = v * (y0 + size/2.0)
    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            psi[i, j] = psi_obstacle

    # SOR iteration
    for it in range(max_iter):
        psi_old = psi.copy()
        # Update interior points (excluding boundaries and obstacle)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Skip obstacle cells (they remain fixed)
                if (i0 <= i <= i1) and (j0 <= j <= j1):
                    continue
                # Laplace equation discretization: d2psi/dx2 + d2psi/dy2 = 0
                psi_ij = ( (psi[i+1, j] + psi[i-1, j]) * dy**2 +
                           (psi[i, j+1] + psi[i, j-1]) * dx**2 ) / (2.0 * (dx**2 + dy**2))
                psi[i, j] = omega * psi_ij + (1 - omega) * psi[i, j]

        # Neumann condition at right boundary (∂ψ/∂x = 0)
        for j in range(1, ny-1):
            psi[nx-1, j] = psi[nx-2, j]

        # Re-apply obstacle condition (ensures values remain fixed)
        for i in range(i0, i1+1):
            for j in range(j0, j1+1):
                psi[i, j] = psi_obstacle

        # Check convergence
        diff = np.max(np.abs(psi - psi_old))
        if diff < tol:
            print(f"Converged after {it+1} iterations, max change = {diff:.2e}")
            break
    else:
        print(f"Maximum iterations ({max_iter}) reached, final change = {diff:.2e}")

    return psi, X, Y

def compute_velocity(psi, dx, dy):
    """
    Compute velocity components u, v from streamfunction.
    u = ∂ψ/∂y,  v = -∂ψ/∂x
    """
    u = np.gradient(psi, dy, axis=1)          # ∂ψ/∂y
    v = -np.gradient(psi, dx, axis=0)         # -∂ψ/∂x
    return u, v

def plot_flow(X, Y, psi, u, v, obstacle, v_inflow, every=5):
    """
    Plot streamlines, velocity vectors, and obstacle.
    """
    plt.figure(figsize=(10, 5))
    # Streamlines
    plt.contour(X, Y, psi, levels=20, colors='blue', alpha=0.6)
    # Velocity vectors (downsampled)
    step = every
    plt.quiver(X[::step, ::step], Y[::step, ::step],
               u[::step, ::step], v[::step, ::step],
               alpha=0.5, color='red')
    # Square obstacle
    x0, y0, size = obstacle
    rect = Rectangle((x0, y0), size, size,
                     facecolor='gray', edgecolor='black', alpha=0.7)
    plt.gca().add_patch(rect)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Potential flow around square obstacle (inflow speed v = {v_inflow} m/s)')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # Domain and grid parameters
    Lx = 2.0       # length of tube [m]
    Ly = 1.0       # height of tube [m]
    nx = 101       # number of grid points in x
    ny = 51        # number of grid points in y
    v_inflow = 1.0 # inflow velocity [m/s]

    # Square obstacle: (x0, y0, size)
    obstacle = (0.8, 0.35, 0.3)   # position lower-left corner, side length 0.3 m

    # Solve for streamfunction
    psi, X, Y = solve_streamfunction(Lx, Ly, nx, ny, v_inflow, obstacle,
                                     tol=1e-6, max_iter=30000, omega=1.9)

    # Compute velocities
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    u, v = compute_velocity(psi, dx, dy)

    # Plot results
    plot_flow(X, Y, psi, u, v, obstacle, v_inflow, every=4)

if __name__ == "__main__":
    main()
