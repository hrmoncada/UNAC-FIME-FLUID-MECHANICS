/*
Compilar:
$ g++ -O2 -o flow_solver flow_solver.cpp
Ejecutar:
$ ./flow_solver

Se crea un archivo de datos de salida
flow_data.csv

Graficar : lee el archivo flow_data.csv
$ python flujo_2_cpp.py
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Solve streamfunction psi for potential flow using SOR
void solve_streamfunction(
    double Lx, double Ly, int nx, int ny, double v_inflow,
    double obst_x0, double obst_y0, double obst_size,
    double tol, int max_iter, double omega,
    vector<vector<double>>& psi)
{
    double dx = Lx / (nx - 1);
    double dy = Ly / (ny - 1);

    // Initialize psi array with zeros
    psi.assign(nx, vector<double>(ny, 0.0));

    // Left boundary: psi = v_inflow * y
    for (int j = 0; j < ny; ++j) {
        double y = j * dy;
        psi[0][j] = v_inflow * y;
    }
    // Top boundary: psi = v_inflow * Ly
    for (int i = 0; i < nx; ++i)
        psi[i][ny-1] = v_inflow * Ly;
    // Bottom boundary: psi = 0
    for (int i = 0; i < nx; ++i)
        psi[i][0] = 0.0;

    // Square obstacle indices
    int i0 = max(1, (int)(obst_x0 / dx) + 1);
    int i1 = min(nx-2, (int)((obst_x0 + obst_size) / dx));
    int j0 = max(1, (int)(obst_y0 / dy) + 1);
    int j1 = min(ny-2, (int)((obst_y0 + obst_size) / dy));
    double psi_obstacle = v_inflow * (obst_y0 + obst_size/2.0);

    // Set obstacle cells to constant psi
    for (int i = i0; i <= i1; ++i)
        for (int j = j0; j <= j1; ++j)
            psi[i][j] = psi_obstacle;

    // SOR iteration
    double diff;
    for (int iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;
        vector<vector<double>> psi_old = psi;

        // Update interior points (excluding boundaries)
        for (int i = 1; i < nx-1; ++i) {
            for (int j = 1; j < ny-1; ++j) {
                // Skip obstacle cells
                if (i >= i0 && i <= i1 && j >= j0 && j <= j1)
                    continue;

                double psi_ij = ((psi[i+1][j] + psi[i-1][j]) * dy*dy +
                                 (psi[i][j+1] + psi[i][j-1]) * dx*dx) /
                                (2.0 * (dx*dx + dy*dy));
                psi[i][j] = omega * psi_ij + (1.0 - omega) * psi[i][j];
            }
        }

        // Neumann condition at right boundary: dpsi/dx = 0
        for (int j = 1; j < ny-1; ++j)
            psi[nx-1][j] = psi[nx-2][j];

        // Re‑apply obstacle condition
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
                psi[i][j] = psi_obstacle;

        // Compute maximum change
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                diff = max(diff, fabs(psi[i][j] - psi_old[i][j]));

        if (diff < tol) {
            cout << "Converged after " << iter+1 << " iterations, max change = " << diff << endl;
            break;
        }
        if (iter == max_iter-1)
            cout << "Max iterations reached, final change = " << diff << endl;
    }
}

// Compute velocity components u = dpsi/dy , v = -dpsi/dx
void compute_velocity(const vector<vector<double>>& psi, double dx, double dy,
                      vector<vector<double>>& u, vector<vector<double>>& v)
{
    int nx = psi.size();
    int ny = psi[0].size();
    u.assign(nx, vector<double>(ny, 0.0));
    v.assign(nx, vector<double>(ny, 0.0));

    for (int i = 1; i < nx-1; ++i) {
        for (int j = 1; j < ny-1; ++j) {
            u[i][j] = (psi[i][j+1] - psi[i][j-1]) / (2.0 * dy);
            v[i][j] = -(psi[i+1][j] - psi[i-1][j]) / (2.0 * dx);
        }
    }
    // Simple boundary handling: copy nearest interior values
    for (int j = 0; j < ny; ++j) u[0][j] = u[1][j];
    for (int i = 0; i < nx; ++i) u[i][0] = u[i][1];
    for (int i = 0; i < nx; ++i) u[i][ny-1] = u[i][ny-2];
    for (int j = 0; j < ny; ++j) v[0][j] = v[1][j];
    for (int i = 0; i < nx; ++i) v[i][0] = v[i][1];
    for (int i = 0; i < nx; ++i) v[i][ny-1] = v[i][ny-2];
}

// Write results to CSV file
void write_to_csv(const string& filename,
                  const vector<double>& x, const vector<double>& y,
                  const vector<vector<double>>& psi,
                  const vector<vector<double>>& u,
                  const vector<vector<double>>& v)
{
    ofstream file(filename);
    file << fixed << setprecision(8);
    file << "x,y,psi,u,v\n";
    int nx = x.size();
    int ny = y.size();
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            file << x[i] << "," << y[j] << ","
                 << psi[i][j] << "," << u[i][j] << "," << v[i][j] << "\n";
        }
    }
    file.close();
    cout << "Data written to " << filename << endl;
}

int main() {
    // Domain and grid
    const double Lx = 2.0;
    const double Ly = 1.0;
    const int nx = 101;
    const int ny = 51;
    const double v_inflow = 1.0;

    // Square obstacle: lower-left corner (x0,y0) and side length
    const double obst_x0 = 0.8;
    const double obst_y0 = 0.35;
    const double obst_size = 0.3;

    // Solver parameters
    const double tol = 1e-6;
    const int max_iter = 30000;
    const double omega = 1.9;  // SOR over‑relaxation

    vector<vector<double>> psi, u, v;
    solve_streamfunction(Lx, Ly, nx, ny, v_inflow,
                         obst_x0, obst_y0, obst_size,
                         tol, max_iter, omega, psi);

    // Compute velocities
    double dx = Lx / (nx - 1);
    double dy = Ly / (ny - 1);
    compute_velocity(psi, dx, dy, u, v);

    // Create grid coordinates
    vector<double> x(nx), y(ny);
    for (int i = 0; i < nx; ++i) x[i] = i * dx;
    for (int j = 0; j < ny; ++j) y[j] = j * dy;

    // Save to CSV
    write_to_csv("flow_data.csv", x, y, psi, u, v);

    return 0;
}
