import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.auxil import equid_nodes, analytical, first_der, left_hat, right_hat, L2_H1_errors, inf_error, rel_l2_error


methods = {"std": "Standard", "supg": "SUPG"}


parser = argparse.ArgumentParser(description="Run FEM")

parser.add_argument("--method", choices=methods.keys(), default="std", help="Finite element method")
parser.add_argument("--epsilon", type=float, default=0.1, help="Diffusion coefficient")
parser.add_argument("--mesh_width", type=float, default=0.125, help="Mesh width")
parser.add_argument("--n_ref", type=int, default=20000)

args = parser.parse_args()

method = args.method
eps = args.epsilon
mesh_width = args.mesh_width
n_ref = args.n_ref

if eps <= 0:
    parser.error("eps must be positive")

if mesh_width <= 0 or mesh_width > 1:
    parser.error("mesh_width must be in the interval (0, 1]")

if n_ref <= 0:
    parser.error("n_ref must be positive")

h_ref = 1 / n_ref


## function for the right-hand side
def f(x):
    return 1

## boundary conditions
def g(x):
    return 0


def p1_fem_for_1D_steady_conv_diff(f, g, start, end, h, epsilon, method, delta=None):
    nodes, _, inner_nodes = equid_nodes(start, end, h)
    n = int(len(nodes))
    A = np.zeros((n, n))
    RHS = np.zeros(n)
    left_der = -1/h
    right_der = 1/h
    derivatives = [left_der, right_der]
    if method=="std":
        factor = epsilon
    elif method=="supg":
        factor = epsilon+delta
    for k in range(n-1):
        indices = [k, k+1]
        for nr, i in enumerate(indices):
            for no, j in enumerate(indices):
                if nr==0:
                    val = factor*quad(lambda x: derivatives[nr]*derivatives[no], nodes[k], nodes[k+1])[0] \
                    + quad(lambda x: derivatives[no]*left_hat(x, nodes[k+1], h), nodes[k], nodes[k+1])[0]
                    A[i][j] += val
                elif nr==1:
                    val = factor*quad(lambda x: derivatives[nr]*derivatives[no], nodes[k], nodes[k+1])[0] \
                    + quad(lambda x: derivatives[no]*right_hat(x, nodes[k], h), nodes[k], nodes[k+1])[0]
                    A[i][j] += val
    for idx in range(1, len(nodes)-1, 1):
        first_int = quad(lambda x: right_hat(x, nodes[idx-1], h), nodes[idx-1], nodes[idx])[0]
        second_int = quad(lambda x: left_hat(x, nodes[idx+1], h), nodes[idx], nodes[idx+1])[0]
        RHS[idx] += first_int + second_int
    RHS[0] += g(nodes[0])
    RHS[n-1] += g(nodes[n-1])
    B = np.zeros_like(A)
    B[1:n-1, :] = A[1:n-1, :]
    B[1,0] = 0
    B[-2,-1] = 0
    B[0, 0] = 1
    B[n-1, n-1] = 1
    B = csr_matrix(B)
    u = spsolve(B, RHS)
    return u, inner_nodes, nodes


Delta = mesh_width/2
approx_sol, inner_nodes, nodes = p1_fem_for_1D_steady_conv_diff(f, g, 0, 1, mesh_width, eps, method, Delta)

L2_error, H1_error = L2_H1_errors(analytical, approx_sol, 0, 1, mesh_width, eps, first_der)

u_h = approx_sol[1:-1]
u = []
for node in inner_nodes:
    u.append(analytical(node, eps))

infinity_error = inf_error(np.array(u), np.array(u_h))

X = np.linspace(0, 1, 1024)
Y = analytical(X, eps)

grid_ref = np.linspace(0, 1, n_ref + 1)
interpol_u_h = interp1d(nodes, approx_sol, kind="linear")(grid_ref)
u_ref = analytical(grid_ref, eps)
ref_error = rel_l2_error(u_ref, interpol_u_h)
interpol_L2_error, interpol_H1_error = L2_H1_errors(analytical, interpol_u_h, 0, 1, h_ref, eps, first_der)

eps_folder = f"{eps:.0e}".replace("e-0", "e-")

run_name = f"mesh_width_{mesh_width}"
run_dir = Path("outputs") / "FEM" / eps_folder / method / run_name

run_dir.mkdir(parents=True, exist_ok=True)


plt.plot(X, Y, label="Analytical Solution")
plt.plot(nodes, approx_sol, label=f"{methods[method]}, mesh width {mesh_width}")

plt.title(rf"Model problem with $\varepsilon$ = {eps}")
plt.legend()
plt.savefig(run_dir / "solution.png")
plt.close()


metrics = {
    "method": method,
    "epsilon": eps,
    "mesh_width": mesh_width,
    "L2 error": float(L2_error),
    "H1 error": float(H1_error),
    "inf_error": float(infinity_error),
    "ref rel l2 error": float(ref_error),
    "ref L2 error": float(interpol_L2_error),
    "ref H1 error": float(interpol_H1_error)}

with open(run_dir / "error.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Method: {method}")
print(f"eps: {eps}")
print(f"Mesh width: {mesh_width}")
print(f"L2 error: {L2_error:.6e}")
print(f"Results saved in: {run_dir}")


