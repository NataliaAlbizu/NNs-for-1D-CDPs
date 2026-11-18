import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sympy import coth
import matplotlib.pyplot as plt
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.auxil import equid_nodes, analytical, inf_error


methods = {
    "cds": "Central difference",
    "upw": "Upwind",
    "ias": "Il’in-Allen-Southwell"}

parser = argparse.ArgumentParser(description="Run FDM")

parser.add_argument("--method", choices=methods.keys(), default="cds", help="Finite difference method")
parser.add_argument("--epsilon", type=float, default=0.1, help="Diffusion coefficient")
parser.add_argument("--mesh_width", type=float, default=0.125, help="Mesh width")

args = parser.parse_args()

method = args.method
eps = args.epsilon
mesh_width = args.mesh_width

if eps <= 0:
    parser.error("eps must be positive")

if mesh_width <= 0 or mesh_width > 1:
    parser.error("mesh_width must be in the interval (0, 1]")


## function for the right-hand side
def f(x):
    return 1


## boundary conditions
def g(x):
    return 0


## vector on the right-hand side
def get_right_hand_side(f, i_nodes):
    rhs = []
    for node in i_nodes:
        rhs.append(f(node))
    return np.array(rhs)


## matrix of the reduced system
def system_matrix(n, h, epsilon, method):

    if method=="upw":
        matrix = diags([-epsilon/(h**2) - 1/h, (2*epsilon)/(h**2) + 1/h, -epsilon/(h**2)], [-1, 0, 1], shape=(n, n))

    elif method=="cds":
        matrix = diags([(-epsilon/(h**2)) - 1/(2*h), (2*epsilon)/(h**2), (-epsilon/(h**2)) + 1/(2*h)], [-1, 0, 1], shape=(n, n))

    elif method=="ias":
        sub = float((-1/(2*h)) * (1 + coth(h/(2*epsilon))))
        dia = float((1/h) * coth(h/(2*epsilon)))
        sup = float((1/(2*h)) * (1 - coth(h/(2*epsilon))))
        matrix = diags([sub, dia, sup], [-1, 0, 1], shape=(n, n))

    return matrix



def fdm_with_equid_nodes(f, g, x_start, x_end, h, epsilon, method):
    nodes, boundary_nodes, inner_nodes = equid_nodes(x_start, x_end, h)
    n = int(len(inner_nodes))
    F = get_right_hand_side(f, inner_nodes)
    A = system_matrix(n, h, epsilon, method)
    if method=="upw":
        F[0] = F[0] - (((-epsilon/(h**2)) - 1/h)*g(boundary_nodes[0]))
        F[-1] = F[-1] - ((-epsilon/(h**2))*g(boundary_nodes[1]))
    elif method=="cds":
        F[0] = F[0] - (((-epsilon/(h**2)) - 1/(2*h))*g(boundary_nodes[0]))
        F[-1] = F[-1] - (((-epsilon/(h**2)) + 1/(2*h))*g(boundary_nodes[1]))
    elif method=="ias":
        F[0] = F[0] - (((-1/(2*h))*(1 + (coth(h/(2*epsilon)))))*g(boundary_nodes[0]))
        F[-1] = F[-1] - (((1/(2*h))*(1 - (coth(h/(2*epsilon)))))*g(boundary_nodes[1]))
    u = spsolve(A, F)
    full_u = []
    full_u.append(g(boundary_nodes[0]))
    for idx, _ in enumerate(inner_nodes):
        full_u.append(u[idx])
    full_u.append(g(boundary_nodes[1]))
    full_u = [float(x) for x in full_u]
    return full_u, inner_nodes, nodes


X = np.linspace(0, 1, 1024)
Y = analytical(X, eps)

approx_sol, inner_nodes, nodes = fdm_with_equid_nodes(f, g, 0, 1, mesh_width, eps, method)

u_h = approx_sol[1:-1]
u = []
for node in inner_nodes:
    u.append(analytical(node, eps))

error = inf_error(np.array(u), np.array(u_h))


run_name = f"eps_{eps}_mesh_width_{mesh_width}_{method}"
run_dir = Path("outputs") / "FDM" / run_name
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
    "inf_error": float(error)}

with open(run_dir / "error.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Method: {method}")
print(f"eps: {eps}")
print(f"Mesh width: {mesh_width}")
print(f"Infinity error: {error:.6e}")
print(f"Results saved in: {run_dir}")