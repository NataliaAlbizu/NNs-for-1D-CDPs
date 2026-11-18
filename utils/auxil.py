from pathlib import Path
import numpy as np
import copy
from scipy.integrate import quad
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch



def analytical(x, epsilon):
    val = x - (np.exp((x-1)/epsilon) - np.exp(-1/epsilon))/(1 - np.exp(-1/epsilon))
    return val


def first_der(x, epsilon):
    val = 1 - (np.exp((x-1)/epsilon))/(epsilon*(1 - np.exp(-1/epsilon)))
    return val


def equid_nodes(start, end, h):
    inner_nodes = []
    nr_of_coords = ((end - start) / h) + 1
    for i in range(int(nr_of_coords)-2):
      inner_nodes.append(start + (i+1)*h)
    nodes = copy.deepcopy(inner_nodes)
    nodes.insert(0, start)
    nodes.append(end)
    boundary_nodes = [start, end]
    return nodes, boundary_nodes, inner_nodes


def left_hat(x, end_node, h):
    return (end_node - x)  / h


def right_hat(x, start_node, h):
    return (x - start_node)  / h


def L2_H1_errors(u, u_h, start, end, h, epsilon, der):
    nodes = equid_nodes(start, end, h)[0]
    ln = len(nodes)
    L2 = 0
    H1 = 0
    left_der = -1/h
    right_der = 1/h
    L2 += quad(lambda x: (u(x, epsilon) - u_h[1]*right_hat(x, nodes[0], h))**2, nodes[0], nodes[1])[0]
    H1 += quad(lambda x: (der(x, epsilon) - u_h[1]*right_der)**2, nodes[0], nodes[1])[0]
    for k in range(1,ln-2):
        local_error = quad(lambda x: (u(x, epsilon) - u_h[k]*left_hat(x, nodes[k+1], h) - u_h[k+1]*right_hat(x, nodes[k], h))**2, nodes[k], nodes[k+1])[0]
        L2 += local_error
        local_error_h1 = quad(lambda x: (der(x, epsilon) - u_h[k]*left_der - u_h[k+1]*right_der)**2, nodes[k], nodes[k+1])[0]
        H1 += local_error_h1
    L2 += quad(lambda x: (u(x, epsilon) - u_h[ln-2]*left_hat(x, nodes[ln-1], h))**2, nodes[ln-2], nodes[ln-1])[0]
    H1 += quad(lambda x: (der(x, epsilon) - u_h[ln-2]*left_der)**2, nodes[ln-2], nodes[ln-1])[0]
    return np.sqrt(L2), np.sqrt(H1)


def inf_error(u, u_h):
    diff = u - u_h
    return LA.norm(diff, np.inf)


def eucl_error(u, u_h):
    diff = u - u_h
    return LA.norm(diff, 2) 


def rel_l2_error(u, u_h):
    diff = u - u_h
    return LA.norm(diff, 2) / LA.norm(u, 2)


def plot_and_save(y_nn, y_theory, plotting_grid, n, epsilon, save_dir, lines=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_nn = np.asarray(y_nn).reshape(-1)
    y_theory = np.asarray(y_theory).reshape(-1)
    plotting_grid = np.asarray(plotting_grid).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(plotting_grid, y_nn,
        color="blue", linewidth=2, label="Neural network")
    ax.plot(plotting_grid, y_theory,
        color="red", linewidth=2, label="Analytical solution")

    if lines:
        for x in np.linspace(0, 1, n + 1):
            ax.axvline(x, color="black", linewidth=0.3, alpha=0.5)

    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc="upper left")
    ax.set_title(rf"Model vs. Analytical Solution, $\varepsilon$ = {epsilon}",
        fontsize=18)

    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_dir / "solution.png",
        dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost2(savecost, save_dir, adam_steps=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(savecost, torch.Tensor):
        cost = savecost.detach().cpu().numpy()
    else:
        cost = np.asarray(savecost, dtype=float)

    cost = cost.reshape(-1)

    if cost.size == 0:
        print("No loss history available; skipping loss plot.")
        return

    # Avoid issues with log scaling if a loss is exactly zero.
    cost = np.maximum(cost, np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cost, label="Training loss")
    ax.set_yscale("log")

    if adam_steps is not None and 0 < adam_steps < len(cost):
        ax.axvline(
            adam_steps - 1,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="End of first optimizer")
        ax.legend()

    ax.set_xlabel("Optimizer step / objective evaluation")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (log scale)")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        save_dir / "loss.png",
        dpi=300,
        bbox_inches="tight")
    plt.close(fig)