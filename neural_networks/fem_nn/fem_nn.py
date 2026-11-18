import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import yaml
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.auxil import analytical, plot_cost2, plot_and_save, rel_l2_error, L2_H1_errors, first_der


parser = argparse.ArgumentParser(description="Run FEM-NN",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--seed", type=int, default=None,
                    help="Random seed")
parser.add_argument("--dtype", type=str, choices=["float32", "float64"],
                    default=None, help="PyTorch data type")

parser.add_argument("--n", type=int, default=None,
                    help="Number of equidistant intervals")
parser.add_argument("--epsilon", type=float, default=None,
                    help="Diffusion coefficient")
parser.add_argument("--method", choices=["std", "supg"], default=None,
                    help="Standard or SUPG FEM")
parser.add_argument("--BC-appr", dest="BC_approach",
                    choices=["soft", "H", "I"], default=None,
                    help="Enforcing of BC")
parser.add_argument("--alpha", type=float, default=None,
                    help="Alpha parameter for I indicator function")
parser.add_argument("--beta", type=float, default=None,
                    help="Beta parameter for I indicator function")

parser.add_argument("--n-ref", dest="n_ref", type=int, default=None,
                    help="Number of equidistant intervals for evaluation")

parser.add_argument("--weight-init", dest="weight_init",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="Enable or disable Kaiming weight initialization")
parser.add_argument("--bias-init", dest="bias_init",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="Enable or disable random bias initialization")

parser.add_argument("--backprop", choices=["manual", "auto"], default=None,
                    help="Gradient computation method")

parser.add_argument("--optimizer", choices=["Adam", "GD"], default=None,
                    help="First-stage optimizer")
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate for Adam or GD")
parser.add_argument("--iterations", type=int, default=None,
                    help="Number of first-stage iterations")

parser.add_argument("--lbfgs-iterations", dest="lbfgs_iterations",
                    type=int, default=None,
                    help="Number of outer L-BFGS iterations")
parser.add_argument("--lbfgs-lr", dest="lbfgs_lr",
                    type=float, default=None,
                    help="L-BFGS learning rate")
parser.add_argument("--lbfgs-max-iter", dest="lbfgs_max_iter",
                    type=int, default=None,
                    help="Maximum internal iterations per L-BFGS step")
parser.add_argument("--lbfgs-history-size", dest="lbfgs_history_size",
                    type=int, default=None,
                    help="L-BFGS history size")
parser.add_argument("--lbfgs-tolerance-grad", dest="lbfgs_tolerance_grad",
                    type=float, default=None,
                    help="L-BFGS gradient tolerance")
parser.add_argument("--lbfgs-tolerance-change", dest="lbfgs_tolerance_change",
                    type=float, default=None,
                    help="L-BFGS function-change tolerance")

parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML configuration file")

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

overrides = {
    ("seed",): args.seed,
    ("dtype",): args.dtype,
    ("problem", "n"): args.n,
    ("problem", "epsilon"): args.epsilon,
    ("problem", "method"): args.method,
    ("problem", "BC_approach"): args.BC_approach,
    ("problem", "alpha"): args.alpha,
    ("problem", "beta"): args.beta,
    ("evaluation", "n_ref"): args.n_ref,
    ("model", "weight_init"): args.weight_init,
    ("model", "bias_init"): args.bias_init,
    ("loss", "backprop"): args.backprop,
    ("training", "optimizer"): args.optimizer,
    ("training", "learning_rate"): args.lr,
    ("training", "iterations"): args.iterations,
    ("lbfgs", "iterations"): args.lbfgs_iterations,
    ("lbfgs", "learning_rate"): args.lbfgs_lr,
    ("lbfgs", "max_iter"): args.lbfgs_max_iter,
    ("lbfgs", "history_size"): args.lbfgs_history_size,
    ("lbfgs", "tolerance_grad"): args.lbfgs_tolerance_grad,
    ("lbfgs", "tolerance_change"): args.lbfgs_tolerance_change}

for path, value in overrides.items():
    if value is not None:
        section = config
        for key in path[:-1]:
            section = section[key]
        section[path[-1]] = value


output_cfg = config["output"]
results_root = Path(output_cfg["results_dir"])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_name = (f"{timestamp}_N{config['problem']['n']}"
    f"_eps{config['problem']['epsilon']}"
    f"_{config['problem']['method']}"
    f"_{config['problem']['BC_approach']}")

run_dir = results_root / run_name
run_dir.mkdir(parents=True, exist_ok=False)

print(f"Saving results to: {run_dir}")

if output_cfg["save_config"]:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


torch.manual_seed(config["seed"])

dtype = getattr(torch, config["dtype"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = config["problem"]["epsilon"]

n = config["problem"]["n"]
h = 1 / n

method = config["problem"]["method"]
BC_appr = config["problem"]["BC_approach"]
if config["problem"]["BC_approach"] in ("soft", "H"):
    config["problem"]["alpha"] = None
    config["problem"]["beta"] = None
alpha = config["problem"]["alpha"]
beta = config["problem"]["beta"]

n_ref = config["evaluation"]["n_ref"]
h_ref = 1 / n_ref

weight_init = config["model"]["weight_init"]
bias_init = config["model"]["bias_init"]

backprop = config["loss"]["backprop"]

optimizer_1 = config["training"]["optimizer"]
lr = config["training"]["learning_rate"]
Niter_opt1 = config["training"]["iterations"]

lbfgs_cfg = config["lbfgs"]
Niter_lbfgs = lbfgs_cfg["iterations"]

CP = torch.linspace(0, 1, n+1, dtype=dtype, device=device).unsqueeze(1)

if weight_init==True:
    W1 = torch.empty(3 * (n - 1), 1, dtype=dtype, device=device, requires_grad=True)
    W2 = torch.empty(1, 3 * (n - 1), dtype=dtype, device=device, requires_grad=True)
    torch.nn.init.kaiming_normal_(W1, nonlinearity='relu')
    torch.nn.init.kaiming_normal_(W2, nonlinearity='relu');
else:
    W1 = torch.randn(3 * (n - 1), 1, dtype=dtype, device=device, requires_grad=True)
    W2 = torch.randn(1, 3 * (n - 1), dtype=dtype, device=device, requires_grad=True)

if bias_init==True:
    b1 = torch.randn(3 * (n - 1), dtype=dtype, device=device, requires_grad=True)
else:
    b1 = torch.zeros(3 * (n - 1), dtype=dtype, device=device, requires_grad=True)


savecost = torch.zeros(Niter_opt1 + Niter_lbfgs, dtype=dtype, device=device)

if method=="std":
    beta_1 = (-epsilon/h) - 0.5
    beta0 = (2*epsilon)/h
    beta1 = (-epsilon/h) + 0.5
elif method=="supg":
    beta_1 = (-epsilon/h) - 1
    beta0 = (2*epsilon)/h + 1
    beta1 = (-epsilon/h)


def relu(x):
    return torch.relu(x)

def relu_prime(x):
    return (x > 0).to(dtype=x.dtype)


def forward_all(x, b, w1, w2, BCs):
    z = w1 @ x.T + b.unsqueeze(1)
    sigma_z = relu(z)
    if BCs=="soft":
        out = w2 @ sigma_z
        out = out.squeeze(0)
    elif BCs=="H":
        out = w2 @ sigma_z
        x = x.squeeze(1)
        out = x * (1.0 - x) * out.squeeze(0)
    elif BCs=="I":
        out = w2 @ sigma_z
        x = x.squeeze(1)
        out = (1.0 - torch.exp(-(10**alpha) * x))*(1.0 - torch.exp(-(10**beta) * (1.0 - x)))*out.squeeze(0)
    return out, z, sigma_z

def bc_transform(x, BCs):
    x = x.squeeze(1)
    if BCs=="H":
        val = x * (1.0 - x)
    elif BCs=="I":
        val = (1.0 - torch.exp(-(10**alpha) * x))*(1.0 - torch.exp(-(10**beta) * (1.0 - x)))
    return val

def cost2(Fiter, BCs):
    Dj = beta_1 * Fiter[:-2] + beta0 * Fiter[1:-1] + beta1 * Fiter[2:] - h
    CP_loss = (Dj ** 2).sum()
    if BCs=="soft":
        costval = Fiter[0]**2 + Fiter[-1]**2
        CP_loss = costval + CP_loss
    return CP_loss, Dj


if optimizer_1=="Adam":
    optimizer = torch.optim.Adam([W1, W2, b1], lr=lr)   
elif optimizer_1=="GD":
    optimizer = torch.optim.SGD([W1, W2, b1], lr=lr)


pbar = tqdm(range(Niter_opt1))
for i in pbar:

    F_iter, z_all, sigma_z = forward_all(CP, b1, W1, W2, BC_appr)
    cost, Dj = cost2(F_iter, BC_appr)

    if backprop=="manual":

        sigma_prime_z = relu_prime(z_all)
        if BC_appr=="H":
            tr_CP = bc_transform(CP, BC_appr)
        elif BC_appr=="I":
            tr_CP = bc_transform(CP, BC_appr)

        if BC_appr=="soft":
            term_sigma_prime = (2 * F_iter[0] * sigma_prime_z[:, 0]
                + 2 * F_iter[-1] * sigma_prime_z[:, -1])
            Dj_terms = (beta_1 * sigma_prime_z[:, :-2]
                + beta0 * sigma_prime_z[:, 1:-1]
                + beta1 * sigma_prime_z[:, 2:])
        else:
            term_sigma_prime = 0
            Dj_terms = (beta_1 * tr_CP[:-2] * sigma_prime_z[:, :-2]
                + beta0 * tr_CP[1:-1] * sigma_prime_z[:, 1:-1]
                + beta1 * tr_CP[2:] * sigma_prime_z[:, 2:])
        term_sigma_prime += 2 * torch.sum(Dj * Dj_terms, dim=1)
        db1 = W2.squeeze(0).T * term_sigma_prime

        if BC_appr=="soft":
            term_dw1 = (2 * F_iter[-1] * sigma_prime_z[:, -1] * CP[-1, 0])
            Dj_terms_w1 = (beta_1 * CP[:-2, 0] * sigma_prime_z[:, :-2]
                + beta0 * CP[1:-1, 0] * sigma_prime_z[:, 1:-1]
                + beta1 * CP[2:, 0] * sigma_prime_z[:, 2:])
        else:
            term_dw1 = 0
            Dj_terms_w1 = (beta_1 * tr_CP[:-2] * CP[:-2, 0] * sigma_prime_z[:, :-2]
                + beta0 *  tr_CP[1:-1] * CP[1:-1, 0] * sigma_prime_z[:, 1:-1]
                + beta1 * tr_CP[2:] * CP[2:, 0] * sigma_prime_z[:, 2:])
        term_dw1 += 2 * torch.sum(Dj * Dj_terms_w1, dim=1)
        dw1 = (W2.squeeze(0).T * term_dw1).unsqueeze(1)

        if BC_appr=="soft":
            dw2 = (2 * F_iter[0] * sigma_z[:, 0] + 2 * F_iter[-1] * sigma_z[:, -1])
            Dj_terms_w2 = (beta_1 * sigma_z[:, :-2]
                + beta0 * sigma_z[:, 1:-1]
                + beta1 * sigma_z[:, 2:])
        else:
            dw2 = 0
            Dj_terms_w2 = (beta_1 * tr_CP[:-2] * sigma_z[:, :-2]
                + beta0 *  tr_CP[1:-1] * sigma_z[:, 1:-1]
                + beta1 * tr_CP[2:] * sigma_z[:, 2:])
        dw2 += 2 * torch.sum(Dj * Dj_terms_w2, dim=1)
        dw2 = dw2.unsqueeze(0)

        optimizer.zero_grad()  
        with torch.no_grad():
            W1.grad = dw1.clone()
            W2.grad = dw2.clone()
            b1.grad = db1.clone()

        optimizer.step()

    elif backprop=="auto":
        optimizer.zero_grad()
        cost.backward()

        optimizer.step()

    # Recalculate loss after the parameter update
    with torch.no_grad():
        F_iter, _, _ = forward_all(CP, b1, W1, W2, BC_appr)
        cost, _ = cost2(F_iter, BC_appr)

    savecost[i] = cost

    if i % 1000 == 0:
        pbar.set_description(f"{optimizer_1} Iter {i}, cost={cost.item():.3e}")


optimizer = torch.optim.LBFGS(
    [W1, W2, b1],
    lr=lbfgs_cfg["learning_rate"],
    max_iter=lbfgs_cfg["max_iter"],
    history_size=lbfgs_cfg["history_size"],
    tolerance_grad=lbfgs_cfg["tolerance_grad"],
    tolerance_change=lbfgs_cfg["tolerance_change"])

def closure():
    optimizer.zero_grad()
    F_iter, _, _ = forward_all(CP, b1, W1, W2, BC_appr)
    cost, _ = cost2(F_iter, BC_appr)
    cost.backward()  
    return cost


pbar = tqdm(range(Niter_lbfgs))
for i in pbar:

    optimizer.step(closure)
    # Recalculate the loss using the updated parameters
    with torch.no_grad():
        F_iter, _, _ = forward_all(CP, b1, W1, W2, BC_appr)
        cost, _ = cost2(F_iter, BC_appr)

    savecost[Niter_opt1 + i] = cost
    if i % 10 == 0:
        pbar.set_description(f"L-BFGS Iter {i}, cost={cost.item():.3e}")


plotting_grid = np.linspace(0, 1, n_ref + 1, dtype=np.float64)
x_tensor = torch.tensor(plotting_grid, dtype=dtype, device=device).unsqueeze(1)

with torch.no_grad():
    y_nn, _, _ = forward_all(x_tensor, b1, W1, W2, BC_appr)
    y_nn = y_nn.detach().cpu().numpy()

pred_left = float(y_nn[0])
pred_right = float(y_nn[-1])

y_theory = analytical(plotting_grid, epsilon)

if output_cfg["save_plots"]:
    plot_cost2(savecost, run_dir, adam_steps=Niter_opt1)
    plot_and_save(y_nn, y_theory, plotting_grid, n, epsilon, run_dir, lines=False)


rel_l2 = rel_l2_error(y_theory, y_nn)

l2_error, h1_error = L2_H1_errors(analytical, y_nn, 0, 1, h_ref, epsilon, first_der)

metrics = {
    "final_cost": float(savecost[-1].item()),
    "relative l2 error": float(rel_l2),
    "L2 error": float(l2_error),
    "H1 error": float(h1_error),
    "predicted boundary value at x=0": pred_left,
    "predicted boundary value at x=1": pred_right}

print(metrics)

if output_cfg["save_metrics"]:
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)







