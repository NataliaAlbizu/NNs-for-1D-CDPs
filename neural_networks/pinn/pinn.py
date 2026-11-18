import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import argparse
import yaml
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from utils.auxil import analytical, rel_l2_error, plot_and_save, plot_cost2, L2_H1_errors, first_der

CONFIG_PATH = SCRIPT_DIR / "config.yaml"


parser = argparse.ArgumentParser(description="Run PINN")

parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to configuration file")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--dtype", choices=["float32", "float64"], default=None)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--BC_appr", choices=["soft", "H", "I"], default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--N_c", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--adam_epochs", type=int, default=None)
parser.add_argument("--lbfgs_epochs", type=int, default=None)
parser.add_argument("--n_ref", type=int, default=None)

args = parser.parse_args()

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Apply only explicitly supplied command-line overrides
for key in ["seed", "dtype", "epsilon", "BC_appr", "alpha", "beta",
    "n", "N_c", "lr", "adam_epochs", "lbfgs_epochs", "n_ref"]:
    value = getattr(args, key)
    if value is not None:
        config[key] = value

if config["epsilon"] <= 0:
    parser.error("epsilon must be positive")

if config["n"] < 3:
    parser.error("n must be at least 3")

if not 1 <= config["N_c"] <= config["n"] - 1:
    parser.error("N_c must be between 1 and n - 1")

if config["adam_epochs"] < 0 or config["lbfgs_epochs"] < 0:
    parser.error("Epoch counts must be non-negative")

if config["lr"] <= 0:
    parser.error("lr must be positive")

if config["BC_appr"] == "I" and (
    config["alpha"] is None or config["beta"] is None
):
    parser.error("alpha and beta are required for BC_approach: I")

if config["n_ref"] < 1:
    parser.error("n_ref must be positive")

output_cfg = config["output"]
results_root = Path(output_cfg["results_dir"])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_name = (
    f"{timestamp}_Nc{config['N_c']}"
    f"_eps{config['epsilon']}"
    f"_{config['BC_appr']}")

run_dir = results_root / run_name
run_dir.mkdir(parents=True, exist_ok=False)

print(f"Saving results to: {run_dir}")

if output_cfg.get("save_config", False):
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# Reproducibility
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64}
dtype = dtype_map[config["dtype"]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = config["seed"]
epsilon = config["epsilon"]
BC_appr = config["BC_appr"]
alpha = config["alpha"]
beta = config["beta"]

n = config["n"]
N_c = config["N_c"]
n_ref = config["n_ref"]
h_ref = 1 / n_ref

layers = config["layers"]

lr = config["lr"]
max_epochs_adam = config["adam_epochs"]
max_epochs_lbfgs = config["lbfgs_epochs"]

grid = np.linspace(0, 1, n+1).reshape(-1, 1)

# Define the BC:
idx_bc = np.where((grid==0)|(grid==1.0))[0]
X_bc = grid[idx_bc]
u_bc = np.zeros((X_bc.shape[0], 1))

# Sample collocation points from the interior only
X_interior = grid[(grid[:, 0] > 0) & (grid[:, 0] < 1)]
idx_Xf = np.random.choice(X_interior.shape[0], N_c, replace=False)
X_colloc_train = X_interior[idx_Xf]

X_bc = torch.tensor(X_bc, dtype=dtype, device=device)
u_bc = torch.tensor(u_bc, dtype=dtype, device=device)
nb_bc = X_bc.shape[0]

X_colloc = torch.tensor(X_colloc_train, dtype=dtype, device=device)
nb_colloc = X_colloc.shape[0]


def net_transform(X_f, model_nn, BC_type):
    if BC_type == "soft":
        result = model_nn(X_f)

    elif BC_type == "H":
        x = X_f[:, 0:1]
        result = x * (1.0 - x) * model_nn(X_f)

    elif BC_type == "I":
        x = X_f[:, 0:1]

        a = torch.pow(torch.tensor(10.0, dtype=X_f.dtype, device=X_f.device),
            alpha)
        b = torch.pow(torch.tensor(10.0, dtype=X_f.dtype, device=X_f.device),
            beta)

        term_left = 1.0 - torch.exp(-a * x)
        term_right = 1.0 - torch.exp(-b * (1.0 - x))

        result = term_left * term_right * model_nn(X_f)

    else:
        raise ValueError(f"Unknown BC_type: {BC_type}")

    return result


def f_user(X_f, model_nn, BC_type):
    x_temp = X_f.detach().clone().requires_grad_(True)

    u = net_transform(x_temp, model_nn, BC_type)

    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x_temp,
        grad_outputs=torch.ones_like(u),
        create_graph=True)[0]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_temp,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True)[0]

    f = -epsilon * u_xx + u_x
    return f


def loss_fct(f):
    return torch.mean(torch.square(f - 1))


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        modules = []

        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(linear.weight)  
            nn.init.zeros_(linear.bias)

            modules.append(linear)
            modules.append(nn.Tanh())

        output_layer = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        modules.append(output_layer)

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


net_u = PINN(layers).to(device=device, dtype=dtype)


torch_optimizer = torch.optim.Adam(net_u.parameters(),lr=lr)


def loss_pinns(X_f, model_nn, u_pred_bc, u_star_bc, BC_type):
    loss_f = loss_fct(f_user(X_f, model_nn, BC_type))

    if BC_type == "soft":
        loss_bc = torch.mean(torch.square(u_pred_bc - u_star_bc))
        return loss_f + loss_bc

    elif BC_type in ("H", "I"):
        return loss_f

    else:
        raise ValueError(f"Unknown BC_type: {BC_type}")


def get_grad(X_f, BC_type):
    if BC_type == "soft":
        u_pred_bc = net_transform(X_bc, net_u, BC_type)
    else:
        u_pred_bc = None

    loss_value = loss_pinns(X_f, net_u, u_pred_bc, u_bc, BC_type)

    grads = torch.autograd.grad(
        outputs=loss_value,
        inputs=tuple(net_u.parameters()),
        create_graph=False)

    return loss_value, grads


def evaluate_loss(X_f, BC_type):
    with torch.enable_grad():
        loss_value, _ = get_grad(X_f, BC_type) 
    return loss_value.detach().item()


def train(max_epochs_adam=0, max_epochs_lbfgs=0, print_per_epochs=1000):

    loss_array = np.array([])
    epoch = 0
    current_loss = None
    final_loss = None

    parameters = tuple(net_u.parameters())

    def train_step(X_f, BC_type):
        nonlocal loss_array

        loss_value, grads = get_grad(X_f, BC_type) 

        torch_optimizer.zero_grad(set_to_none=True)

        for parameter, grad in zip(parameters, grads):
            parameter.grad = grad.detach().clone()

        torch_optimizer.step()

        return loss_value  


    for epoch_adam in range(max_epochs_adam):
        loss_value = train_step(X_colloc, BC_appr)  

        loss_scalar = loss_value.detach().item()
        loss_array = np.append(loss_array, loss_scalar)

        if epoch_adam % print_per_epochs == 0:
            print(f"Loss PINN at epoch {epoch_adam} (Adam): "
                f"{loss_scalar}")

        final_loss = loss_scalar
        epoch += 1
    
    if max_epochs_adam > 0:
        final_loss = evaluate_loss(X_colloc, BC_appr)

        # Align the last recorded Adam loss with the updated network.
        loss_array[-1] = final_loss

    
    if max_epochs_lbfgs > 0:

        optimizer_lbfgs = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=1,
            max_eval=20,
            history_size=100,
            line_search_fn="strong_wolfe")


        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)

            if BC_appr == "soft":
                u_pred_bc = net_transform(X_bc, net_u, BC_appr)
            else:
                u_pred_bc = None

            loss_value = loss_pinns(X_colloc, net_u, u_pred_bc, u_bc, BC_appr)

            loss_value.backward()

            return loss_value


        for epoch_lbfgs in range(max_epochs_lbfgs):

            optimizer_lbfgs.step(closure)

            current_loss = evaluate_loss(X_colloc, BC_appr)

            loss_array = np.append(loss_array, current_loss)

            if (epoch_lbfgs + 1) % print_per_epochs == 0:
                print(
                    f"Loss PINN at epoch {epoch + epoch_lbfgs + 1} "
                    f"(L-BFGS): {current_loss:.6e}")

        final_loss = current_loss

    if final_loss is None:
        raise ValueError("Set max_epochs_adam or max_epochs_lbfgs to a positive value.")

    return float(final_loss), loss_array, max_epochs_adam


final_loss, loss_history, adam_steps = train(
    max_epochs_adam=max_epochs_adam,
    max_epochs_lbfgs=max_epochs_lbfgs,
    print_per_epochs=400)


plotting_grid = np.linspace(0, 1, n_ref + 1)
x_tensor = torch.tensor(plotting_grid, dtype=dtype, device=device).unsqueeze(1)

net_u.eval()
with torch.no_grad():
    pred = net_transform(x_tensor, net_u, BC_appr).cpu().numpy()

pred_flat = pred.reshape(-1)
pred_left = float(pred_flat[0])
pred_right = float(pred_flat[-1])

y_theory = analytical(plotting_grid, epsilon)
error = rel_l2_error(y_theory, pred_flat)

l2_error, h1_error = L2_H1_errors(analytical, pred_flat, 0, 1, h_ref, epsilon, first_der)


if output_cfg.get("save_plots", True):
    plot_and_save(
        pred_flat,
        y_theory,
        plotting_grid,
        n,
        epsilon,
        run_dir,
        lines=False)
    plot_cost2(
        loss_history,
        run_dir,
        adam_steps=adam_steps)


metrics = {
    "final loss": final_loss,
    "relative l2 error": float(error),
    "L2 error": float(l2_error),
    "H1 error": float(h1_error),
    "predicted boundary value at x=0": pred_left,
    "predicted boundary value at x=1": pred_right}

if output_cfg.get("save_metrics", True):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

print(f"Final loss: {final_loss:.6e}")
print(f"Relative l2 error: {error:.6e}")
print(f"Predicted boundary value at x=0: {pred_left:.8e}")
print(f"Predicted boundary value at x=1: {pred_right:.8e}")
print(f"Results saved in: {run_dir}")