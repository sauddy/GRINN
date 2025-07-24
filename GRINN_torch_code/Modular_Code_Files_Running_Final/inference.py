import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model_architecture import PINN
from data_generator import alpha_generator
from losses import ASTPN
from config import xmin, ymin, tmin, rho_o
from visualisation import plot_function, rel_misfit
from solver import req_consts_calc  # <-- import for jeans and alpha

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

alpha_min, alpha_max = 0.1, 0.8
_, _, alpha_list = alpha_generator(alpha_min=alpha_min, alpha_max=alpha_max, N=20)
alpha_list = alpha_list.to(device)

# Domain setup (smaller: 2000 points)
lam = 7.0
rho_1 = 0.03
num_of_waves = 2
tmax = 1.5
N_0 = N_b = 1000
N_r = 10000
xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves

# Model and domain
model_1D = ASTPN(rmin=[xmin, tmin], rmax=[xmax, tmax], N_0=N_0, N_b=N_b, N_r=N_r, dimension=1)
domain = model_1D.geo_time_coord(option="Domain")

# Load model
net = PINN()
net.load_state_dict(torch.load('Case2_final_part1.pth', map_location=device))
net = net.to(device)
net.eval()

'''x = domain[0][:, 0:1].to(device)  # [2000, 1]
results = []
t_values = [0.0, 0.5]

with torch.no_grad():
    for alpha_val in alpha_list:
        alpha_val_scalar = alpha_val.item() if alpha_val.numel() == 1 else alpha_val.cpu().numpy().item()
        for t_val in t_values:
            t = torch.full_like(x, t_val).to(device)
            alpha = torch.full_like(x, alpha_val_scalar).to(device)
            output = net([x, t, alpha])
            rho = output[:, 0:1].cpu().numpy()
            vel = output[:, 1:2].cpu().numpy()
            phi = output[:, 2:3].cpu().numpy()
            results.append({
                "x": x.cpu().numpy(),
                "t": t_val,
                "alpha": alpha_val_scalar,
                "rho": rho,
                "vel": vel,
                "phi": phi
            })'''

'''print(f"Total results: {len(results)} (should be {len(alpha_list)*len(t_values)})")
print("Sample entry:")
for k, v in results[0].items():
    if isinstance(v, np.ndarray):
        print(f"{k}: shape {v.shape}")
    else:
        print(f"{k}: {v}")'''

# Compute jeans and plot_v1 for the chosen alpha

alpha_flat = alpha_list.view(-1)
plot_rhos = ((alpha_flat[:-1] + alpha_flat[1:]) / 2).tolist()
plot_rhos = [round(i, 3) for i in plot_rhos]

jeans, alpha_val = req_consts_calc(lam)

# Arrays to store results for each time step
avg_misfit_rho_by_time = {0.5: [], 1.0: [], 1.5: [], 2.0: []}
max_misfit_rho_by_time = {0.5: [], 1.0: [], 1.5: [], 2.0: []}

nu = 0.5
N = 8000
num_of_waves = 2
rho_1 = 0.03

time_array_misfit = np.array([0.5, 1.0, 1.5, 2.0])

'''for i in range(len(plot_rhos)):
    plot_rho = plot_rhos[i]
    plot_v1 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho)
    initial_params = (xmin, xmax, rho_1, plot_rho, plot_v1, jeans, lam, tmax, device)

    misfit_rho, misfit_vel = rel_misfit(net, time_array_misfit, initial_params, N, nu, num_of_waves, rho_1, show=False)

    # Calculate average relative misfit for each time step separately
    for j, time in enumerate(time_array_misfit):
        avg_misfit = np.mean(np.abs(misfit_rho[j]))  # Average across spatial points for this time step
        max_misfit = np.max(np.abs(misfit_rho[j]))
        avg_misfit_rho_by_time[time].append(avg_misfit)
        max_misfit_rho_by_time[time].append(max_misfit)

#print("Max misfits by time: ", max_misfit_rho_by_time)

# Plot the results for each time step
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Flatten axes for easier iteration
axes_flat = axes.flatten()

for idx, (time, misfits) in enumerate(avg_misfit_rho_by_time.items()):
    axes_flat[idx].plot(plot_rhos, misfits, color='red')
    # Add blue dots at the exact data points
    axes_flat[idx].scatter(plot_rhos, misfits, color='blue', s=50, zorder=5)
    axes_flat[idx].set_xlabel('Alpha')
    axes_flat[idx].set_ylabel('Avg Misfit (%)')
    axes_flat[idx].set_title(r'Avg Misfit vs $\rho_{1a}$' + f' (t={time})')
    axes_flat[idx].grid()

plt.tight_layout()
plt.show()'''

plot_rho1 = plot_rhos[1]
plot_rho2 = plot_rhos[6]
plot_rho3 = plot_rhos[-1]

plot_v1 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho1)
plot_v2 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho2)
plot_v3 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho3)

initial_params1 = xmin, xmax, rho_1, plot_rho1, plot_v1, jeans, lam, tmax, device
initial_params2 = xmin, xmax, rho_1, plot_rho2, plot_v2, jeans, lam, tmax, device
initial_params3 = xmin, xmax, rho_1, plot_rho3, plot_v3, jeans, lam, tmax, device

rel_misfit(net, time_array_misfit, initial_params1, N, nu, num_of_waves, rho_1)
rel_misfit(net, time_array_misfit, initial_params2, N, nu, num_of_waves, rho_1)
rel_misfit(net, time_array_misfit, initial_params3, N, nu, num_of_waves, rho_1)