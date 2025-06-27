import logging
import sys
import os
import numpy as np
from dmosopt import dmosopt
from brian2 import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_stdp import Model
from utils import process_spikes, compute_rates
from scipy.signal import find_peaks, peak_widths
from mea_layout import generate_layout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connectivity_values = []
noise_amp_values = []
transmitter_leakage_values = []
transmitter_restoration_values = []
maximum_weight_values = []
tau_noise_values = []
distance_factor_values = []

def extract_objectives(time, firing_rate):
    # time = time[-len(time)//2:]
    firing_rate = firing_rate[-len(firing_rate)//2:]
    peaks, props = find_peaks(firing_rate, height=5.0)
    peak_heights = firing_rate[peaks]
    peak_times   = time[peaks]

    if len(peaks) == 0:
        return [1e4, 1e4, 1e4]

    # Objective 1：峰值偏离 20 Hz 的 MSE
    obj1 = np.mean((peak_heights - 5.0)**2)

    # Objective 2：相邻峰间隔偏离 1s 的方差
    intervals = np.diff(peak_times)
    obj2 = np.var(intervals - 2.0)

    # Objective 3：非 burst 时段（排除每个峰的宽度区间）rate 超过 1Hz 的最大超出量
    #    用 peak_widths 估算每个峰左右宽度
    widths, width_heights, left_ips, right_ips = peak_widths(firing_rate, peaks, rel_height=0.5)
    mask = np.ones_like(firing_rate, bool)
    for l, r in zip(left_ips.astype(int), right_ips.astype(int)):
        mask[l:r] = False
    if np.any(firing_rate[mask] > 1.0):
        obj3 = np.max(firing_rate[mask])
    else:
        obj3 = 0.0

    return [obj1, obj2*0, obj3*100]

def run_simulation(params):
    connectivity = params["connectivity_range"]
    noise_amp = params["noise_amp_range"]
    maximum_weight = params["maximum_weight_range"]
    tau_noise = params["tau_noise_range"]
    distance_factor = params["distance_factor_range"]

    # Store parameter values for plotting
    connectivity_values.append(connectivity)
    noise_amp_values.append(noise_amp)
    maximum_weight_values.append(maximum_weight)
    tau_noise_values.append(tau_noise)
    distance_factor_values.append(distance_factor)
    # Setup oscillator population
    defaultclock.dt = 2 * ms
    dApre = 0.1
    num_neuron = 1000

    layout = generate_layout(height=1600,
                            electrode_t=64,
                            electrode_b=64,
                            electrode_per_row_b=8,
                            electrode_per_row_t=8,
                            num_neuron=num_neuron,
                            spacing=100,
                            plot_result=False)

    model = Model(hebbian_lr=0, 
                refractory=10, 
                sigma_noise=noise_amp,
                connectivity=connectivity,
                distance_factor=distance_factor,
                taupre=300,
                taupost=400,
                dApre=dApre,
                tau_noise=tau_noise,
                transmitter_leakage=0,
                transmitter_restoration=600,
                stimulation_voltage=80,
                stimulation_frequency=6,
                layout=layout,
                gmax=maximum_weight)
    
    # Run simulation
    time_spon = 10
    model.train(duration=time_spon)
    time, firing_rate = compute_rates(model)
    objective = extract_objectives(time, firing_rate)

    # For dmosopt, we return the objectives as a list
    logger.info(
        f"Parameters: connectivity={connectivity:.3f}, noise_amp={noise_amp:.3f}, "
        f"maximum_weight={maximum_weight:.3f},"
        f"→ objective={objective}"
    )
    return np.array(objective)


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = {k: v for k, v in pp.items()}
    return run_simulation(param_values)


if __name__ == "__main__":
    parameter_space = {
        "connectivity_range": [2e-2, 5e-1],
        "noise_amp_range": [5, 20],
        "maximum_weight_range": [0.02, 2],
        "tau_noise_range": [1, 50],
        "distance_factor_range": [300, 2000],
    }

    problem_parameters = {}
    objective_names = ["y1", "y2", "y3"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "spontaneous_behavior_opt",
        "obj_fun_name": "dmosopt_run.obj_fun",
        "problem_parameters": problem_parameters,
        "space": parameter_space,
        "objective_names": objective_names,
        "population_size": 60,
        "num_generations": 15,
        "initial_maxiter": 5,
        "optimizer_name": "age",
        "termination_conditions": True,
        "n_initial": 3,
        "n_epochs": 4,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print(best)
        bestx, besty = best
        x, y = dmosopt.dopt_dict["spontaneous_behavior_opt"].optimizer_dict[0].get_evals()

        besty_dict = dict(besty)  # besty 是目标值的键值对，如 {"y1": ..., "y2": ..., "y3": ...}

        # 创建 3D 图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 所有评估点
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], c='blue', alpha=0.6, label='Evaluated points')

        # 最优点
        ax.scatter(
            besty_dict["y1"], besty_dict["y2"], besty_dict["y3"],
            c='red', s=80, label='Best point'
        )

        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title("3D Objective Space")
        ax.legend()

        plt.tight_layout()
        plt.savefig("spontaneous_behavior_3dplot.svg")
        plt.show()
