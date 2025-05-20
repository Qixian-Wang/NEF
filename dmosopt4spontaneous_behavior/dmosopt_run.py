import logging
import numpy as np
from dmosopt import dmosopt
from brian2 import *
from model_stdp_spon import Model
from NEF.spontaneous_behavior.utils import process_spikes, compute_rates
from scipy.signal import find_peaks, peak_widths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connectivity_values = []
noise_amp_values = []
transmitter_leakage_values = []
transmitter_restoration_values = []
maximum_weight_values = []
taupre_values = []
taupost_values = []

def extract_objectives(time, firing_rate):
    peaks, props = find_peaks(firing_rate, height=1.0)
    peak_heights = firing_rate[peaks]
    peak_times   = time[peaks]

    if len(peaks) == 0:
        return [1e3, 1e3, 1e3]

    # Objective 1：峰值偏离 20 Hz 的 MSE
    obj1 = np.mean((peak_heights - 10.0)**2)

    # Objective 2：相邻峰间隔偏离 1s 的方差
    intervals = np.diff(peak_times)
    obj2 = np.var(intervals - 1.0)

    # Objective 3：非 burst 时段（排除每个峰的宽度区间）rate 超过 1Hz 的最大超出量
    #    用 peak_widths 估算每个峰左右宽度
    widths, width_heights, left_ips, right_ips = peak_widths(firing_rate, peaks, rel_height=0.5)
    mask = np.ones_like(firing_rate, bool)
    for l, r in zip(left_ips.astype(int), right_ips.astype(int)):
        mask[l:r] = False
    if np.any(firing_rate[mask] > 1.0):
        obj3 = np.max(firing_rate[mask] - 1.0)
    else:
        obj3 = 0.0

    return [obj1, obj2, obj3*10]

def run_simulation(params):
    connectivity = params["connectivity_range"]
    noise_amp = params["noise_amp_range"]
    transmitter_leakage = params["transmitter_leakage_range"]
    transmitter_restoration = params["transmitter_restoration_range"]
    maximum_weight = params["maximum_weight_range"]
    taupre = params["taupre_range"]
    taupost = params["taupost_range"]


    # Store parameter values for plotting
    connectivity_values.append(connectivity)
    noise_amp_values.append(noise_amp)
    transmitter_leakage_values.append(transmitter_leakage)
    transmitter_restoration_values.append(transmitter_restoration)
    maximum_weight_values.append(maximum_weight)
    taupre_values.append(taupre)
    taupost_values.append(taupost)

    # Setup oscillator population
    defaultclock.dt = 1 * ms
    groups_per_row = 8
    neuron_per_group = 10
    gmax = 0.5

    model = Model(hebbian_lr=1e-3, refractory=2, sigma_i=noise_amp,
                  connectivity=connectivity,
                  groups_per_row=groups_per_row,
                  n_per_group=neuron_per_group,
                  taupre=taupre,
                  taupost=taupost,
                  transmitter_leakage=transmitter_leakage,
                  transmitter_restoration=transmitter_restoration,
                  gmax=gmax)

    # Run simulation
    time_spon = 10
    model.train(duration=time_spon)
    time, firing_rate = compute_rates(model['excitatory_spike'], n_groups=groups_per_row ** 2, n_per_group=neuron_per_group)
    objective = extract_objectives(time, firing_rate)

    # For dmosopt, we return the objectives as a list
    logger.info(
        f"Parameters: connectivity={connectivity:.3f}, noise_amp={noise_amp:.3f}, "
        f"transmitter_leakage={transmitter_leakage:.3f},"
        f"transmitter_restoration={transmitter_restoration:.3f}, transmitter_restoration={transmitter_restoration:.3f},"
        f"→ objective={objective}"
    )
    return np.array(objective)


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = {k: v for k, v in pp.items()}
    return run_simulation(param_values)


if __name__ == "__main__":
    parameter_space = {
        "connectivity_range": [1e-1, 2e-1],
        "noise_amp_range": [20, 40],
        "transmitter_leakage_range": [0.1, 0.4],
        "transmitter_restoration_range": [100, 500],
        "maximum_weight_range": [0.05, 1],
        "taupre_range": [100, 1000],
        "taupost_range": [100, 1000],
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
        "population_size": 50,
        "num_generations": 10,
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