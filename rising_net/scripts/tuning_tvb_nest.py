"""
File: tuning_tvb_nest_2023.py
Author: Alice Geminiani
Email: alice.geminiani@unipv.it
Date: oct 2023
Description: script for tuning TVB to NEST cosim interface
"""

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *  # build_NEST_network, plot_nest_results

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

# Assuming (unless explicitly modified in code or by user inputs):
# DEFAULT_ARGS = {'G': 6.0, 'STIMULUS': 0.1,
#                 'I_e': -0.35, 'I_s': 0.085,
#                 'w_ie': -3.0, 'w_rs': -2.0,
#                 'CONN_LOG': True, 'FIC': 1.11,  'FIC_SPLIT': 0.31,  #'fit',
#                 'PRIORS_DIST': 'uniform',
#                 'output_folder': "", 'verbose': 1, 'plot_flag': True}

# config.TRANSIENT_RATIO =0.25
# # TVB - NEST interface parameters:
# config.MOSSY_MAX_RATE = 122.0  # Hz
# config.w_TVB_to_NEST = 0.04

SIMULATION_LENGTH = 30000.0
TUNED_VALUES_TVB_TO_NEST = [0.02, 0.03, 0.04, 0.045, 0.05, 0.06]  # 0.0425, 0.04375,


def tuning_tvb_nest(w_TVB_to_NEST=0.04, **kwargs):
    # RMSEs = []

    # for w_TVB_to_NEST in TUNED_VALUES_TVB_TO_NEST:
    # Get configuration
    config, plotter = configure(output_folder='nest_tvb_' + str(w_TVB_to_NEST), verbose=2,
                                STIMULUS=0.0,  # We are fitting in resting state!!!
                                w_TVB_to_NEST=w_TVB_to_NEST,
                                **kwargs)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

    # Build TVB-NEST interfaces
    nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
    simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
    # Simulate TVB-NEST model
    results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)

    # Compute error

    # Get spike events from NEST spike recorders
    events = nest_network.output_devices['mossy_fibers']['Right Ansiform lobule'].get_events()

    # Compute approximate average rate of mossy fibers as:
    # number_of_spikes / (number_of_neurons * time_length_in_ms) * 1000 (to convert to spikes/sec)
    duration = nest_network.nest_instance.GetKernelStatus("biological_time") - transient
    n_spikes = np.sum(events['times'] > transient)
    rate = 1000 * n_spikes / \
           (nest_network.output_devices['mossy_fibers']['Right Ansiform lobule'].number_of_neurons
            * duration)
    print("Approximate mossy_fibers rate during the last %g ms = %g" % (duration, rate))
    RMSE = (rate - 3.9) ** 2

    print("RMSE for TVB-NEST gain = %g is %g" % (config.w_TVB_to_NEST, RMSE))

    # RMSEs.append(RMSE)
    # print("RMSE for TVB-NEST gains ", TUNED_VALUES_TVB_TO_NEST, " are ", RMSEs)

    dump_pickled_dict({"mossy_rate": rate, "RMSE": RMSE}, os.path.join(config.out.FOLDER_RES, "rate_RMSE.pkl"))

    # Plot and save:
    plot_tvb(transient, inds, results,
             simulator=simulator, plotter=plotter, config=config, write_files=True)

    return rate, RMSE


if __name__ == '__main__':
    # Example use:
    # $ python tuning_tvb_nest.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
    # Called tuning_tvb_nest.py with:
    # keyword argument: w_TVB_to_NEST=world
    # keyword argument: simulation_length=300.0

    import sys

    kwargs = {}
    for arg in sys.argv[1:]:
        keyval = arg.split("=")
        kwargs[keyval[0]] = float(keyval[1])

    tuning_tvb_nest(**kwargs)
