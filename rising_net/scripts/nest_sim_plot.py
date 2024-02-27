# -*- coding: utf-8 -*-

import time
import datetime

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.cosim_run_plot import *

from tvb_multiscale.core.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from examples.plot_write_results import plot_write_spiking_network_results


# PATHWAY_GAIN = 50.0
# INDEGREE_GAIN = 1.0
# # Assuming:
# kwargs = {'G': 6.0, 'STIMULUS': 0.4, 'STIMULUS_BASELINE': 1.0,
#           'I_e': -0.35, 'I_s': 0.085,
#           'w_ie': -3.0, 'w_rs': -2.0,
#           'CONN_LOG': True,
#           'FIC': 1.11,
#           'FIC_SPLIT': 0.31,  #'fit',
#           'PONS': 0.0,
#           "PATHWAY_GAIN": PATHWAY_GAIN,
#           "INDEGREE_GAIN": INDEGREE_GAIN,
#           "SENSTRIG": 2.0,
#           "CEREB": 2.0,
#           "simulation_length": 10000.0,
#           'plot_flag': True}


if __name__ == "__main__":

    # Assuming:
    kwargs = {'G': 6.0, 'STIMULUS': 0.4, 'STIMULUS_BASELINE': 1.0,
              'I_e': -0.35, 'I_s': 0.085,
              'w_ie': -3.0, 'w_rs': -2.0,
              'CONN_LOG': True,
              'FIC': 1.11,
              'FIC_SPLIT': 0.31,  #'fit',
              "simulation_length": 1000.0,
              'plot_flag': True}

    config, plotter = configure(**kwargs)
    config.NEST_PERIPHERY = "Input TVB to parrot_medulla"
    config.NEST_PERIPHERY_MANY_NEURONS = False
    config.NEST_BACKGROUND_FREQ = 0.0
    config.NEST_STIMULUS_FILE = "../working_files/tvb_to_nest_input.npz"
    config.figures.LARGE_SIZE = (10, 100)
    config.figures.DEFAULT_SIZE = (10, 100)

    # results = run_nest_workflow(PSD_target=None, model_params={}, config=config)
    plot_flag = kwargs["plot_flag"]
    if config.VERBOSE:
        print("\n\n------------------------------------------------\n\n"+
              "Running NEST workflow for plot_flag=%s, \nand model_params=\n%s...\n"
              % (str(plot_flag), str(config.model_params)))
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Build the NEST network
    nest_network, nest_nodes_inds, neuron_models, neuron_number, start_id_scaffold = build_NEST_network(config)

    input_rates = list([list(), list()])
    input_times = list()
    for iR in range(2):
        input_rates[iR] = nest_network.input_devices[0][iR].device.rate_values
    input_times = nest_network.input_devices[0][0].device.rate_times
    input_rates = np.array(input_rates)
    input_times = np.array(input_times)
    if input_rates.ndim > 2:
        input_rates = input_rates[:, 0, :].squeeze()
        input_times = input_times[0, :].squeeze()

    # # Simulate the NEST network
    # nest_network = simulate_nest_network(nest_network, config, neuron_models, neuron_number)
    nest_network.configure()
    nest_network.nest_instance.Prepare()

    input_times += nest_network.nest_instance.GetKernelStatus("biological_time") - input_times[0] + 0.1

    print("Simulating with %s with %d stimulating device(s)!..." %
          (config.NEST_PERIPHERY, len(nest_network.input_devices[0][0].device)))
    tic = time.time()
    for tt in range(12498):
        nest_time = nest_network.nest_instance.GetKernelStatus("biological_time")
        if np.mod(nest_time, 500) == 0:
            toc = time.time() - tic
            print("%g of 12500 ms in %s" % (nest_time, str(datetime.timedelta(seconds=toc)).split(".")[0]))
        for iR in range(2):
            nest_network.input_devices[0][iR].device.set({"rate_values": input_rates[iR, 10*tt:10*tt+10].tolist(),
                                                          "rate_times": input_times[10*tt:10*tt+10].tolist()})
        nest_network.Run(1.0)

    toc = time.time() - tic
    print("%g of 12500 ms in %s" % (nest_time, str(datetime.timedelta(seconds=toc)).split(".")[0]))

    simulation_length, transient = configure_simulation_length_with_transient(config)
    plot_write_spiking_network_results(nest_network, connectivity=connectivity,
                                       time=None, transient=transient, monitor_period=simulator.monitors[0].period,
                                       plot_per_neuron=False, plotter=plotter, writer=None, config=config)

    plot_nest_results_raster(nest_network, neuron_models, neuron_number, config)
