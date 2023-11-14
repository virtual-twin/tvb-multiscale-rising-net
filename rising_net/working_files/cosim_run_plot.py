# coding: utf-8

import pickle

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results

from tvb_multiscale.core.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def cosim_run_plot(**kwargs):

    # Assuming:
    # DEFAULT_ARGS = {'G': 6.0, 'STIMULUS': 0.1,
    #                 'I_e': -0.35, 'I_s': 0.085,
    #                 'w_ie': -3.0, 'w_rs': -2.0,
    #                 'CONN_LOG': True, 'FIC': 1.11,  'FIC_SPLIT': 0.31,  #'fit',
    #                 'PRIORS_DIST': 'uniform',
    #                 'output_folder': "", 'verbose': 1, 'plot_flag': True}

    # config.TRANSIENT_RATIO = 0.25
    # # TVB - NEST interface parameters:
    # config.MOSSY_MAX_RATE = 122.0  # Hz
    # config.w_TVB_to_NEST = 0.04

    seed = int(kwargs.pop("seed", 10))
    test_name = kwargs.pop("test_name", "cosim")  # 'cosim', 'tvb-only', 'cerebOFF'
    if test_name == 'cosim':
        COMPUTE_REF = False          # True if you want to run TVB-only
        CEREB_OFF = False
    elif test_name == 'tvb-only':
        COMPUTE_REF = True
        CEREB_OFF = False
    elif test_name == 'cerebOFF':
        COMPUTE_REF = True
        CEREB_OFF = True

    resfilename = "results_%s_noise_seed_%d" % (test_name, seed)
    path = os.path.join(os.getcwd(), resfilename)
    # Get configuration
    config, plotter = configure(output_folder=path, verbose=2,
                                DEFAULT_TVB_NOISE_SEED=seed,
                                NEST_MASTER_SEED=143202461+seed,
                                SIMULATION_LENGTH=kwargs.get("simulation_length", 30000.0),
                                **kwargs)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    # # Scale up connections from principal sensory trigeminal nucleus to ansiform lobule
    # reg1='Left Ansiform lobule'
    # reg2='Right Ansiform lobule'
    # reg3 = 'Left Principal sensory nucleus of the trigeminal'
    # reg4 = 'Right Principal sensory nucleus of the trigeminal'
    # reg5 = 'Left Spinal nucleus of the trigeminal'
    # reg6 = 'Right Spinal nucleus of the trigeminal'
    # #find the indices in region labels of these strings
    # iR1 = np.where([reg1 in reg for reg in connectivity.region_labels])[0]
    # iR2 = np.where([reg2 in reg for reg in connectivity.region_labels])[0]
    # iR3 = np.where([reg3 in reg for reg in connectivity.region_labels])[0]
    # iR4 = np.where([reg4 in reg for reg in connectivity.region_labels])[0]
    # iR5 = np.where([reg5 in reg for reg in connectivity.region_labels])[0]
    # iR6 = np.where([reg6 in reg for reg in connectivity.region_labels])[0]
    #
    # pathway_gain = 1
    # '''# PST to AN
    # connectivity.weights[iR1, iR3] *= pathway_gain
    # connectivity.weights[iR1, iR4] *= pathway_gain
    # connectivity.weights[iR2, iR3] *= pathway_gain
    # connectivity.weights[iR2, iR4] *= pathway_gain
    # # SNT to PST
    # connectivity.weights[iR3, iR5] *= pathway_gain
    # connectivity.weights[iR3, iR6] *= pathway_gain
    # connectivity.weights[iR4, iR5] *= pathway_gain
    # connectivity.weights[iR4, iR6] *= pathway_gain
    # '''
    # # SNT to AN
    # connectivity.weights[iR1, iR5] *= pathway_gain
    # connectivity.weights[iR1, iR6] *= pathway_gain
    # connectivity.weights[iR2, iR5] *= pathway_gain
    # connectivity.weights[iR2, iR6] *= pathway_gain

    # # To have the full sensory whisking pathway
    # reg7 = 'Left Primary somatosensory area, barrel field'
    # reg8 = 'Right Primary somatosensory area, barrel field'
    # iR7 = np.where([reg7 in reg for reg in connectivity.region_labels])[0]
    # iR8 = np.where([reg8 in reg for reg in connectivity.region_labels])[0]
    # # S1 to PST
    # connectivity.weights[iR3, iR7] *= pathway_gain
    # connectivity.weights[iR3, iR8] *= pathway_gain
    # connectivity.weights[iR4, iR7] *= pathway_gain
    # connectivity.weights[iR4, iR8] *= pathway_gain
    # # SNT to S1
    # connectivity.weights[iR7, iR5] *= pathway_gain
    # connectivity.weights[iR7, iR6] *= pathway_gain
    # connectivity.weights[iR8, iR5] *= pathway_gain
    # connectivity.weights[iR8, iR6] *= pathway_gain


    # Put cereb weights to 0 if CEREB_OFF
    if CEREB_OFF:
        #reg1='Cerebell*'
        reg1='Left Cerebellar Cortex'
        reg2='Left Cerebellar Nuclei'
        reg3='Left Ansiform lobule'
        reg4='Left Interposed nucleus'
        reg5='Right Cerebellar Cortex'
        reg6='Right Cerebellar Nuclei'
        reg7='Right Ansiform lobule'
        reg8='Right Interposed nucleus'
        #find the indices in region labels of these strings
        iR1 = np.where([reg1 in reg for reg in connectivity.region_labels])[0]
        iR2 = np.where([reg2 in reg for reg in connectivity.region_labels])[0]
        iR3 = np.where([reg3 in reg for reg in connectivity.region_labels])[0]
        iR4 = np.where([reg4 in reg for reg in connectivity.region_labels])[0]
        iR5 = np.where([reg5 in reg for reg in connectivity.region_labels])[0]
        iR6 = np.where([reg6 in reg for reg in connectivity.region_labels])[0]
        iR7 = np.where([reg7 in reg for reg in connectivity.region_labels])[0]
        iR8 = np.where([reg8 in reg for reg in connectivity.region_labels])[0]
        # for reg1, reg2, sc in config.BRAIN_CONNECTIONS_TO_SCALE:
        #     iR1 = np.where([reg in reg1 for reg in connectivity.region_labels])[0]
        #     iR2 = np.where([reg in reg2 for reg in connectivity.region_labels])[0]
        #     connectivity.weights[iR1, iR2] *= 0
        #iR1
        connectivity.weights.shape
        for i in [iR1, iR2, iR3, iR4, iR5, iR6, iR7, iR8]:
            connectivity.weights[i,:]=0
            connectivity.weights[:,i]=0
                            # , iR2, iR3, iR4, iR5, iR6, iR7, iR8] = 0
        connectivity.weights[iR1,:]


    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)

    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

    if COMPUTE_REF:
        # Run simulation and get results for reference values
        results, transient = simulate(simulator, config)
    else:
        # Build TVB-NEST interfaces
        nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
        simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
        # Simulate TVB-NEST model
        results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)

    print(results)



    # Target values: ansilob=-0.3263, interposed=-0.3209, oliv=-0.3284


    # Compute coherence
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2

    results = plot_tvb(transient, inds,
                       results=results, simulator=simulator, plotter=plotter, config=config, write_files=True)



    results_path = os.path.join(config.out.FOLDER_RES, '%s.pickle' % resfilename)
    with open(results_path, 'wb') as handle:
       pickle.dump(results, handle)
    print(results_path)
    # results = pickle.load(results_path, 'rb'))  # to load results


    cohfilename = "coherence_30sec_%s_noise_seed_%d" % (test_name, seed)
    coherence_path = os.path.join(config.out.FOLDER_RES, '%s.pickle' % cohfilename)
    # Save coherence
    CxyR = results["CxyR_M1_S1"]
    fR = results["fR"]
    CxyL = results["Cxyl_M1_S1"]
    fL = results["fL"]
    with open(coherence_path, 'wb') as handle:
        pickle.dump([CxyR, fR, fL, CxyL], handle)
    print(coherence_path)

    return results


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
        if keyval[0] != "test_name":
            key = float(keyval[1])
        else:
            key = keyval[1]
        kwargs[keyval[0]] = key

    cosim_run_plot(**kwargs)
