# coding: utf-8

import warnings
import glob
import pickle
import os
import shutil
from matplotlib import pyplot

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray
from examples.plot_write_results import plot_write_spiking_network_results


def get_config(**kwargs):

    # DEFAULT_ARGS = {  # TVB model:
    #     'I_s': 0.1,
    #     'I_e': -0.35,
    #     'w_ie': -3.0,
    #     # TVB network:
    #     'G': 6.0,
    #     'FIC': 2.0,
    #     # Pathway gains:
    #     "PATHWAY_GAIN": 1,
    #     "TRIG_GAIN": 50.0, "MEDULLA_GAIN": 50.0, "CEREB_GAIN": 50.0,
    #     "TRIGS1_GAIN": 5.0, "MEDULLAS1_GAIN": 5.0, "CNS1_GAIN": 5.0,
    #     "CNM1_GAIN": 10.0,
    #     "M1S1_GAIN": 50.0,
    #     "M1FACIAL_GAIN": 50.0,
    #     "FACIALTRIG_GAIN": 50.0,
    #     # TVB <-> NEST Interface:
    #     "w_TVB_to_NEST": 35.0, "w_TVB_to_NEST_rest": 0.15,
    #     "MAX_RATES": {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
    #                   "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0},  # Hz
    #     # WORKFLOW:
    #     "TASK": True,
    #     'output_folder': "", 'verbose': 1, 'plot_flag': True}

    # Get configuration
    config, plotter = configure(verbose=0, **kwargs)
    config.VERBOSE = 2.0

    print(config.model_params)
    print(config)

    return config, plotter


def cosim_run_plot(**kwargs):

    config, plotter = get_config(**kwargs)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

    if "CEREBOFF" in config.MODE:
        inds_off = np.sort(inds['cereb_crtx'].tolist() +
                           inds['cereb_nuclei'].tolist() +
                           inds['ansilob'].tolist())
        simulator.connectivity.weights[inds_off, :] = 0
        simulator.connectivity.weights[:, inds_off] = 0
        if config.VERBOSE:
            print("\n")
            print("-" * 25)
            print("-" * 25)
            print("Setting to 0.0 connections in and out of cerebellum\n"
                  "['Left/Right Cerebellar Cortex'\n"
                  "'Left/Right Cerebellar Nuclei'\n"
                  "'Left Ansiform lobule']!!!:\n"
                  "IN: %s\n"
                  "OUT: %s" % (str(simulator.connectivity.weights[inds_off, :]),
                               str(simulator.connectivity.weights[:, inds_off])))
        simulator.connectivity.configure()
        simulator.configure()

    nest_network = None
    if "COSIM" in config.MODE:
        # Build NEST network
        nest_network, nest_nodes_inds, neuron_models, neuron_number, start_id_scaffold = build_NEST_network(config)
        # Build TVB-NEST interfaces
        simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config,
                                                            neuron_models, start_id_scaffold)
        if "CEREBOFF" in config.MODE:
            for hemi in ["Right", "Left"]:
                nest_network.brain_regions['%s Cerebellar Nuclei' % hemi]['dcn_cell_glut_large'].Set({"V_th": 35.0})
                print('%s Cerebellar Nuclei - dcn_cell_glut_large' % hemi)
                print(nest_network.brain_regions['%s Cerebellar Nuclei' % hemi]['dcn_cell_glut_large'].Get("V_th"))
        # Simulate TVB-NEST model
        results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)
    else:
        # Run simulation and get results for reference values
        results, transient = simulate(simulator, config)

    # Target values: ansilob=-0.3263, interposed=-0.3209, oliv=-0.3284

    # Compute coherence
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2

    results = plot_tvb(transient, inds,
                       results=results, simulator=simulator, plotter=plotter, config=config, write_files=True)

    # if "COSIM" in config.MODE::
    #     plot_write_spiking_network_results(nest_network, connectivity=connectivity,
    #                                        time=None, transient=transient, monitor_period=simulator.monitors[0].period,
    #                                        plot_per_neuron=False, plotter=plotter, writer=None, config=config)

    results_path = os.path.join(config.out.FOLDER_RES, 'results.pickle')
    with open(results_path, 'wb') as handle:
       pickle.dump(results, handle)
    print(results_path)
    # results = pickle.load(results_path, 'rb'))  # to load results

    coherence_path = os.path.join(config.out.FOLDER_RES, 'coherence.pkl')
    # Save coherence
    CxyR = results["CxyR_M1_S1"]
    fR = results["fR"]
    CxyL = results["Cxyl_M1_S1"]
    fL = results["fL"]
    with open(coherence_path, 'wb') as handle:
        pickle.dump([CxyR, fR, fL, CxyL], handle)
    print(coherence_path)

    return results, simulator, nest_network, config, inds


def plot_comparison(tests, **kwargs):

    TESTS = tests
    colors = []
    for test, col in zip(["COSIM", "COSIM_CEREBOFF", "TVB", "TVB_CEREBOFF"], ["b", "m", "g", "r"]):
        if test in TESTS:
            colors.append(col)
    TESTSFOLDER = "-".join(TESTS)

    # CONFIGURATION:
    config, plotter = get_config(output_folder=TESTSFOLDER, **kwargs)

    # CONNECTIVITY:
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=None)
    connectivity = build_connectivity(connectome, inds, config)

    # Results path:
    BASEPATH = os.path.dirname(config.out.FOLDER_RES.split("res")[0][:-1])
    print(BASEPATH)
    TESTSPATH = os.path.join(BASEPATH, TESTSFOLDER)

    # Task related regions' labels:
    REGION_LABELS = connectivity.region_labels[config.TASKINDS]
    # Task related regions' abreviated labels:
    SHORT_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"]) for reg in REGION_LABELS]

    THETA = [5.0, 10.0]  # Hz
    GAMMA = [25.0, 45.0]  # Hz

    # results dictionary:
    results = {"inds": config.TASKINDS,
               "region_labels": REGION_LABELS, "short_labels": SHORT_LABELS,
               "theta": THETA, "gamma": GAMMA}

    for test_name in TESTS:

        results[test_name] = {}
        Ps = []
        Cs = []

        testpath_old = os.path.join(BASEPATH, test_name)
        testpath = os.path.join(TESTSPATH, test_name)
        if os.path.isdir(testpath_old):
            shutil.move(testpath_old, testpath)
        for path in glob.glob(os.path.join(testpath, "nsd*")):
            resultsfile = os.path.join(path, "res/source_ts.pkl")
            print(resultsfile)
            with open(resultsfile, 'rb') as handle:
                source_ts = pickle.load(handle)  # to load results
            Pxx_den, Cxy, f, ij = compute_selected_spectra_coherence(
                                        source_ts["data"], config.TASKINDS,
                                        transient=source_ts["data"].shape[0]-2**15, # 2**15 final length
                                        sample_period=source_ts["sample_period"],
                                        nperseg=512, fmin=0.0, fmax=50.0)
            Ps.append(Pxx_den)
            Cs.append(Cxy)

        results[test_name]['PSD'] = np.array(Ps)
        results[test_name]['COH'] = np.array(Cs)
        fth = np.where(np.logical_and(f > THETA[0], f < THETA[1]))[0]
        results[test_name]['COHth'] = results[test_name]['COH'][:, :, fth].mean(axis=2)  # TODO Find out if it is correct!
        fgm = np.where(np.logical_and(f > GAMMA[0], f < GAMMA[1]))[0]
        results[test_name]['COHgm'] = results[test_name]['COH'][:, :, fgm].mean(axis=2) # TODO Find out if it is correct!

    results["f"] = f      # frequency vector
    results["fth"] = fth  # theta frequency vector inds
    results["fgm"] = fgm  # gamma frequency vector inds
    results["ij"] = ij    # pairs of regions of coherences where i, j in [0, config.TASKINDS.size]

    dump_pickled_dict(results, os.path.join(config.out.FOLDER_RES, "res_PSD_COH.pkl"))

    figR, axR, figL, axL = plot_pathway_psd_coh(results, inds,
                                                tests=TESTS, colors=colors,
                                                percentile_min=1, percentile_max=99, n=1,
                                                plot_mean=True, plot_median=False, mode="semilog",
                                                alpha=0.5, figsize=config.figures.LARGE_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        for fig, hemi in zip([figR, figL], ["Right", "Left"]):
            plt.figure(fig.number)
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "PathwayPSD_COH_%s.png" % hemi))

    figPSD, axesPSD = psd_percent_plot(results,
                                        inds=None,
                                        tests=TESTS, colors=colors,
                                        percentile_min=1, percentile_max=99, n=1,
                                        plot_mean=False, plot_median=True,
                                        alpha=0.5, figsize=config.figures.DEFAULT_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        plt.figure(figPSD.number)
        plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "PSDs.png"))

    figCOH, axesCOH = coherence_networks_plot(results,
                                              tests=TESTS,
                                              resnames=['COHth', 'COHgm'],
                                              bands=["theta", "gamma"],
                                              figsize=config.figures.DEFAULT_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        plt.figure(figCOH.number)
        plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "COHs.png"))


if __name__ == '__main__':
    # Example use:
    # $ python tuning_tvb_nest.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
    # Called tuning_tvb_nest.py with:
    # keyword argument: w_TVB_to_NEST=world
    # keyword argument: simulation_length=300.0

    import sys

    kwargs = {}
    ntests = 0
    for arg in sys.argv[1:]:
        keyval = arg.split("=")
        if keyval[0] not in ["MODE"]:
            key = float(keyval[1])
        else:
            key = keyval[1].split(" ")
            if keyval[0] == "MODE":
                ntests = len(key)
                if ntests == 1:
                    key = key[0]
        kwargs[keyval[0]] = key

    if ntests > 1:
        plot_comparison(kwargs.pop("MODE"), **kwargs)
    else:
        cosim_run_plot(**kwargs)
