# coding: utf-8

import warnings
import glob
import pickle
import os
import shutil
from matplotlib import pyplot
import torch

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.sbi_script import build_priors, prepare_for_sbi, simulate_for_sbi
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray
from examples.plot_write_results import plot_write_spiking_network_results


def sim_filepath(iR, config, filepath=None, extension=None, filename=None):
    if filepath is None or extension is None:
        filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, filename))
    return config.SIM_FILE_FORMAT % (filepath, iR, extension)


def priors_filepath(iR, config, filepath=None, extension=None):
    return sim_filepath(iR, config, filepath, extension, config.PRIORS_SAMPLES_FILE)


def sample_priors_for_sbi(config=None):
    config = assert_config(config, return_plotter=False)
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)
    dummy_sim = lambda priors: priors
    priors = build_priors(config)
    simulator, priors = prepare_for_sbi(dummy_sim, priors)
    priors_samples, sim_res = simulate_for_sbi(dummy_sim, proposal=priors,
                                               num_simulations=1,
                                               num_workers=config.SBI_NUM_WORKERS)
    return priors_samples, sim_res


def priors_samples(iR, priors_samples=None, config=None, write_to_files=True):
    config = assert_config(config, return_plotter=False)
    if priors_samples is None:
        priors_samples = sample_priors_for_sbi(config)[0]
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.PRIORS_SAMPLES_FILE))
    if write_to_files:
        torch.save(priors_samples, priors_filepath(iR, config, filepath, extension))
    return priors_samples


def generate_priors_samples(config=None):
    from collections import OrderedDict

    config = assert_config(config, return_plotter=False, MODE="PRIORS")

    samples = []
    for iR in range(config.N_SIMULATIONS):
        samples.append(priors_samples(iR, config=config, write_to_files=True).numpy())
    samples = np.array(samples).squeeze()
    print("samples.shape=%s" % str(samples.shape))
    stats = OrderedDict()
    for p in ["min", "max", "mean", "std"]:
        stats[p] = []
        stats[p] = getattr(samples, p)(axis=0)
        print("\nsamples.%s() =\n%s" % (p, str(stats[p])))

    return samples, stats


def load_priors_samples(iR, config=None):
    config = assert_config(config, return_plotter=False, MODE="PRIORS")
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.PRIORS_SAMPLES_FILE))
    return torch.load(priors_filepath(iR, config, filepath, extension))


def sim_res_filepath(iR, config, filepath=None, extension=None):
    return sim_filepath(iR, config, filepath, extension, config.SIM_RES_FILE)


def get_config(iR=None, **kwargs):

    # DEFAULT_ARGS = {  # TVB model:
    #     'I_s': 0.1,  # 0.085,
    #     'I_e': -0.35,
    #     "STIMULUS": 0.0,
    #     "STIMULUS_BASELINE": 1.0,
    #     "tau_w": 10.0,
    #     "I_w": -0.35,
    #     "G_w": 3.0,
    #     # TVB network:
    #     'G': 6.0,
    #     'FIC': 1.11,  # 2.0,
    #     'FIC_SPLIT': 0.31,  # 0.0,
    #     # Pathway gains:
    #     "PATHWAY_GAIN": 1,
    #     "TRIG_GAIN": 50.0, "MEDULLA_GAIN": 50.0, "CEREB_GAIN": 50.0,
    #     "TRIGS1_GAIN": 10.0, "MEDULLAS1_GAIN": 10.0, "CNS1_GAIN": 30.0,
    #     "CNM1_GAIN": 50.0,
    #     "M1S1_GAIN": 10.0,
    #     "M1FACIAL_GAIN": 50.0,  # 50.0,
    #     "FACIALTRIG_GAIN": 1.0,  # 50.0,
    #     "WHISKERS_GAIN": 50.0,
    #     # TVB <-> NEST Interface:
    #     "w_TVB_to_NEST": 35.0, "w_TVB_to_NEST_rest": 0.15,
    #     "MAX_RATES": {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
    #                   "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0},  # Hz
    #     # WORKFLOW:
    #     "NOISE": 1e-6,
    #     "SIMULATION_LENGTH": 2 ** 10 + 1.0,
    #     "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
    #     'output_folder': "", 'verbose': 1, 'plot_flag': True}

    # Get configuration
    if iR is not None:
        config = configure(MODE="PRIORS", plot_flag=False, verbose=0)[0]
        priors = dict(zip(config.PRIORS_PARAMS_NAMES, load_priors_samples(iR, config).numpy().squeeze()))
        print("PRIORS_%05d:\n%s" % (iR, str(priors)))
        kwargs.update(priors)
        kwargs["plot_flag"] = False
        verbose = 0
    else:
        verbose = kwargs.pop("verbose", 1)
    config, plotter = configure(verbose=0, SEED=iR, **kwargs)

    config.VERBOSE = verbose
    config.BOLD_PERIOD = None  # None, If None, BOLD will not be computed
    config.AFFERENT_MONITOR = False

    print(config.model_params)
    print(config)

    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config.__dict__, file, recurse=1)

    return config, plotter


def gain_run_plot(iR=0, **kwargs):

    config, plotter = get_config(iR, **kwargs)

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

    # Compute transient
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD

    if plotter:
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

    else:

        if isinstance(results, (list, tuple)):
            results = tvb_res_to_time_series(results, simulator, config=config, write_files=False)

        n_transient = int(np.ceil(transient / results["source_ts"].sample_period))

        source_ts = results["source_ts"][n_transient:, [0], config.TASKINDS]
        results["source_ts"] = source_ts
        taskinds = np.arange(results["source_ts"].shape[2]).astype("i")

        # Power Spectra and Coherence for M1 - S1 barrel field
        PSD, COH, f, pairs = \
            compute_selected_spectra_coherence(results["source_ts"], taskinds, results["source_ts"].sample_period,
                                               transient=0, nperseg=1024, ftarg=config.FREQS)
        results["PSD"] = PSD
        results["f"] = f
        results["COH"] = COH
        results["pairs"] = pairs

        TaskMetrics = compute_task_transfer_metrics(results["source_ts"], 0,
                                                    simulator.connectivity.region_labels[taskinds],
                                                    taskinds, config.THETA, config.GAMMA, config.FREQS,
                                                    Pxx_den=PSD, methods=(5, 2, 3), plot_flag=False)
        results["TaskMetrics"] = TaskMetrics

        dump_pickled_dict(results, sim_res_filepath(iR, config))

    return results, simulator, nest_network, config, inds


if __name__ == '__main__':
    # Example use:
    # $ python gain_run_plot.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
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

    gain_run_plot(**kwargs)
