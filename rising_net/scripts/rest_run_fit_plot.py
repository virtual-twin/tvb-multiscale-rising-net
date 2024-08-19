# coding: utf-8
import random
import time

import warnings
import glob
import pickle
import os
import shutil
from collections import OrderedDict

import dill
import numpy
from matplotlib import pyplot

from rising_net.scripts.base import assert_config, configure
from rising_net.scripts.filepaths import simres_filepath
from rising_net.scripts.tvb_script import run_workflow, load_connectome, prepare_connectome, build_connectivity, \
    build_model, build_simulator, simulate, plot_tvb, tvb_res_to_time_series
from rising_net.scripts.nest_script import build_NEST_network
from rising_net.scripts.tvb_nest_script import build_tvb_nest_interfaces, simulate_tvb_nest
from rising_net.scripts.sbi_script import build_priors, \
    sbi_estimate, sbi_train, sbi_infer, write_posterior, compute_diagnostics, write_posterior_samples, \
    load_posterior, load_posterior_samples, load_posterior_samples_all_runs
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from examples.plot_write_results import plot_write_spiking_network_results

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from tvb_multiscale.core.utils.file_utils import dump_pickled_dict


def get_config(iR=None, **kwargs):

    # DEFAULT_ARGS = {  # TVB model:
    #     'I_s': 0.1,
    #     'I_e': -0.35,
    #     'w_ie': -3.0,
    #     "tau_w": 10.0,
    #     "I_w": -0.35,
    #     "G_w": 0.0,
    #     # TVB network:
    #     'G': 6.0,
    #     'FIC': 1.11,  # 2.0,
    #     'FIC_SPLIT': 0.31,  # 0.0,
    #     # Pathway gains:
    #     "PATHWAY_GAIN": 0,
    #     "TRIG_GAIN": 50.0, "MEDULLA_GAIN": 50.0, "CEREB_GAIN": 50.0,
    #     "TRIGS1_GAIN": 10.0, "MEDULLAS1_GAIN": 10.0, "CNS1_GAIN": 30.0,
    #     "CNM1_GAIN": 50.0,
    #     "M1S1_GAIN": 10.0,
    #     "M1FACIAL_GAIN": 50.0,
    #     "FACIALTRIG_GAIN": 1.0,
    #     "WHISKERS_GAIN": 0.0,
    #     # TVB <-> NEST Interface:
    #     "w_TVB_to_NEST": 35.0, "w_TVB_to_NEST_rest": 0.15,
    #     "MAX_RATES": {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
    #                   "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0},  # Hz
    #     "NOISE": 1e-6,
    #     "SIMULATION_LENGTH": 2 ** 10 + 1.0,
    #     "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
    #     'output_folder': "", 'verbosity': 1, 'plot_flag': True}

    # Get configuration
    verbosity = kwargs.pop("verbosity", 1)
    kwargs['PATHWAY_GAIN'] = 0
    kwargs['WHISKER'] = 0.0
    kwargs["G_w"] = 5.0
    if iR is not None:
        config, plotter = configure(verbosity=0, SEED=int(iR), **kwargs)
    else:
        config, plotter = configure(verbosity=0, **kwargs)
    config.VERBOSITY = verbosity

    print(config.model_params)
    print(config)

    return config, plotter


def simres_filepath(iR, config, filepath=None, extension=None):
    return construct_filepath(iR, config, filepath, extension, config.SIM_RES_FILE)


def cosim_run_plot(iR=None, **kwargs):

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
        if config.VERBOSITY:
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

        # results_path = os.path.join(config.out.FOLDER_RES, 'results.pkl')
        # with open(results_path, 'wb') as handle:
        #    pickle.dump(results, handle)
        # print(results_path)
        # # results = pickle.load(results_path, 'rb'))  # to load results
        #
        # coherence_path = os.path.join(config.out.FOLDER_RES, 'coherence.pkl')
        # # Save coherence
        # CxyR = results["CxyR_M1_S1"]
        # fR = results["fR"]
        # CxyL = results["Cxyl_M1_S1"]
        # fL = results["fL"]
        # with open(coherence_path, 'wb') as handle:
        #     pickle.dump([CxyR, fR, fL, CxyL], handle)
        # print(coherence_path)

    # else:

    if "REST" in config.MODE:
        # Return only the M1<-> PSD fitting target
        if isinstance(results, (list, tuple)):
            PSD, PSD_target = compute_PSD_target_and_data(config, results[0], inds, transient,
                                                          write_files=True, plotter=None)
            results = {"PSD": PSD, "f": PSD_target['f'],
                       "regions": simulator.connectivity.regions[inds["m1s1brl"]]}
            dump_pickled_dict(results, simres_filepath(iR, config))
    else:
        # Return PSD and COH along the task pathay fitting targets:
        if isinstance(results, (list, tuple)):
            results = tvb_res_to_time_series(results, simulator, config=config, write_files=False)

        n_transient = int(np.ceil(transient / results["source_ts"].sample_period))

        source_ts = results["source_ts"][n_transient:, [0], config.TASKINDS]
        taskinds = np.arange(source_ts.shape[2]).astype("i")

        # Power Spectra and Coherence for M1 - S1 barrel field
        PSD, COH, f, pairs = \
            compute_selected_spectra_coherence(source_ts, taskinds, source_ts.sample_period,
                                               transient=0, nperseg=1024, ftarg=config.FREQS)
        results["PSD"] = PSD
        results["f"] = f
        results["COH"] = COH
        results["pairs"] = pairs
        results["regions"] = simulator.connectivity.regions[config.TASKINDS]
        del results["source_ts"]

        # TaskMetrics = compute_task_transfer_metrics(results["source_ts"], 0,
        #                                             simulator.connectivity.region_labels[taskinds],
        #                                             taskinds, config.THETA, config.GAMMA, config.FREQS,
        #                                             Pxx_den=PSD, methods=(5, 2, 3), plot_flag=False)
        # results["TaskMetrics"] = TaskMetrics

        dump_pickled_dict(results, simres_filepath(iR, config))

    return results, simulator, nest_network, config, inds


def loadsim_to_xarrays_fun(path):
    res = load_pickled_dict(path)
    return DataArray(res["PSD"], dims=["Region", "f"],
                     coords={"Region": res["regions"],
                             "f": res["f"]},
                     name="PSD")


def load_allsims_to_xarrays(config, **kwargs):
    config = assert_config(config, return_plotter=False, **kwargs)
    res_files = glob.glob(path)
    # TODO: Make this temporary hack more robust!!!:
    sims = []
    for res_file in res_files:
        sims.append(int(res_file.split("_")[-1].split(".pkl")[0]))
    sims = np.sort(sims)
    res = []
    failed = []
    ifailed = []
    for iiR, iR in enumerate(sims):
        try:
            res.append(loadsim_to_xarrays(simres_filepath(iR, config)))
        except Exception as e:
            failed.append(iR)
            ifailed.append(iiR)
            warnings.warn(str(e))
            continue
    sims = np.delete(sims, ifailed)
    res = concat(res, dim=Index(sims, name="Simulation"))

    nFailed = len(failed)
    if nFailed:
        warnings.warn("There are %d simulations that failed to provide a sample!:\n%s" % (nFailed, str(failed)))

    return PSDs, failed


def target_PSD_fun(config, target=None):
    # Load the target
    PSD_target = np.load(config.PSD_TARGET_PATH, allow_pickle=True).item()
    # If we are fitting for a connected network...
    # Duplicate the target for the two M1 regions (right, left) and the two S1 regions (right, left)
    #                                        right                       left
    return np.concatenate([PSD_target["PSD_M1_target"], PSD_target["PSD_M1_target"],  # M1
                           PSD_target["PSD_S1_target"], PSD_target["PSD_S1_target"]])  # S1


def sim_res_PSD_fun(sim_res, config):
    return sim_res.stack(Regions=("Region", "f")).values


def load_priors_and_simulations_for_sbi(iG=None, priors=None, priors_samples=None, sim_res=None, config=None, label=""):
    config = assert_config(config, return_plotter=False)
    if priors is None:
        priors = build_priors(config)
    # Load priors' samples if not given in the input:
    if priors_samples is None:
        # Load priors' samples
        priors_samples = load_priors_samples(config=config, iR=None, iG=iG, label=label)
    # Load priors' samples if not given in the input:
    if sim_res is None:
        # Load priors' samples
        sim_res = []
        for iR in range(config.N_SIMULATIONS):
            sim_res.append(load_pickled_dict(simres_filepath(iR, config))["PSD"])
        sim_res = torch.from_numpy(np.concatenate(sim_res).astype('float32'))
    return priors, priors_samples, sim_res


def load_posterior_samples_all_Gs(iGs=None, runs=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    samples = OrderedDict()
    if iGs is None:
        iGs = range(len(config.Gs))
    for iG in iGs:
        G = config.Gs[iG]
        try:
            if runs is False:
                samples[G] = load_posterior_samples(iG, None, label, config)
            else:
                samples[G] = load_posterior_samples_all_runs(iG, runs, label, config=config)
        except Exception as e:
            warnings.warn("Failed to load posterior samples for iG=%d, G=%g!\n%s" % (iG, G, str(e)))
    return samples


def get_train_test_samples_for_iG(iG, config, n_train_samples=None):
    priors, priors_samples, sim_res = load_priors_and_simulations_for_sbi(iG, config=config)
    n_samples = sim_res.shape[0]
    # Test samples are always going to be the LAST samples:
    n_test_samples = int(np.floor(config.TEST_SAMPLES_RATIO * n_samples))
    test_samples = priors_samples[-n_test_samples:]
    test_res = sim_res[-n_test_samples:]
    if n_train_samples is None:
        n_train_samples = n_samples - n_test_samples
    all_inds = list(range(n_samples))
    sampl_inds = random.sample(all_inds, n_train_samples)
    train_samples = priors_samples[sampl_inds]
    train_res = sim_res[sampl_inds]
    return train_samples, train_res, test_samples, test_res


def num_train_sample_to_label(nts, format="", config=None):
    if len(format) == 0:
        config = assert_config(config, return_plotter=False)
        format = config.N_TRAIN_SAMPLES_LABEL
    return format % nts


def sbi_train_for_iG(iG, config, iR=None, n_train_samples=None):
    tic = time.time()
    if config.VERBOSITY:
        if iR is None:
            run_str = ""
        else:
            run_str = ", iR=%d" % iR
        print("\nTraining network with %d samples for iG=%d%s!" % (n_train_samples, iG, run_str))
    train_samples, train_res, test_samples, test_res = get_train_test_samples_for_iG(iG, config, n_train_samples)
    label = num_train_sample_to_label(train_res.shape[0], format=config.N_TRAIN_SAMPLES_LABEL)
    # Train:
    priors = build_priors(config)
    posterior = sbi_train(priors, train_samples, train_res, config.VERBOSITY)

    if config.VERBOSITY:
        print("\nDone with training with all samples in %g sec!" % (time.time() - tic))
        print("\n\nFind results in %s!" % config.out.FOLDER_RES)
    return posterior, priors, train_samples, train_res, test_samples, test_res


def sbi_test_for_iG(iG, config,
                    iR=None, label="",
                    posterior=None, priors=None, test_samples=None, test_res=None):
    if posterior is None:
        posterior = load_posterior(iG, iR=iR, label=label, config=config)
    if test_samples is None or test_res is None:
        test_samples, test_res = get_train_test_samples_for_iG(iG, config)[-2:]
    if priors is None:
        priors = build_priors(config)
    if config.VERBOSITY:
        if iR is None:
            run_str = ""
        else:
            run_str = ", iR=%d" % iR
        print("\nTesting network for iG=%d%s by sampling %d posterior samples for %d testing samples!" %
              (iG, run_str, config.N_POSTERIOR_SAMPLES_PER_RUN, test_samples.shape[0]))
    samples_fit = None
    nts = len(test_samples)
    for iT, (ts, rs) in enumerate(zip(test_samples, test_res)):
        if config.VERBOSITY:
            print("\nTesting sample... %d/%d" % (iT+1, nts))
        posterior, posterior_samples, map = sbi_estimate(posterior, rs.numpy(), config.N_POSTERIOR_SAMPLES_PER_RUN,
                                                         config.VERBOSITY)
        # write_posterior(posterior, iG, iR=iR, label=label, config=config)
        diagnostics = compute_diagnostics(posterior_samples, config, priors, map, ts.numpy())
        samples_fit = write_posterior_samples(diagnostics, config, iG, iR, label,
                                              samples_fit=samples_fit, save_samples=False)
    return samples_fit


def sbi_train_and_test_for_iG(iG, config, iR=None, n_train_samples=None):
    tic = time.time()
    # Train:
    posterior, priors, train_samples, train_res, test_samples, test_res = \
        sbi_train_for_iG(iG, config, iR, n_train_samples)
    label = num_train_sample_to_label(train_res.shape[0], format=config.N_TRAIN_SAMPLES_LABEL)
    # Test:
    samples_fit = sbi_test_for_iG(iG, config, iR, label, posterior, priors, test_samples, test_res)
    if config.VERBOSITY:
        print("Done with fitting with all samples in %g sec!"  % (time.time() - tic))
    if config.VERBOSITY:
        print("\n\nFinished after %g sec!" % (time.time() - tic))
        print("\n\nFind results in %s!" % config.out.FOLDER_RES)
    return samples_fit


def plot_sbi_fit(config=None):
    FIGWIDTH = 15
    FIGHEIGHT_PER_PRIOR = 5
    RUNS_COLOR = 'k'
    LAST_RUN_COLOR = 'r'
    SAMPLES_MARKER_SIZE = 0.1
    MARKER_MEAN = 'o'
    MARKER_MAP = 'x'
    MARKER_SIZE = 5.0
    SAMPLES_ALPHA = 0.1
    RUNS_ALPHA = 0.5
    PLOT_RUNS = True
    PLOT_SAMPLES = True

    def plot_run(ax, G, map, mean, std, samples=None, is_last=False):
        color = RUNS_COLOR
        alpha = 1.0
        if is_last:
            color = LAST_RUN_COLOR
            alpha = RUNS_ALPHA
        if samples is not None:
            ax.plot([G] * len(samples), samples, marker=MARKER_MEAN,
                    markersize=SAMPLES_MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
                    linestyle='None', alpha=SAMPLES_ALPHA)
        ax.plot(G, mean, marker=MARKER_MEAN,
                markersize=MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
                linestyle='None', alpha=alpha)
        ax.plot(G, map, marker=MARKER_MAP,
                markersize=MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
                linestyle='None', alpha=alpha)
        ax.plot([G] * 2, [mean - std, mean + std], color=color, linestyle='-', linewidth=1, alpha=alpha)
        return ax

    def plot_G(ax, samples, iP):
        n_runs = len(samples['mean'])
        if PLOT_RUNS:
            iR_start = 0
        else:
            iR = n_runs - 1
        for iR in range(iR_start, n_runs):
            ax = plot_run(ax, samples['G'], samples['map'][iR][iP], samples['mean'][iR][iP], samples['std'][iR][iP],
                          samples=samples['samples'][iR][:, iP] if PLOT_SAMPLES else None, is_last=iR == n_runs - 1)
        return ax

    def plot_parameter(ax, iP, pname, samples, is_last=False):
        for G, sg in samples.items():
            ax = plot_G(ax, sg, iP)
        Gs = list(samples.keys())
        if is_last:
            ax.set_xticks(Gs)
            ax.set_xticklabels(Gs)
            ax.set_xlabel("G")
        ax.set_ylabel(pname)
        return ax

    config = assert_config(config, return_plotter=False)
    samples = load_posterior_samples_all_Gs(config)
    fig, axes = plt.subplots(config.n_priors, 1, figsize=(FIGWIDTH, FIGHEIGHT_PER_PRIOR * config.n_priors))
    axes = ensure_list(axes)
    for iP, ax in enumerate(axes):
        axes[iP] = plot_parameter(ax, iP, config.PRIORS_PARAMS_NAMES[iP], samples,
                                  is_last=iP == config.n_priors - 1)
    fig.tight_layout()
    if config.figures.SHOW_FLAG:
        fig.show()
    if config.figures.SAVE_FLAG:
        plt.savefig(config.SBI_FIT_PLOT_PATH)

    return fig, axes


def simulate_after_fitting_for_iG(iG, iR=None, label="", config=None,
                                  workflow_fun=None, model_params={}, FIC=None, FIC_SPLIT=None,
                                  plot_flag=True, **config_args):

    config = assert_config(config, return_plotter=False)
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)

    samples_fit = load_posterior_samples_all_runs(iG, iR, label, config=config)
    if iR is None:
        iR = slice(None)
    # Get the default values for the parameter except for G
    pvals = np.concatenate(samples_fit[config.OPT_RES_MODE][iR]).mean(axis=0)

    # Get G for this run:
    G = samples_fit.get("G", config.Gs[iG])

    # Get the default values for the parameter except for G
    params = config.model_params.copy()
    params['G'] = G
    # Set the posterior means or maps of the parameters:
    for pname, pval in zip(config.PRIORS_PARAMS_NAMES, pvals):
        if isinstance(pval, np.floating):
            np_pval = pval
        else:
            np_pval = pval.numpy()
        if pname == "FIC":
            config.FIC = np_pval
        elif pname == "FIC_SPLIT":
            config.FIC_SPLIT = np_pval
        else:
            params[pname] = np_pval
    if FIC is not None:
        config.FIC = FIC
    if FIC_SPLIT is not None:
        config.FIC_SPLIT = FIC_SPLIT
    # Run one simulation with the posterior means:
    if config.VERBOSITY:
        print("Simulating using the estimate of the %s of the parameters' posterior distribution!"
              % config.OPT_RES_MODE)
        print("params =\n", params)
        print("FIC=%g" % config.FIC)
        print("FIC_SPLIT=%g" % config.FIC_SPLIT)
    if workflow_fun is None:
        workflow_fun = run_workflow
    # Specify other parameters or overwrite some:
    params.update(model_params)
    if len(label):
        label = "_%s" % label
    outputs = workflow_fun(model_params=params, config=config,
                           output_folder="%s/G%g/STIM%0.2f_Is%0.2f_FIC%0.2f_FIC_SPLIT%0.2f%s" %
                                         (config.output_base, params['G'], params["STIMULUS"],
                                          params['I_s'], config.FIC, config.FIC_SPLIT, label),
                           plot_flag=plot_flag, **config_args)
    outputs["samples_fit"] = samples_fit
    return outputs


def ppt_batch_sim_res_filepath(iB, config, iG=None, filepath=None, extension=None):
    return batch_filepath(iB, config, iG, filepath, extension, config.PPT_BATCH_SIM_RES_FILE)


def write_ppt_batch_sim_res_to_file(sim_res, iB, iG=None, config=None):
    np.save(ppt_batch_sim_res_filepath(iB, assert_config(config, return_plotter=False), iG), sim_res, allow_pickle=True)


def write_ppt_batch_sim_res_to_file_per_iG(sim_res, iB, iG, config=None):
    write_ppt_batch_sim_res_to_file(sim_res, iB, iG, config)


def posterior_predictive_check_simulations_for_iG_iB(iB, iG, num_train_samples=None, iR=None,
                                                     workflow_fun=run_workflow, write_to_file=True, config=None):
    config = assert_config(config, return_plotter=False)
    if num_train_samples is not None:
        label = num_train_sample_to_label(num_train_samples, config.N_TRAIN_SAMPLES_LABEL)
    else:
        label = ""
    if iR is not None:
        iR = ensure_list(iR)
    samples_fit = load_posterior_samples_all_runs(iG, runs=iR, label=label, samples=None, config=config)
    samples = np.hstack(samples_fit["samples"])[0].copy()
    del samples_fit
    n_samples = samples.shape[0]
    # Split the total number of samples into N_PPT_SIM_BATCHES consecutive segments...
    n_possible_samples_per_batch = int(n_samples / config.N_PPT_SIM_BATCHES)
    # ...and choose the segment that corresponds to this batch iB:
    samples = samples[iB*n_possible_samples_per_batch:(iB+1)*n_possible_samples_per_batch]
    # Now choose N_PPT_SIMS_PER_BATCH randomly among the samples meant for this batch
    sampl_inds = random.sample(list(range(n_possible_samples_per_batch)), config.N_PPT_SIMS_PER_BATCH)
    samples = samples[sampl_inds]
    if write_to_file:
        write_to_file = write_ppt_batch_sim_res_to_file_per_iG
    return simulate_batch(iB, iG, samples, workflow_fun, write_to_file, config)


def read_ppt_batch_sim_res_from_file(iB, iG=None, config=None):
    return np.load(ppt_batch_sim_res_filepath(iB, assert_config(config, return_plotter=False), iG))


def read_ppt_batch_sim_res_from_file_for_iG(iB, iG, config=None):
    return read_ppt_batch_sim_res_from_file(iB, iG, config)


def read_all_ppt_batch_sim_res_files_for_iG(iG, config=None):
    config = assert_config(config, return_plotter=False)
    pptPSDs = []
    for iB in range(config.N_PPT_SIM_BATCHES):
        pptPSDs.append(read_ppt_batch_sim_res_from_file_for_iG(iB, iG, config))
    return pptPSDs


def plot_ppt_PSDs(iGs=None, config=None, confidence="5%", figsize=None):
    config = assert_config(config, return_plotter=False)

    if figsize is None:
        figsize = config.figures.DEFAULT_SIZE

    connectome, _, _, inds = load_connectome(config)
    inds = inds['m1s1brl']
    region_labels = connectome["region_labels"][inds]
    del connectome, inds

    PSD_target = np.load(config.PSD_TARGET_PATH, allow_pickle=True).item()
    f = PSD_target["f"]
    nfs = f.size
    PSD_target = [PSD_target["PSD_M1_target"], PSD_target["PSD_S1_target"]]*2

    if iGs is None:
        iGs = np.arange(len(config.Gs))

    def plot_ax(plot_fun, axes, axi, axj, finds, f, PSDs, PSDtarg, reg_lbl, conf=None):
        pfun = getattr(axes[axi, axj], plot_fun)
        pfun(f, PSDs[finds, 0], color='b', linewidth=1, linestyle='-', alpha=0.1, label="PSD samples")
        pfun(f, PSDs[finds, 1:], color='b', linewidth=1, linestyle='-', alpha=0.1)
        pfun(f, PSDtarg, color='r', linewidth=2, linestyle='-', label="PSD target")
        if conf is not None:
            pfun(f, conf[0][finds], color='k', linewidth=2, linestyle='-')
            pfun(f, conf[1][finds], color='k', linewidth=2, linestyle='-')
        if axi == 1:
            axes[axi, axj].set_xlabel('f (Hz)', fontsize=12)
        if axi == 1 and axj == 1:
            axes[axi, axj].legend(prop={'size': 12})
        axes[axi, axj].set_title(reg_lbl, fontsize=12)
        return axes

    conf = None
    for iG in ensure_list(iGs):
        PSDs = np.vstack(read_all_ppt_batch_sim_res_files_for_iG(iG, config)).T
        if confidence:
            if confidence == "std":
                conf = np.std(PSDs, axis=1)
                mean = np.mean(PSDs, axis=1)
                conf = np.array([mean-conf, mean+conf])
                del mean
            else:
                percent = float(confidence.split("%")[0])
                conf = np.percentile(PSDs, [percent, 100-percent], axis=1)
                del percent
        fig_lin, axes_lin = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        fig_lin.suptitle('Power spectral densities', fontsize=14)
        fig_semilog, axes_semilog = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        fig_semilog.suptitle('Logpower spectral densities', fontsize=14)
        for iR in range(4):
            finds = np.arange(nfs*iR, nfs*(iR+1)).astype("i")
            axi = int(iR > 1)
            axj = 1-np.mod(iR, 2)
            axes_lin = plot_ax("plot", axes_lin, axi, axj, finds, f, PSDs, PSD_target[iR],
                               region_labels[iR], conf=conf)
            axes_semilog = plot_ax("semilogy", axes_semilog, axi, axj,  finds, f, PSDs, PSD_target[iR],
                                    region_labels[iR], conf=conf)

    for fig, figname in zip([fig_lin, fig_semilog], ["pptPSDlin", "pptPSDsemilog"]):
        plt.figure(fig.number)
        if config.figures.SAVE_FLAG:
            plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, figname+".png"))
        if config.figures.SHOW_FLAG:
            plt.show()
    return fig_lin, axes_lin, fig_semilog, axes_semilog


def plot_diagnostic_for_iG(iG, diagnostic, config, num_train_samples=None, params=None, runs=None, confidence=True,
                           colors=['b', "g", "m"], marker='.', linestyle='-', title=True, xlabel=True, ylabel=True,
                           ax=None, figsize=None):

    if num_train_samples is None:
        num_train_samples = config.N_TRAIN_SAMPLES_LIST

    if params is None:
        params = config.PRIORS_PARAMS_NAMES

    if figsize is None:
        figsize = config.figures.DEFAULT_SIZE

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = None

    res = []
    for nts in num_train_samples:
        samples_fit = \
            load_posterior_samples_all_runs(iG, runs,
                                            label=num_train_sample_to_label(nts,
                                                                            format=config.N_TRAIN_SAMPLES_LABEL),
                                            config=config)
        res.append(np.concatenate(samples_fit[diagnostic]))
    res = np.stack(res)
    mean = np.mean(res, axis=1)
    err = None
    if confidence:
        if confidence == "std":
            std = np.std(res, axis=1)
            err = np.array([std]*2)
        else:
            percent = float(confidence.split("%")[0])
            err = np.percentile(res, [percent, 100-percent], axis=1)
            err[0] = mean - err[0]
            err[1] = err[1] - mean

    for iP, (param, col) in enumerate(zip(params, colors)):
        if err is not None:
            ax.errorbar(num_train_samples, mean[:, iP], yerr=err[:, :, iP], capsize=5,
                        color=col, marker=marker, markersize=5, linestyle=linestyle, linewidth=2,
                        label="%s" % param)
        else:
            ax.plot(num_train_samples, mean[:, iP],
                    color=col, marker=marker, markersize=5, linestyle=linestyle, linewidth=2,
                    label="%s" % param)
        if title:
            ax.set_title("G=%g" % config.Gs[iG], fontsize=14)
        if xlabel:
            ax.set_xlabel("N training samples", fontsize=14)
        if ylabel:
            ax.set_ylabel(diagnostic, fontsize=14)
        ax.legend(prop={'size': 14})

    if fig is None:
        return ax
    else:
        return ax, fig


def plot_all_together(config, iGs=None, diagnostics=["diff", "accuracy", "zscore_prior", "zscore", "shrinkage"],
                      params=None, num_train_samples=None, runs=None, confidence="5%",
                      colors=['b', "g", "m"], marker='.', linestyle='-', figsize=None):

    if iGs is None:
        iGs = list(range(len(config.Gs)))

    if num_train_samples is None:
        num_train_samples = config.N_TRAIN_SAMPLES_LIST

    if params is None:
        params = config.PRIORS_PARAMS_NAMES

    if figsize is None:
        figsize = config.figures.DEFAULT_SIZE

    figsize = np.array(figsize)
    nGs = len(iGs)
    nDs = len(diagnostics)
    figsize[0] = figsize[1] * nDs
    figsize[1] = figsize[0] * nGs
    figsize = tuple(figsize.tolist())

    fig, axes = plt.subplots(nrows=nDs, ncols=nGs, figsize=figsize)
    if nGs == 1 and nDs == 1:
        axes = np.array([[axes]])
    elif nDs == 1:
        axes = axes[np.newaxis]
    elif nGs == 1:
        axes = axes[:, np.newaxis]

    for iD, diagnostic in enumerate(diagnostics):
        for iiG, iG in enumerate(iGs):
            axes[iD, iiG] = plot_diagnostic_for_iG(iG, diagnostic, config, num_train_samples, params, runs,
                                                   confidence, colors, marker, linestyle,
                                                   title=False, xlabel=False, ylabel=False,
                                                   ax=axes[iD, iiG])
        if iD == 0:
            axes[iD, iiG].set_title("G=%g" % config.Gs[iG], fontsize=14)
        if iD == nDs-1:
            axes[iD, iiG].set_xlabel("N training samples", fontsize=14)
        if iiG == 0:
            axes[iD, iiG].set_ylabel(diagnostic, fontsize=14)

    plt.figure(fig.number)
    if config.figures.SAVE_FLAG:
        plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, "Diagnostics_%s.png" % "_".join(diagnostics)))
    if config.figures.SHOW_FLAG:
        plt.show()

    return fig, axes


# TODO: Figure this out!:
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
        if keyval[0] not in ["MODE", "plot_flag"]:
            key = float(keyval[1])
        else:
            key = keyval[1].split(" ")
            if keyval[0] == "MODE":
                ntests = len(key)
                if ntests == 1:
                    key = key[0]
            elif keyval[0] == "plot_flag":
                if keyval[1] == "False":
                    key = False
                else:
                    key = True
        kwargs[keyval[0]] = key

    cosim_run_plot(**kwargs)


# if __name__ == "__main__":
#     parser = args_parser("rest_run_fit_plot")
#     parser.add_argument('--script_id', '-scr',
#                         dest='script_id', metavar='script_id',
#                         type=int, required=False, default=1,  # nargs=1,
#                         help="Integer 0 or 1 (default) to select simulate_TVB_for_sbi_batch "
#                              "or sbi_infer_for_iG, respectively")
#     parser.add_argument('--iB', '-ib',
#                         dest='iB', metavar='iB',
#                         type=int, required=False, default=0,  # nargs=1,
#                         help="Batch integer indice. Default=0.")
#     parser.add_argument('--iG', '-ig',
#                         dest='iG', metavar='iG',
#                         type=int, required=False, default=-1,  # nargs=1,
#                         help="G values' integer indice. Default = -1 will be interpreted as None.")
#     parser.add_argument('--iR', '-ir',
#                         dest='iR', metavar='iR',
#                         type=int, required=False, default=-1,  # nargs=1,
#                         help="Run. Default = -1 will be interpreted as None.")
#     parser.add_argument('--num_train_samples', '-nts',
#                         dest='num_train_samples', metavar='num_train_samples',
#                         type=int, required=False, default=-1,  # nargs=1,
#                         help="Number of training samples. Default = -1 will be interpreted as None.")
#
#     args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
#     verbosity = args.get('verbosity', DEFAULT_ARGS['verbosity'])
#     if verbosity:
#         print("Running %s with arguments:\n" % parser.description)
#         print(parser_args, "\n")
#         print(args, "\n")
#     config = configure(**args)[0]
#     iR = parser_args.iR
#     if iR == -1:
#         iR = None
#     iG = parser_args.iG
#     if parser_args.script_id == 0:
#         if iG == -1:
#             iG = None
#         sim_res = simulate_TVB_for_sbi_batch(parser_args.iB, iG, config=config, write_to_file=True)
#     else:
#         if iG == -1:
#             raise ValueError("iG=-1 is not possible for running sbi_infer_for_iG!")
#         elif parser_args.script_id == 1:
#             samples_fit_Gs, results, fig, simulator, output_config = sbi_infer_for_iG(iG, config)
#         elif parser_args.script_id == 2:
#             simulate_after_fitting_for_iG(iG, iR=iR, config=config,
#                                           workflow_fun=None, model_params={}, FIC=None, FIC_SPLIT=None)
#         elif parser_args.script_id == 3:
#             num_train_samples = parser_args.num_train_samples
#             samples_fit = sbi_train_and_test_for_iG(iG, config, iR=iR, n_train_samples=num_train_samples)
#         elif parser_args.script_id == 4:
#             num_train_samples = parser_args.num_train_samples
#             samples_fit = sbi_test_for_iG(iG, config, iR,
#                                           num_train_sample_to_label(num_train_samples,
#                                                                     format=config.N_TRAIN_SAMPLES_LABEL),
#                                           posterior=None, priors=None, test_samples=None, test_res=None)
#         elif parser_args.script_id == 5:
#             if iG == -1:
#                 iG = None
#             num_train_samples = parser_args.num_train_samples
#             sim_res = \
#                 posterior_predictive_check_simulations_for_iG_iB(parser_args.iB,
#                                                                  num_train_samples=num_train_samples, iR=iR,
#                                                                  workflow_fun=run_workflow, write_to_file=True,
#                                                                  config=config)
#         else:
#             raise ValueError("Input argument script_id=%s is neither 0 for simulate_TVB_for_sbi_batch "
#                              "nor 1 for sbi_infer_for_iG!")
