# coding: utf-8

import sys
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
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot

from rising_net.scripts.base import assert_config, configure, DEFAULT_ARGS, args_parser, parse_args
from rising_net.scripts.filepaths import istr, get_path, simres_filepath, construct_filepath
from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR, iGstr, get_simres_folder_name, \
    simres_folder, get_stats_params, process_funcmode, find_all_folders, load_sims_to_xarrays_for_iP, \
    run_fit_plot_args_parser
from rising_net.scripts.tvb_script import run_workflow, load_connectome, prepare_connectome, build_connectivity, \
    build_model, build_simulator, simulate, plot_tvb, tvb_res_to_time_series, \
    compute_target_PSDs, compute_PSD_target_and_data
from rising_net.scripts.nest_script import build_NEST_network
from rising_net.scripts.tvb_nest_script import build_tvb_nest_interfaces, simulate_tvb_nest
from rising_net.scripts.sbi_script import build_priors, \
    load_train_params_samples, load_train_params_samples_selection, \
    sbi_estimate, sbi_train, sbi_infer, write_posterior, compute_diagnostics, write_posterior_samples, \
    load_posterior, load_posterior_samples, infer_workflow, infer_nRuns, \
    plot_stats, plot_best_stat_sims_params_target, correlation_distance
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *
from rising_net.scripts.utils import joinstr

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.data_structures_utils import narray_summary_info
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict, load_pickled_dict
from examples.plot_write_results import plot_write_spiking_network_results

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def rest_simres_filepath(config, iG=None, iP=None, iR=None, FUNCMODE="TRAINSIM",
                         label="", filepath=None, extension=None):
    folder = simres_folder(config, iG, iP, iR, FUNCMODE, label)
    return simres_filepath(config, config.SIM_RES_FILE, folder,
                           iR=iR, label="",
                           filepath=filepath, extension=extension)


def get_G(config, iG=None, **kwargs):
    if iG is None:
        G = kwargs.get("G", None)
        if G is not None:
            try:
                iG = config.Gs.tolist().index(G)
            except:
                iG = None
    else:
        kwargs["G"] = config.Gs[int(iG)]
    return iG, kwargs


# iP: parameter sample index
# iR: simulation repetition and noise seed index
def get_config(iG=None, iP=None, iR=None, FUNCMODE="SIM", fitlabel="", iF=None,
               # parameters_iR=None, parameters_label, parameters_filepath=None, parameters_filepath_ext=None,
               **kwargs):

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

    # Make sure we work in REST condition:
    kwargs['PATHWAY_GAIN'] = 0
    kwargs['WHISKER'] = 0.0
    kwargs["G_w"] = 5.0
    MODE = kwargs.pop("MODE", "")  # make sure we don't overshadow MODE
    if "REST" not in MODE.upper():
        MODE = joinstr(["REST", MODE])

    verbosity = kwargs.pop("verbosity", 1)
    config, plotter = configure(MODE=MODE, verbosity=0, **kwargs)

    iG, kwargs = get_G(config, iG=iG, **kwargs)

    iRpath, iR, iP, params, params_string, fitlabel, kwargs = \
        process_funcmode(FUNCMODE, MODE, config, verbosity, iP, iR, iF, iG, fitlabel, **kwargs)

    if "SIM" in FUNCMODE:
        kwargs["output_folder"] = os.path.dirname(
            os.path.dirname(
                rest_simres_filepath(config, iG, iP, iRpath, FUNCMODE, fitlabel)))
    config, plotter = configure(MODE=MODE, SEED=int(iR), verbosity=verbosity, **kwargs)

    print(config.model_params)

    return config, plotter


def load_sims_PSD_to_xarrays(folder, config, iR=None, resstr=RESSTR, **kwargs):  # measures = "PSD" not really needed!
    path = construct_filepath(os.path.join(folder, resstr), default_filename=config.SIM_RES_FILE, iR=iR)
    res = load_pickled_dict(path)
    # TODO: Remove this transitory hack:
    f = res.get("f", np.arange(5.0, 48.0, 1.0))
    regions = res.get("regions",
                      np.array(['Right Primary motor area',
                                'Left Primary motor area',
                                'Right Primary somatosensory area, barrel field',
                                'Left Primary somatosensory area, barrel field']))
    indf = pd.Index(f, name='f')
    indr = pd.Index(regions, name='Regions')
    indPSD = pd.MultiIndex.from_product([indr, indf], names=[indf.name, indr.name])
    name = joinstr(indPSD.names, " - ")
    # To unravel index:
    # PSD.unstack(PSD.dims[0]).shape = (nregs, nfreqs)
    return {"PSD": xr.DataArray(res["PSD"], dims=[name], coords={name: indPSD}, name="PSD: %s" % path)}


def load_sims_to_xarrays_for_iG(path=None, config=None, iG=None, iP=None, iR=None,
                                average_repetitions=True, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if path is None:
        path = config.out.FOLDER_RES
    if iG is None:
        iG = find_all_folders(path, igstr)
    else:
        iG = np.sort(ensure_list(iG)).tolist()
    res = []
    iPs = []
    if len(iG):
        for iiG in iG:
            res_ig, iP_ig = load_sims_to_xarrays_for_iP(load_sims_PSD_to_xarrays, "PSD",
                    os.path.join(path, iGstr(iiG, Ngs=len(config.Gs), igstr=igstr)),
                    config, iP=iP, iR=iR,
                    average_repetitions=average_repetitions,
                    folderstr=folderstr, resstr=resstr)
            res.append(res_ig["PSD"])
            iPs.append(iP_ig)
        res = xr.concat(res, dim=pd.Index(iG, name="Global coupling scaling parameter (G) index iG"))
        if len(iG) == 1:
            res = res.squeeze()
            res.name = path + ", G index: %d" % iG[0]
        else:
            res.name = path + ", G indices: %s" % \
                       list(narray_summary_info(np.array(iG), omit_shape=True).values())[0]
    return res, iG, iPs


def target_PSD_fun(config, target=None):
    # Load, interpolate and normalize Popa 2013 m1 and s1 power spectra:
    psd_m1_target, psd_s1_target = compute_target_PSDs(config)
    # If we are fitting for a connected network...
    # Duplicate the target for the two M1 regions (right, left) and the two S1 regions (right, left)
    #                                        right                       left
    return np.concatenate([psd_m1_target, psd_m1_target,  # M1
                           psd_s1_target, psd_s1_target])  # S1


def load_priors_target_and_sims_for_sbi_for_iG(iG,
                                               priors=None, train_params_samples=None, sim_res=None, sim_res_path=None,
                                               target=None,
                                               config=None,
                                               igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if sim_res_path is None:
        sim_res_path = get_path(config, folder=config.TRAIN_SIMS_FOLDER)
    # Rebuild priors if not provided in the input:
    if priors is None:
        priors = build_priors(config)
    # Load training parameters' samples if not provided in the input:
    if train_params_samples is None:
        train_params_samples = load_train_params_samples(config,
                                                         # iR=parameters_iR,
                                                         # label=parameters_label,
                                                         # filepath=parameters_filepath,
                                                         # extension=parameters_filepath_ext
                                                         ).numpy().squeeze().astype('float32')
    # Load training simulation results if not provided in the input:
    if sim_res is None:
        # Load priors' samples
        # By default, we load all parameters and all simulation repetitions and we average across repetitions.
        sim_res = \
            load_sims_to_xarrays_for_iG(path=sim_res_path, config=config,
                                        iG=iG, iP=None, iR=None, average_repetitions=True,
                                        igstr=igstr, folderstr=folderstr, resstr=resstr)[0].values.astype('float32')
    if target is None:
        target = target_PSD_fun(config)
    return priors, train_params_samples, sim_res, target


def infer_workflow_for_iG(iG,
                          priors=None, train_params_samples=None, sim_res=None, sim_res_path=None,
                          target=None,
                          config=None,
                          igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                          ground_truth=None,
                          label="", n_samples_per_run=None, measure_labels=None,
                          results=None, iR=None, save_samples=True, plot_flag=True, verbosity=None):
    priors, train_params_samples, sim_res, target = \
        load_priors_target_and_sims_for_sbi_for_iG(iG,
                                                   priors, train_params_samples, sim_res, sim_res_path, target,
                                                   config,
                                                   igstr, folderstr, resstr)
    label = joinstr([label, iGstr(iG, Ngs=len(config.Gs))], igstr=igstr)
    return infer_workflow(train_params_samples, sim_res, priors, target, ground_truth, config,
                          label, n_samples_per_run, measure_labels,
                          results, iR, save_samples, plot_flag, verbosity)


def infer_nRuns_for_iG(iG,
                       priors=None, train_params_samples=None, sim_res=None, sim_res_path=None,
                       target=None,
                       config=None,
                       igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                       ground_truth=None,
                       label="", n_samples_per_run=None, measure_labels=None,
                       save_samples=True, plot_flag=True, verbosity=None):
    priors, train_params_samples, sim_res, target = \
        load_priors_target_and_sims_for_sbi_for_iG(iG,
                                                   priors, train_params_samples, sim_res, sim_res_path, target,
                                                   config,
                                                   igstr, folderstr, resstr)
    label = joinstr([label, iGstr(iG, Ngs=len(config.Gs))], igstr=igstr)
    return infer_nRuns(train_params_samples, sim_res, priors, target,
                       ground_truth, config,
                       label, n_samples_per_run, measure_labels,
                       save_samples, plot_flag, verbosity)


def load_stat_sims_for_iG(iG, stat="PPC", label="", sim_res_path=None, iP=None, config=None,
                          igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if sim_res_path is None:
        sim_res_path = get_path(config, folder=get_simres_folder_name(config, "%sSIM" % stat.upper()))
    if len(label):
        sim_res_path = os.path.join(sim_res_path, label)
    # Load training simulation results:
    # By default, we load all parameters and all simulation repetitions and we average across repetitions.
    sim_res, iG, iPs = \
        load_sims_to_xarrays_for_iG(path=sim_res_path, config=config,
                                    iG=iG, iP=iP, iR=None, average_repetitions=True,
                                    igstr=igstr, folderstr=folderstr, resstr=resstr)
    if sim_res.ndim < 2:
        sim_res = sim_res.expand_dims(dim=None, axis=0, create_index_for_new_dim=True, Simulations=np.array([0]))
    if len(iG) == 1:
        iG = iG[0]
    if len(iPs) == 1:
        iPs = iPs[0]
    return sim_res, iG, iPs


def load_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="", sim_res_path=None, config=None,
                                        target=None,
                                        igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    sim_res, iG, iP = load_stat_sims_for_iG(iG, stat=stat, label=label, sim_res_path=sim_res_path, iP=iP,
                                            config=config,  igstr=igstr, folderstr=folderstr, resstr=resstr)
    sim_res = sim_res.values.astype('float32')
    params, fitlabel, params_string = \
        get_stats_params(config, stat=stat, FUNCMODE=None, iG=iG, iP=iP, iF=iF, fitlabel=label)

    if target is None:
        target = target_PSD_fun(config)
    return sim_res, iP, target, params


def load_and_plot_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="",
                                                 sim_res_path=None, config=None,
                                                 target=None,
                                                 igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                                                 measure_labels=None):
    config = assert_config(config, return_plotter=False)
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_iG(iG, stat, iF, iP, label, sim_res_path, config,
                                            target, igstr, folderstr, resstr)
    params_vals = np.array(list(params.values())).T
    if params_vals.ndim < 2:
        params_vals = params_vals[np.newaxis]
    return plot_stats(sim_res, stat, target, params_vals,
                      joinstr([iGstr(iG, Ngs=len(config.Gs), igstr=igstr), label]), measure_labels, config)


def load_and_plot_best_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="",
                                                      sim_res_path=None, config=None,
                                                      target=None, target_dist_fun=correlation_distance, Nbest=None,
                                                      igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                                                      measure_labels=None):
    config = assert_config(config, return_plotter=False)
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_iG(iG, stat, iF, iP, label, sim_res_path, config,
                                            target, igstr, folderstr, resstr)
    return plot_best_stat_sims_params_target(sim_res, target, stat, params=np.array(list(params.values())).T,
                                             label=joinstr([iGstr(iG, Ngs=len(config.Gs), igstr=igstr), label]),
                                             target_dist_fun=target_dist_fun, Nbest=Nbest,
                                             measure_labels=measure_labels, config=config)


# -------------------------------------------- TODO: REMOVE THESE!: ----------------------------------------------------
#
#
# def get_train_test_samples_for_iG(iG, config, n_train_samples=None):
#     priors, priors_samples, sim_res = load_priors_and_simulations_for_sbi(iG, config=config)
#     n_samples = sim_res.shape[0]
#     # Test samples are always going to be the LAST samples:
#     n_test_samples = int(np.floor(config.TEST_SAMPLES_RATIO * n_samples))
#     test_samples = priors_samples[-n_test_samples:]
#     test_res = sim_res[-n_test_samples:]
#     if n_train_samples is None:
#         n_train_samples = n_samples - n_test_samples
#     all_inds = list(range(n_samples))
#     sampl_inds = random.sample(all_inds, n_train_samples)
#     train_samples = priors_samples[sampl_inds]
#     train_res = sim_res[sampl_inds]
#     return train_samples, train_res, test_samples, test_res
#
#
# def num_train_sample_to_label(nts, format="", config=None):
#     if len(format) == 0:
#         config = assert_config(config, return_plotter=False)
#         format = config.N_TRAIN_SAMPLES_LABEL
#     return format % nts
#
#
# def sbi_train_for_iG(iG, config, iR=None, n_train_samples=None):
#     tic = time.time()
#     if config.VERBOSITY:
#         if iR is None:
#             run_str = ""
#         else:
#             run_str = ", iR=%d" % iR
#         print("\nTraining network with %d samples for iG=%d%s!" % (n_train_samples, iG, run_str))
#     train_samples, train_res, test_samples, test_res = get_train_test_samples_for_iG(iG, config, n_train_samples)
#     label = num_train_sample_to_label(train_res.shape[0], format=config.N_TRAIN_SAMPLES_LABEL)
#     # Train:
#     priors = build_priors(config)
#     posterior = sbi_train(priors, train_samples, train_res, config.VERBOSITY)
#
#     if config.VERBOSITY:
#         print("\nDone with training with all samples in %g sec!" % (time.time() - tic))
#         print("\n\nFind results in %s!" % config.out.FOLDER_RES)
#     return posterior, priors, train_samples, train_res, test_samples, test_res
#
#
# def sbi_test_for_iG(iG, config,
#                     iR=None, label="",
#                     posterior=None, priors=None, test_samples=None, test_res=None):
#     if posterior is None:
#         posterior = load_posterior(iG, iR=iR, label=label, config=config)
#     if test_samples is None or test_res is None:
#         test_samples, test_res = get_train_test_samples_for_iG(iG, config)[-2:]
#     if priors is None:
#         priors = build_priors(config)
#     if config.VERBOSITY:
#         if iR is None:
#             run_str = ""
#         else:
#             run_str = ", iR=%d" % iR
#         print("\nTesting network for iG=%d%s by sampling %d posterior samples for %d testing samples!" %
#               (iG, run_str, config.N_POSTERIOR_SAMPLES_PER_RUN, test_samples.shape[0]))
#     samples_fit = None
#     nts = len(test_samples)
#     for iT, (ts, rs) in enumerate(zip(test_samples, test_res)):
#         if config.VERBOSITY:
#             print("\nTesting sample... %d/%d" % (iT+1, nts))
#         posterior, posterior_samples, map = sbi_estimate(posterior, rs.numpy(), config.N_POSTERIOR_SAMPLES_PER_RUN,
#                                                          config.VERBOSITY)
#         # write_posterior(posterior, iG, iR=iR, label=label, config=config)
#         diagnostics = compute_diagnostics(posterior_samples, config, priors, map, ts.numpy())
#         samples_fit = write_posterior_samples(diagnostics, config, iG, iR, label,
#                                               samples_fit=samples_fit, save_samples=False)
#     return samples_fit
#
#
# def sbi_train_and_test_for_iG(iG, config, iR=None, n_train_samples=None):
#     tic = time.time()
#     # Train:
#     posterior, priors, train_samples, train_res, test_samples, test_res = \
#         sbi_train_for_iG(iG, config, iR, n_train_samples)
#     label = num_train_sample_to_label(train_res.shape[0], format=config.N_TRAIN_SAMPLES_LABEL)
#     # Test:
#     samples_fit = sbi_test_for_iG(iG, config, iR, label, posterior, priors, test_samples, test_res)
#     if config.VERBOSITY:
#         print("Done with fitting with all samples in %g sec!"  % (time.time() - tic))
#     if config.VERBOSITY:
#         print("\n\nFinished after %g sec!" % (time.time() - tic))
#         print("\n\nFind results in %s!" % config.out.FOLDER_RES)
#     return samples_fit
#
#
# def plot_sbi_fit(config=None):
#     FIGWIDTH = 15
#     FIGHEIGHT_PER_PRIOR = 5
#     RUNS_COLOR = 'k'
#     LAST_RUN_COLOR = 'r'
#     SAMPLES_MARKER_SIZE = 0.1
#     MARKER_MEAN = 'o'
#     MARKER_MAP = 'x'
#     MARKER_SIZE = 5.0
#     SAMPLES_ALPHA = 0.1
#     RUNS_ALPHA = 0.5
#     PLOT_RUNS = True
#     PLOT_SAMPLES = True
#
#     def plot_run(ax, G, map, mean, std, samples=None, is_last=False):
#         color = RUNS_COLOR
#         alpha = 1.0
#         if is_last:
#             color = LAST_RUN_COLOR
#             alpha = RUNS_ALPHA
#         if samples is not None:
#             ax.plot([G] * len(samples), samples, marker=MARKER_MEAN,
#                     markersize=SAMPLES_MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
#                     linestyle='None', alpha=SAMPLES_ALPHA)
#         ax.plot(G, mean, marker=MARKER_MEAN,
#                 markersize=MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
#                 linestyle='None', alpha=alpha)
#         ax.plot(G, map, marker=MARKER_MAP,
#                 markersize=MARKER_SIZE, markeredgecolor=color, markerfacecolor=color,
#                 linestyle='None', alpha=alpha)
#         ax.plot([G] * 2, [mean - std, mean + std], color=color, linestyle='-', linewidth=1, alpha=alpha)
#         return ax
#
#     def plot_G(ax, samples, iP):
#         n_runs = len(samples['mean'])
#         if PLOT_RUNS:
#             iR_start = 0
#         else:
#             iR = n_runs - 1
#         for iR in range(iR_start, n_runs):
#             ax = plot_run(ax, samples['G'], samples['map'][iR][iP], samples['mean'][iR][iP], samples['std'][iR][iP],
#                           samples=samples['samples'][iR][:, iP] if PLOT_SAMPLES else None, is_last=iR == n_runs - 1)
#         return ax
#
#     def plot_parameter(ax, iP, pname, samples, is_last=False):
#         for G, sg in samples.items():
#             ax = plot_G(ax, sg, iP)
#         Gs = list(samples.keys())
#         if is_last:
#             ax.set_xticks(Gs)
#             ax.set_xticklabels(Gs)
#             ax.set_xlabel("G")
#         ax.set_ylabel(pname)
#         return ax
#
#     config = assert_config(config, return_plotter=False)
#     samples = load_posterior_samples_all_Gs(config)
#     fig, axes = plt.subplots(config.n_priors, 1, figsize=(FIGWIDTH, FIGHEIGHT_PER_PRIOR * config.n_priors))
#     axes = ensure_list(axes)
#     for iP, ax in enumerate(axes):
#         axes[iP] = plot_parameter(ax, iP, config.PRIORS_PARAMS_NAMES[iP], samples,
#                                   is_last=iP == config.n_priors - 1)
#     fig.tight_layout()
#     if config.figures.SHOW_FLAG:
#         fig.show()
#     if config.figures.SAVE_FLAG:
#         plt.savefig(config.SBI_FIT_PLOT_PATH)
#
#     return fig, axes
#
#
# def simulate_after_fitting_for_iG(iG, iR=None, label="", config=None,
#                                   workflow_fun=None, model_params={}, FIC=None, FIC_SPLIT=None,
#                                   plot_flag=True, **config_args):
#
#     config = assert_config(config, return_plotter=False)
#     with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
#         dill.dump(config, file, recurse=1)
#
#     samples_fit = load_posterior_samples_all_runs(iG, iR, label, config=config)
#     if iR is None:
#         iR = slice(None)
#     # Get the default values for the parameter except for G
#     pvals = np.concatenate(samples_fit[config.OPT_RES_MODE][iR]).mean(axis=0)
#
#     # Get G for this run:
#     G = samples_fit.get("G", config.Gs[iG])
#
#     # Get the default values for the parameter except for G
#     params = config.model_params.copy()
#     params['G'] = G
#     # Set the posterior means or maps of the parameters:
#     for pname, pval in zip(config.PRIORS_PARAMS_NAMES, pvals):
#         if isinstance(pval, np.floating):
#             np_pval = pval
#         else:
#             np_pval = pval.numpy()
#         if pname == "FIC":
#             config.FIC = np_pval
#         elif pname == "FIC_SPLIT":
#             config.FIC_SPLIT = np_pval
#         else:
#             params[pname] = np_pval
#     if FIC is not None:
#         config.FIC = FIC
#     if FIC_SPLIT is not None:
#         config.FIC_SPLIT = FIC_SPLIT
#     # Run one simulation with the posterior means:
#     if config.VERBOSITY:
#         print("Simulating using the estimate of the %s of the parameters' posterior distribution!"
#               % config.OPT_RES_MODE)
#         print("params =\n", params)
#         print("FIC=%g" % config.FIC)
#         print("FIC_SPLIT=%g" % config.FIC_SPLIT)
#     if workflow_fun is None:
#         workflow_fun = run_workflow
#     # Specify other parameters or overwrite some:
#     params.update(model_params)
#     if len(label):
#         label = "_%s" % label
#     outputs = workflow_fun(model_params=params, config=config,
#                            output_folder="%s/G%g/STIM%0.2f_Is%0.2f_FIC%0.2f_FIC_SPLIT%0.2f%s" %
#                                          (config.output_base, params['G'], params["STIMULUS"],
#                                           params['I_s'], config.FIC, config.FIC_SPLIT, label),
#                            plot_flag=plot_flag, **config_args)
#     outputs["samples_fit"] = samples_fit
#     return outputs
#
#
# def ppt_batch_sim_res_filepath(iB, config, iG=None, filepath=None, extension=None):
#     return batch_filepath(iB, config, iG, filepath, extension, config.PPT_BATCH_SIM_RES_FILE)
#
#
# def write_ppt_batch_sim_res_to_file(sim_res, iB, iG=None, config=None):
#     np.save(ppt_batch_sim_res_filepath(iB, assert_config(config, return_plotter=False), iG), sim_res, allow_pickle=True)
#
#
# def write_ppt_batch_sim_res_to_file_per_iG(sim_res, iB, iG, config=None):
#     write_ppt_batch_sim_res_to_file(sim_res, iB, iG, config)
#
#
# def posterior_predictive_check_simulations_for_iG_iB(iB, iG, num_train_samples=None, iR=None,
#                                                      workflow_fun=run_workflow, write_to_file=True, config=None):
#     config = assert_config(config, return_plotter=False)
#     if num_train_samples is not None:
#         label = num_train_sample_to_label(num_train_samples, config.N_TRAIN_SAMPLES_LABEL)
#     else:
#         label = ""
#     if iR is not None:
#         iR = ensure_list(iR)
#     samples_fit = load_posterior_samples_all_runs(iG, runs=iR, label=label, samples=None, config=config)
#     samples = np.hstack(samples_fit["samples"])[0].copy()
#     del samples_fit
#     n_samples = samples.shape[0]
#     # Split the total number of samples into N_PPT_SIM_BATCHES consecutive segments...
#     n_possible_samples_per_batch = int(n_samples / config.N_PPT_SIM_BATCHES)
#     # ...and choose the segment that corresponds to this batch iB:
#     samples = samples[iB*n_possible_samples_per_batch:(iB+1)*n_possible_samples_per_batch]
#     # Now choose N_PPT_SIMS_PER_BATCH randomly among the samples meant for this batch
#     sampl_inds = random.sample(list(range(n_possible_samples_per_batch)), config.N_PPT_SIMS_PER_BATCH)
#     samples = samples[sampl_inds]
#     if write_to_file:
#         write_to_file = write_ppt_batch_sim_res_to_file_per_iG
#     return simulate_batch(iB, iG, samples, workflow_fun, write_to_file, config)
#
#
# def read_ppt_batch_sim_res_from_file(iB, iG=None, config=None):
#     return np.load(ppt_batch_sim_res_filepath(iB, assert_config(config, return_plotter=False), iG))
#
#
# def read_ppt_batch_sim_res_from_file_for_iG(iB, iG, config=None):
#     return read_ppt_batch_sim_res_from_file(iB, iG, config)
#
#
# def read_all_ppt_batch_sim_res_files_for_iG(iG, config=None):
#     config = assert_config(config, return_plotter=False)
#     pptPSDs = []
#     for iB in range(config.N_PPT_SIM_BATCHES):
#         pptPSDs.append(read_ppt_batch_sim_res_from_file_for_iG(iB, iG, config))
#     return pptPSDs
#
#
# def plot_ppt_PSDs(iGs=None, config=None, confidence="5%", figsize=None):
#     config = assert_config(config, return_plotter=False)
#
#     if figsize is None:
#         figsize = config.figures.DEFAULT_SIZE
#
#     connectome, _, _, inds = load_connectome(config)
#     inds = inds['m1s1brl']
#     region_labels = connectome["region_labels"][inds]
#     del connectome, inds
#
#     PSD_target = np.load(config.PSD_TARGET_PATH, allow_pickle=True).item()
#     f = PSD_target["f"]
#     nfs = f.size
#     PSD_target = [PSD_target["PSD_M1_target"], PSD_target["PSD_S1_target"]]*2
#
#     if iGs is None:
#         iGs = np.arange(len(config.Gs))
#
#     def plot_ax(plot_fun, axes, axi, axj, finds, f, PSDs, PSDtarg, reg_lbl, conf=None):
#         pfun = getattr(axes[axi, axj], plot_fun)
#         pfun(f, PSDs[finds, 0], color='b', linewidth=1, linestyle='-', alpha=0.1, label="PSD samples")
#         pfun(f, PSDs[finds, 1:], color='b', linewidth=1, linestyle='-', alpha=0.1)
#         pfun(f, PSDtarg, color='r', linewidth=2, linestyle='-', label="PSD target")
#         if conf is not None:
#             pfun(f, conf[0][finds], color='k', linewidth=2, linestyle='-')
#             pfun(f, conf[1][finds], color='k', linewidth=2, linestyle='-')
#         if axi == 1:
#             axes[axi, axj].set_xlabel('f (Hz)', fontsize=12)
#         if axi == 1 and axj == 1:
#             axes[axi, axj].legend(prop={'size': 12})
#         axes[axi, axj].set_title(reg_lbl, fontsize=12)
#         return axes
#
#     conf = None
#     for iG in ensure_list(iGs):
#         PSDs = np.vstack(read_all_ppt_batch_sim_res_files_for_iG(iG, config)).T
#         if confidence:
#             if confidence == "std":
#                 conf = np.std(PSDs, axis=1)
#                 mean = np.mean(PSDs, axis=1)
#                 conf = np.array([mean-conf, mean+conf])
#                 del mean
#             else:
#                 percent = float(confidence.split("%")[0])
#                 conf = np.percentile(PSDs, [percent, 100-percent], axis=1)
#                 del percent
#         fig_lin, axes_lin = plt.subplots(nrows=2, ncols=2, figsize=figsize)
#         fig_lin.suptitle('Power spectral densities', fontsize=14)
#         fig_semilog, axes_semilog = plt.subplots(nrows=2, ncols=2, figsize=figsize)
#         fig_semilog.suptitle('Logpower spectral densities', fontsize=14)
#         for iR in range(4):
#             finds = np.arange(nfs*iR, nfs*(iR+1)).astype("i")
#             axi = int(iR > 1)
#             axj = 1-np.mod(iR, 2)
#             axes_lin = plot_ax("plot", axes_lin, axi, axj, finds, f, PSDs, PSD_target[iR],
#                                region_labels[iR], conf=conf)
#             axes_semilog = plot_ax("semilogy", axes_semilog, axi, axj,  finds, f, PSDs, PSD_target[iR],
#                                     region_labels[iR], conf=conf)
#
#     for fig, figname in zip([fig_lin, fig_semilog], ["pptPSDlin", "pptPSDsemilog"]):
#         plt.figure(fig.number)
#         if config.figures.SAVE_FLAG:
#             plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, figname+".png"))
#         if config.figures.SHOW_FLAG:
#             plt.show()
#     return fig_lin, axes_lin, fig_semilog, axes_semilog
#
#
# def plot_diagnostic_for_iG(iG, diagnostic, config, num_train_samples=None, params=None, runs=None, confidence=True,
#                            colors=['b', "g", "m"], marker='.', linestyle='-', title=True, xlabel=True, ylabel=True,
#                            ax=None, figsize=None):
#
#     if num_train_samples is None:
#         num_train_samples = config.N_TRAIN_SAMPLES_LIST
#
#     if params is None:
#         params = config.PRIORS_PARAMS_NAMES
#
#     if figsize is None:
#         figsize = config.figures.DEFAULT_SIZE
#
#     if ax is None:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#     else:
#         fig = None
#
#     res = []
#     for nts in num_train_samples:
#         samples_fit = \
#             load_posterior_samples_all_runs(iG, runs,
#                                             label=num_train_sample_to_label(nts,
#                                                                             format=config.N_TRAIN_SAMPLES_LABEL),
#                                             config=config)
#         res.append(np.concatenate(samples_fit[diagnostic]))
#     res = np.stack(res)
#     mean = np.mean(res, axis=1)
#     err = None
#     if confidence:
#         if confidence == "std":
#             std = np.std(res, axis=1)
#             err = np.array([std]*2)
#         else:
#             percent = float(confidence.split("%")[0])
#             err = np.percentile(res, [percent, 100-percent], axis=1)
#             err[0] = mean - err[0]
#             err[1] = err[1] - mean
#
#     for iP, (param, col) in enumerate(zip(params, colors)):
#         if err is not None:
#             ax.errorbar(num_train_samples, mean[:, iP], yerr=err[:, :, iP], capsize=5,
#                         color=col, marker=marker, markersize=5, linestyle=linestyle, linewidth=2,
#                         label="%s" % param)
#         else:
#             ax.plot(num_train_samples, mean[:, iP],
#                     color=col, marker=marker, markersize=5, linestyle=linestyle, linewidth=2,
#                     label="%s" % param)
#         if title:
#             ax.set_title("G=%g" % config.Gs[iG], fontsize=14)
#         if xlabel:
#             ax.set_xlabel("N training samples", fontsize=14)
#         if ylabel:
#             ax.set_ylabel(diagnostic, fontsize=14)
#         ax.legend(prop={'size': 14})
#
#     if fig is None:
#         return ax
#     else:
#         return ax, fig
#
#
# def plot_all_together(config, iGs=None, diagnostics=["diff", "accuracy", "zscore_prior", "zscore", "shrinkage"],
#                       params=None, num_train_samples=None, runs=None, confidence="5%",
#                       colors=['b', "g", "m"], marker='.', linestyle='-', figsize=None):
#
#     if iGs is None:
#         iGs = list(range(len(config.Gs)))
#
#     if num_train_samples is None:
#         num_train_samples = config.N_TRAIN_SAMPLES_LIST
#
#     if params is None:
#         params = config.PRIORS_PARAMS_NAMES
#
#     if figsize is None:
#         figsize = config.figures.DEFAULT_SIZE
#
#     figsize = np.array(figsize)
#     nGs = len(iGs)
#     nDs = len(diagnostics)
#     figsize[0] = figsize[1] * nDs
#     figsize[1] = figsize[0] * nGs
#     figsize = tuple(figsize.tolist())
#
#     fig, axes = plt.subplots(nrows=nDs, ncols=nGs, figsize=figsize)
#     if nGs == 1 and nDs == 1:
#         axes = np.array([[axes]])
#     elif nDs == 1:
#         axes = axes[np.newaxis]
#     elif nGs == 1:
#         axes = axes[:, np.newaxis]
#
#     for iD, diagnostic in enumerate(diagnostics):
#         for iiG, iG in enumerate(iGs):
#             axes[iD, iiG] = plot_diagnostic_for_iG(iG, diagnostic, config, num_train_samples, params, runs,
#                                                    confidence, colors, marker, linestyle,
#                                                    title=False, xlabel=False, ylabel=False,
#                                                    ax=axes[iD, iiG])
#         if iD == 0:
#             axes[iD, iiG].set_title("G=%g" % config.Gs[iG], fontsize=14)
#         if iD == nDs-1:
#             axes[iD, iiG].set_xlabel("N training samples", fontsize=14)
#         if iiG == 0:
#             axes[iD, iiG].set_ylabel(diagnostic, fontsize=14)
#
#     plt.figure(fig.number)
#     if config.figures.SAVE_FLAG:
#         plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, "Diagnostics_%s.png" % "_".join(diagnostics)))
#     if config.figures.SHOW_FLAG:
#         plt.show()
#
#     return fig, axes


# # TODO: Figure this out!:
# if __name__ == '__main__':
#     # Example use:
#     # $ python tuning_tvb_nest.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
#     # Called tuning_tvb_nest.py with:
#     # keyword argument: w_TVB_to_NEST=world
#     # keyword argument: simulation_length=300.0
#
#     STRING_ARGS = {"MODE": "SIM",
#                    "MODE": "TVB_REST",
#                    "BASENAME": "",
#                    "parameters_label": "",
#                    "parameters_filepath": None,
#                    "parameters_filepath_ext": None}
#     INT_ARGS = {"iP": None, "iR": None, }
#     kwargs = {}
#     ntests = 0
#     for arg in sys.argv[1:]:
#         keyval = arg.split("=")
#         if keyval[0] not in ["MODE", "plot_flag"]:
#             key = float(keyval[1])
#         else:
#             key = keyval[1].split(" ")
#             if keyval[0] == "MODE":
#                 ntests = len(key)
#                 if ntests == 1:
#                     key = key[0]
#             elif keyval[0] == "plot_flag":
#                 if keyval[1] == "False":
#                     key = False
#                 else:
#                     key = True
#         kwargs[keyval[0]] = key
#
#     cosim_run_plot(**kwargs)
#
#
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
#     defargs, parser_args, parser = parse_args(parser, defargs=DEFAULT_ARGS)
#     verbosity = defargs.get('verbosity', DEFAULT_ARGS['verbosity'])
#     if verbosity:
#         print("Running %s with arguments:\n" % parser.description)
#         print(parser_args, "\n")
#         print(defargs, "\n")
#     config = configure(**defargs)[0]
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


if __name__ == "__main__":
    parser, defargs = run_fit_plot_args_parser("rest_run_fit_plot")
    args, parser_args, parser = parse_args(parser, argsnames=list(defargs.keys()))
    funcname = args.pop("function", "sim_run_plot")
    verbosity = args.get('verbosity', defargs['verbosity'])
    if verbosity:
        print("Running function %s of script %s with REST_or_TASK='REST' "
              "and user provided arguments:\n" % (funcname, parser.description))
        print(args, "\n")
    globals()[funcname](REST_or_TASK="REST", **args)
