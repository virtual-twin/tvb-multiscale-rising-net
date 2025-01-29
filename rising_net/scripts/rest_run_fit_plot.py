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
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot

from rising_net.scripts.base import assert_config, configure, DEFAULT_ARGS, args_parser, parse_args
from rising_net.scripts.filepaths import get_path, simres_filepath, construct_filepath
from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR, iGstr, get_simres_folder_name, \
    simres_folder, get_stats_params, process_funcmode, find_all_folders, load_sims_to_xarrays_for_iP, \
    run_fit_plot_args_parser, sim_run_plot, get_G, load_stats_per_iG
from rising_net.scripts.tvb_script import run_workflow, load_connectome, prepare_connectome, build_connectivity, \
    build_model, build_simulator, simulate, plot_tvb, tvb_res_to_time_series, \
    compute_target_PSDs, compute_PSD_target_and_data
from rising_net.scripts.tvb_nest_script import build_tvb_nest_interfaces, simulate_tvb_nest
from rising_net.scripts.sbi_script import fitfigs_filepath, build_prior, \
    load_train_params_samples, load_train_params_samples_selection, \
    sbi_estimate, sbi_train, sbi_infer, write_posterior, compute_diagnostics, write_posterior_samples, \
    load_inference, load_proposal, load_posterior, load_posterior_samples, infer_workflow, infer_nRuns, \
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


REST_FIT_MEASURE_LABELS_FOR_PLOT = ["log(PSD) RM1", "log(PSD) LM1", "log(PSD) RS1", "log(PSD) LS1"]


def load_params_from_fit_rest(iG, stat="mean", fitlabel="allsamples", BASENAME="FIT_REST", verbosity=1):
    from rising_net.scripts.rest_run_fit_plot import get_config as get_config_fit_rest
    FUNCMODE = "%sSIM" % stat.upper()
    configFitRest = get_config_fit_rest(iG=iG, FUNCMODE=FUNCMODE,
                                        fitlabel=fitlabel, BASENAME=BASENAME, plot_flag=False, verbosity=0)[0]
    params = {"I_s": configFitRest.model_params["I_s"], "FIC": configFitRest.FIC, "FIC_SPLIT": configFitRest.FIC_SPLIT}
    if verbosity:
        if len(fitlabel):
            labelstr = " for label %s" % fitlabel
        else:
            labelstr = ""
        print("\nLoading %s parameters from %s%s...:\n%s" % (stat, BASENAME, fitlabel, str(params)))
    return params


def rest_simres_filepath(config, iG=None, iP=None, iR=None, FUNCMODE="TRAINSIM",
                         label="", filepath=None, extension=None):
    folder = simres_folder(config, iG, iP, iR, FUNCMODE, label)
    return simres_filepath(config, config.SIM_RES_FILE, folder,
                           iR=iR, label="",
                           filepath=filepath, extension=extension)


# iP: parameter sample index
# iR: simulation repetition and noise seed index
def get_config(iG=None, iP=None, iR=None, FUNCMODE="SIM", fitlabel="", iF=None,
               REST_BASENAME="", restfitlabel="", fit_round=0,
               # parameters_iR=None, parameters_filepath=None, parameters_filepath_ext=None,
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
    #     "SIMULATION_LENGTH": 2 ** 13 + 1.0,
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

    if FUNCMODE == "BOLDSIM":
        kwargs["TIME_SERIES_MONITORS"] = kwargs.get("TIME_SERIES_MONITORS", False)
        kwargs["SIMULATION_LENGTH"] = kwargs.get("SIMULATION_LENGTH", config.BOLD_SIMULATION_LENGTH)
        kwargs["TRANSIENT_RATIO"] = kwargs.get("TRANSIENT_RATIO",
                                               np.minimum(0.25, 2**14 / kwargs["SIMULATION_LENGTH"]).item())
        effective_FUNCMODE = "%sSIM" % config.OPT_RES_MODE.upper()
    else:
        effective_FUNCMODE = FUNCMODE

    iRpath, iR, iP, params, params_string, fitlabel, kwargs = \
        process_funcmode(effective_FUNCMODE, MODE, config, verbosity, iP, iR, iF, iG, fitlabel, fit_round, **kwargs)

    if len(REST_BASENAME):
        # Load REST parameters from previous REST fitting:
        paramsRest = {}
        paramsRest.update(load_params_from_fit_rest(iG,
                                                    stat=config.OPT_RES_MODE,
                                                    fitlabel=restfitlabel,
                                                    BASENAME=REST_BASENAME,
                                                    verbosity=verbosity))
        params.update(paramsRest)
    kwargs.update(params)

    if "SIM" in FUNCMODE:
        kwargs["output_folder"] = os.path.dirname(
            os.path.dirname(
                rest_simres_filepath(config, iG, iP, iRpath, FUNCMODE, fitlabel)))
    config, plotter = configure(MODE=MODE, SEED=int(iR), verbosity=verbosity, **kwargs)

    if config.VERBOSITY:
        print(config.model_params)

    return config, plotter


def load_sims_PSD_to_xarrays(folder, config, iR=None, resstr=RESSTR, **kwargs):  # measures = "PSD" not really needed!
    path = construct_filepath(os.path.join(folder, resstr), default_filename=config.SIM_RES_FILE, iR=iR)
    res = load_pickled_dict(path)
    indPSD = pd.MultiIndex.from_product([pd.Index(res["regions"], name='Regions'),
                                         pd.Index(res["f"], name='f')])
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


def load_prior_target_and_sims_for_sbi_for_iG(iG,
                                              train_params_samples=None,
                                              round=0, prior=None, inference=None, proposal=None,
                                              sim_res=None, sim_res_path=None,
                                              target=None,
                                              config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False, FUNCMODE="FIT")
    # Rebuild proposal if not provided in the input:
    if prior is None:
        prior = build_prior(config)
    if target is None:
        target = target_PSD_fun(config)
    if round > 0:
        parameters_label = iGstr(iG, Ngs=len(config.Gs), igstr=igstr)
        if proposal is None:
            proposal = load_proposal(iR=None, label=parameters_label, config=config)
        proposal.set_default_x(target)
        if inference is None:
            inference = load_inference(iR=None, label=parameters_label, config=config)
    else:
        proposal = prior
        parameters_label = ""
    # Load training parameters' samples if not provided in the input:
    if train_params_samples is None:
        train_params_samples = load_train_params_samples(config,
                                                         # iR=parameters_iR,
                                                         label=parameters_label,
                                                         # filepath=parameters_filepath,
                                                         # extension=parameters_filepath_ext
                                                         ).numpy().squeeze().astype('float32')
    # Load training simulation results if not provided in the input:
    if sim_res is None:
        if sim_res_path is None:
            sim_res_path = os.path.join(config.HEADPATH, config.TRAIN_SIMS_FOLDER)
        # Load proposal' samples
        # By default, we load all parameters and all simulation repetitions and we average across repetitions.
        try:
            sim_res = \
                load_sims_to_xarrays_for_iG(path=str(sim_res_path), config=config,
                                            iG=iG, iP=None, iR=None, average_repetitions=True,
                                            igstr=igstr, folderstr=folderstr, resstr=resstr)[0].values.astype('float32')
        except Exception as e:
            warnings.warn(str(e))
            sim_res_path = os.path.join(config.HEADPATH, config.TRAIN_SIMS_FOLDER)
            sim_res = \
                load_sims_to_xarrays_for_iG(path=str(sim_res_path), config=config,
                                            iG=iG, iP=None, iR=None, average_repetitions=True,
                                            igstr=igstr, folderstr=folderstr, resstr=resstr)[0].values.astype('float32')
    return train_params_samples, sim_res, prior, inference, proposal, target


def plot_PSDs_samples_measures_and_targets(measures, target=None, label="",
                                           measure_labels=REST_FIT_MEASURE_LABELS_FOR_PLOT, config=None):
    config = assert_config(config, return_plotter=False)
    if target is None:
        target = target_PSD_fun(config)
    Nf = int(target.shape[0] / 4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    inds = np.arange(Nf).astype("i")
    for iM, ml in enumerate(measure_labels):
        iC = np.mod(iM, 2)
        iR = int(iM / 2)
        axes[iR, iC] = percent_plot(config.TARGET_FREQS, measures[:, inds],
                                    percentile_min=10, percentile_max=90, n=5,
                                    plot_mean=True, plot_median=False,
                                    color='b', alpha=0.5, ax=axes[iR, iC], mode="linear")
        if target is not None:
            axes[iR, iC].plot(config.TARGET_FREQS, target[inds], color='r', linewidth=2)
        inds += Nf
        axes[iR, iC].set_title(ml)
    if config.figures.SAVE_FLAG:
        plt.savefig(fitfigs_filepath(config, "measure_vs_target_plot.png", label=label))
    if config.figures.SHOW_FLAG:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def infer_workflow_for_iG(iG,
                          train_params_samples=None, round=0, prior=None, inference=None, proposal=None,
                          sim_res=None, sim_res_path=None,
                          target=None, ground_truth=None,
                          config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                          label="", n_samples_per_run=None,
                          results=None, iR=None, save_samples=True,
                          plot_flag=True, plot_diagnostics_flag=True, verbosity=None):
    train_params_samples, sim_res, prior, inference, proposal, target = \
        load_prior_target_and_sims_for_sbi_for_iG(iG, train_params_samples,
                                                  round, prior, inference, proposal,
                                                  sim_res, sim_res_path, target,
                                                  config, igstr, folderstr, resstr)
    label = joinstr([label, iGstr(iG, Ngs=len(config.Gs), igstr=igstr)])
    return infer_workflow(train_params_samples, sim_res, prior, inference, proposal, target, ground_truth,
                          config, label, n_samples_per_run, REST_FIT_MEASURE_LABELS_FOR_PLOT,
                          results, iR, save_samples, plot_flag, plot_PSDs_samples_measures_and_targets,
                          plot_diagnostics_flag, verbosity)


def infer_nRuns_for_iG(iG,
                       train_params_samples=None, round=0, prior=None, inference=None, proposal=None,
                       sim_res=None, sim_res_path=None,
                       target=None, ground_truth=None,
                       config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                       label="", n_samples_per_run=None,
                       save_samples=True, plot_flag=True, verbosity=None):
    train_params_samples, sim_res, prior, inference, proposal, target = \
        load_prior_target_and_sims_for_sbi_for_iG(iG, train_params_samples,
                                                  round, prior, inference, proposal,
                                                  sim_res, sim_res_path, target,
                                                  config, igstr, folderstr, resstr)
    label = joinstr([label, iGstr(iG, Ngs=len(config.Gs), igstr=igstr)])
    return infer_nRuns(train_params_samples, sim_res, prior, inference, proposal, target, ground_truth,
                       config, label, n_samples_per_run, REST_FIT_MEASURE_LABELS_FOR_PLOT, save_samples,
                       plot_flag, plot_PSDs_samples_measures_and_targets, verbosity)


def load_stat_sims_for_iG(iG, stat="PPC", label="", sim_res_path=None, iP=None,
                          config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    stat = stat.upper()
    if sim_res_path is None:
        sim_res_path = os.path.join(config.HEADPATH, getattr(config, "%s_FOLDER" % stat))
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


def load_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="", sim_res_path=None, target=None,
                                        config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    sim_res, iG, iP = load_stat_sims_for_iG(iG, stat=stat, label=label, sim_res_path=sim_res_path, iP=iP,
                                            config=config,  igstr=igstr, folderstr=folderstr, resstr=resstr)
    sim_res = sim_res.values.astype('float32')
    params, iP, fitlabel, params_string = \
        get_stats_params(config, stat=stat, FUNCMODE=None, iG=iG, iP=iP, iF=iF, fitlabel=label)

    if target is None:
        target = target_PSD_fun(config)
    return sim_res, iP, target, params


def load_and_plot_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="",
                                                 sim_res_path=None, target=None,
                                                 config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_iG(iG, stat, iF, iP, label, sim_res_path, target,
                                             config, igstr, folderstr, resstr)
    params_vals = np.array(list(params.values())).T
    if params_vals.ndim < 2:
        params_vals = params_vals[np.newaxis]
    return plot_stats(sim_res, stat, target, params_vals,
                      joinstr([iGstr(iG, Ngs=len(config.Gs), igstr=igstr), label]),
                      REST_FIT_MEASURE_LABELS_FOR_PLOT, plot_PSDs_samples_measures_and_targets, config)


def load_and_plot_best_stat_sims_params_target_for_iG(iG, stat="PPC", iF=None, iP=None, label="",
                                                      sim_res_path=None,
                                                      target=None, target_dist_fun=correlation_distance, Nbest=None,
                                                      config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_iG(iG, stat, iF, iP, label, sim_res_path,  target,
                                            config, igstr, folderstr, resstr)
    return plot_best_stat_sims_params_target(sim_res, target, stat, params=np.array(list(params.values())).T,
                                             label=joinstr([iGstr(iG, Ngs=len(config.Gs), igstr=igstr), label]),
                                             target_dist_fun=target_dist_fun, Nbest=Nbest,
                                             measure_labels=REST_FIT_MEASURE_LABELS_FOR_PLOT,
                                             measures_plot_fun=plot_PSDs_samples_measures_and_targets,
                                             config=config)


def rest_run_fit_plot_args_parser(funname, defargs=DEFAULT_ARGS):

    parser, args = run_fit_plot_args_parser(funname, defargs)

    arguments = {'REST_BASENAME': ['rbsnm', str, 'Rest fitting base folder name', ""],
                 'restfitlabel':  ['rflbl', str,
                                   'Specific fitting label name for rest fitting results to load', "allsamples"]
                 }
    for arg, vals in arguments.items():
        args[arg] = vals[-1]
        parser.add_argument('--%s' % arg,
                            '-%s' % vals[0],
                            dest=arg, metavar=arg,
                            type=vals[1],
                            #default=args[arg],
                            required=False,  # nargs=1,
                            help=vals[2])
    return parser, args


if __name__ == "__main__":
    parser, defargs = rest_run_fit_plot_args_parser("rest_run_fit_plot")
    args, parser_args, parser = parse_args(parser, argsnames=list(defargs.keys()))
    funcname = args.pop("function", "sim_run_plot")
    verbosity = args.get('verbosity', defargs['verbosity'])
    if verbosity:
        print("Running function %s of script %s with REST_or_TASK='REST' "
              "and user provided arguments:\n" % (funcname, parser.description))
        print(args, "\n")
    globals()[funcname](REST_or_TASK="REST", **args)
