# coding: utf-8

import warnings
import glob
import pickle
import os
import shutil

import numpy
from matplotlib import pyplot
import torch
import sbi
from xarray import DataArray, concat
from pandas import Index
from scipy.interpolate import interp1d

from rising_net.scripts.base import assert_config, configure, DEFAULT_ARGS, args_parser, parse_args
from rising_net.scripts.filepaths import simres_filepath, construct_filepath
from rising_net.scripts.tvb_script import prepare_connectome, build_connectivity
from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.sbi_script import build_priors, \
    load_train_params_samples, load_train_params_samples_selection, \
    sbi_estimate, sbi_train, sbi_infer, write_posterior, compute_diagnostics, write_posterior_samples, \
    load_posterior, load_posterior_samples, infer_workflow, infer_nRuns, \
    plot_stats, plot_best_stat_sims_params_target, correlation_distance
from rising_net.scripts.plot_utils import shorten_region_name, plot_pathway_psd_coh, psd_percent_plot, \
    coherence_networks_plot
from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR, iGstr, iPstr, get_G, \
    get_simres_folder_name, simres_folder, process_funcmode, get_stats_params, sim_run_plot, run_fit_plot_args_parser, \
    load_sims_to_xarrays_for_iP, sim_run_plot
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import load_pickled_dict, dump_pickled_dict

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


MODES = ["TVB", "TVB_CEREBOFF", "COSIM", "COSIM_CEREBOFF"]
SIMULATION_MODE_STR = "Simulation mode"


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


def task_simres_filepath(config, mode=None, iG=None, iP=None, iR=None, FUNCMODE="TRAINSIM",
                         label="", filepath=None, extension=None):
    folder = simres_folder(config, iG, iP, iR, FUNCMODE, label)
    if mode is not None:
        folder = os.path.join(folder, mode)
    return simres_filepath(config, config.SIM_RES_FILE, folder,
                           iR=iR, label="",
                           filepath=filepath, extension=extension)


# iP: parameter sample index
# iR: simulation repetition and noise seed index
def get_config(iG=None, iP=None, iR=None, FUNCMODE="SIM", fitlabel="", iF=None,
               REST_BASENAME="", restfitlabel="",
               # parameters_iR=None, parameters_label, parameters_filepath=None, parameters_filepath_ext=None,
               **kwargs):

    # DEFAULT_ARGS = {  # TVB model:
    #     'I_s': 0.1,  # 0.085,
    #     'I_e': -0.35,
    #     "tau_w": 10.0,
    #     "I_w": -0.35,
    #     "G_w": 5.0,
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
    #     "SIMULATION_LENGTH": 2 ** 13 + 1.0,
    #     "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
    #     'output_folder': "", 'verbosity': 1, 'plot_flag': True}

    # Get configuration
    MODE = kwargs.pop("MODE", "")  # make sure we don't overshadow MODE
    if "TASK" not in MODE.upper():
        MODE = joinstr(["TASK", MODE])
    verbosity = kwargs.pop("verbosity", 1)
    config, plotter = configure(MODE=MODE, verbosity=0, **kwargs)

    iG, kwargs = get_G(config, iG=iG, **kwargs)

    iRpath, iR, iP, params, params_string, fitlabel, kwargs = \
        process_funcmode(FUNCMODE, MODE, config, verbosity, iP, iR, iF, iG, fitlabel, **kwargs)

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
        for md in ["COSIM_CEREBOFF", "TVB_CEREBOFF", "COSIM", "TVB"]:
            if md in MODE:
                break
            else:
                md = None
        kwargs["output_folder"] = os.path.dirname(
            os.path.dirname(
                task_simres_filepath(config, md, iG, iP, iRpath, FUNCMODE, fitlabel)))
    config, plotter = configure(MODE=MODE, SEED=int(iR), verbosity=verbosity, **kwargs)

    if config.VERBOSITY:
        print(config.model_params)

    return config, plotter


def find_all_modes_folders(path, modes=MODES):
    modes_found = []
    for mode in modes:
        folder = os.path.join(path, mode)
        if os.path.isdir(folder):
            resfiles = glob.glob(os.path.join(folder, "res", "res_*.pkl"))
            Nf = len(resfiles)
            if Nf == 1:
                modes_found.append(mode)
            elif Nf > 1:
                raise ValueError("\nMore than one (%d) results files found in path %s!:\n%s\n" %
                                 (Nf, folder, str(resfiles)))
    return modes_found


def correct_regions(config):
    # TODO: REMOVE THIS HACK!
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=None)
    connectivity = build_connectivity(connectome, inds, config)
    return connectivity.region_labels[config.TASKINDS]


def coh_to_xarray(res):
    Nregs = len(res["regions"])
    COH = DataArray(np.zeros((Nregs, Nregs, len(res["f"]))),
                    dims=["Region1", "Region2", "f"],
                    coords={"Region1": res["regions"],
                            "Region2": res["regions"],
                            "f": res["f"]},
                    name="COH")
    for iP, pair in enumerate(res["pairs"]):
        COH[pair[0], pair[1]] = res["COH"][iP]
        COH[pair[1], pair[0]] = COH[pair[0], pair[1]]
    return COH


def psd_to_xarray(res):
    return DataArray(res["PSD"],
                     dims=["Region", "f"],
                     coords={"Region": res["regions"], "f": res["f"]},
                     name="PSD")


def load_task_sims_to_xarrays(folder, config, iR=None, resstr=RESSTR, modes=None, measures="COH"):
    if modes is None:
        modes = find_all_modes_folders(folder, modes=MODES)
    if len(modes) == 0:
        raise ValueError("No modes found in path %s!" % folder)
    measures = ensure_list(measures)
    for iM, measure in enumerate(measures):
        measures[iM] = measure.upper()
    res = dict(zip(measures, [list() for _ in range(len(measures))]))
    for mode in modes:
        path = construct_filepath(os.path.join(folder, mode, resstr), default_filename=config.SIM_RES_FILE, iR=iR)
        res_i = load_pickled_dict(path)
        for measure in measures:
            if "COH" in measure:
                res[measure].append(coh_to_xarray(res_i))
            elif "PSD" in measure:
                res[measure].append(psd_to_xarray(res_i))
            res[measure][-1].name += " %s: %s" % (mode, path)
    for measure in measures:
        res[measure] = concat(res[measure], dim=Index(modes, name=SIMULATION_MODE_STR))
        res[measure].name = "%s %s: %s" % (measure, str(modes), folder)
    return res


def load_sims_to_xarrays(path=None, config=None, iG=None, iP=None, iR=None, label="", modes=None, measures="COH",
                         average_repetitions=True, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if path is None:
        path = config.out.FOLDER_RES
    if len(label):
        path = os.path.join(path, label)
    if iG is not None:
        path = os.path.join(path, iGstr(iG, Ngs=len(config.Gs), igstr=igstr))
    res, iPs = load_sims_to_xarrays_for_iP(load_task_sims_to_xarrays, measures, path,
                                           config, iP=iP, iR=iR,
                                           average_repetitions=average_repetitions,
                                           folderstr=folderstr, resstr=resstr, modes=modes)
    return res, iPs, path


def pathway_pairs_fun():
    pathway_pairs_R = [[0, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [0, 13], [13, 18]]
    pathway_pairs_L = [[1, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [1, 12], [12, 19]]
    pathway_pairs = np.array(pathway_pairs_R + pathway_pairs_L)
    return pathway_pairs


def all_task_pairs_fun(N):
    if N > 1:
        inds = np.arange(N).astype("i")
        mask = np.triu(np.ones((N, N)), 1).astype("bool")
        X, Y = np.meshgrid(inds, inds)
        return np.array(list(zip(Y[mask].flatten(), X[mask].flatten())))
    else:
        return np.array([[]])


def load_results_for_tests(TESTS=["TVB", "TVB_CEREBOFF"], path=None, config=None, iG=None, iP=None, iR=None, label="",
                           measures=["COH", "PSD"], igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR, **kwargs):
    # CONFIGURATION:
    config = assert_config(config, return_plotter=False, **kwargs)
    if path is None:
        FUNCMODE = kwargs.get("FUNCMODE", "SIM").upper()
        path = config.HEADPATH
        folder = get_simres_folder_name(config, FUNCMODE=FUNCMODE)
        if len(folder):
            path = os.path.join(path, folder)

    res, iPs, path = load_sims_to_xarrays(path, config, iG=iG, iP=iP, iR=iR, label=label,
                                          modes=TESTS, measures=measures,
                                          average_repetitions=False, igstr=igstr, folderstr=folderstr, resstr=resstr)
    return res, iPs, path, config


def get_task_regions(config=None):
    if config is None:
        config = configure(plot_flag=False, PATHWAY_GAIN=1.0, verbosity=0, BASENAME="tmp123456789")[0]
        shutil.rmtree(config.HEADPATH)
        connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=None)
        connectivity = build_connectivity(connectome, inds, config)
        return config.TASKINDS, connectivity.region_labels[config.TASKINDS], inds
    else:
        try:
            connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=None)
            connectivity = build_connectivity(connectome, inds, config)
            return config.TASKINDS, connectivity.region_labels[config.TASKINDS], inds
        except Exception as e:
            warnings.warn(str(e))
            return get_task_regions()


def plot_comparisons(COH, PSD, config, plotter, folder=None):

    if folder is None:
        folder = config.figures.FOLDER_FIGURES

    # # Results path:
    # if config.VERBOSITY > 1: print("FOLDER_RES: ", config.out.FOLDER_RES)
    # BASEPATH = os.path.dirname(config.out.FOLDER_RES.split("/res")[0])
    # if config.VERBOSITY > 1: print("BASEPATH: ", BASEPATH)  # e.g. "../outputs"
    # TESTSPATH = os.path.join(BASEPATH, TESTSFOLDER)
    # if config.VERBOSITY > 1: print("TESTSPATH: ", TESTSPATH)
    # config.out._out_base = TESTSPATH
    # config.figures._out_base = TESTSPATH

    # Task related regions' labels, and indices:
    TASKINDS, REGION_LABELS, inds = get_task_regions(config)
    # Task related regions' abreviated labels:
    SHORT_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"]) for reg in REGION_LABELS]

    THETA = config.THETA[[0, -1]]  # Hz
    GAMMA = config.GAMMA[[0, -1]]  # Hz

    # results dictionary:
    results = {"inds": TASKINDS,
               "region_labels": REGION_LABELS, "short_labels": SHORT_LABELS,
               "theta": THETA, "gamma": GAMMA}
    f = COH.coords["f"].values
    fth = np.where(np.logical_and(f > THETA[0], f < THETA[1]))[0]
    fgm = np.where(np.logical_and(f > GAMMA[0], f < GAMMA[1]))[0]
    modes = COH.coords[COH.dims[COH.dims.index(SIMULATION_MODE_STR)]].values.tolist()
    colors = []
    for mode, col in zip(MODES, ["g", "r", "b", "m"]):
        if mode in modes:
            colors.append(col)

    def assert_ndims(datarr, ndim):
        while datarr.ndim < ndim:
            datarr = datarr[np.newaxis]
        return datarr

    PSD = assert_ndims(PSD.values, 4)
    COH = assert_ndims(COH.values, 5)
    for iM, mode in enumerate(modes):
        results[mode] = {}
        results[mode]['PSD'] = PSD[:, iM]
        results[mode]['COH'] = COH[:, iM]
        results[mode]['COHth'] = results[mode]['COH'][:, :, :, fth].mean(axis=-1)
        results[mode]['COHgm'] = results[mode]['COH'][:, :, :, fgm].mean(axis=-1)
        results["f"] = f  # frequency vector
        results["fth"] = fth  # theta frequency vector inds
        results["fgm"] = fgm  # gamma frequency vector inds
        # pairs of regions of coherences where i, j in [0, TASKINDS.size]:
        results["ij"] = all_task_pairs_fun(len(results["inds"]))  # pathway_pairs_fun()
    figR, axR, figL, axL = plot_pathway_psd_coh(results, inds,
                                                tests=modes, colors=colors,
                                                percentile_min=1, percentile_max=99, n=1,
                                                plot_mean=True, plot_median=False, modePSD="semilog", modeCOH="linear",
                                                alpha=0.5, figsize=config.figures.LARGE_SIZE, fontsize=16)
    if plotter.config.SAVE_FLAG:
        for fig, hemi in zip([figR, figL], ["Right", "Left"]):
            plt.figure(fig.number)
            plt.savefig(os.path.join(folder, "Pathway_PSD_COH_%s.png" % hemi))
    figPSD, axesPSD = psd_percent_plot(results,
                                       inds=None,
                                       tests=modes, colors=colors,
                                       percentile_min=1, percentile_max=99, n=1,
                                       plot_mean=False, plot_median=True,
                                       alpha=0.5, figsize=config.figures.DEFAULT_SIZE, fontsize=16)
    if plotter.config.SAVE_FLAG:
        plt.figure(figPSD.number)
        plt.savefig(os.path.join(folder, "PSDs.png"))

    figCOH, axesCOH = coherence_networks_plot(results,
                                              tests=modes,
                                              resnames=['COHth', 'COHgm'],
                                              bands=["theta", "gamma"],
                                              figsize=config.figures.DEFAULT_SIZE, fontsize=16)
    if plotter.config.SAVE_FLAG:
        plt.figure(figCOH.number)
        plt.savefig(os.path.join(folder, "COHs.png"))


def load_and_plot_comparisons(TESTS=["TVB", "TVB_CEREBOFF"], path=None, config=None, iG=None, iP=None, iR=None,
                              label="", igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR, **kwargs):
    label = kwargs.get("fitlabel", label)
    if config is None:
        kwargs["plot_flag"] = True
        config, plotter = get_config(iG=iG, **kwargs)
    else:
        config, plotter = assert_config(config, return_plotter=True, **kwargs)
    # config, plotter = assert_config(config, return_plotter=True, **kwargs)
    res, iPs, figsfolder, config = load_results_for_tests(TESTS=TESTS, path=path, config=config,
                                                          iG=iG, iP=iP, iR=iR, label=label, measures=["COH", "PSD"],
                                                          igstr=igstr, folderstr=folderstr, resstr=resstr, **kwargs)
    if len(label):
        figsfolder = os.path.join(figsfolder, label)
    if iPs is None:
        figsfolder = os.path.join(figsfolder, "figs")
        safe_makedirs(figsfolder)
        plot_comparisons(res["COH"], res["PSD"], config, plotter, figsfolder)
    else:
        iPs = ensure_list(iPs)
        for iiP, iP in enumerate(iPs):
            figsfolder_iiP = os.path.join(figsfolder, iPstr(iP, Nsims=config.N_SIMULATIONS, resstr=resstr), "figs")
            safe_makedirs(figsfolder_iiP)
            plot_comparisons(res["COH"][iiP], res["PSD"][iiP], config, plotter, figsfolder_iiP)


def M1S1_pairs_fun():
    return np.array([[0, 1], [18, 19]]).T


def _get_sim_res_COHgamma(COHs, pathway_pairs, config):
    gammaInds = np.logical_and(COHs.coords["f"] >= config.GAMMA[0], COHs.coords["f"] <= config.GAMMA[-1])
    COHs = COHs[:, :, :, :, gammaInds].isel(
        Region1=DataArray(pathway_pairs[:, 0], dims="Region1-Region2"),
        Region2=DataArray(pathway_pairs[:, 1], dims="Region1-Region2"))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        COHs = np.arctanh(COHs)
    return COHs


def get_sim_res_COHgamma(COHs, pathway_pairs, config):
    return _get_sim_res_COHgamma(COHs, pathway_pairs, config).values


def get_sim_res_COHgammaPathway_params(COHs, config):
    return COHs.mean(axis=-1).values


def get_sim_res_COHgammaM1S1diff(COHs, config):
    COHs, params = _get_sim_res_COHgamma(COHs, M1S1_pairs_fun(), config)
    COHs = COHs[0] - COHs[1]   # TVB
    COHs = COHs.mean(axis=-1)  # average over gamma band
    return COHs.values


def get_sim_res_COHM1S1diff(COHs, config):
    pathway_pairs = M1S1_pairs_fun()
    COHs = COHs[:, :, :, :, :].isel(
        Region1=DataArray(pathway_pairs[:, 0], dims="Region1-Region2"),
        Region2=DataArray(pathway_pairs[:, 1], dims="Region1-Region2"))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        COHs = np.arctanh(COHs)
    thetaInds = np.logical_and(COHs.coords["f"] >= config.THETA[0], COHs.coords["f"] <= config.THETA[-1])
    betaInds = np.logical_and(COHs.coords["f"] >= config.BETA[0], COHs.coords["f"] <= config.BETA[-1])
    gammaInds = np.logical_and(COHs.coords["f"] >= config.GAMMA[0], COHs.coords["f"] <= config.GAMMA[-1])
    COHsDiffsPerBand = []
    for inds in [thetaInds, betaInds, gammaInds]:
        # Average over freq band:
        temp = COHs[:, :, :, inds].mean(axis=-1)
        # Diff conditions
        COHsDiffsPerBand.append((temp[0] - temp[1]).values.squeeze())
    COHs = np.hstack(COHsDiffsPerBand)
    return COHs


def get_sim_res_COHM1S1andDiff(COHs, config):
    pathway_pairs = M1S1_pairs_fun()
    COHs = COHs[:, :, :, :, :].isel(
        Region1=DataArray(pathway_pairs[:, 0], dims="Region1-Region2"),
        Region2=DataArray(pathway_pairs[:, 1], dims="Region1-Region2"))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        COHs = np.arctanh(COHs)
    thetaInds = np.logical_and(COHs.coords["f"] >= config.THETA[0], COHs.coords["f"] <= config.THETA[-1])
    betaInds = np.logical_and(COHs.coords["f"] >= config.BETA[0], COHs.coords["f"] <= config.BETA[-1])
    gammaInds = np.logical_and(COHs.coords["f"] >= config.GAMMA[0], COHs.coords["f"] <= config.GAMMA[-1])
    COHsPerBand = []
    COHsDiffsPerBand = []
    for inds in [thetaInds, betaInds, gammaInds]:
        # Average over freq band:
        temp = COHs[:, :, :, inds].mean(axis=-1)
        COHsPerBand.append(temp[0].values.squeeze())
        # Diff conditions
        COHsDiffsPerBand.append((temp[0] - temp[1]).values.squeeze())
    COHs = np.hstack([np.hstack(COHsPerBand), np.hstack(COHsDiffsPerBand)])
    return COHs


def get_sim_res_COHM1S1diffratio(COHs, config):
    pathway_pairs = M1S1_pairs_fun()
    COHs = COHs[:, :, :, :, :].isel(
        Region1=DataArray(pathway_pairs[:, 0], dims="Region1-Region2"),
        Region2=DataArray(pathway_pairs[:, 1], dims="Region1-Region2"))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        COHs = np.arctanh(COHs)
    thetaInds = np.logical_and(COHs.coords["f"] >= config.THETA[0], COHs.coords["f"] <= config.THETA[-1])
    betaInds = np.logical_and(COHs.coords["f"] >= config.BETA[0], COHs.coords["f"] <= config.BETA[-1])
    gammaInds = np.logical_and(COHs.coords["f"] >= config.GAMMA[0], COHs.coords["f"] <= config.GAMMA[-1])
    # COHsPerBand = []
    COHsDiffsPerBandRatio = []
    for inds in [thetaInds, betaInds, gammaInds]:
        # Average the coherence band:
        COHsPerBand = COHs[:, :, :, inds].mean(axis=-1)
        # Diff conditions and normalize with TVB with CEREBON coherence, and average over frequency band
        COHsDiffsPerBandRatio.append(
            ( (COHsPerBand[0] - COHsPerBand[1]) / COHsPerBand[0] ).values.squeeze()
        )
    COHs = np.hstack(COHsDiffsPerBandRatio)
    return COHs


def get_sim_res_COHM1S1diffratioDist(COHs, config):
    COHs =  get_sim_res_COHM1S1diffratio(COHs, config)
    target = target_COHM1S1diffratio_fun(config).numpy()
    for iB, w in enumerate(config.FREQ_BAND_FITNESS_WEIGHTS):
        iC = 2*iB
        for iH in range(2):
            COHs[:, iC+iH] = w * (COHs[:, iC+iH] - target[iC+iH])
    return COHs


def get_sim_res_COHM1S1diffratioDist2Sum(COHs, config):
    return np.sqrt((get_sim_res_COHM1S1diffratioDist(COHs, config)**2).sum(axis=1))[:, np.newaxis]


def get_sim_res_COHM1S1diffratioDistRatio(COHs, config):
    COHs =  get_sim_res_COHM1S1diffratio(COHs, config)
    target = target_COHM1S1diffratio_fun(config).numpy()
    for iB, w in enumerate(config.FREQ_BAND_FITNESS_WEIGHTS):
        iC = 2*iB
        for iH in range(2):
            COHs[:, iC+iH] = w * (COHs[:, iC+iH] - target[iC+iH])/target[iC+iH]
    return COHs


def get_sim_res_COHM1S1diffratioDistRatioDist(COHs, config):
    return get_sim_res_COHM1S1diffratioDistRatio(COHs, config).sum(axis=1)[:, np.newaxis]


def get_sim_res_COHM1S1diffratioDistRatioDist2(COHs, config):
    return (get_sim_res_COHM1S1diffratioDistRatio(COHs, config)**2).sum(axis=1)[:, np.newaxis]


def target_COHgammaPathway_fun(config, target=0.5):
    target = target*np.ones((16, ))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        target = np.arctanh(target)
    return torch.Tensor(target)


def load_Popa_etal_COH(config):
    with open(os.path.join(config.TARGET_POPA_PATH, 'COH.npy'), 'rb') as f:
        COH = np.load(f)
    # # Compute coherence interpolation...
    # interp = interp1d(COH.T[0], COH.T[1], kind='linear', axis=0,
    #                   copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)
    return COH  # interp(config.TARGET_FREQS)


def target_COHgammaM1S1diff_fun(config, target=0.1):
    if target is None:
        COH = load_Popa_etal_COH(config)
        if config.COHERENCE_FISHER_Z_TRANSFORM:
            COH = np.arctanh(COH)
        gammaInds = np.logical_and(config.TARGET_FREQS >= config.GAMMA[0], config.TARGET_FREQS <= config.GAMMA[-1])
        target = np.array([[np.mean(COH[0, gammaInds] - COH[1, gammaInds])] * 2]).flatten()
    else:
        target = target * np.ones((2,))
    return torch.Tensor(target)


def target_COHM1S1diff_fun(config, target=None):
    if target is None:
        COH = load_Popa_etal_COH(config)
        if config.COHERENCE_FISHER_Z_TRANSFORM:
            COH = np.arctanh(COH)
        thetaInds = np.logical_and(config.TARGET_FREQS >= config.THETA[0], config.TARGET_FREQS <= config.THETA[-1])
        betaInds = np.logical_and(config.TARGET_FREQS >= config.BETA[0], config.TARGET_FREQS <= config.BETA[-1])
        gammaInds = np.logical_and(config.TARGET_FREQS >= config.GAMMA[0], config.TARGET_FREQS <= config.GAMMA[-1])
        target = np.array([[np.mean(COH[0, thetaInds] - COH[1, thetaInds])]*2,
                           [np.mean(COH[0, betaInds] - COH[1, betaInds])]*2,
                           [np.mean(COH[0, gammaInds] - COH[1, gammaInds])]*2]).flatten()
    else:
        target = target * np.ones((6,))
    return torch.Tensor(target)


def target_COHM1S1andDiff_fun(config, target=None):
    if target is None:
        COH = load_Popa_etal_COH(config)
        if config.COHERENCE_FISHER_Z_TRANSFORM:
            COH = np.arctanh(COH)
        thetaInds = np.logical_and(config.TARGET_FREQS >= config.THETA[0], config.TARGET_FREQS <= config.THETA[-1])
        betaInds = np.logical_and(config.TARGET_FREQS >= config.BETA[0], config.TARGET_FREQS <= config.BETA[-1])
        gammaInds = np.logical_and(config.TARGET_FREQS >= config.GAMMA[0], config.TARGET_FREQS <= config.GAMMA[-1])
        target = np.array([[np.mean(COH[0, thetaInds])]*2,
                           [np.mean(COH[0, betaInds])]*2,
                           [np.mean(COH[0, gammaInds])]*2,
                           [np.mean(COH[0, thetaInds] - COH[1, thetaInds])]*2,
                           [np.mean(COH[0, betaInds] - COH[1, betaInds])]*2,
                           [np.mean(COH[0, gammaInds] - COH[1, gammaInds])]*2]).flatten()
    else:
        target = target * np.ones((12,))
    return torch.Tensor(target)


def target_COHM1S1diffratio_fun(config, target=None):
    if target is None:
        COH = load_Popa_etal_COH(config)
        if config.COHERENCE_FISHER_Z_TRANSFORM:
            COH = np.arctanh(COH)
        thetaInds = np.logical_and(config.TARGET_FREQS >= config.THETA[0], config.TARGET_FREQS <= config.THETA[-1])
        betaInds = np.logical_and(config.TARGET_FREQS >= config.BETA[0], config.TARGET_FREQS <= config.BETA[-1])
        gammaInds = np.logical_and(config.TARGET_FREQS >= config.GAMMA[0], config.TARGET_FREQS <= config.GAMMA[-1])
        target = np.array([[np.mean((COH[0, thetaInds] - COH[1, thetaInds])/COH[0, thetaInds])]*2,
                           [np.mean((COH[0, betaInds] - COH[1, betaInds])/COH[0, betaInds])]*2,
                           [np.mean((COH[0, gammaInds] - COH[1, gammaInds])/COH[0, gammaInds])]*2]).flatten()
    else:
        target = target * np.ones((6,))
    return torch.Tensor(target)


def target_COHM1S1diffratioDist_fun(config, target=None):
    return torch.Tensor(np.zeros((6,)))


def target_COHM1S1diffratioDist2Sum_fun(config, target=None):
    return torch.Tensor(np.zeros((1,)))


def target_COHM1S1diffratioDistRatioDist(config, target=None):
    return torch.Tensor(np.zeros((1,)))


def load_sims_for_sbi(sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                      config=None, iG=None, iP=None, label="", igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if sim_res_path is None:
        sim_res_path = os.path.join(config.HEADPATH, config.TRAIN_SIMS_FOLDER)
    # Load priors' samples
    # By default, we load all parameters and all simulation repetitions and we average across repetitions.
    sim_res, iPs = load_sims_to_xarrays(sim_res_path, config, iG=iG, iP=iP, iR=None, label=label, modes=None,
                                        measures="COH",  average_repetitions=True,
                                        igstr=igstr, folderstr=folderstr, resstr=resstr)[:2]
    sim_res = sim_res["COH"]
    # Reverse the dimensions of modes and parameters:
    return sim_res_fun(sim_res.transpose(*np.array(sim_res.dims)[[1, 0, 2, 3, 4]].tolist()), config), iPs


def load_priors_target_and_sims_for_sbi(priors=None, train_params_samples=None,
                                        sim_res=None, sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                                        target=None, target_fun=target_COHM1S1diffratio_fun,
                                        config=None, label="", iG=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False, FUNCMODE="FIT")
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
        sim_res = load_sims_for_sbi(sim_res_path=sim_res_path, sim_res_fun=sim_res_fun,
                                    config=config, iG=iG, iP=None, label=label,
                                    igstr=igstr, folderstr=folderstr, resstr=resstr)[0]
    if target is None:
        target = target_fun(config, target)
    return priors, train_params_samples, sim_res, target


def infer_workflow_for_task(priors=None, train_params_samples=None,
                            sim_res=None, sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                            target=None, target_fun=target_COHM1S1diffratio_fun, ground_truth=None,
                            config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                            label="", n_samples_per_run=None, measure_labels=None,
                            results=None, iG=None, iR=None, save_samples=True,
                            plot_flag=True, plot_diagnostics_flag=True, verbosity=None):
    priors, train_params_samples, sim_res, target = \
        load_priors_target_and_sims_for_sbi(priors, train_params_samples,
                                            sim_res, sim_res_path, sim_res_fun,
                                            target, target_fun,
                                            config, label, iG, igstr, folderstr, resstr)
    if iG is not None:
        label = joinstr([label, iGstr(iG, Ngs=len(config.Gs), igstr=igstr)])
    return infer_workflow(train_params_samples, sim_res, priors, target, ground_truth,
                          config, label, n_samples_per_run, measure_labels,
                          results, iR, save_samples, plot_flag, None, plot_diagnostics_flag, verbosity)


def infer_nRuns_for_task(priors=None, train_params_samples=None,
                         sim_res=None, sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                         target=None, target_fun=target_COHM1S1diffratio_fun, ground_truth=None,
                         config=None,  igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                         label="", n_samples_per_run=None, measure_labels=None, iG=None,
                         save_samples=True, plot_flag=True, verbosity=None):
    priors, train_params_samples, sim_res, target = \
        load_priors_target_and_sims_for_sbi(priors, train_params_samples,
                                            sim_res, sim_res_path, sim_res_fun,
                                            target, target_fun,
                                            config, label, iG, igstr, folderstr, resstr)
    if iG is not None:
        label = joinstr([label, iGstr(iG, Ngs=len(config.Gs), igstr=igstr)])
    return infer_nRuns(train_params_samples, sim_res, priors, target, ground_truth,
                       config, label, n_samples_per_run, measure_labels, save_samples, plot_flag, None, verbosity)


def load_stat_sims_for_task(stat="PPC", label="",
                            sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                            iG=None, iP=None, config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    stat = stat.upper()
    if sim_res_path is None:
        sim_res_path = os.path.join(config.HEADPATH, getattr(config, "%s_FOLDER" % stat))
    # Load training simulation results:
    # By default, we load all parameters and all simulation repetitions and we average across repetitions.
    sim_res, iPs = load_sims_for_sbi(sim_res_path=sim_res_path, sim_res_fun=sim_res_fun,
                                     config=config, iG=iG, iP=iP, label=label,
                                     igstr=igstr, folderstr=folderstr, resstr=resstr)
    if sim_res.ndim < 2:
        if isinstance(sim_res, np.ndarray):
            sim_res = sim_res[np.newaxis]
        else:
            sim_res = sim_res.expand_dims(dim=None, axis=0, create_index_for_new_dim=True, Simulations=np.array([0]))
    if len(iPs) == 1:
        iPs = iPs[0]
    return sim_res, iPs


def load_stat_sims_params_target_for_task(stat="PPC", iF=None, iG=None, iP=None, label="",
                                          sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                                          target=None, target_fun=target_COHM1S1diffratio_fun,
                                          config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    sim_res, iP = load_stat_sims_for_task(stat=stat, label=label,
                                          sim_res_path=sim_res_path, sim_res_fun=sim_res_fun,
                                          iG=iG, iP=iP, config=config, igstr=igstr, folderstr=folderstr, resstr=resstr)
    if not isinstance(sim_res, numpy.ndarray):
        sim_res = sim_res.values.astype('float32')
    params, iP, fitlabel, params_string = \
        get_stats_params(config, stat=stat, FUNCMODE=None, iG=iG, iP=iP, iF=iF, fitlabel=label)

    if target is None:
        target = target_fun(config, target)
    return sim_res, iP, target, params


def load_and_plot_stat_sims_params_target_for_task(stat="PPC", iF=None, iG=None, iP=None, label="",
                                                   sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                                                   target=None, target_fun=target_COHM1S1diffratio_fun,
                                                   config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                                                   measure_labels=None, plot_comparisons=True):
    config = assert_config(config, return_plotter=False, FUNCMODE="FIT")
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_task(stat=stat, iF=iF, iG=iG, iP=iP, label=label,
                                              sim_res_path=sim_res_path, sim_res_fun=sim_res_fun,
                                              target=target, target_fun=target_fun,
                                              config=config, igstr=igstr, folderstr=folderstr, resstr=resstr)
    params_vals = np.array(list(params.values())).T
    if params_vals.ndim < 2:
        params_vals = params_vals[np.newaxis]
    outputs = plot_stats(sim_res, stat, target, params_vals, label, measure_labels, None, config)
    if plot_comparisons:
        load_and_plot_comparisons(TESTS=["TVB", "TVB_CEREBOFF"], path=None, config=config,
                                  iG=iG, iP=iP, iR=None, label=label,
                                  igstr=igstr, folderstr=folderstr, resstr=resstr, FUNCMODE="%sSIM" % stat.upper())
    return outputs


def load_and_plot_best_stat_sims_params_target_for_task(stat="PPC", iF=None, iG=None, iP=None, label="",
                                                        sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                                                        target=None, target_fun=target_COHM1S1diffratio_fun,
                                                        target_dist_fun=correlation_distance, Nbest=None,
                                                        config=None, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                                                        measure_labels=None):
    config = assert_config(config, return_plotter=False)
    sim_res, iP, target, params = \
        load_stat_sims_params_target_for_task(stat=stat, iF=iF, iG=iG, iP=iP, label=label,
                                              sim_res_path=sim_res_path, sim_res_fun=sim_res_fun,
                                              target=target, target_fun=target_fun,
                                              config=config, igstr=igstr, folderstr=folderstr, resstr=resstr)
    return plot_best_stat_sims_params_target(sim_res, target, stat, params=np.array(list(params.values())).T,
                                             label=label,
                                             target_dist_fun=target_dist_fun, Nbest=Nbest,
                                             measure_labels=measure_labels, measures_plot_fun=None,
                                             config=config)


def task_run_fit_plot_args_parser(funname, defargs=DEFAULT_ARGS):

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
    parser, defargs = task_run_fit_plot_args_parser("task_run_fit_plot")
    args, parser_args, parser = parse_args(parser, argsnames=list(defargs.keys()))
    funcname = args.pop("function", "sim_run_plot")
    verbosity = args.get('verbosity', defargs['verbosity'])
    if verbosity:
        print("Running function %s from script %s with REST_or_TASK='TASK' "
              "and user provided arguments:\n" % (funcname, parser.description))
        print(args, "\n")
    globals()[funcname](REST_or_TASK="TASK", **args)
