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
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.plot_utils import shorten_region_name, plot_pathway_psd_coh, psd_percent_plot, \
    coherence_networks_plot
from rising_net.scripts.run_fit_plot import RESSTR, NSDSTR, iPstr, get_simres_folder_name, simres_folder, \
    process_funcmode, sim_run_plot, run_fit_plot_args_parser, load_sims_to_xarrays_for_iP
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import load_pickled_dict, dump_pickled_dict

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


MODES = ["TVB", "TVB_CEREBOFF", "COSIM", "COSIM_CEREBOFF"]


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
def get_config(iP=None, iR=None, FUNCMODE="SIM", fitlabel="", iF=None,
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
    #     "SIMULATION_LENGTH": 2 ** 10 + 1.0,
    #     "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
    #     'output_folder': "", 'verbosity': 1, 'plot_flag': True}

    # Get configuration
    MODE = kwargs.pop("MODE", "")  # make sure we don't overshadow MODE
    if "TASK" not in MODE.upper():
        MODE = joinstr(["TASK", MODE])
    verbosity = kwargs.pop("verbosity", 1)
    config, plotter = configure(MODE=MODE, verbosity=0, **kwargs)

    iG = kwargs.pop("iG", None)
    iRpath, iR, iP, params, params_string, fitlabel, kwargs = \
        process_funcmode(FUNCMODE, MODE, config, verbosity, iP, iR, iF, iG, fitlabel, **kwargs)

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

    print(config.model_params)

    return config, plotter


def find_all_modes_folders(path, modes=MODES):
    modes_found = []
    for mode in modes:
        folder = os.path.join(path, mode)
        if os.path.isdir(folder):
            resfiles = glob.glob(os.path.join(folder, "res", "res_*.pkl"))
            Nf =len(resfiles)
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
    measures = ensure_list(measures)
    for iM, measure in enumerate(measures):
        measures[iM] = measure.upper()
    res = dict(zip(measures, [list() for _ in range(len(measures))]))
    for mode in modes:
        path = construct_filepath(os.path.join(folder, mode, resstr), default_filename=config.SIM_RES_FILE, iR=iR)
        res_i = load_pickled_dict(path)
        # TODO: REMOVE THIS HACK!
        res_i["regions"] = correct_regions(config)
        for measure in measures:
            if "COH" in measure:
                res[measure].append(coh_to_xarray(res_i))
            elif "PSD" in measure:
                res[measure].append(psd_to_xarray(res_i))
            res[measure][-1].name += " %s: %s" % (mode, path)
    for measure in measures:
        res[measure] = concat(res[measure], dim=Index(modes, name="Simulation mode"))
        res[measure].name = "%s %s: %s" % (measure, str(modes), folder)
    return res


def load_sims_to_xarrays(path=None, config=None, iP=None, iR=None, modes=None, measures="COH",
                         average_repetitions=True, folderstr=NSDSTR, resstr=RESSTR):
    config = assert_config(config, return_plotter=False)
    if path is None:
        path = config.out.FOLDER_RES
    return load_sims_to_xarrays_for_iP(load_task_sims_to_xarrays, measures, path,
                                       config, iP=iP, iR=iR,
                                       average_repetitions=average_repetitions,
                                       folderstr=folderstr, resstr=resstr, modes=modes)


def pathway_pairs_fun():
    pathway_pairs_R = [[0, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [0, 13], [13, 18]]
    pathway_pairs_L = [[1, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [1, 12], [12, 19]]
    pathway_pairs = np.array(pathway_pairs_R + pathway_pairs_L)
    return pathway_pairs


def plot_comparisons(COH, PSD, config, plotter, folder=None):

    if folder is None:
        folder = config.figures.FOLDER_FIGURES

    # CONNECTIVITY:
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=None)
    connectivity = build_connectivity(connectome, inds, config)

    # # Results path:
    # if config.VERBOSITY > 1: print("FOLDER_RES: ", config.out.FOLDER_RES)
    # BASEPATH = os.path.dirname(config.out.FOLDER_RES.split("/res")[0])
    # if config.VERBOSITY > 1: print("BASEPATH: ", BASEPATH)  # e.g. "../outputs"
    # TESTSPATH = os.path.join(BASEPATH, TESTSFOLDER)
    # if config.VERBOSITY > 1: print("TESTSPATH: ", TESTSPATH)
    # config.out._out_base = TESTSPATH
    # config.figures._out_base = TESTSPATH

    # Task related regions' labels:
    REGION_LABELS = connectivity.region_labels[config.TASKINDS]
    # Task related regions' abreviated labels:
    SHORT_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"]) for reg in REGION_LABELS]

    THETA = config.THETA[[0, -1]]  # Hz
    GAMMA = config.GAMMA[[0, -1]]  # Hz

    # results dictionary:
    results = {"inds": config.TASKINDS,
               "region_labels": REGION_LABELS, "short_labels": SHORT_LABELS,
               "theta": THETA, "gamma": GAMMA}
    f = COH.coords["f"].values
    fth = np.where(np.logical_and(f > THETA[0], f < THETA[1]))[0]
    fgm = np.where(np.logical_and(f > GAMMA[0], f < GAMMA[1]))[0]
    modes = COH.coords[COH.dims[1]].values.tolist()
    colors = []
    for mode, col in zip(MODES, ["g", "r", "b", "m"]):
        if mode in modes:
            colors.append(col)
    for iM, mode in enumerate(modes):
        results[mode] = {}
        results[mode]['PSD'] = PSD[:, iM].values
        results[mode]['COH'] = COH[:, iM].values
        results[mode]['COHth'] = results[mode]['COH'][:, :, :, fth].mean(axis=-1)
        results[mode]['COHgm'] = results[mode]['COH'][:, :, :, fgm].mean(axis=-1)
        results["f"] = f  # frequency vector
        results["fth"] = fth  # theta frequency vector inds
        results["fgm"] = fgm  # gamma frequency vector inds
        results["ij"] = pathway_pairs_fun()  # pairs of regions of coherences where i, j in [0, config.TASKINDS.size]

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


def load_and_plot_comparisons(TESTS=None, config=None, iP=None, iR=None, folderstr=NSDSTR, resstr=RESSTR, **kwargs):

    # CONFIGURATION:
    FUNCMODE = kwargs.get("FUNCMODE", "SIM")
    config, plotter = assert_config(config, return_plotter=True, **kwargs)
    folder = get_simres_folder_name(config, FUNCMODE=FUNCMODE)
    path = config.out.FOLDER_RES
    figsfolder = config.figures.FOLDER_FIGURES
    if len(folder):
        path = os.path.join(os.path.dirname(config.out.FOLDER_RES), folder)

    res, iPs = load_sims_to_xarrays(path, config, iP=iP, iR=iR, modes=TESTS, measures=["COH", "PSD"],
                                    average_repetitions=False, folderstr=folderstr, resstr=resstr)
    Nps = len(iPs)
    if Nps == 0:
        iPs = [None]

    for iiP, iP in enumerate(iPs):
        if iP is not None:
            figsfolder = os.path.join(path, iPstr(iP, Nsims=Nps, resstr=resstr))
            figsfolder = os.path.join(figsfolder, "figs")
            safe_makedirs(figsfolder)
        plot_comparisons(res["COH"][iiP], res["PSD"][iiP], config, plotter, figsfolder)


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


def get_sim_res_COHgamma_params_from_path(pathway_pairs, config,
                                          Nsims=None, conds=["TVB"], path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=conds, path=path, assert_params=assert_params)
    return get_sim_res_COHgamma_params(COHs, pathway_pairs, config), params.values


def get_sim_res_COHgammaPathway_params(COHs, config):
    return COHs.mean(axis=-1).values


def get_sim_res_COHgammaPathway_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params = get_sim_res_COHgamma_params_from_path(pathway_pairs_fun(), config,
                                                         Nsims=Nsims, conds=["TVB"],
                                                         path=path, assert_params=assert_params)
    COHs = COHs.mean(axis=-1)  # average over gamma band
    return COHs.mean(axis=-1), params


def get_sim_res_COHgammaM1S1diff(COHs, config):
    COHs, params = _get_sim_res_COHgamma(COHs, M1S1_pairs_fun(), config)
    COHs = COHs[0] - COHs[1]   # TVB
    COHs = COHs.mean(axis=-1)  # average over gamma band
    return COHs.values


def get_sim_res_COHgammaM1S1diff_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params = get_sim_res_COHgamma_params_from_path(M1S1_pairs_fun(), config,
                                                         Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                         path=path, assert_params=assert_params)
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        COHs = np.arctanh(COHs)
    COHs = COHs[0] - COHs[1]   # TVB
    COHs = COHs.mean(axis=-1)  # average over gamma band
    return COHs, params


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


def get_sim_res_COHM1S1diff_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diff(COHs, config), params.values


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


def get_sim_res_COHM1S1andDiff_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1andDiff(COHs, config), params.values


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


def get_sim_res_COHM1S1diffratio_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratio(COHs, config), params.values


def get_sim_res_COHM1S1diffratioDist(COHs, config):
    COHs =  get_sim_res_COHM1S1diffratio(COHs, config)
    target = target_COHM1S1diffratio_fun(config).numpy()
    for iB, w in enumerate(config.FREQ_BAND_FITNESS_WEIGHTS):
        iC = 2*iB
        for iH in range(2):
            COHs[:, iC+iH] = w * (COHs[:, iC+iH] - target[iC+iH])
    return COHs


def get_sim_res_COHM1S1diffratioDist_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratioDist(COHs, config), params.values


def get_sim_res_COHM1S1diffratioDist2Sum(COHs, config):
    return np.sqrt((get_sim_res_COHM1S1diffratioDist(COHs, config)**2).sum(axis=1))[:, np.newaxis]


def get_sim_res_COHM1S1diffratioDist2Sum_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratioDist2Sum(COHs, config), params.values



def get_sim_res_COHM1S1diffratioDistRatio(COHs, config):
    COHs =  get_sim_res_COHM1S1diffratio(COHs, config)
    target = target_COHM1S1diffratio_fun(config).numpy()
    for iB, w in enumerate(config.FREQ_BAND_FITNESS_WEIGHTS):
        iC = 2*iB
        for iH in range(2):
            COHs[:, iC+iH] = w * (COHs[:, iC+iH] - target[iC+iH])/target[iC+iH]
    return COHs


def get_sim_res_COHM1S1diffratioDistRatio_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratioDistRatio(COHs, config), params.values


def get_sim_res_COHM1S1diffratioDistRatioDist(COHs, config):
    return get_sim_res_COHM1S1diffratioDistRatio(COHs, config).sum(axis=1)[:, np.newaxis]


def get_sim_res_COHM1S1diffratioDistRatioDist_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratioDistRatioDist(COHs, config), params.values


def get_sim_res_COHM1S1diffratioDistRatioDist2(COHs, config):
    return (get_sim_res_COHM1S1diffratioDistRatio(COHs, config)**2).sum(axis=1)[:, np.newaxis]


def get_sim_res_COHM1S1diffratioDistRatioDist2_params_from_path(config, Nsims=None, path=None, assert_params=True):
    COHs, params, failed = load_allsims_to_xarrays(Nsims=Nsims, conds=["TVB", "TVB_CEREBOFF"],
                                                   path=path, assert_params=assert_params)
    return get_sim_res_COHM1S1diffratioDistRatioDist2(COHs, config), params.values


def train_posterior(sim_res_fun, config,
                    Nsims=None, path=None, assert_params=True, target_fun=None, target=None,
                    label="", measure_labels=None, plot_flag=True):
    COHs, params = sim_res_fun(config, Nsims=Nsims, path=path, assert_params=assert_params)
    params = params.T

    if plot_flag:
        fig1, axes1, fig2, axes2 = \
            plot_samples_measures_and_targets(config, params=params, COHs=COHs, target_fun=target_fun, target=target,
                                              label=label, measure_labels=measure_labels)

    priors = build_priors(config)
    posterior = sbi_train(priors,
                          torch.Tensor(params),
                          torch.Tensor(COHs),
                          config.VERBOSITY)
    return posterior, COHs, params, priors


def target_COHgammaPathway_fun(config, target=0.5):
    target = target*np.ones((16, ))
    if config.COHERENCE_FISHER_Z_TRANSFORM:
        target = np.arctanh(target)
    return torch.Tensor(target)


def load_Popa_etal_COH(config):
    with open(os.path.join(config.TARGET_PSD_POPA_PATH, 'COH.npy'), 'rb') as f:
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
        # TODO: Update when we have the CEREBOFF Popa et al data:
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
        # TODO: Update when we have the CEREBOFF Popa et al data:
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
        # TODO: Update when we have the CEREBOFF Popa et al data:
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
        # TODO: Update when we have the CEREBOFF Popa et al data:
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


def estimate_posterior_samples(target, target_fun, posterior, config, n_samples_per_run=None, verbosity=1):
    if n_samples_per_run is None:
        n_samples_per_run = config.N_POSTERIOR_SAMPLES_PER_RUN
    return sbi_estimate(posterior, target_fun(config, target), n_samples_per_run, verbosity)


def load_posterior_samples_all_runs(runs=None, label="", samples=None, config=None):
    config = assert_config(config, return_plotter=False)
    if samples is None:
        samples = OrderedDict()
    if runs is None:
        runs = list(range(config.N_FIT_RUNS))
    for iR in ensure_list(runs):
        try:
            samples_iR = load_posterior_samples(iG=None, iR=iR, label=label, config=config)
            samples = add_posterior_samples_iR(samples, samples_iR)
        except Exception as e:
            warnings.warn("Failed to load posterior samples for iR=%d!\n%s" % (iR, str(e)))
    return samples


def iRstrfun(iR):
    if iR is None:
        iRstr = ""
    else:
        iRstr = "%02d" % iR
    return iRstr


def plot_infer(runs=None, samples=None, sampleslabel="", figlabel="", config=None):
    config = assert_config(config, return_plotter=False)

    if samples is None:
        samples = load_posterior_samples_all_runs(runs, sampleslabel, samples, config)

    n_samples_runs = len(samples[config.OPT_RES_MODE])
    if runs is None:
        iiR = slice(None)
        n_runs = n_samples_runs
    else:
        iiR = ensure_list(runs)
        n_runs = len(iiR)
    if n_runs == n_samples_runs:
        iiR = slice(None)
    elif n_runs > n_samples_runs:
        raise ValueError("The number of runs requested to plot (=%d) "
                         "are more than the runs (=%d) in the given or loaded samples!" % (n_runs, n_samples_runs))

    # Get the default values for the parameter
    if config.figures.SAVE_FLAG:
        figname = 'params_pairplot'
        if len(sampleslabel):
            figname += "_%s" % sampleslabel
        if len(figlabel):
            figname += "_%s" % figlabel
        figname += '.png'
    return params_pairplot_from_samples_fit_dict(samples, points=None, metric=None, inds=iiR,
                                                 config=config, figname=figname, figpath=None)


def infer_for_iR(sim_res_fun, target_fun, target=None,
                 n_training_samples_per_run=None, n_posterior_samples_per_run=None,
                 samples_fit=None, iR=None, path=None, assert_params=True,
                 label="", measure_labels=None, config=None, plot_flag=True):
    ticR = time.time()
    iRstr = iRstrfun(iR)
    if config.VERBOSITY:
        print("\n\nFitting %s!..\n" % iRstr)
    config = assert_config(config, return_plotter=False)
    posterior, COHs, params, priors = \
        train_posterior(sim_res_fun, config, Nsims=n_training_samples_per_run,
                        path=path, assert_params=assert_params,
                        target_fun=target_fun, target=target, label=label,
                        measure_labels=measure_labels, plot_flag=plot_flag)
    write_posterior(posterior, iG=None, iR=iR, label=label, config=config)
    posterior, posterior_samples, map = \
        estimate_posterior_samples(target, target_fun, posterior, config, n_posterior_samples_per_run, config.VERBOSITY)
    results = compute_diagnostics(posterior_samples, config, priors=priors, map=map, ground_truth=None)
    results["COHs"] = COHs
    results["params"] = params
    samples_fit = write_posterior_samples(results, config,
                                          iG=None, iR=iR, label=label,
                                          samples_fit=samples_fit, save_samples=True)
    # Plot posterior:
    if config.VERBOSITY:
        print("Plotting...")
    plot_infer(runs=iR, samples=samples_fit, sampleslabel=label, figlabel=iRstr, config=config)
    if config.VERBOSITY:
        print("DONE with %s in %g sec!" % (iRstr, time.time() - ticR))
    return samples_fit


def infer(sim_res_fun, target_fun, target=None, path=None, assert_params=True,
          label="", measure_labels=None, config=None, plot_flag=True):
    config = assert_config(config, return_plotter=False)
    samples_fit = None
    if config.N_FIT_RUNS:
        for iR in range(config.N_FIT_RUNS):
            # For every fitting run...
            samples_fit = infer_for_iR(sim_res_fun, target_fun, target,
                                       samples_fit=samples_fit, iR=iR,
                                       n_training_samples_per_run=config.N_TRAINING_SAMPLES_PER_RUN,
                                       n_posterior_samples_per_run=config.N_POSTERIOR_SAMPLES_PER_RUN,
                                       path=path, assert_params=assert_params,
                                       label=label, measure_labels=measure_labels,
                                       config=config, plot_flag=plot_flag)
        # Plot with samples from all runs!:
        if config.VERBOSITY:
            print("Plotting samples from all %d runs together..." % config.N_FIT_RUNS)
        plot_infer(runs=None, samples=samples_fit,
                   sampleslabel=label, figlabel="Allruns", config=config)

    if config.VERBOSITY:
        print("\n\nFitting with all samples!..\n")
    samples_fit_all = infer_for_iR(sim_res_fun, target_fun, target,
                                  samples_fit=None, iR=None,
                                  n_training_samples_per_run=None,
                                  n_posterior_samples_per_run=config.N_FIT_RUNS*config.N_POSTERIOR_SAMPLES_PER_RUN,
                                  path=path, assert_params=assert_params,
                                  label=label + "Allsamples", measure_labels=measure_labels,
                                  config=config, plot_flag=plot_flag)

    return samples_fit_all, samples_fit


# -------------------------------------------- TODO: REMOVE THESE!: ----------------------------------------------------
#
# def load_allsims_to_xarrays(config, **kwargs):
#     config = assert_config(config, return_plotter=False, **kwargs)
#     res_files = glob.glob(path)
#     # TODO: Make this temporary hack more robust!!!:
#     sims = []
#     for res_file in res_files:
#         sims.append(int(res_file.split("_")[-1].split(".pkl")[0]))
#     sims = np.sort(sims)
#     res = []
#     failed = []
#     ifailed = []
#     for iiR, iR in enumerate(sims):
#         try:
#             res.append(loadsim_to_xarrays(rest_simres_filepath(iR, config)))
#         except Exception as e:
#             failed.append(iR)
#             ifailed.append(iiR)
#             warnings.warn(str(e))
#             continue
#     sims = np.delete(sims, ifailed)
#     res = concat(res, dim=Index(sims, name="Simulation"))
#
#     nFailed = len(failed)
#     if nFailed:
#         warnings.warn("There are %d simulations that failed to provide a sample!:\n%s" % (nFailed, str(failed)))
#
#     return PSDs, failed
#
# def params_path_fun(iR=None, path=None):
#     if path is None:
#         path = os.getcwd()
#     if iR is not None:
#         return os.path.join(path, "PRIORS/res/ps_%05d.pt" % iR)
#     else:
#         return os.path.join(path, "PRIORS/res/ps_*.pt")
#
#
# def assert_params_fun(config, params):
#     for iP, p in enumerate(config["PRIORS_PARAMS_NAMES"]):
#
#         if iP < 2:
#             pval = config["model_params"][p]
#         else:
#             pval = config[p]
#         try:
#             #         print(p)
#             #         print(pval)
#             #         print(params[iP])
#             assert pval == params[iP]
#         except Exception as e:
#             print(p)
#             print(pval)
#             print(params[iP])
#             raise e
#
#
# def load_allsims_to_xarrays(Nsims=None, conds=["TVB", "TVB_CEREBOFF"], path=None, assert_params=True):
#     res_files = glob.glob(sim_res_path_fun(iR=None, path=path, mode=conds[0])[0])
#     # TODO: Make this temporary hack more robust!!!:
#     NallSims = int(np.sort(res_files)[-1].split("_")[-1].split(".pkl")[0])
#     # NallSims = len(res_files)
#     if Nsims is None:
#         Nsims = NallSims
#     if Nsims < NallSims:
#         sims = np.random.permutation(NallSims)[:Nsims]
#     else:
#         if Nsims > NallSims:
#             warnings.warn("There are no %d samples available! Switching to %d samples!" % (Nsims, NallSims))
#             Nsims = NallSims
#         sims = np.arange(Nsims).astype("i")
#
#     if Nsims == 0:
#         _, config, _ = loadsim(0, mode=conds[0], path=path, assert_params=False)
#
#     params = []
#     # TSs = []
#     # PSDs = []
#     COHs = []
#     # TMs = []
#
#     failed = []
#     ifailed = []
#     for iiR, iR in enumerate(sims):
#         COHiR = []
#         try:
#             for iC, cond in enumerate(conds):
#                 # TS1, PSD1, COH1, TM1, param, config
#                 COHiC, param, config = \
#                     loadsim_to_xarrays(iR, mode=cond, path=path, assert_params=assert_params)
#                 COHiR.append(COHiC)
#         except Exception as e:
#             failed.append(iR)
#             ifailed.append(iiR)
#             warnings.warn(str(e))
#             continue
#         params.append(param)
#         # TSs.append(concat([TS1, TS2], dim=Index(conds, name="Condition")))
#         # PSDs.append(concat([PSD1, PSD2], dim=Index(conds, name="Condition")))
#         COHs.append(concat(COHiR, dim=Index(conds, name="Condition")))
#         # TMs.append(concat([TM1, TM2], dim=Index(conds, name="Condition")))
#     sims = np.delete(sims, ifailed)
#
#     params = DataArray(np.array(params).T, dims=["Parameter", "Simulation"],
#                        coords={"Parameter": config['PRIORS_PARAMS_NAMES'],
#                                "Simulation": sims})
#     # TSs = concat(TSs, dim=Index(sims, name="Simulation"))
#     # TSs = TSs.transpose(*(tuple(np.array(TSs.dims)[[1, 0]].tolist()) + TSs.dims[2:]))
#     # PSDs = concat(PSDs, dim=Index(sims, name="Simulation"))
#     # PSDs = PSDs.transpose(*(tuple(np.array(PSDs.dims)[[1, 0]].tolist()) + PSDs.dims[2:]))
#     COHs = concat(COHs, dim=Index(sims, name="Simulation"))
#     COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
#     # TMs = concat(TMs, dim=Index(sims, name="Simulation"))
#     # TMs = TMs.transpose(*(tuple(np.array(TMs.dims)[[1, 0]].tolist()) + TMs.dims[2:]))
#
#     nFailed = len(failed)
#     if nFailed:
#         warnings.warn("There are %d simulations that failed to provide a sample!:\n%s" % (nFailed, str(failed)))
#
#     return COHs, params, failed  # TSs, PSDs, COHs, TMs, params, failed
#
#
#
#
# def sim_res_PSD_fun(sim_res, config):
#     return sim_res.stack(Regions=("Region", "f")).values


# def load_priors_and_simulations_for_sbi(iG=None, priors=None, train_params_samples=None, sim_res=None, config=None, label=""):
#     config = assert_config(config, return_plotter=False)
#     if priors is None:
#         priors = build_priors(config)
#     # Load priors' samples if not given in the input:
#     if train_params_samples is None:
#         # Load priors' samples
#         train_params_samples = load_priors_samples(config=config, iR=None, iG=iG, label=label)
#     # Load priors' samples if not given in the input:
#     if sim_res is None:
#         # Load priors' samples
#         sim_res = []
#         for iR in range(config.N_SIMULATIONS):
#             sim_res.append(load_pickled_dict(rest_simres_filepath(iR, config))["PSD"])
#         sim_res = torch.from_numpy(np.concatenate(sim_res).astype('float32'))
#     return priors, train_params_samples, sim_res


# def load_posterior_samples_all_Gs(iGs=None, runs=None, label="", config=None):
#     config = assert_config(config, return_plotter=False)
#     samples = OrderedDict()
#     if iGs is None:
#         iGs = range(len(config.Gs))
#     for iG in iGs:
#         G = config.Gs[iG]
#         try:
#             if runs is False:
#                 samples[G] = load_posterior_samples(iG, None, label, config)
#             else:
#                 samples[G] = load_posterior_samples_all_runs(iG, runs, label, config=config)
#         except Exception as e:
#             warnings.warn("Failed to load posterior samples for iG=%d, G=%g!\n%s" % (iG, G, str(e)))
#     return samples
#
#
# def load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config,
#                                                 target=None, label=""):
#     samples_fit = load_posterior_samples_all_runs(None, label, None, config)
#     samples = np.hstack(samples_fit["samples"])[0].copy()
#     map_or_mean = np.hstack(samples_fit[config.OPT_RES_MODE])[0].copy().mean(axis=0)
#     del samples_fit
#     basepath = config.out.FOLDER_RES.split("/res")[0]
#     pptpath = os.path.join(basepath, "PPC", label)
#     cereboffpaths = glob.glob(os.path.join(pptpath, "TVB_CEREBOFF_[0-9]*/nsd*/res/res_*.pkl"))
#     Nsims = len(cereboffpaths)
#     print("Number of PPC simulations: %d" % Nsims)
#     COHs = []
#     sample_inds = []
#     for cereboffpath in cereboffpaths:
#         tvbpath = cereboffpath.replace("_CEREBOFF", "")
#         try:
#             sample_inds.append(int(tvbpath.split("TVB_")[-1][:5]))
#         except Exception as e:
#             raise ValueError("No sample number found in path %s!\n" % tvbpath + str(e))
#         COHsCond = []
#         for path in [tvbpath, cereboffpath]:
#             COHsCond.append(coh_to_xarray(load_pickled_dict(path)))
#         COHs.append(concat(COHsCond, dim=Index(["TVB", "TVB_CEREBOFF"], name="Condition")))
#     COHs = concat(COHs, dim=Index(sample_inds, name="Sample"))
#     COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
#     COHs = sim_res_fun(COHs, config)
#     target = target_fun(config, target).numpy()
#     try:
#         assert COHs.shape[0] == Nsims
#     except Exception as e:
#         print(COHs.shape)
#         raise e
#     return COHs, target, map_or_mean, samples[sample_inds]
#
#
# def plot_posterior_predictive_check(sim_res_fun, target_fun, target=None, label="", measure_labels=None, config=None):
#     config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
#     COHS, target, map_or_mean, samples = \
#         load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config, target, label)
#     fig1, axes1 = sbi_pairplot(COHs, points=target, metric="target", labels=measure_labels,
#                                figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
#                                                     "PPC", label, 'ppc_measures_pairplot.png'),
#                                save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
#     fig2, axes2 = params_pairplot(samples, points=map_or_mean, metric=config.OPT_RES_MODE, config=config,
#                                   figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
#                                                        "PPC", label, "ppc_params_pairplot.png"))
#     return fig1, axes1, fig2, axes2
#
#
# def stats_simulation(metric, metric_vals, statsName, sim_res_fun, target_fun, config, target=None,
#                      Nsims=1, label="", iR=0, measure_labels=None, **kwargs):
#     results = {}
#     pathlabel = "%s/%s/%s" % (statsName, label, metric)
#     fitpathlabel = "FIT/%s" % pathlabel
#     kwargs.update(dict(zip(config.PRIORS_PARAMS_NAMES, metric_vals)))
#     tests = ["TVB", "TVB_CEREBOFF"]
#     COHs = []
#     sims = np.arange(iR * Nsims, (iR + 1) * Nsims).astype('i')
#     for test in tests:
#         results[test] = []
#     for iS in range(Nsims):
#         COHsCond = []
#         for test in tests:
#             res = cosim_run_plot(iR=sims[iS],
#                                  MODE="%s/%s" % (fitpathlabel, test), **kwargs)[0]
#             COHsCond.append(coh_to_xarray(res))
#             results[test].append(res)
#         COHs.append(concat(COHsCond, dim=Index(tests, name="Condition")))
#     plot_comparison(tests, MODE=fitpathlabel)
#     COHs = concat(COHs, dim=Index(sims, name="Simulation"))
#     COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
#     COHs = sim_res_fun(COHs, config)
#     target = target_fun(config, target)
#     fig, axes = sbi_pairplot(COHs, points=target, metric="target", labels=measure_labels,
#                             figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
#                                                  pathlabel, "-".join(tests), "pairplot%d.png" % iR),
#                              save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
#     return results
#
#
# def map_or_mean_simulations(sim_res_fun, target_fun, target=None,
#                             runs=None, Nsims=1, map_mean=None, label="", config=None, measure_labels=None, **kwargs):
#     results = {}
#     config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
#     if map_mean is None or len(map_mean) == 0:
#         map_mean = [config.OPT_RES_MODE]
#     else:
#         map_mean = ensure_list(map_mean)
#     samples_fit = load_posterior_samples_all_runs(None, label, None, config)
#     if runs:
#         for iR in runs:
#             results[iR] = {}
#             for metric in map_mean:
#                 metric_vals = samples_fit[metric][iR][0].squeeze()
#                 results[iR][metric] = stats_simulation(metric, metric_vals, "MAPmean",
#                                                        sim_res_fun, target_fun, config,
#                                                        target=target, Nsims=Nsims, label=label, iR=iR+1,
#                                                        measure_labels=measure_labels, **kwargs)
#     for metric in map_mean:
#         metric_vals = np.vstack(samples_fit[metric]).mean(axis=0).squeeze()
#         results[metric] = stats_simulation(metric, metric_vals, "MAPmean",
#                                            sim_res_fun, target_fun, config,
#                                            target=target, Nsims=Nsims, label=label,
#                                            measure_labels=measure_labels, **kwargs)
#     return results
#
#
# def ppc_best_simulations(sim_res_fun, target_fun, target=None,
#                          Nmetrics=1, Nsims=1, label="", config=None, measure_labels=None, **kwargs):
#     config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
#     COHs, target, _, samples = \
#         load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config, target, label)
#     target = target[np.newaxis]
#     dist = np.sqrt(np.sum((COHs-target)**2, axis=1))
#     inds = np.argsort(dist, axis=0)[:Nmetrics]
#     results = {}
#     for iS in range(Nmetrics):
#         results[iS] = stats_simulation("%02dbestPPCsim" % (iS+1),
#                                        samples[inds[iS]],
#                                        "PPC",
#                                        sim_res_fun, target_fun, config,
#                                        target=target, Nsims=Nsims, label=label,
#                                        measure_labels=measure_labels, **kwargs)
#     return results
#
#
# if __name__ == '__main__':
#     # Example use:
#     # $ python task_run_fit_plot.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
#     # Called tuning_tvb_nest.py with:
#     # keyword argument: w_TVB_to_NEST=world
#     # keyword argument: simulation_length=300.0
#
#     import sys
#
#     MODE = ""
#
#     kwargs = {}
#     ntests = 0
#     for arg in sys.argv[1:]:
#         keyval = arg.split("=")
#         if keyval[0] not in ["MODE", "label"]:
#             key = float(keyval[1])
#         else:
#             key = keyval[1].split(" ")
#             if keyval[0] == "MODE":
#                 if key[0] == "PPC":
#                     MODE = "PPC"
#                     continue
#                 else:
#                     ntests = len(key)
#                     if ntests == 1:
#                         key = key[0]
#             elif keyval[0] == "label":
#                 key = key[0]
#         kwargs[keyval[0]] = key
#
#     if MODE == "PPC":
#         posterior_predictive_check_simulations(**kwargs)
#     else:
#         cosim_run_plot(**kwargs)


if __name__ == "__main__":
    parser, defargs = run_fit_plot_args_parser("task_run_fit_plot")
    args, parser_args, parser = parse_args(parser, argsnames=list(defargs.keys()))
    funcname = args.pop("function", "sim_run_plot")
    verbosity = args.get('verbosity', defargs['verbosity'])
    if verbosity:
        print("Running function %s from script %s with REST_or_TASK='TASK' "
              "and user provided arguments:\n" % (funcname, parser.description))
        print(args, "\n")
    globals()[funcname](REST_or_TASK="TASK", **args)
