# coding: utf-8

import warnings
import glob
import pickle
import os
import shutil
from matplotlib import pyplot
import torch
import sbi
from xarray import DataArray, concat
from pandas import Index
from scipy.interpolate import interp1d

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.sbi_script import \
    build_priors, prepare_for_sbi, simulate_for_sbi, sbi_train, sbi_estimate, \
    write_posterior, write_posterior_samples, add_posterior_samples_iR, load_posterior_samples, compute_diagnostics
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *
from rising_net.scripts.cosim_run_plot import plot_comparison

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import load_pickled_dict

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
    #     'output_folder': "", 'verbosity': 1, 'plot_flag': True}

    # Get configuration
    if "PRIORS" in kwargs.get("MODE", "") and iR is not None:
        config = configure(MODE="PRIORS", plot_flag=False, verbosity=0)[0]
        priors = dict(zip(config.PRIORS_PARAMS_NAMES, load_priors_samples(iR, config).numpy().squeeze()))
        print("PRIORS_%05d:\n%s" % (iR, str(priors)))
        kwargs.update(priors)
        kwargs["plot_flag"] = False
        verbosity = 0
    else:
        verbosity = kwargs.pop("verbosity", 1)

    config, plotter = configure(verbosity=0, SEED=iR, **kwargs)

    config.VERBOSITY = verbosity
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

    if isinstance(results, (list, tuple)):
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=False)

    n_transient = int(np.ceil(transient / results["source_ts"].sample_period))

    source_ts = results["source_ts"][n_transient:, [0], config.TASKINDS]
    results["source_ts"] = source_ts
    taskinds = np.arange(source_ts.shape[2]).astype("i")

    # Power Spectra and Coherence for M1 - S1 barrel field
    PSD, COH, f, pairs = \
        compute_selected_spectra_coherence(results["source_ts"], taskinds, results["source_ts"].sample_period,
                                           transient=0, nperseg=1024, ftarg=config.FREQS)
    results["PSD"] = PSD
    results["f"] = f
    results["COH"] = COH
    results["pairs"] = pairs

    # TaskMetrics = compute_task_transfer_metrics(results["source_ts"], 0,
    #                                             simulator.connectivity.region_labels[taskinds],
    #                                             taskinds, config.THETA, config.GAMMA, config.FREQS,
    #                                             Pxx_den=PSD, methods=(5, 2, 3), plot_flag=False)
    # results["TaskMetrics"] = TaskMetrics

    dump_pickled_dict(results, sim_res_filepath(iR, config))

    return results, simulator, nest_network, config, inds


def params_path_fun(iR=None, path=None):
    if path is None:
        path = os.getcwd()
    if iR is not None:
        return os.path.join(path, "PRIORS/res/ps_%05d.pt" % iR)
    else:
        return os.path.join(path, "PRIORS/res/ps_*.pt")


def sim_res_path_fun(iR=None, path=None, mode="TVB"):
    if path is None:
        path = os.getcwd()
    if iR is not None:
        ipath = os.path.join(path, mode, "nsd%05d" % iR, "res")
        return os.path.join(ipath, "res_%05d.pkl" % iR), os.path.join(ipath, "config.pkl")
    else:
        ipath = os.path.join(path, mode, "nsd*", "res")
        return os.path.join(ipath, "res_*.pkl"), os.path.join(ipath, "config.pkl")


def all_paths_fun(iR=None, path=None, mode="TVB"):
    if path is None:
        path = os.getcwd()
    params_path = params_path_fun(iR, path)
    res_path, config_path = sim_res_path_fun(iR, path, mode)
    return params_path, res_path, config_path


def assert_params_fun(config, params):
    for iP, p in enumerate(config["PRIORS_PARAMS_NAMES"]):

        if iP < 2:
            pval = config["model_params"][p]
        else:
            pval = config[p]
        try:
            #         print(p)
            #         print(pval)
            #         print(params[iP])
            assert pval == params[iP]
        except Exception as e:
            print(p)
            print(pval)
            print(params[iP])
            raise e


def loadsim(iR, mode="TVB", path=None, assert_params=True):
    params_path, res_path, config_path = all_paths_fun(iR, path, mode)

    params = torch.load(params_path).numpy().squeeze()
    config = load_pickled_dict(config_path)
    res = load_pickled_dict(res_path)

    if assert_params:
        assert_params_fun(config, params)

    return params, config, res


def coh_to_xarray(res):
    COH = DataArray(np.zeros((20, 20, 96)), dims=["Region1", "Region2", "f"],
                    coords={"Region1": res["source_ts"].labels_dimensions["Region"],
                            "Region2": res["source_ts"].labels_dimensions["Region"],
                            "f": res["f"]},
                    name="COH")
    for iP, pair in enumerate(res["pairs"]):
        COH[pair[0], pair[1]] = res["COH"][iP]
    return COH


def loadsim_to_xarrays(iR, mode="TVB", path=None, assert_params=True):
    if path is None:
        path = os.getcwd()
    params, config, res = loadsim(iR, mode=mode, path=path, assert_params=assert_params)

    # TS = res["source_ts"]._data[:, :, :, 0].transpose("Region", "Time", "State Variable")

    # PSD = DataArray(res["PSD"], dims=["Region", "f"],
    #                 coords={"Region": res["source_ts"].labels_dimensions["Region"],
    #                         "f": res["f"]},
    #                 name="PSD")

    # TM = res["TaskMetrics"].T

    return coh_to_xarray(res), params, config  # TS, PSD, COH, TM, params, config


def load_allsims_to_xarrays(Nsims=None, conds=["TVB", "TVB_CEREBOFF"], path=None, assert_params=True):
    res_files = glob.glob(sim_res_path_fun(iR=None, path=path, mode=conds[0])[0])
    # TODO: Make this temporary hack more robust!!!:
    NallSims = int(np.sort(res_files)[-1].split("_")[-1].split(".pkl")[0])
    # NallSims = len(res_files)
    if Nsims is None:
        Nsims = NallSims
    if Nsims < NallSims:
        sims = np.random.permutation(NallSims)[:Nsims]
    else:
        if Nsims > NallSims:
            warnings.warn("There are no %d samples available! Switching to %d samples!" % (Nsims, NallSims))
            Nsims = NallSims
        sims = np.arange(Nsims).astype("i")

    if Nsims == 0:
        _, config, _ = loadsim(0, mode=conds[0], path=path, assert_params=False)

    params = []
    # TSs = []
    # PSDs = []
    COHs = []
    # TMs = []

    failed = []
    ifailed = []
    for iiR, iR in enumerate(sims):
        COHiR = []
        try:
            for iC, cond in enumerate(conds):
                # TS1, PSD1, COH1, TM1, param, config
                COHiC, param, config = \
                    loadsim_to_xarrays(iR, mode=cond, path=path, assert_params=assert_params)
                COHiR.append(COHiC)
        except Exception as e:
            failed.append(iR)
            ifailed.append(iiR)
            warnings.warn(str(e))
            continue
        params.append(param)
        # TSs.append(concat([TS1, TS2], dim=Index(conds, name="Condition")))
        # PSDs.append(concat([PSD1, PSD2], dim=Index(conds, name="Condition")))
        COHs.append(concat(COHiR, dim=Index(conds, name="Condition")))
        # TMs.append(concat([TM1, TM2], dim=Index(conds, name="Condition")))
    sims = np.delete(sims, ifailed)

    params = DataArray(np.array(params).T, dims=["Parameter", "Simulation"],
                       coords={"Parameter": config['PRIORS_PARAMS_NAMES'],
                               "Simulation": sims})
    # TSs = concat(TSs, dim=Index(sims, name="Simulation"))
    # TSs = TSs.transpose(*(tuple(np.array(TSs.dims)[[1, 0]].tolist()) + TSs.dims[2:]))
    # PSDs = concat(PSDs, dim=Index(sims, name="Simulation"))
    # PSDs = PSDs.transpose(*(tuple(np.array(PSDs.dims)[[1, 0]].tolist()) + PSDs.dims[2:]))
    COHs = concat(COHs, dim=Index(sims, name="Simulation"))
    COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
    # TMs = concat(TMs, dim=Index(sims, name="Simulation"))
    # TMs = TMs.transpose(*(tuple(np.array(TMs.dims)[[1, 0]].tolist()) + TMs.dims[2:]))

    nFailed = len(failed)
    if nFailed:
        warnings.warn("There are %d simulations that failed to provide a sample!:\n%s" % (nFailed, str(failed)))

    return COHs, params, failed  # TSs, PSDs, COHs, TMs, params, failed


def define_colors(N):
    from cycler import cycler
    import matplotlib as mpl

    cmap = mpl.colormaps['jet']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, N))

    custom_cycler = cycler(color=colors)  # or simply color=colorlist
    plt.rc('axes', prop_cycle=custom_cycler)

    return custom_cycler


def pathway_pairs_fun():
    pathway_pairs_R = [[0, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [0, 13], [13, 18]]
    pathway_pairs_L = [[1, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [1, 12], [12, 19]]
    pathway_pairs = np.array(pathway_pairs_R + pathway_pairs_L)
    return pathway_pairs


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


def params_pairplot(samples, points=None, metric=None, config=None, figname=None, figpath=None):
    config = assert_config(config, return_plotter=False)
    limits = []
    ticks = []
    labels = []
    if points is not None:
        for pmin, point, pmax in zip(config.prior_min, points, config.prior_max):
            limits.append([pmin, pmax])
            ticks.append(np.sort([pmin, point, pmax]).tolist())
        if metric is None:
            metric = config.OPT_RES_MODE
        for p, point in zip(config.PRIORS_PARAMS_NAMES, points):
            try:
                labels.append("%s %s = %g" % (p, metric, point))
            except Exception as e:
                print(p)
                print(metric)
                print(point)
                print(point.shape)
                raise e
    else:
        for pmin, pmax in zip(config.prior_min, config.prior_max):
            limits.append([pmin, pmax])
        ticks = deepcopy(limits)
        for p in config.PRIORS_PARAMS_NAMES:
            try:
                labels.append("%s" % p)
            except Exception as e:
                print(p)
                raise e
    if config.figures.SAVE_FLAG:
        if figpath is None:
            if figname is None:
                figname = "params_pairplot.png"
            figpath = os.path.join(config.figures.FOLDER_FIGURES, figname)
    return sbi_pairplot(samples, figpath=figpath,
                        save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG,
                        limits=limits, ticks=ticks, points=points, labels=labels)


def params_pairplot_from_samples_fit_dict(samples_fit, points=None, metric=None, inds=slice(None),
                                          config=None, figname=None, figpath=None):
    config = assert_config(config, return_plotter=False)
    samples = np.hstack(np.array(samples_fit['samples'])[inds].tolist()).squeeze()
    if metric is None:
        metric = config.OPT_RES_MODE
    try:
        if points is None:
            points = np.concatenate(np.array(samples_fit[config.OPT_RES_MODE])[inds].tolist()).mean(axis=0).squeeze()
    except Exception as e:
        warning.warn("Failed to get metric %s!\n%s" % str(e))
        metric = None
        points = None
    return params_pairplot(samples, points=points, metric=metric, config=config, figname=figname, figpath=figpath)


def plot_samples_measures_and_targets(config, params=None, COHs=None,
                                      Nsims=None, path=None, assert_params=True,
                                      sim_res_fun=None, target_fun=None, target=None,
                                      label="", measure_labels=None):
    if params is None or COHs is None:
        COHs, params = sim_res_fun(config, Nsims=Nsims, path=path, assert_params=assert_params)
        params = params.T
    fig1, axes1 = params_pairplot(params, points=params.mean(axis=0), metric="mean",
                                  config=config, figname="%s_%s" % (label, 'priors_pairplot.png'),
                                  figpath=None)
    metric = "target"
    points = target
    if points is None:
        if target_fun is not None:
            points = target_fun(config, target)
        else:
            metric = "mean"
            points = COHs.mean(axis=0)
    fig2, axes2 = sbi_pairplot(COHs, points=points, metric=metric, labels=measure_labels,
                              figpath=os.path.join(config.figures.FOLDER_FIGURES,
                                                   "%s_%s" % (label, 'measures_pairplot.png')),
                              save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
    return fig1, axes1, fig2, axes2


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


def posterior_predictive_check_simulations(label="", config=None, **kwargs):
    config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
    samples_fit = load_posterior_samples_all_runs(None, label, None, config)
    samples = np.hstack(samples_fit["samples"])[0].copy()
    n_samples = samples.shape[0]
    sampleslist = list(range(n_samples))
    del samples_fit
    basepath = config.out.FOLDER_RES.split("/res")[0]
    ppcpath = os.path.join(basepath, "PPC", label)
    paths = glob.glob(os.path.join(ppcpath, "TVB_[0-9]*"))
    runs = []
    for path in paths:
        runs.append(int(path.split("_")[-1]))
    iR = len(runs)
    sampleslist = np.delete(sampleslist, runs).tolist()
    sampl_ind = random.sample(sampleslist, 1)[0]
    # Now choose N_PPT_SIMS_PER_BATCH randomly among the samples meant for this batch
    samples = samples[sampl_ind].squeeze()
    kwargs.update(dict(zip(config.PRIORS_PARAMS_NAMES, samples)))
    results = {}
    for mode in ["TVB", "TVB_CEREBOFF"]:
        results[mode] = gain_run_plot(iR=iR,
                                      MODE="FIT/PPC/%s/%s_%05d" % (label, mode, sampl_ind),
                                      plot_flag=False,
                                      **kwargs)
    return results


def load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config,
                                                target=None, label=""):
    samples_fit = load_posterior_samples_all_runs(None, label, None, config)
    samples = np.hstack(samples_fit["samples"])[0].copy()
    map_or_mean = np.hstack(samples_fit[config.OPT_RES_MODE])[0].copy().mean(axis=0)
    del samples_fit
    basepath = config.out.FOLDER_RES.split("/res")[0]
    pptpath = os.path.join(basepath, "PPC", label)
    cereboffpaths = glob.glob(os.path.join(pptpath, "TVB_CEREBOFF_[0-9]*/nsd*/res/res_*.pkl"))
    Nsims = len(cereboffpaths)
    print("Number of PPC simulations: %d" % Nsims)
    COHs = []
    sample_inds = []
    for cereboffpath in cereboffpaths:
        tvbpath = cereboffpath.replace("_CEREBOFF", "")
        try:
            sample_inds.append(int(tvbpath.split("TVB_")[-1][:5]))
        except Exception as e:
            raise ValueError("No sample number found in path %s!\n" % tvbpath + str(e))
        COHsCond = []
        for path in [tvbpath, cereboffpath]:
            COHsCond.append(coh_to_xarray(load_pickled_dict(path)))
        COHs.append(concat(COHsCond, dim=Index(["TVB", "TVB_CEREBOFF"], name="Condition")))
    COHs = concat(COHs, dim=Index(sample_inds, name="Sample"))
    COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
    COHs = sim_res_fun(COHs, config)
    target = target_fun(config, target).numpy()
    try:
        assert COHs.shape[0] == Nsims
    except Exception as e:
        print(COHs.shape)
        raise e
    return COHs, target, map_or_mean, samples[sample_inds]


def plot_posterior_predictive_check(sim_res_fun, target_fun, target=None, label="", measure_labels=None, config=None):
    config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
    COHS, target, map_or_mean, samples = \
        load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config, target, label)
    fig1, axes1 = sbi_pairplot(COHs, points=target, metric="target", labels=measure_labels,
                               figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
                                                    "PPC", label, 'ppc_measures_pairplot.png'),
                               save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
    fig2, axes2 = params_pairplot(samples, points=map_or_mean, metric=config.OPT_RES_MODE, config=config,
                                  figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
                                                       "PPC", label, "ppc_params_pairplot.png"))
    return fig1, axes1, fig2, axes2


def stats_simulation(metric, metric_vals, statsName, sim_res_fun, target_fun, config, target=None,
                     Nsims=1, label="", iR=0, measure_labels=None, **kwargs):
    results = {}
    pathlabel = "%s/%s/%s" % (statsName, label, metric)
    fitpathlabel = "FIT/%s" % pathlabel
    kwargs.update(dict(zip(config.PRIORS_PARAMS_NAMES, metric_vals)))
    tests = ["TVB", "TVB_CEREBOFF"]
    COHs = []
    sims = np.arange(iR * Nsims, (iR + 1) * Nsims).astype('i')
    for test in tests:
        results[test] = []
    for iS in range(Nsims):
        COHsCond = []
        for test in tests:
            res = gain_run_plot(iR=sims[iS],
                                MODE="%s/%s" % (fitpathlabel, test), **kwargs)[0]
            COHsCond.append(coh_to_xarray(res))
            results[test].append(res)
        COHs.append(concat(COHsCond, dim=Index(tests, name="Condition")))
    plot_comparison(tests, MODE=fitpathlabel)
    COHs = concat(COHs, dim=Index(sims, name="Simulation"))
    COHs = COHs.transpose(*(tuple(np.array(COHs.dims)[[1, 0]].tolist()) + COHs.dims[2:]))
    COHs = sim_res_fun(COHs, config)
    target = target_fun(config, target)
    fig, axes = sbi_pairplot(COHs, points=target, metric="target", labels=measure_labels,
                            figpath=os.path.join(config.figures.FOLDER_FIGURES.split("figs")[0],
                                                 pathlabel, "-".join(tests), "pairplot%d.png" % iR),
                             save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
    return results


def map_or_mean_simulations(sim_res_fun, target_fun, target=None,
                            runs=None, Nsims=1, map_mean=None, label="", config=None, measure_labels=None, **kwargs):
    results = {}
    config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
    if map_mean is None or len(map_mean) == 0:
        map_mean = [config.OPT_RES_MODE]
    else:
        map_mean = ensure_list(map_mean)
    samples_fit = load_posterior_samples_all_runs(None, label, None, config)
    if runs:
        for iR in runs:
            results[iR] = {}
            for metric in map_mean:
                metric_vals = samples_fit[metric][iR][0].squeeze()
                results[iR][metric] = stats_simulation(metric, metric_vals, "MAPmean",
                                                       sim_res_fun, target_fun, config,
                                                       target=target, Nsims=Nsims, label=label, iR=iR+1,
                                                       measure_labels=measure_labels, **kwargs)
    for metric in map_mean:
        metric_vals = np.vstack(samples_fit[metric]).mean(axis=0).squeeze()
        results[metric] = stats_simulation(metric, metric_vals, "MAPmean",
                                           sim_res_fun, target_fun, config,
                                           target=target, Nsims=Nsims, label=label,
                                           measure_labels=measure_labels, **kwargs)
    return results


def ppc_best_simulations(sim_res_fun, target_fun, target=None,
                         Nmetrics=1, Nsims=1, label="", config=None, measure_labels=None, **kwargs):
    config = assert_config(config, return_plotter=False, plot_flag=False, MODE="FIT")
    COHs, target, _, samples = \
        load_posterior_predictive_check_simulations(sim_res_fun, target_fun, config, target, label)
    target = target[np.newaxis]
    dist = np.sqrt(np.sum((COHs-target)**2, axis=1))
    inds = np.argsort(dist, axis=0)[:Nmetrics]
    results = {}
    for iS in range(Nmetrics):
        results[iS] = stats_simulation("%02dbestPPCsim" % (iS+1),
                                       samples[inds[iS]],
                                       "PPC",
                                       sim_res_fun, target_fun, config,
                                       target=target, Nsims=Nsims, label=label,
                                       measure_labels=measure_labels, **kwargs)
    return results


if __name__ == '__main__':
    # Example use:
    # $ python gain_run_fit_plot.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
    # Called tuning_tvb_nest.py with:
    # keyword argument: w_TVB_to_NEST=world
    # keyword argument: simulation_length=300.0

    import sys

    kwargs = {}
    ntests = 0
    for arg in sys.argv[1:]:
        keyval = arg.split("=")
        if keyval[0] not in ["MODE", "label"]:
            key = float(keyval[1])
        else:
            key = keyval[1].split(" ")
            if keyval[0] == "MODE":
                if key[0] == "PPC":
                    MODE = "PPC"
                    continue
                else:
                    ntests = len(key)
                    if ntests == 1:
                        key = key[0]
            elif keyval[0] == "label":
                key = key[0]
        kwargs[keyval[0]] = key

    if MODE == "PPC":
        posterior_predictive_check_simulations(**kwargs)
    else:
        gain_run_plot(**kwargs)
