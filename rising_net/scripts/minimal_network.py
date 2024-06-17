#!/usr/bin/env python
# coding: utf-8

import glob
import pickle
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        # build_NEST_network, plot_nest_results
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical


#
#
# def foldername(**kwargs):
#     name = ""
#     for key, val in kwargs.items():
#         name += "%s%d" % (key, intval(val))
#     return name


def get_config(**kwargs):

    # Assuming:
    # DEFAULT_ARGS = {'G': 6.0, 'STIMULUS': 0.1, 'STIMULUS_BASELINE': 1.0,
    #                 'I_e': -0.35, 'I_s': 0.085,
    #                 'w_ie': -3.0, 'w_rs': -2.0,
    #                 'CONN_LOG': True, 'FIC': 1.11,  'FIC_SPLIT': 0.31,  #'fit',
    #                 'PRIORS_DIST': 'uniform',
    #                 'output_folder': "", 'verbosity': 1, 'plot_flag': True}

    # config.TRANSIENT_RATIO =0.25

    simulation_length = kwargs.pop("simulation_length", 3000.0)
    STIMULUS = kwargs.get("STIMULUS", 0.5)
    STIMULUS_BASELINE = kwargs.get("STIMULUS_BASELINE", 1.0)
    NOISE = int(kwargs.pop("NOISE", 6))
    CNS1TH = float(kwargs.pop("CNS1TH", 1.0))
    PONS = float(kwargs.pop("PONS", 0.0))
    SENSTRIG = float(kwargs.pop("SENSTRIG", 1.0))
    G = float(kwargs.get("G", 6.0))
    FIC = float(kwargs.get("FIC", 0.0))  # 1.11
    SET_WEIGHTS = kwargs.pop("SET_WEIGHTS", True)
    HEMISPHERES = int(kwargs.pop("HEMISPHERES", -1))

    seed = int(kwargs.pop("seed", -1))
    if seed >= 0:
        SEED = True
    else:
        SEED = False
        seed = 10

    # "s1stim", "s1m1stim",
    # "trigs1stim", "trigs1m1stim"
    # "cerebON", "cerebOFF"
    test_name = kwargs.pop("test_name", "")

    experiment_name = ""

    if SET_WEIGHTS:
        experiment_name = "_".join([experiment_name, "SetW"])

    # experiment_name = "_".join([experiment_name,
    #                             "stimbase%d_stim%d_noise%d_G%d" %
    #                             (intval(STIMULUS_BASELINE), intval(STIMULUS), NOISE, intval(G))])

    experiment_name = "_".join([experiment_name,
                                "stimbase%d_PONS%d_noise%d_stim%d" %
                                (intval(STIMULUS_BASELINE), intval(PONS), NOISE, intval(STIMULUS))])

    if experiment_name[0] == "_":
        experiment_name = experiment_name[1:]

    if FIC:
        experiment_name = "_".join([experiment_name, "FIC"])

    if len(test_name):
        experiment_name = "_".join([experiment_name, test_name])
    else:
        test_name = "cerebON"

    path = os.path.join(os.getcwd(), experiment_name)

    if test_name == "cerebON":
        CEREB = 2
    elif "cereb" in test_name:
        CEREB = 1
    else:
        CEREB = 0
    if test_name.find("trig") > -1 or CEREB:
        TRIGEMINAL = True
    else:
        TRIGEMINAL = False
    if test_name.find("m1stim") > -1:
        M1STIM = True
    else:
        M1STIM = False

    if SEED:
        path = os.path.join(path, "nsd%d" % seed)

    # Get configuration
    config, plotter = configure(output_folder=path, verbosity=0,
                                NOISE=10 ** (-NOISE),
                                SIMULATION_LENGTH=simulation_length,
                                **kwargs)
    config.RANDOM_SEED_TVB = seed
    config.RANDOM_SEED_NEST = seed
    config.CEREB = CEREB
    config.CNS1TH = CNS1TH
    config.PONS = PONS
    config.SENSTRIG = SENSTRIG
    config.TRIGEMINAL = TRIGEMINAL
    config.M1STIM = M1STIM
    config.HEMISPHERES = HEMISPHERES
    config.SET_WEIGHTS = SET_WEIGHTS
    config.STIMULUS_RATE = 8.0  # Hz
    config.VERBOSITY = 2.0

    print(config.model_params)
    print(config)

    return config, plotter


def getflags_from_config(config):
    CEREB = getattr(config, "CEREB", 2)
    TRIGEMINAL = getattr(config, "TRIGEMINAL", True)
    M1STIM = getattr(config, "M1STIM", True)
    HEMISPHERES = getattr(config, "HEMISPHERES", -1)
    CNS1TH = getattr(config, "CNS1TH", 1.0)
    SENSTRIG = getattr(config, "SENSTRIG", 1.0)
    PONS = getattr(config, "PONS", 0.0)
    return CEREB, TRIGEMINAL, M1STIM, HEMISPHERES, CNS1TH, SENSTRIG, PONS


def newconn_and_inds(config, plotter):

    CEREB, TRIGEMINAL, M1STIM, HEMISPHERES, CNS1TH, SENSTRIG, PONS = getflags_from_config(config)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config,
                                      hemispheres=HEMISPHERES,
                                      cereb_nuclei_to_s1thal=False,
                                      trigeminal_to_m1thal=TRIGEMINAL and M1STIM)

    if not TRIGEMINAL:
        connectivity.weights[inds["trigeminal"]] = 0.0
        connectivity.weights[:, inds["trigeminal"]] = 0.0

    # Keep only the selected regions.
    indegree = {}
    taskinds = []
    regs = ["m1", "s1brl", "m1thal", "s1brlthal", "trigeminal"]
    if CEREB:
        regs += ["ansilob", "cereb_nuclei"]
    if SENSTRIG > 0.0:
        regs += ["ponssens_trigeminal"]
    if PONS > 0.0:
        regs += ["ponssens", "ponsmotor"]
    for reg in regs:
        for iH, hemi in zip([0, 1], ["right_", "left_"]):
            indegree[hemi + reg] = connectivity.weights[inds[reg][iH]].sum()
        taskinds += inds[reg].tolist()
    taskinds = np.sort(taskinds)

    def newind(ind, inds):
        return inds.tolist().index(ind)

    connectivity.region_labels = connectivity.region_labels[taskinds]
    connectivity.centres = connectivity.centres[taskinds]
    # connectivity.orientations = connectivity.orientations[taskinds]
    # connectivity.areas = connectivity.areas[taskinds]
    connectivity.hemispheres = connectivity.hemispheres[taskinds]
    # connectivity.cortical = connectivity.cortical[taskinds]
    connectivity.weights = connectivity.weights[taskinds][:, taskinds]
    connectivity.tract_lengths = connectivity.tract_lengths[taskinds][:, taskinds]
    connectivity.configure()

    newinds = {}
    newtaskinds = []
    for reg in regs:
        newinds[reg] = []
        for iH, hemi in zip([0, 1], ["right_", "left_"]):
            ind = newind(inds[reg][iH], taskinds)
            #         if ("thal" not in reg or "w" not in config.THAL_CRTX_FIX) and indegree[hemi + reg] > 0.0:
            #             if reg in ["m1", "s1brl"] and "w" in config.THAL_CRTX_FIX:
            #                 connectivity.weights[ind] *= (indegree[hemi + reg] -1.0) / (connectivity.weights[ind].sum() - 1.0)
            #             else:
            #                 connectivity.weights[ind] *= indegree[hemi + reg] / connectivity.weights[ind].sum()
            #                 assert np.isclose(connectivity.weights[ind].sum(), indegree[hemi + reg])
            newtaskinds.append(ind)
            newinds[reg].append(ind)
        newinds[reg] = np.sort(newinds[reg])

    newinds["m1s1brl"] = np.sort(np.concatenate([newinds["m1"], newinds["s1brl"]]))
    newinds["crtx"] = newinds["m1s1brl"]
    newinds["thalspec"] = np.sort(np.concatenate([newinds["m1thal"], newinds["s1brlthal"]]))
    newinds["subcrtx"] = np.sort(np.concatenate([newinds["thalspec"],
                                                 newinds["trigeminal"] if TRIGEMINAL else []])).astype('i')
    newinds['crtx_and_subcrtx'] = np.sort(np.concatenate([newinds["crtx"],
                                                          newinds["trigeminal"] if TRIGEMINAL else []])).astype('i')
    if CEREB:
        newinds["cereb"] = np.sort(np.concatenate([newinds["ansilob"], newinds["cereb_nuclei"]]))
    else:
        newinds["cereb"] = []
        newinds["ansilob"] = []
    newinds["motor"] = np.sort(np.concatenate([newinds["m1"], newinds["m1thal"],
                                               newinds["trigeminal"] if TRIGEMINAL else [],
                                               newinds["cereb"] if CEREB else [],
                                               newinds["ponssens_trigeminal"] if SENSTRIG > 0.0 else []])).astype('i')
    newinds["sens"] = np.sort(np.concatenate([newinds["s1brl"], newinds["s1brlthal"],
                                              newinds["trigeminal"] if TRIGEMINAL else [],
                                              newinds["ponssens_trigeminal"] if SENSTRIG > 0.0 else []])).astype('i')

    newinds["facial"] = []

    newmaps = {}
    newmaps["is_cortical"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_cortical"][newinds["crtx"]] = True
    newmaps["is_thalamic"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_thalamic"][newinds["thalspec"]] = True
    newmaps["is_subcortical"] = np.logical_not(newmaps["is_cortical"])
    newmaps["is_subcortical_not_thalspec"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_subcortical_not_thalspec"][newinds["trigeminal"].tolist() +
                                           newinds["ponssens_trigeminal"].tolist() if SENSTRIG > 0.0 else [] +
                                           newinds["ponsmotor"].tolist() if PONS > 0.0 else [] +
                                           newinds["ponssens"].tolist() if PONS > 0.0 else [] +
                                           newinds["cereb"].tolist() if CEREB else []] = True
    newinds["subcrtx_not_thalspec"] = np.where(newmaps["is_subcortical_not_thalspec"])[0].astype('i')

    if "w" in config.THAL_CRTX_FIX:
        connectivity.weights[newinds["crtx"], newinds["thalspec"]] = 1.0

    print(newinds)

    return connectivity, newinds, newmaps


def print_weight_to_indegree(src, trg, inds, w, hemispheres=1):
    print("\n" + "-"*25)
    print("%s -> %s" % (src, trg))
    print(w[inds[trg]][:, inds[src]])
    print("%:")
    print(w[inds[trg]][:, inds[src]] / w[inds[trg]].sum() * 200 / (2 - np.abs(hemispheres)))


def simulate_minimal(**kwargs):

    # Configuration
    config, plotter = get_config(**kwargs)

    CEREB, TRIGEMINAL, M1STIM, HEMISPHERES, CNS1TH, SENSTRIG, PONS = getflags_from_config(config)

    # CONNECTOME:
    connectivity, inds, maps = newconn_and_inds(config, plotter)

    # # # For symmetric connectomme:
    # # connectivity.weights = np.sqrt(connectivity.weights * connectivity.weights.T)
    # # connectivity.tract_lengths = np.sqrt(connectivity.tract_lengths * connectivity.tract_lengths.T)
    # # connectivity.configure()

    if config.SET_WEIGHTS:
        # Force connectivity:
        connectivity.weights *= 0.0

        # M1 <-> S1 ipsilateral:
        connectivity.weights[inds["m1"], inds["s1brl"]] = 0.3
        connectivity.weights[inds["s1brl"], inds["m1"]] = 0.3

        # M1 <-> S1 contralateral:
        connectivity.weights[inds["m1"], inds["s1brl"][::-1]] = 0.2
        connectivity.weights[inds["s1brl"], inds["m1"][::-1]] = 0.2

        # SpecThal -> Crtx Ipsilateral only:
        connectivity.weights[inds["m1"], inds["m1thal"]] = 1.0
        connectivity.weights[inds["s1brl"], inds["s1brlthal"]] = 1.0

        # Crtx -> SpecThal Ipsilateral only:
        connectivity.weights[inds["m1thal"], inds["m1"]] = 1.0
        connectivity.weights[inds["s1brlthal"], inds["s1brl"]] = 1.0

        if CEREB:
            connectivity.weights[inds["cereb_nuclei"], inds["ansilob"]] = 2.0
            if CEREB > 1:
                connectivity.weights[inds["m1thal"], inds["cereb_nuclei"][::-1]] = 2.0
                if CNS1TH > 0.0:
                    connectivity.weights[inds["s1brlthal"], inds["cereb_nuclei"][::-1]] = CNS1TH

        if TRIGEMINAL:
            # Trigeminal -> SpecThal contralateral only:
            connectivity.weights[inds["s1brlthal"], inds["trigeminal"][::-1]] = 1.0
            if SENSTRIG > 0.0:
                connectivity.weights[inds["ponssens_trigeminal"], inds["trigeminal"]] = 2.0
                connectivity.weights[inds["s1brlthal"], inds["ponssens_trigeminal"][::-1]] = SENSTRIG
            if CEREB > 1:
                connectivity.weights[inds["ansilob"], inds["trigeminal"]] = 2.0 - SENSTRIG
                if SENSTRIG > 0.0:
                    connectivity.weights[inds["ansilob"], inds["ponssens_trigeminal"]] = SENSTRIG
            elif M1STIM:
                connectivity.weights[inds["m1thal"], inds["trigeminal"][::-1]] = 1.0

        if PONS > 0.0:
            connectivity.weights[inds["ponsmotor"], inds["m1"]] = 1.0
            connectivity.weights[inds["ponssens"], inds["s1brl"]] = 1.0
            if CEREB > 1:
                connectivity.weights[inds["ansilob"], inds["ponssens"]] = PONS
                connectivity.weights[inds["ansilob"], inds["ponsmotor"]] = PONS

    plotter.plot_tvb_connectivity(connectivity)

    # MODEL:
    dummy = np.ones((connectivity.number_of_regions,))

    model_params = {}
    model_params.update(config.model_params)
    STIMULUS = model_params.pop("STIMULUS")

    model_params = {}
    for p, pval in config.model_params.items():
        if p != "STIMULUS":
            if pval is not None:
                pval = np.array([pval]).flatten()
                if p == 'G':
                    # G normalized by the number of regions as in Griffiths et al paper
                    # Geff = G /(number_of_regions - inds['thalspec'].size)
                    pval = pval / (connectivity.number_of_regions - inds['thalspec'].size)
                model_params[p] = pval

    # Stimuli:
    A_st = 0 * dummy.astype("f")
    B_st = 0 * dummy.astype("f")
    f_st = 0 * dummy.astype("f")
    if TRIGEMINAL:
        # Stimulus to trigeminal:
        A_st[inds["trigeminal"]] = STIMULUS
        B_st[inds["trigeminal"]] = config.STIMULUS_BASELINE
        f_st[inds["trigeminal"]] = config.STIMULUS_RATE  # Hz
    else:
        A_st[inds["s1brlthal"]] = STIMULUS
        B_st[inds["s1brlthal"]] = config.STIMULUS_BASELINE
        f_st[inds["s1brlthal"]] = config.STIMULUS_RATE  # Hz
        if M1STIM:
            A_st[inds["m1thal"]] = STIMULUS
            B_st[inds["m1thal"]] = config.STIMULUS_BASELINE
            f_st[inds["m1thal"]] = config.STIMULUS_RATE  # Hz
    model_params.update({"A_st": A_st, "B_st": B_st, "f_st": f_st})

    model = WilsonCowanThalamoCortical(is_cortical=maps['is_cortical'][:, np.newaxis],
                                       is_thalamic=maps['is_thalamic'][:, np.newaxis],
                                       **model_params)

    model.dt = config.DEFAULT_DT

    # Remove Specific thalamic relay -> nonspecific subcortical structures connections!
    w_se = model.w_se * dummy
    w_se[inds['subcrtx']] = 0.0  #  model.G[0]
    model.w_se = w_se
    # Remove specific thalamic relay -> inhibitory nonspecific subcortical structures connections
    w_si = model.w_si * dummy
    w_si[inds['subcrtx']] = 0.0  # * model.G[0]
    model.w_si = w_si

    # Long range connections to specific thalamic relay and reticular structures connections' weights:
    model.G = model.G * dummy
    model.G[inds["thalspec"]] = 0.0
    # Retain connections
    # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
    model.G[inds["s1brlthal"]] = model.G[inds["crtx"][0]]
    # from Cerebellar Nuclei to M1:
    model.G[inds["m1thal"]] = model.G[inds["crtx"][0]]

    model.configure()

    # SIMULATOR:
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter)

    print("\n" + "-" * 50)
    print("Pathway connections to indegree:")

    for conns in [["s1brl", "m1"], ["m1", "s1brl"]]:
        print_weight_to_indegree(conns[0], conns[1], inds, connectivity.weights, hemispheres=0)

    conns = [["s1brlthal", "s1brl"],
             ["s1brl", "s1brlthal"], ["trigeminal", "s1brlthal"],
             ["m1thal", "m1"],
             ["m1", "m1thal"]]
    if SENSTRIG > 0.0:
        conns += [["trigeminal", "ponssens_trigeminal"], ["ponssens_trigeminal", "s1brlthal"]]
    if PONS > 0.0:
        conns += [["s1brl", "ponssens"], ["m1", "ponsmotor"]]
    if CEREB:
        conns += [["ansilob", "cereb_nuclei"], ["cereb_nuclei", "m1thal"]]
        if TRIGEMINAL:
            conns += [["trigeminal", "ansilob"]]
        if SENSTRIG > 0.0:
            conns += [["ponssens_trigeminal", "ansilob"]]
        if PONS > 0.0:
            conns += [["ponssens", "ansilob"], ["ponsmotor", "ansilob"]]
        if CNS1TH > 0.0:
            conns += [["cereb_nuclei", "s1brlthal"]]
    elif M1STIM:
        conns += [["trigeminal", "m1thal"]]
    for conn in conns:
        print_weight_to_indegree(conn[0], conn[1], inds, simulator.connectivity.weights, hemispheres=HEMISPHERES)

    # SIMULATION:
    results, transient = simulate(simulator, config)

    # RESULTS:
    # Compute coherence
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2

    results = plot_tvb(transient, inds,
                       results=results, simulator=simulator, plotter=plotter, config=config, write_files=True)

    return results, simulator, config


def get_reg_ind(reg, inds, hemi, results):
    if hemiI:
        ind = inds[reg][hemi]
    else:
        ind = inds[reg][1 - hemi]
    return np.where(results["inds"] == ind)[0].item()


def get_reg_pairs_inds(regs, inds, hemis, results):
    for reg, hemi in zip(regs, hemis):
        ind = inds[reg][hemi]
        pair.append(np.where(results["inds"] == ind)[0].item())
    try:
        iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                     results["ij"][:, 1].flatten() == pair[1]))[0].item()
    except:
        pair = pair[::-1]
        iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                     results["ij"][:, 1].flatten() == pair[1]))[0].item()
    return iR


def plot_comparison(tests, **kwargs):

    # CONFIGURATION:
    config, plotter = get_config(**kwargs)

    CEREB, TRIGEMINAL, M1STIM, HEMISPHERES, CNS1TH, SENSTRIG, PONS = getflags_from_config(config)

    # CONNECTIVITY:
    connectivity, inds, maps = newconn_and_inds(config, None)

    # Results path:
    BASEPATH = config.out.FOLDER_RES.split("res")[0][:-1]
    print(BASEPATH)

    # Task related regions' indices:
    TASKINDS = np.concatenate([inds["m1"], inds["s1brl"],
                               inds["m1thal"], inds["s1brlthal"]])
    for reg in ["trigeminal", "ponssens_trigeminal",
                "ansilob", "cereb_nuclei",
                "ponsmotor", "ponssens"]:
        this_inds = inds.get(reg, [])
        if len(this_inds):
            TASKINDS = np.concatenate([TASKINDS, this_inds])

    # Task related regions' labels:
    REGION_LABELS = connectivity.region_labels[TASKINDS]
    # Task related regions' abreviated labels:
    SHORT_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"]) for reg in REGION_LABELS]

    THETA = [3.0, 10.0]  # Hz
    GAMMA = [25.0, 45.0]  # Hz

    # results dictionary:
    results = {"inds": TASKINDS,
               "region_labels": REGION_LABELS, "short_labels": SHORT_LABELS,
               "theta": THETA, "gamma": GAMMA}

    TESTS = tests
    for test_name in TESTS:

        results[test_name] = {}
        Ps = []
        Cs = []

        testpath_old = "_".join([BASEPATH, test_name])
        testpath = os.path.join(BASEPATH, test_name)
        if os.path.isdir(testpath_old):
            shutil.move(testpath_old, testpath)
        for path in glob.glob(os.path.join(testpath, "nsd*")):
            resultsfile = os.path.join(path, "res/source_ts.pkl")
            with open(resultsfile, 'rb') as handle:
                source_ts = pickle.load(handle)  # to load results
            Pxx_den, Cxy, f, ij = compute_selected_spectra_coherence(
                                        source_ts["time_series"], TASKINDS,
                                        transient=source_ts["time_series"].shape[0]-2**15, # 2**15 final length
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
    results["ij"] = ij    # pairs of regions of coherences where i, j in [0, TASKINDS.size]

    dump_pickled_dict(results, os.path.join(config.out.FOLDER_RES, "res_PSD_COH.pkl"))

    figR, axR, figL, axL = plot_pathway_psd_coh(
        results, inds, CNS1TH=CNS1TH, SENSTRIG=SENSTRIG, PONS=PONS,
        tests=TESTS, colors=["g", "r"],
        percentile_min=1, percentile_max=99, n=1,
        plot_mean=True, plot_median=False, mode="semilog",
        alpha=0.5, figsize=config.figures.LARGE_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        for fig, hemi in zip([figR, figL], ["Right", "Left"]):
            plt.figure(fig.number)
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "PathwayPSD_COH_%s.png" % hemi))

    figPSD, axesPSD = psd_percent_plot(results,
                                        inds=None,
                                        tests=TESTS, colors=["g", "r"],
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
    # $ python minimal_network.py test_name=s1stim 'simulation_length'=3000.0
    # Called tuning_tvb_nest.py with:
    # keyword argument: test_name="s1stim"
    # keyword argument: simulation_length=3000.0

    import sys

    kwargs = {}
    ntests = 0
    for arg in sys.argv[1:]:
        keyval = arg.split("=")
        if keyval[0] not in ["test_name"]:
            key = float(keyval[1])
        else:
            key = keyval[1].split(" ")
            if keyval[0] == "test_name":
                ntests = len(key)
                if ntests == 1:
                    key = key[0]
        kwargs[keyval[0]] = key

    if ntests > 1:
        plot_comparison(kwargs.pop("test_name"), **kwargs)
    else:
        simulate_minimal(**kwargs)
