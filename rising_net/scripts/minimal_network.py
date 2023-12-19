#!/usr/bin/env python
# coding: utf-8

import glob
import pickle
import os
import shutil
from matplotlib import pyplot as plt

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        # build_NEST_network, plot_nest_results
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical


def get_config(**kwargs):

    # Assuming:
    # DEFAULT_ARGS = {'G': 6.0, 'STIMULUS': 0.1, 'STIMULUS_BASELINE': 0.0,
    #                 'I_e': -0.35, 'I_s': 0.085,
    #                 'w_ie': -3.0, 'w_rs': -2.0,
    #                 'CONN_LOG': True, 'FIC': 1.11,  'FIC_SPLIT': 0.31,  #'fit',
    #                 'PRIORS_DIST': 'uniform',
    #                 'output_folder': "", 'verbose': 1, 'plot_flag': True}

    # config.TRANSIENT_RATIO =0.25

    simulation_length = kwargs.pop("simulation_length", 3000.0)
    pathway_gain = kwargs.pop("pathway_gain", 1.0)
    pathway_mode = kwargs.pop("pathway_mode", "task")
    STIMULUS = kwargs.get("STIMULUS", 0.1)
    STIMULUS_BASELINE = kwargs.get("STIMULUS_BASELINE", 0.1)
    NOISE = int(kwargs.pop("NOISE", 4))
    CNS1TH = float(kwargs.pop("CNS1TH", 1.0))
    G = float(kwargs.get("G", 1.0))
    FIC = float(kwargs.get("FIC", 1.11))
    SET_WEIGHTS = kwargs.pop("SET_WEIGHTS", True)
    # SET_DELAYS = kwargs.pop("SET_DELAYS", False)
    hemispheres = int(kwargs.pop("hemispheres", -1))

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

    # if SET_DELAYS:
    #     experiment_name = "_".join([experiment_name, "SetDel"])

    if pathway_gain > 1.0:
        if pathway_mode == "stim":
            experiment_name = "stimgain"
        else:
            experiment_name = "gain"
        experiment_name += "%d" % int(pathway_gain)

    experiment_name = "_".join([experiment_name,
                                "stimbase%d_stim%d_noise%d_G%d" %
                                (int(STIMULUS_BASELINE), int(10*STIMULUS), NOISE, int(G))])
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
    config, plotter = configure(output_folder=path, verbose=0,
                                NOISE=10 ** (-NOISE),
                                SIMULATION_LENGTH=simulation_length,
                                **kwargs)
    config.RANDOM_SEED_TVB = seed
    config.RANDOM_SEED_NEST = seed
    config.CEREB = CEREB
    config.CNS1TH = CNS1TH
    config.TRIGEMINAL = TRIGEMINAL
    config.M1STIM = M1STIM
    config.HEMISPHERES = hemispheres
    config.PATHWAY_GAIN = pathway_gain
    config.PATHWAY_MODE = pathway_mode
    config.SET_WEIGHTS = SET_WEIGHTS
    # config.SET_DELAYS = SET_DELAYS
    config.STIMULUS_RATE = 8.0  # Hz
    config.VERBOSITY = 2.0

    print(config.model_params)
    print(config)

    return config, plotter


def newconn_and_inds(config, plotter):

    CEREB = getattr(config, "CEREB", True)
    TRIGEMINAL = getattr(config, "TRIGEMINAL", True)
    M1STIM = getattr(config, "M1STIM", True)
    HEMISPHERES = getattr(config, "HEMISPHERES", -1)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config,
                                      hemispheres=HEMISPHERES,
                                      cereb_nuclei_to_s1thal=False,
                                      trigeminal_to_m1thal=TRIGEMINAL and M1STIM)

    if not TRIGEMINAL:
        connectivity.weights[inds["trigeminal"]] = 0.0
        connectivity.weights[:, inds["trigeminal"]] = 0.0

    # Keep only the spinal trigeminal and M1 and S1 barrel field and their specific thalami.
    # Maintain the total indegree, though.
    indegree = {}
    taskinds = []
    regs = ["m1", "s1brl", "m1thal", "s1brlthal", "trigeminal"]
    if CEREB:
        regs += ["ansilob", "cereb_nuclei"]
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
                                               newinds["cereb"] if CEREB else []])).astype('i')
    newinds["sens"] = np.sort(np.concatenate([newinds["s1brl"], newinds["s1brlthal"],
                                              newinds["trigeminal"] if TRIGEMINAL else []])).astype('i')

    newinds["facial"] = []

    newmaps = {}
    newmaps["is_cortical"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_cortical"][newinds["crtx"]] = True
    newmaps["is_thalamic"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_thalamic"][newinds["thalspec"]] = True
    newmaps["is_subcortical"] = np.logical_not(newmaps["is_cortical"])
    newmaps["is_subcortical_not_thalspec"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_subcortical_not_thalspec"][newinds["trigeminal"].tolist() +
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

    CEREB = getattr(config, "CEREB", True)
    TRIGEMINAL = getattr(config, "TRIGEMINAL", True)
    M1STIM = getattr(config, "M1STIM", True)
    hemispheres = getattr(config, "HEMISPHERES", -1)
    CNS1TH = getattr(config, "CNS1TH", 1.0)

    # CONNECTOME:
    connectivity, inds, maps = newconn_and_inds(config, plotter)

    # # # For symmetric connectomme:
    # # connectivity.weights = np.sqrt(connectivity.weights * connectivity.weights.T)
    # # connectivity.tract_lengths = np.sqrt(connectivity.tract_lengths * connectivity.tract_lengths.T)
    # # connectivity.configure()

    pathway_gain = config.PATHWAY_GAIN
    pathway_mode = config.PATHWAY_MODE
    if pathway_gain > 1.0:
        from cosim_run_plot import apply_pathway_gain_to_target
        hemispheres = np.abs(hemispheres)
        print("\n" + "-"*50)
        print("Applying pathway gain = %g" % pathway_gain)
        print("-" * 50 + "\n")

        # A. INPUT SENSORY PATHWAY:

    #     # 3. S1 brl thal <- Trigeminal (stimulus)
    #     connectivity.weights = \
    #         apply_pathway_gain_to_target(newinds["trigeminal"][::-1],      # contralaterally or bilaterally
    #                                      newinds["s1brlthal"],
    #                                      pathway_gain, connectivity.weights,
    #                                      hemispheres=hemispheres,
    #                                      fix_inds=newinds["s1brl"])


        # C. INPUT MOTOR PATHWAY:

    #     if TRIGEMINAL:
    #         # 1.  M1 thal <- trigeminal
    #         connectivity.weights = \
    #             apply_pathway_gain_to_target(newinds["trigeminal"][::-1],  # contralaterally or bilaterally
    #                                         newinds["m1thal"],
    #                                         pathway_gain, connectivity.weights,
    #                                         hemispheres=hemispheres,
    #                                             fix_inds=newinds["m1"])

        if pathway_mode != "stim":

            connectivity.weights = \
                apply_pathway_gain_to_target(inds["s1brl"],  # bilaterally
                                             inds["m1"],
                                             pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=inds["m1thal"])

            connectivity.weights = \
                apply_pathway_gain_to_target(inds["m1"],  # bilaterally
                                             inds["s1brl"],
                                             pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=inds["s1brlthal"])

            connectivity.weights = \
                apply_pathway_gain_to_target(None,  # bilaterally
                                             inds["m1"],
                                             1.0/pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=inds["m1thal"],
                                             preserve_indegree=False)

            connectivity.weights = \
                apply_pathway_gain_to_target(None,  # bilaterally
                                             inds["s1brl"],
                                             1.0/pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=inds["s1brlthal"],
                                             preserve_indegree=False)

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

        if TRIGEMINAL:
            # Trigeminal -> SpecThal contralateral only:
            connectivity.weights[inds["s1brlthal"], inds["trigeminal"][::-1]] = 1.0
            if CEREB:
                connectivity.weights[inds["cereb_nuclei"], inds["ansilob"]] = 2.0
            if CEREB > 1:
                connectivity.weights[inds["ansilob"], inds["trigeminal"]] = 2.0
                connectivity.weights[inds["m1thal"], inds["cereb_nuclei"][::-1]] = 2.0
                if CNS1TH > 0.0:
                    connectivity.weights[inds["s1brlthal"], inds["cereb_nuclei"][::-1]] = CNS1TH
            elif M1STIM:
                connectivity.weights[inds["m1thal"], inds["trigeminal"][::-1]] = 1.0

    # if config.SET_DELAYS:
    #     # M1 <-> S1 ipsilateral:
    #     connectivity.tract_lengths[inds["m1"], inds["s1brl"]] = set_delays(10.0)
    #     connectivity.tract_lengths[inds["s1brl"], inds["m1"]] = set_delays(10.0)
    #
    #     # M1 <-> S1 contralateral:
    #     connectivity.tract_lengths[inds["m1"], inds["s1brl"][::-1]] = set_delays(20.0)
    #     connectivity.tract_lengths[inds["s1brl"], inds["m1"][::-1]] = set_delays(20.0)
    #
    #     # SpecThal -> Crtx Ipsilateral only:
    #     connectivity.tract_lengths[inds["m1"], inds["m1thal"]] = set_delays(20.0)
    #     connectivity.tract_lengths[inds["s1brl"], inds["s1brlthal"]] = set_delays(20.0)
    #
    #     # Crtx -> SpecThal Ipsilateral only:
    #     connectivity.tract_lengths[inds["m1thal"], inds["m1"]] = set_delays(20.0)
    #     connectivity.tract_lengths[inds["s1brlthal"], inds["s1brl"]] = set_delays(20.0)
    #
    #     if TRIGEMINAL:
    #         # Trigeminal -> SpecThal contralateral only:
    #         connectivity.tract_lengths[inds["s1brlthal"], inds["trigeminal"][::-1]] = set_delays(10.0)
    #         if CEREB:
    #             connectivity.tract_lengths[inds["cereb_nuclei"], inds["ansilob"]] = set_delays(5.0)
    #         if CEREB > 1:
    #             connectivity.tract_lengths[inds["ansilob"], inds["trigeminal"]] = set_delays(10.0)
    #             connectivity.tract_lengths[inds["m1thal"], inds["cereb_nuclei"][::-1]] = set_delays(10.0)
    #             if CNS1TH > 0.0:
    #                 onnectivity.tract_lengths[inds["s1brlthal"], inds["cereb_nuclei"][::-1]] = set_delays(10.0)
    #         elif M1STIM:
    #             connectivity.tract_lengths[inds["m1thal"], inds["trigeminal"][::-1]] = set_delays(10.0)

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
    if CEREB:
        conns += [["trigeminal", "ansilob"], ["ansilob", "cereb_nuclei"], ["cereb_nuclei", "m1thal"]]
        if CNS1TH > 0.0:
            conns += [["cereb_nuclei", "s1brlthal"]]
    elif M1STIM:
        conns += [["trigeminal", "m1thal"]]
    for conn in conns:
        print_weight_to_indegree(conn[0], conn[1], inds, simulator.connectivity.weights, hemispheres=hemispheres)

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


def prepare_plot_pathway(tests=["cerebON", "cerebOFF"], CNS1TH=1.0):
    CEREB = False
    for test in tests:
        if test.find("cereb") > -1:
            CEREB = True

    REGIONS = ["s1brl", "m1",
               "s1brlthal", "m1thal"]
    # PSD plots:
    subplotsPSD = [[0, 0], [0, 2],  # S1, M1

                   [2, 0], [2, 2]]  # S1Th, M1Th
    ipsiPSD = [True, True,
               True, True]

    REGPAIRS = [["s1brl", "m1"],
                ["s1brl", "s1brlthal"], ["m1", "m1thal"],
                ["trigeminal", "s1brlthal"]]

    subplotsCOH = [[0, 1],  # S1M1
                   [1, 0], [1, 2],  # S1S1th, M1M1Th

                   [3, 0]]  # TRS1Th
    ipsiCOH = [[True, True],
               [True, True], [True, True],
               [False, True]]
    if CEREB:
        nRows = 6
        REGIONS += ["ansilob", "cereb_nuclei", "trigeminal"]
        ipsiPSD += [False, False, False]
        # PSD plots:
        subplotsPSD += [[4, 1], [3, 1],  # AL, CN
                        [5, 0]]  # TR

        REGPAIRS += [["ansilob", "cereb_nuclei"], ["cereb_nuclei", "m1thal"], ["trigeminal", "ansilob"]]
        # COH plots:
        subplotsCOH += [[4, 2], [3, 2],  # ALCN, CNM1Th

                        [5, 1]]  # TRAL
        ipsiCOH += [[False, False], [False, True],
                    [False, False]]

        if CNS1TH > 0.0:
            REGPAIRS += [["cereb_nuclei", "s1brlthal"]]
            subplotsCOH += [[5, 2]]
            ipsiCOH += [[False, True]]
    else:
        nRows = 4
        # PSD plots:
        REGIONS += ["trigeminal"]
        subplotsPSD += [[3, 1]]  # TR
        ipsiPSD += [False]

        REGPAIRS += [["trigeminal", "m1thal"]]
        subplotsCOH += [[3, 2]]
        ipsiCOH += [[False, True]]

    mosaic = np.tile(["."], (nRows, 3)).astype('O')

    # PSD plots:
    for ax, reg in zip(subplotsPSD, REGIONS):
        mosaic[ax[0], ax[1]] = reg
    # COH plots:
    for ax, regs, in zip(subplotsCOH, REGPAIRS):
        mosaic[ax[0], ax[1]] = "-".join(regs)

    return mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH


def plot_pathway_psd_coh_minimal(results, inds, CNS1TH=1.0, tests=["cerebON", "cerebOFF"], colors=["g", "r"],
                                 percentile_min=1, percentile_max=99, n=1,
                                 plot_mean=False, plot_median=True, mode="semilog",
                                 alpha=0.5, figsize=(10, 10), fontsize=16, **line_kwargs):

    mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH = prepare_plot_pathway(tests, CNS1TH)

    figR, axR = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    figL, axL = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)

    print(axR)
    print(axL)
    # PSD plots:

    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for reg, hemiI in zip(REGIONS, ipsiPSD):
            if hemiI:
                ind = inds[reg][hemi]
            else:
                ind = inds[reg][1 - hemi]
            iR = np.where(results["inds"] == ind)[0].item()
            for col, test in zip(colors, tests):
                percent_plot(results["f"], results[test]['PSD'][:, iR, :].squeeze(),
                             percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                             plot_mean=plot_mean, plot_median=plot_median,
                             color=col, alpha=alpha, ax=axH[reg], mode=mode,
                             **line_kwargs)
            axH[reg].set_title(results['short_labels'][iR])
            if mode == "semilog":
                axH[reg].set_ylabel('log(PSD)', fontsize=fontsize)
            else:
                axH[reg].set_ylabel('PSD', fontsize=fontsize)

    # COH plots:
    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for regs, hemiIs in zip(REGPAIRS, ipsiCOH):
            pair = []
            for reg, hemiI in zip(regs, hemiIs):
                if hemiI:
                    ind = inds[reg][hemi]
                else:
                    ind = inds[reg][1 - hemi]
                pair.append(np.where(results["inds"] == ind)[0].item())
            try:
                iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                             results["ij"][:, 1].flatten() == pair[1]))[0].item()
            except:
                pair = pair[::-1]
                iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                             results["ij"][:, 1].flatten() == pair[1]))[0].item()
            ax = axH["-".join(regs)]
            for col, test in zip(colors, tests):
                percent_plot(results["f"], results[test]['COH'][:, iR, :].squeeze(),
                             percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                             plot_mean=plot_mean, plot_median=plot_median,
                             color=col, alpha=alpha, ax=ax, mode=mode,
                             **line_kwargs)
                for band, COH, f in zip(["theta", "gamma"],
                                        ["COHth", "COHgm"],
                                        ["fth", "fgm"]):
                    if mode == "semilog":
                        mean = np.log(results[test]['COH'][:, iR, results[f]]).mean()
                    else:
                        mean = results[test]['COH'][:, iR, results[f]].mean()
                    ax.plot(results[band], [mean] * 2,
                            color=col, linewidth=2.0)
            ax.set_title("%s - %s" % (results['short_labels'][pair[0]],
                                      results['short_labels'][pair[1]]), fontsize=fontsize)
            if mode == "semilog":
                ax.set_ylabel('log(COH)', fontsize=fontsize)
            else:
                ax.set_ylabel('COH', fontsize=fontsize)
        figH.tight_layout()

    return figR, axR, figL, axL


def plot_pathway_sync_minimal(results, inds, CNS1TH=1.0, tests=["cerebON", "cerebOFF"], colors=["g", "r"],
                              percentile_min=1, percentile_max=99, n=1,
                              plot_mean=False, plot_median=True, mode="semilog",
                              alpha=0.5, figsize=(10, 10), fontsize=16, **line_kwargs):

    mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH = prepare_plot_pathway(tests, CNS1TH)

    figR, axR = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    figL, axL = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)

    print(axR)
    print(axL)
    # PSD plots:
    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for reg, hemiI in zip(REGIONS, ipsiPSD):
            if hemiI:
                ind = inds[reg][hemi]
            else:
                ind = inds[reg][1 - hemi]
            iR = np.where(results["inds"] == ind)[0].item()
            for col, test in zip(colors, tests):
                percent_plot(results["f"], results[test]['PSD'][:, iR, :].squeeze(),
                             percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                             plot_mean=plot_mean, plot_median=plot_median,
                             color=col, alpha=alpha, ax=axH[reg], mode=mode,
                             **line_kwargs)
            axH[reg].set_title(results['short_labels'][iR])
            if mode == "semilog":
                axH[reg].set_ylabel('log(PSD)', fontsize=fontsize)
            else:
                axH[reg].set_ylabel('PSD', fontsize=fontsize)

    # SYNC plots:
    results["syncij"] = []
    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for regs, hemiIs in zip(REGPAIRS, ipsiCOH):
            pair = []
            for reg, hemiI in zip(regs, hemiIs):
                if hemiI:
                    ind = inds[reg][hemi]
                else:
                    ind = inds[reg][1 - hemi]
                pair.append(np.where(results["inds"] == ind)[0].item())
            results["syncij"].append(pair)
            try:
                iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                             results["ij"][:, 1].flatten() == pair[1]))[0].item()
            except:
                pair = pair[::-1]
                iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                             results["ij"][:, 1].flatten() == pair[1]))[0].item()
            ax = axH["-".join(regs)]
            for col, test in zip(colors, tests):
                percent_plot(results["f"], results[test]['COH'][:, iR, :].squeeze(),
                             percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                             plot_mean=plot_mean, plot_median=plot_median,
                             color=col, alpha=alpha, ax=ax, mode=mode,
                             **line_kwargs)
                for band, COH, f in zip(["theta", "gamma"],
                                        ["COHth", "COHgm"],
                                        ["fth", "fgm"]):
                    if mode == "semilog":
                        mean = np.log(results[test]['COH'][:, iR, results[f]]).mean()
                    else:
                        mean = results[test]['COH'][:, iR, results[f]].mean()
                    ax.plot(results[band], [mean] * 2,
                            color=col, linewidth=2.0)
            ax.set_title("%s - %s" % (results['short_labels'][pair[0]],
                                      results['short_labels'][pair[1]]), fontsize=fontsize)
            if mode == "semilog":
                ax.set_ylabel('log(COH)', fontsize=fontsize)
            else:
                ax.set_ylabel('COH', fontsize=fontsize)
        figH.tight_layout()

    return figR, axR, figL, axL


def plot_comparison(tests, **kwargs):

    # CONFIGURATION:
    config, plotter = get_config(**kwargs)

    # CONNECTIVITY:
    connectivity, inds, maps = newconn_and_inds(config, None)

    # Results path:
    BASEPATH = config.out.FOLDER_RES.split("res")[0][:-1]
    print(BASEPATH)

    # Task related regions' indices:
    inds["taskcereb"] = np.sort(np.concatenate([inds['ansilob'], inds['cereb_nuclei']]))  # newinds['cereb_crtx']
    TASKINDS = np.sort(np.concatenate([inds["m1"], inds["s1brl"],
                                       inds["m1thal"], inds["s1brlthal"],
                                       inds["trigeminal"], #, newinds["ponsmotor"], newinds["ponssens"],
                                       inds['taskcereb']
                                      ]))
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
        CsTheta = []
        CsGamma = []

        testpath_old = "_".join([BASEPATH, test_name])
        testpath = os.path.join(BASEPATH, test_name)
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

    figR, axR, figL, axL = plot_pathway_psd_coh_minimal(
        results, inds,
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
        if keyval[0] not in ["test_name", "pathway_mode"]:
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
