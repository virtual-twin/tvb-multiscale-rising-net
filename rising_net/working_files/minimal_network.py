#!/usr/bin/env python
# coding: utf-8

import glob
import pickle
import os
from matplotlib import pyplot as plt

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.utils import *
from rising_net.plot_utils import *

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical


def print_weight_to_indegree(src, trg, inds, w, hemispheres=1):
    print("\n" + "-"*25)
    print("%s -> %s" % (src, trg))
    print(w[inds[trg]][:, inds[src]])
    print("%:")
    print(w[inds[trg]][:, inds[src]] / w[inds[trg]].sum() * 200 / (2 - np.abs(hemispheres)))


def newconn_and_inds(config, plotter):

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
    newinds["motor"] = np.sort(np.concatenate([newinds["m1"], newinds["m1thal"],
                                               newinds["trigeminal"] if TRIGEMINAL else []])).astype('i')
    newinds["sens"] = np.sort(np.concatenate([newinds["s1brl"], newinds["s1brlthal"],
                                              newinds["trigeminal"] if TRIGEMINAL else []])).astype('i')
    newinds["cereb"] = []
    newinds["ansilob"] = []
    newinds["facial"] = []

    newmaps = {}
    newmaps["is_cortical"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_cortical"][newinds["crtx"]] = True
    newmaps["is_thalamic"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_thalamic"][newinds["thalspec"]] = True
    newmaps["is_subcortical"] = np.logical_not(newmaps["is_cortical"])
    newmaps["is_subcortical_not_thalspec"] = np.array([False] * connectivity.number_of_regions).astype("bool")
    newmaps["is_subcortical_not_thalspec"][newinds["trigeminal"]] = True
    newinds["subcrtx_not_thalspec"] = np.where(newmaps["is_subcortical_not_thalspec"])[0].astype('i')

    if "w" in config.THAL_CRTX_FIX:
        connectivity.weights[newinds["crtx"], newinds["thalspec"]] = 1.0

    print(newinds)

    return connectivity, newinds, newmaps


def simulate_minimal(**kwargs):

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

    hemispheres = int(kwargs.pop("hemispheres", -1))

    seed = int(kwargs.pop("seed", 0))
    test_name = kwargs.pop("test_name")              # "s1stim", "s1m1stim", "trigs1stim", "trigs1m1stim"
    if test_name.find("trig") > -1:
        TRIGEMINAL = True
    else:
        TRIGEMINAL = False

    if test_name.find("m1") > -1:
        M1STIM = True
    else:
        M1STIM = False

    resfilename = "%s_stimbase%d_stim%d_nsd%d" % (test_name, int(10*STIMULUS_BASELINE), int(10*STIMULUS), seed)
    path = os.path.join(os.getcwd(), resfilename)
    # Get configuration
    config, plotter = configure(output_folder=path, verbose=2,
                                SIMULATION_LENGTH=simulation_length,
                                **kwargs)
    config.RANDOM_SEED_TVB = seed
    config.RANDOM_SEED_NEST = seed
    config.TRIGEMINAL = TRIGEMINAL
    config.M1STIM = M1STIM
    config.HEMISPHERES = hemispheres

    connectivity, newinds, newmaps = newconn_and_inds(config, plotter)

    print(config.model_params)

    # # # For symmetric connectomme:
    # # connectivity.weights = np.sqrt(connectivity.weights * connectivity.weights.T)
    # # connectivity.tract_lengths = np.sqrt(connectivity.tract_lengths * connectivity.tract_lengths.T)
    # # connectivity.configure()

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
                apply_pathway_gain_to_target(newinds["s1brl"],  # bilaterally
                                             newinds["m1"],
                                             pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=newinds["m1thal"])

            connectivity.weights = \
                apply_pathway_gain_to_target(newinds["m1"],  # bilaterally
                                             newinds["s1brl"],
                                             pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=newinds["s1brlthal"])

            connectivity.weights = \
                apply_pathway_gain_to_target(None,  # bilaterally
                                             newinds["m1"],
                                             1.0/pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=newinds["m1thal"],
                                             preserve_indegree=False)

            connectivity.weights = \
                apply_pathway_gain_to_target(None,  # bilaterally
                                             newinds["s1brl"],
                                             1.0/pathway_gain, connectivity.weights,
                                             hemispheres=0,
                                             fix_inds=newinds["s1brlthal"],
                                             preserve_indegree=False)


    plotter.plot_tvb_connectivity(connectivity)


    dummy = np.ones((connectivity.number_of_regions,))

    model_params = {}
    model_params.update(config.model_params)
    STIMULUS = model_params.pop("STIMULUS", STIMULUS)

    model_params = {}
    for p, pval in config.model_params.items():
        if p != "STIMULUS":
            if pval is not None:
                pval = np.array([pval]).flatten()
                if p == 'G':
                    # G normalized by the number of regions as in Griffiths et al paper
                    # Geff = G /(number_of_regions - inds['thalspec'].size)
                    pval = pval / (connectivity.number_of_regions - newinds['thalspec'].size)
                model_params[p] = pval


    # Stimuli:
    A_st = 0 * dummy.astype("f")
    B_st = 0 * dummy.astype("f")
    f_st = 0 * dummy.astype("f")
    if TRIGEMINAL:
        # Stimulus to trigeminal:
        A_st[newinds["trigeminal"]] = STIMULUS
        B_st[newinds["trigeminal"]] = config.STIMULUS_BASELINE
        f_st[newinds["trigeminal"]] = config.STIMULUS_RATE  # Hz
    else:
        A_st[newinds["s1brlthal"]] = STIMULUS
        B_st[newinds["s1brlthal"]] = config.STIMULUS_BASELINE
        f_st[newinds["s1brlthal"]] = config.STIMULUS_RATE  # Hz
        if M1STIM:
            A_st[newinds["m1thal"]] = STIMULUS
            B_st[newinds["m1thal"]] = config.STIMULUS_BASELINE
            f_st[newinds["m1thal"]] = config.STIMULUS_RATE  # Hz
    model_params.update({"A_st": A_st, "B_st": B_st, "f_st": f_st})

    model = WilsonCowanThalamoCortical(is_cortical=newmaps['is_cortical'][:, np.newaxis],
                                       is_thalamic=newmaps['is_thalamic'][:, np.newaxis],
                                       **model_params)

    model.dt = config.DEFAULT_DT

    # Remove Specific thalamic relay -> nonspecific subcortical structures connections!
    w_se = model.w_se * dummy
    w_se[newinds['subcrtx']] = 0.0  #  model.G[0]
    model.w_se = w_se
    # Remove specific thalamic relay -> inhibitory nonspecific subcortical structures connections
    w_si = model.w_si * dummy
    w_si[newinds['subcrtx']] = 0.0  # * model.G[0]
    model.w_si = w_si

    # Long range connections to specific thalamic relay and reticular structures connections' weights:
    model.G = model.G * dummy
    model.G[newinds["thalspec"]] = 0.0
    # Retain connections
    # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
    model.G[newinds["s1brlthal"]] = model.G[newinds["crtx"][0]]
    # from Cerebellar Nuclei to M1:
    model.G[newinds["m1thal"]] = model.G[newinds["crtx"][0]]

    model.configure()

    simulator = build_simulator(connectivity, model, newinds, newmaps, config, plotter)


    print("\n" + "-" * 50)
    print("Pathway connections to indegree:")

    for conns in [
        ["s1brl", "m1"], ["m1", "s1brl"]]:
        print_weight_to_indegree(conns[0], conns[1], newinds, connectivity.weights, hemispheres=0)

    for conns in [
        ["s1brlthal", "s1brl"],
        ["s1brl", "s1brlthal"], ["trigeminal", "s1brlthal"],
        ["m1thal", "m1"],
        ["m1", "m1thal"], ["trigeminal", "m1thal"],

    ]:
        print_weight_to_indegree(conns[0], conns[1], newinds, simulator.connectivity.weights, hemispheres=hemispheres)

    results, transient = simulate(simulator, config)

    # Compute coherence
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2

    results = plot_tvb(transient, newinds,
                       results=results, simulator=simulator, plotter=plotter, config=config, write_files=True)

    return results, simulator, config


def plot_pathway_psd_coh_minimal(results, inds, tests=["s1m1stim", "s1stim"], colors=["g", "r"],
                                 percentile_min=1, percentile_max=99, n=1,
                                 plot_mean=False, plot_median=True, mode="semilog",
                                 alpha=0.5, figsize=(20, 20), fontsize=16, **line_kwargs):
    REGIONS = ["s1brl", "m1",
               "s1brlthal", "m1thal",
               "trigeminal"]

    REGPAIRS = [["s1brl", "m1"],
                ["s1brl", "s1brlthal"], ["m1", "m1thal"],
                ["trigeminal", "s1brlthal"], ["trigeminal", "m1thal"]]

    # PSD plots:
    subplotsPSD = [[0, 0], [0, 2],
                   [2, 0], [2, 2],
                   [3, 1]]
    ipsiPSD = [True, True,
               True, True,
               False]
    # COH plots:
    subplotsCOH = [[0, 1],
                   [1, 0], [1, 2],
                   [3, 0], [3, 2]]
    ipsiCOH = [[True, True], [True, True],
               [True, True],
               [False, False], [False, False]]

    mosaic = np.tile(["."], (4, 3)).astype('O')

    # PSD plots:
    for ax, reg in zip(subplotsPSD, REGIONS):
        mosaic[ax[0], ax[1]] = reg
    # COH plots:
    for ax, regs, in zip(subplotsCOH, REGPAIRS):
        mosaic[ax[0], ax[1]] = "-".join(regs)

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
        for regs, hemiI in zip(REGPAIRS, ipsiCOH):
            pair = []
            for reg in regs:
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


def plot_comparison(tests, **kwargs):

    STIMULUS = kwargs.get("STIMULUS", 0.1)
    STIMULUS_BASELINE = kwargs.get("STIMULUS_BASELINE", 0.1)

    testnames = "-".join(tests)
    resfilename = "%s_stimbase%d_stim%d" % (testnames, int(10 * STIMULUS_BASELINE), int(10 * STIMULUS))
    path = os.path.join(os.getcwd(), resfilename)
    config, plotter = configure(output_folder=path, verbose=2, **kwargs)

    connectivity, newinds, newmaps = newconn_and_inds(config, None)

    # Results path:
    BASEPATH = "/home/docker/packages/tvb-multiscale/rising_net/working_files"
    # BASEPATH = BASEPATH + "/outputs/tvb-only_VS_cerebOFF_stim_gain_simlen3000/stimbase0"

    # Task related regions' indices:
    # inds["taskcereb"] = np.sort(np.concatenate([newinds['ansilob'], newinds['cereb_nuclei'], newinds['cereb_crtx']]))
    TASKINDS = np.sort(np.concatenate([newinds["m1"], newinds["s1brl"],
                                       newinds["m1thal"], newinds["s1brlthal"],
                                       newinds["trigeminal"] #, newinds["ponsmotor"], newinds["ponssens"],
                                       # newinds['taskcereb']
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

    PATHWAY_GAIN = 0.0 # 50.0# 200.0  # 50.0
    PATHWAY_MODE = "task" # , "stim"

    TESTS = tests
    for test_name in TESTS:

        results[test_name] = {}
        Ps = []
        Cs = []
        CsTheta = []
        CsGamma = []

        resfilename = "%s_stimbase%d_stim%d" % (test_name, int(10*STIMULUS_BASELINE), int(10*STIMULUS))
        if PATHWAY_GAIN > 1.0:
            if PATHWAY_MODE == "stim":
                resfilename += "_stimgain"
            else:
                resfilename += "_gain"
            resfilename += "%d" % int(PATHWAY_GAIN)
        testpath = os.path.join(BASEPATH, resfilename)
        print(testpath)
        for path in glob.glob(testpath + "_nsd*"):
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
        results, newinds,
        tests=TESTS, colors=[ "g", "r"],
        percentile_min=1, percentile_max=99, n=1,
        plot_mean=True, plot_median=False, mode="semilog",
        alpha=0.5, figsize=config.figures.LARGE_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        for fig, hemi in zip([figR, figL], ["Right", "Left"]):
            plt.figure(fig.number)
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "PathwayPSD_COH_%s.png" % hemi))

    figPSD, axesPSD = psd_percent_plot(results,
                                        inds=None,
                                        tests=TESTS, colors=[ "g", "r"],
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
