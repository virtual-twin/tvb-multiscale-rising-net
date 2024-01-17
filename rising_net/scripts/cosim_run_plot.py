# coding: utf-8

import warnings
import glob
import pickle
import os
import shutil
from matplotlib import pyplot

from rising_net.scripts.tvb_nest_script import *
from rising_net.scripts.nest_script import *        #build_NEST_network, plot_nest_results
from rising_net.scripts.utils import *
from rising_net.scripts.plot_utils import *
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def get_config(**kwargs):

    # kwargs = {'G': 6.0, 'STIMULUS': 0.4, 'STIMULUS_BASELINE': 1.0,
    #           'I_e': -0.35, 'I_s': 0.085,
    #           'w_ie': -3.0, 'w_rs': -2.0,
    #           'PONS': False,
    #           'CONN_LOG': True, 'FIC': 1.11, 'FIC_SPLIT': 0.31,  # 'fit',
    #           'output_folder': "", 'verbose': 1, 'plot_flag': True}

    simulation_length = kwargs.pop("simulation_length", 3000.0)

    STIMULUS = kwargs.pop("STIMULUS", 0.4)
    STIMULUS_BASELINE = kwargs.pop("STIMULUS_BASELINE", 1.0)
    PATHWAY_GAIN = kwargs.pop("PATHWAY_GAIN", 1.0)
    INDEGREE_GAIN = kwargs.pop("INDEGREE_GAIN", 1.0)
    NOISE = int(kwargs.pop("NOISE", 6))
    CNS1TH = float(kwargs.pop("CNS1TH", 1.0))
    PONS = float(kwargs.pop("PONS", False))
    SENSTRIG = float(kwargs.pop("SENSTRIG", 1.0))
    CEREB = float(kwargs.pop("CEREB", 1.0))
    G = float(kwargs.get("G", 6.0))

    seed = int(kwargs.pop("seed", -1))
    if seed >= 0:
        SEED = True
    else:
        SEED = False
        seed = 10

    test_name = kwargs.pop("test_name", "")

    experiment_name = "noise%d_PATHWAY_GAIN%d_SENSTRIG%d_CEREB%d" % \
                      (NOISE, intval(PATHWAY_GAIN), intval(SENSTRIG), intval(CEREB))

    if experiment_name[0] == "_":
        experiment_name = experiment_name[1:]

    if len(test_name):
        experiment_name = "_".join([experiment_name, test_name])
    else:
        test_name = "tvb-only"

    path = os.path.join(os.getcwd(), experiment_name)

    if SEED:
        path = os.path.join(path, "nsd%d" % seed)

    # Get configuration
    config, plotter = configure(output_folder=path, verbose=0,
                                STIMULUS=STIMULUS,
                                STIMULUS_BASELINE=STIMULUS_BASELINE,
                                NOISE=10 ** (-NOISE),
                                SIMULATION_LENGTH=simulation_length,
                                **kwargs)
    config.RANDOM_SEED_TVB = seed
    config.RANDOM_SEED_NEST = seed
    config.CEREB = CEREB
    config.CNS1TH = CNS1TH
    config.PONS = PONS
    config.SENSTRIG = SENSTRIG
    config.TRIGEMINAL = True
    config.HEMISPHERES = -1
    config.PATHWAY_GAIN = PATHWAY_GAIN
    config.INDEGREE_GAIN = INDEGREE_GAIN
    config.STIMULUS_RATE = 8.0  # Hz
    config.VERBOSITY = 2.0

    config.CEREB_OFF = False
    if test_name == "cerebOFF":
        config.CEREB_OFF = True

    config.COSIMULATION = False
    if test_name == 'cosim':
        config.COSIMULATION = True

    print(config.model_params)
    print(config)

    return config, plotter


def getflags_from_config(config):
    CEREB = getattr(config, "CEREB", 1.0)
    TRIGEMINAL = getattr(config, "TRIGEMINAL", True)
    HEMISPHERES = getattr(config, "HEMISPHERES", -1)
    CNS1TH = getattr(config, "CNS1TH", 1.0)
    SENSTRIG = getattr(config, "SENSTRIG", 1.0)
    PONS = getattr(config, "PONS", 0.0)
    return CEREB, TRIGEMINAL, HEMISPHERES, CNS1TH, SENSTRIG, PONS


def apply_pathway_gain_to_target(src_inds, trg_inds, pathway_gain, weights, hemispheres=-1,
                                 fix_inds=[], indegree_gain=1.0):
    FIXflag = False
    if len(fix_inds):
        FIXflag = True
    if src_inds is None:
        src_inds = np.arange(weights.shape[1]).astype('i')
        if FIXflag:
            src_inds = np.delete(src_inds, fix_inds)
    for iT, trg in enumerate(trg_inds):
        print("trg = ", trg)
        indegree = weights[trg].sum()                  # initial total indegree
        print("indegree = ", indegree)
        if hemispheres > 0:
            hemi_src_inds = src_inds[slice(np.mod(iT, 2), None, 2)]
        elif hemispheres < 0:
            hemi_src_inds = src_inds[slice(np.abs(np.mod(iT, 2)-1), None, 2)]
        else:
            hemi_src_inds = src_inds
        print("hemi_src_inds = ", hemi_src_inds)
        orig = weights[trg, hemi_src_inds]
        print("orig = ", orig)
        origsum = orig.sum()
        print("origsum = ", origsum)
        nsrc = len(hemi_src_inds)
        if FIXflag:
            hemi_fix_inds = fix_inds[slice(np.mod(iT, 2), None, 2)]  # Only ipsilaterally
            print("hemi_fix_inds = ", hemi_fix_inds)
            fix = weights[trg, hemi_fix_inds]
            print("fix = ", fix)
            fixsum = fix.sum()
            print("fixsum = ", fixsum)
        else:
            fixsum = 0.0
        if indegree_gain != None:
            if indegree < 1.0:
                new_indegree = indegree * indegree_gain
            else:
                new_indegree = indegree/indegree_gain
            print("new_indegree = ", new_indegree)
        elif pathway_gain < 0.1:
            new_indegree = indegree
        else:
            new_indegree = None
        pathway_gain_corr = pathway_gain
        if pathway_gain < 1.0:
            pathway_gain_corr = np.minimum(pathway_gain_corr, 0.99)
            print("pathway_gain_corr %= ", 100*pathway_gain_corr)
            newsum = pathway_gain_corr * new_indegree
            print("wsum = \n", newsum)
            eff_pathway_gain = newsum/origsum
            print("eff_pathway_gain = ", eff_pathway_gain)
            nornom = new_indegree - newsum - fixsum
            if nornom < 0.0:
                new_indegree = newsum + fixsum + 0.01 * indegree
                print("new_indegree_corr = ", new_indegree)
                indegree_gain = indegree / new_indegree
                print("indegree_gain_corr = ", indegree_gain)
                nornom = new_indegree - newsum - fixsum
            norm = nornom / (indegree - origsum - fixsum)
            print("norm = ", norm)
            eff_pathway_gain /= norm
            weights[trg] *= norm
            weights[trg, hemi_src_inds] *= eff_pathway_gain  # increase pathway
            print("w = \n", weights[trg, hemi_src_inds])
            if FIXflag:
                weights[trg, hemi_fix_inds] /= norm  # set fixed connections
                print("wfix = \n", weights[trg, hemi_fix_inds])
        elif pathway_gain >= 1.0:
            if pathway_gain > 1.0:
                pathway_gain_corr = pathway_gain_corr / nsrc
                if new_indegree is not None:
                    pathway_gain_corr = np.minimum(pathway_gain_corr,
                                                   (0.99*new_indegree - fixsum)/origsum)
            print("pathway_gain_corr = ", pathway_gain_corr)
            weights[trg, hemi_src_inds] *= pathway_gain_corr  # increase pathway
            print("w = \n", weights[trg, hemi_src_inds])
            newsum = weights[trg, hemi_src_inds].sum()
            print("wsum = \n", newsum)
            if new_indegree:
                nornom = new_indegree - newsum - fixsum
                if nornom < 0.0:
                    new_indegree = newsum + fixsum + 0.01 * indegree
                    print("new_indegree_corr = ", new_indegree)
                    indegree_gain = indegree / new_indegree
                    print("indegree_gain_corr = ", indegree_gain)
                    nornom = new_indegree - newsum - fixsum
                norm = nornom / (indegree - origsum - fixsum)
                print("norm = ", norm)
                weights[trg] *= norm
                weights[trg, hemi_src_inds] /= norm
                if FIXflag:
                    weights[trg, hemi_fix_inds] /= norm  # set fixed connections
                    print("wfix = \n", weights[trg, hemi_fix_inds])

        final_indegree = weights[trg].sum()
        print("final indegree = ", final_indegree)
        try:
            assert np.all(final_indegree >= 0.0)
        except Exception as e:
            print(weights[trg])
            raise e
        indegree_ratio = indegree / final_indegree
        if indegree_gain is not None:
            try:
                assert np.isclose(indegree_ratio, float(indegree_gain), rtol=1e-03, atol=1e-03)
            except Exception as e:
                print(indegree)
                print(final_indegree)
                raise e
    return np.abs(weights), indegree_ratio


def apply_pathway_gain(weights, inds, pathway_gain, indegree_gain,
                       hemispheres=-1, PONS=False, SENSTRIG=1.0, CEREB=1.0):  # pathway_mode="task"

    hemispheres = np.abs(hemispheres)
    print("\n" + "-" * 50)
    print("Applying pathway gain = %g" % pathway_gain)
    print("-" * 50 + "\n")

    indegree_ratios = {}
    # A. INPUT SENSORY PATHWAY:

    # 1. PosSens Trigeminal <- Trigeminal (stimulus)
    print("-" * 25 + "\n")
    print("trigeminal -> ponssens_trigeminal")

    weights, indegree_ratio = \
        apply_pathway_gain_to_target(inds["trigeminal"],  # ipsilaterally or bilaterally
                                     inds["ponssens_trigeminal"],
                                     SENSTRIG*pathway_gain, weights,
                                     hemispheres=hemispheres,
                                     indegree_gain=indegree_gain)
    indegree_ratios["ponssens_trigeminal"] = indegree_ratio

    if PONS:
        # 2. S1 brl field -> PosSens
        print("-" * 25 + "\n")
        print("s1brl -> ponssens")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["s1brl"],  # ipsilaterally or bilaterally
                                         inds["ponssens"],
                                         pathway_gain, weights,
                                         hemispheres=hemispheres,
                                         indegree_gain=indegree_gain)
        indegree_ratios["ponssens"] = indegree_ratio

    # TODO: Think about these connections as well!
    # # 3. S1 brl thal <- [Trigeminal (stimulus), PonsSens Trigeminal, CerebNuclei]
    # # if pathway_mode != "stim":
    # # 2. S1 brl thal <- [
    print("-" * 25 + "\n")
    print("[Trigeminal (stimulus), PonsSens Trigeminal, CerebNuclei] -> s1brlthal")
    weights, indegree_ratio = \
        apply_pathway_gain_to_target(np.concatenate([inds["trigeminal"],
                                                     inds["ponssens_trigeminal"],
                                                     inds["cereb_nuclei"]]),  # contralaterally or bilaterally
                                     inds["s1brlthal"],
                                     pathway_gain, weights,  # 30.0
                                     hemispheres=-hemispheres,
                                     fix_inds=inds["s1brl"],
                                     indegree_gain=None)
    indegree_ratios["s1brlthal"] = indegree_ratio

    # 5. AnsiLob <- Trigeminal (stimulus)
    print("-" * 25 + "\n")
    # 2. (S1 brl &) PosSens (Trigeminal) + (M1 &) MotorPons -> AnsiLob
    source = np.concatenate([inds["trigeminal"],
                             inds["ponssens_trigeminal"],  # ipsilaterally or bilaterally
                           ])
    if PONS:
        source = np.concatenate([source,
                                 inds["ponssens"][::-1],  # contralaterally or bilaterally
                                 # including B. FEEDBACK SENSORY PATHWAY
                                 # inds["s1brl"][::-1], # TODO: Think about this connection as well!
                                 # including # D. FEEDBACK MOTOR PATHWAY
                                 inds["ponsmotor"][::-1],  # contralaterally or bilaterally
                                 # inds["m1"][::-1], # TODO: Think about this connection as well!
                                 ])
        print("[trigeminal, ponssens_trigeminal, ponssens, ponsmotor] -> ansilob")
    else:
        print("[trigeminal, ponssens_trigeminal] -> ansilob")

    weights, indegree_ratio = \
        apply_pathway_gain_to_target(source,
                                     inds["ansilob"],
                                     CEREB*pathway_gain, weights,
                                     hemispheres=hemispheres,
                                     indegree_gain=indegree_gain)
    indegree_ratios["ansilob"] = indegree_ratio

    #if pathway_mode != "stim":
    # 5. Cereb nuclei <- AnsiLob
    print("-" * 25 + "\n")
    print("ansilob -> cereb_nuclei")
    weights, indegree_ratio = \
        apply_pathway_gain_to_target(inds["ansilob"],  # ipsilaterally or bilaterally
                                     inds["cereb_nuclei"],
                                     CEREB*pathway_gain, weights,
                                     hemispheres=hemispheres,
                                     indegree_gain=indegree_gain)
    indegree_ratios["cereb_nuclei"] = indegree_ratio

    # C. INPUT MOTOR PATHWAY:

    # 1.  M1 thal <- CerebNuclei
    print("-" * 25 + "\n")
    print("cereb_nuclei -> m1thal")
    weights, indegree_ratio = \
        apply_pathway_gain_to_target(inds["cereb_nuclei"],  # contralaterally or bilaterally
                                     inds["m1thal"],
                                     pathway_gain, weights,    # 30.0
                                     hemispheres=-hemispheres,
                                     fix_inds=inds["m1"],
                                     indegree_gain=None)
    indegree_ratios["m1thal"] = indegree_ratio

    if PONS:
        # D. FEEDBACK MOTOR PATHWAY:
        # 1. PonsMotor <- M1
        print("-" * 25 + "\n")
        print("m1 -> ponsmotor")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["m1"],       # ipsilaterally or bilaterally
                                         inds["ponsmotor"],
                                         pathway_gain, weights,
                                         hemispheres=hemispheres,
                                         indegree_gain=indegree_gain)
        indegree_ratios["ponsmotor"] = indegree_ratio

    weights, indegree_ratio = apply_pathway_gain_to_target(
        inds["s1brl"], inds["m1"],
        1.0, # pathway_gain,
        weights, hemispheres=0,
        fix_inds=inds["m1thal"].tolist(),
        indegree_gain=pathway_gain  # 5*indegree_gain
    )
    indegree_ratios["m1"] = indegree_ratio

    weights, indegree_ratio = apply_pathway_gain_to_target(
        inds["m1"], inds["s1brl"],
        1.0,  # pathway_gain,
        weights, hemispheres=0,
        fix_inds=inds["s1brlthal"].tolist(),
        indegree_gain=pathway_gain  # 5*indegree_gain
        )
    indegree_ratios["s1brl"] = indegree_ratio

    return\
        weights, indegree_ratios


def adjust_ficed_params(simulator, indegree_ratios, inds):

    def adjust_fic(weights, indegree_ratio, inds, fie, fwie, I_e, w_ie):
        for ind in inds:
            print("region ind = %d" % ind)
            new_indegree = weights[ind].sum()
            print("new_indegree = %g" % new_indegree)
            print("indegree_ratio = %g" % indegree_ratio)
            indegree = new_indegree * indegree_ratio
            print("indegree = %g" % indegree)
            print("I_e[%d]_old = %g" % (ind, I_e[ind]))
            I_e[ind] = fie(I_e[ind], indegree, new_indegree)
            print("I_e[%d]_adj = %g" % (ind, I_e[ind]))
            print("w_ie[%d]_old = %g" % (ind, w_ie[ind]))
            w_ie[ind] = fwie(w_ie[ind], indegree, new_indegree)
            print("w_ie[%d]_adj = %g" % (ind, w_ie[ind]))
        return I_e, w_ie

    print("-" * 25)
    print("-" * 25)
    print("Adjusting FICed parameters for (non thalamic) regions with modified indegree...")
    # Necessary functions for adjustment of FICed paremeters
    indmax_Ie = inds["crtx_and_subcrtx"][[np.argmax(simulator.model.I_e[inds["crtx_and_subcrtx"]])]][0].item()
    indmax_wie = inds["crtx_and_subcrtx"][[np.argmax(simulator.model.w_ie[inds["crtx_and_subcrtx"]])]][0].item()
    assert indmax_Ie == indmax_wie
    iemax = simulator.model.I_e[indmax_Ie]
    print("Ie[%d]_max = %g" % (indmax_Ie, iemax))
    wiemax = simulator.model.w_ie[indmax_wie]
    print("w_ie[%d]_max = %g" % (indmax_wie, wiemax))
    indegree_min = simulator.connectivity.weights[indmax_Ie].sum()
    print("indegree_min = %g" % indegree_min)
    fie = lambda ie, indegree, new_indegree: np.minimum(iemax,
        iemax + (new_indegree - indegree_min) * np.minimum(0.0, (ie - iemax) / (indegree - indegree_min)))
    #  i.e., ymax  + (     x       -     x0     ) *            (yold - ymax)  / (   xold  -    xmin)
    fwie = lambda wie, indegree, new_indegree: np.minimum(wiemax,
        wiemax + (new_indegree - indegree_min) * np.minimum(0.0, (wie - wiemax) / (indegree - indegree_min)))
    for reg, indratio in indegree_ratios.items():
        if "thal" not in reg and not np.isclose(indratio, 1.0, rtol=1e-03, atol=1e-03):
            print("...adjusting regions %s..." % reg)
            simulator.model.I_e, simulator.model.w_ie = \
                adjust_fic(simulator.connectivity.weights, indratio, inds[reg], fie, fwie,
                           simulator.model.I_e, simulator.model.w_ie)
    return simulator


def print_weight_to_indegree(src, trg, inds, w, hemispheres=1):
    print("\n" + "-"*25)
    print("%s -> %s" % (src, trg))
    print(w[inds[trg]][:, inds[src]])
    print("%:")
    print(w[inds[trg]][:, inds[src]] / w[inds[trg]].sum() * 200 / (2 - np.abs(hemispheres)))


def cosim_run_plot(**kwargs):

    config, plotter = get_config(**kwargs)

    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config,
                                      hemispheres=config.HEMISPHERES,
                                      cereb_nuclei_to_s1thal=config.CNS1TH,
                                      trigeminal_to_m1thal=False)


    # Task related regions' indices:
    config.TASKINDS = np.concatenate([inds["m1"], inds["s1brl"],
                                      inds["m1thal"], inds["s1brlthal"],
                                      inds["trigeminal"], inds['ponssens_trigeminal'],
                                      inds['ansilob'], inds['cereb_nuclei']])  # , inds['cereb_crtx']
    if config.PONS:
        config.TASKINDS = np.concatenate([config.TASKINDS, inds["ponsmotor"], inds["ponssens"]])
    config.TASKINDS = np.sort(config.TASKINDS)

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)

    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

    if config.PATHWAY_GAIN != 1.0:
        simulator.connectivity.weights, indegree_ratios = \
            apply_pathway_gain(simulator.connectivity.weights, inds,
                               config.PATHWAY_GAIN, config.INDEGREE_GAIN,
                               hemispheres=config.HEMISPHERES, PONS=config.PONS,
                               SENSTRIG=config.SENSTRIG, CEREB=config.CEREB)
        print("-"*25)
        print("Indegree ratios:")
        print(indegree_ratios)
        print("-" * 25)
        if not np.isclose(config.FIC, 1.0, rtol=1e-03, atol=1e-03):
            simulator = adjust_ficed_params(simulator, indegree_ratios, inds)
        simulator.configure()

    # Put cereb weights to 0 if CEREB_OFF
    if config.CEREB_OFF:
        print("-" * 25)
        print("-" * 25)
        print("Removing cerebellum connections!..")
        # reg1='Cerebell*'
        reg1 = 'Left Cerebellar Cortex'
        reg2 = 'Left Cerebellar Nuclei'
        reg3 = 'Left Ansiform lobule'
        # reg4 = 'Left Interposed nucleus'
        reg5 = 'Right Cerebellar Cortex'
        reg6 = 'Right Cerebellar Nuclei'
        reg7 = 'Right Ansiform lobule'
        # reg8 = 'Right Interposed nucleus'
        # find the indices in region labels of these strings
        iR1 = np.where([reg1 in reg for reg in connectivity.region_labels])[0].item()
        iR2 = np.where([reg2 in reg for reg in connectivity.region_labels])[0].item()
        iR3 = np.where([reg3 in reg for reg in connectivity.region_labels])[0].item()
        # iR4 = np.where([reg4 in reg for reg in connectivity.region_labels])[0].item()
        iR5 = np.where([reg5 in reg for reg in connectivity.region_labels])[0].item()
        iR6 = np.where([reg6 in reg for reg in connectivity.region_labels])[0].item()
        iR7 = np.where([reg7 in reg for reg in connectivity.region_labels])[0].item()
        # iR8 = np.where([reg8 in reg for reg in connectivity.region_labels])[0].item()
        # for reg1, reg2, sc in config.BRAIN_CONNECTIONS_TO_SCALE:
        #     iR1 = np.where([reg in reg1 for reg in connectivity.region_labels])[0]
        #     iR2 = np.where([reg in reg2 for reg in connectivity.region_labels])[0]
        #     connectivity.weights[iR1, iR2] *= 0
        # iR1
        conns = np.array([iR1, iR2, iR3, iR5, iR6, iR7])  #, iR4, iR8
        for i in conns.tolist():
            simulator.connectivity.weights[i, :] = 0
            simulator.connectivity.weights[:, i] = 0
        simulator.connectivity.configure()
        simulator.configure()
        print([simulator.connectivity.weights[conns].sum(), simulator.connectivity.weights[:, conns].sum()])

    print("\n" + "-" * 50)
    print("Pathway connections to indegree %:")

    for conns in [
        ["s1brl", "m1"], ["m1", "s1brl"],
        ["s1brlthal", "s1brl"],
        ["s1brl", "s1brlthal"],
        ["trigeminal", "ponssens_trigeminal"],
        ["trigeminal", "s1brlthal"], ["ponssens_trigeminal", "s1brlthal"], ["cereb_nuclei", "s1brlthal"],
        ["m1thal", "m1"],
        ["m1", "m1thal"], ["trigeminal", "m1thal"], ["cereb_nuclei", "m1thal"],
        ["trigeminal", "ansilob"], ["ponssens_trigeminal", "ansilob"],
        ["ansilob", "cereb_nuclei"],
        ["s1brl", "ponssens"], ["m1", "ponsmotor"],
        ["ponssens", "ansilob"], ["ponsmotor", "ansilob"]
    ]:
        print_weight_to_indegree(conns[0], conns[1], inds, simulator.connectivity.weights,
                                 hemispheres=config.HEMISPHERES)

    # Plot task network:

    config.TASK_SHORT_REG_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"])
                                    for reg in simulator.connectivity.region_labels[config.TASKINDS]]
    fig, ax = plt.subplots()
    ax = matrix_plot(simulator.connectivity.weights[config.TASKINDS][:, config.TASKINDS].copy(),
                     labels=config.TASK_SHORT_REG_LABELS,
                     label="SC", ax=ax, colorbar=True, fontsize=10)
    fig.tight_layout()
    pyplot.savefig(os.path.join(config.figures.FOLDER_FIGURES, "taskSC.png"), format="png")

    if config.COSIMULATION:
        # Build TVB-NEST interfaces
        nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
        simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
        # Simulate TVB-NEST model
        results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)
    else:
        # Run simulation and get results for reference values
        results, transient = simulate(simulator, config)

    # Target values: ansilob=-0.3263, interposed=-0.3209, oliv=-0.3284

    # Compute coherence
    transient = config.TRANSIENT_RATIO * config.SIMULATION_LENGTH
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2

    results = plot_tvb(transient, inds,
                       results=results, simulator=simulator, plotter=plotter, config=config, write_files=True)

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

    return results, simulator, config, inds


def plot_comparison(tests, **kwargs):

    # CONFIGURATION:
    config, plotter = get_config(**kwargs)

    CEREB, TRIGEMINAL, HEMISPHERES, CNS1TH, SENSTRIG, PONS = getflags_from_config(config)

    # CONNECTIVITY:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=None)
    connectivity = build_connectivity(connectome, inds, config,
                                      hemispheres=config.HEMISPHERES,
                                      cereb_nuclei_to_s1thal=config.CNS1TH,
                                      trigeminal_to_m1thal=False)

    # Results path:
    BASEPATH = config.out.FOLDER_RES.split("res")[0][:-1]
    print(BASEPATH)

    # Task related regions' indices:
    TASKINDS = np.concatenate([inds["m1"], inds["s1brl"],
                               inds["m1thal"], inds["s1brlthal"],
                               inds["trigeminal"], inds['ponssens_trigeminal'],
                               inds['ansilob'], inds['cereb_nuclei']])  # , inds['cereb_crtx']
    if config.PONS:
        TASKINDS = np.concatenate([TASKINDS, inds["ponsmotor"], inds["ponssens"]])
    TASKINDS = np.sort(TASKINDS)

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
    colors = []
    for test, col in zip(["cosim", "tvb-only", "cerebOFF"], ["b", "g", "r"]):
        if test in TESTS:
            colors.append(col)
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
        tests=TESTS, colors=colors,
        percentile_min=1, percentile_max=99, n=1,
        plot_mean=True, plot_median=False, mode="semilog",
        alpha=0.5, figsize=config.figures.LARGE_SIZE, fontsize=16)

    if plotter.config.SAVE_FLAG:
        for fig, hemi in zip([figR, figL], ["Right", "Left"]):
            plt.figure(fig.number)
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "PathwayPSD_COH_%s.png" % hemi))

    figPSD, axesPSD = psd_percent_plot(results,
                                        inds=None,
                                        tests=TESTS, colors=colors,
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
    # $ python tuning_tvb_nest.py w_TVB_to_NEST=0.04375 'simulation_length'='300.0'
    # Called tuning_tvb_nest.py with:
    # keyword argument: w_TVB_to_NEST=world
    # keyword argument: simulation_length=300.0

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
        cosim_run_plot(**kwargs)
