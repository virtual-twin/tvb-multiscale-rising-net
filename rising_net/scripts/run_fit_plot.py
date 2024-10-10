import glob
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

from rising_net.scripts.base import assert_config, DEFAULT_ARGS, args_parser
from rising_net.scripts.filepaths import simres_filepath, istr
from rising_net.scripts.nest_script import build_NEST_network
from rising_net.scripts.sbi_script import load_posterior_samples, load_train_params_samples_selection
from rising_net.scripts.tvb_nest_script import build_tvb_nest_interfaces, simulate_tvb_nest
from rising_net.scripts.tvb_script import prepare_connectome, build_connectivity, build_model, build_simulator, \
    simulate, plot_tvb, tvb_res_to_time_series, compute_PSD_target_and_data
from rising_net.scripts.utils import compute_selected_spectra_coherence, joinstr
from tvb_multiscale.core.utils.data_structures_utils import narray_summary_info
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict, load_pickled_dict

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


GSTR = "iG"
RESSTR = "res"
NSDSTR = "nsd"


def iGstr(iG, Ngs=100, igstr=GSTR):
    return igstr + istr(int(iG), Ns=Ngs)


def iPstr(iP, Nsims=10000, resstr=RESSTR):
    return resstr + istr(int(iP), Ns=Nsims)


def iRstr(iR, Nreps=10, nsdstr=NSDSTR):
    return nsdstr + istr(int(iR), Ns=Nreps)


def get_simres_folder_name(config, FUNCMODE="SIM"):
    if FUNCMODE.upper() == "TRAINSIM":
        return config.TRAIN_SIMS_FOLDER
    elif FUNCMODE.upper() == "PPCSIM":
        return config.PPC_FOLDER
    elif FUNCMODE.upper() == "MEANSIM":
        return config.MEAN_FOLDER
    elif FUNCMODE.upper() == "MAPSIM":
        return config.MAP_FOLDER
    else:
        return ""


def simres_folder(config, iG=None, iP=None, iR=None, FUNCMODE="TRAINSIM", label=""):
    folder = get_simres_folder_name(config, FUNCMODE)
    if len(label):
        folder = os.path.join(folder, label)
    if iG is not None:
        folder = os.path.join(folder,
                              iGstr(iG, Ngs=len(config.Gs)))
    if iP is not None:
        folder = os.path.join(folder,
                              iPstr(iP, Nsims=config.N_SIMULATIONS))
    if iR is not None:
        folder = os.path.join(folder,
                              iRstr(iR, Nreps=config.N_SIMS_PER_PARAM))
    # else:
    #     folder = os.path.join(folder, "res")
    return folder


def get_stats_params(config, stat=None, FUNCMODE=None, iG=None, iP=None, iF=None, fitlabel="", verbosity=None):
    if verbosity is None:
        verbosity = config.VERBOSITY
    if FUNCMODE is None:
        if stat is not None:
            FUNCMODE = "%sSIM" % stat.upper()
        else:
            FUNCMODE = "PPCSIM"
    labeliG = str(fitlabel)
    if iG is not None:
        labeliG = joinstr([iGstr(iG, Ngs=len(config.Gs)), labeliG])
    samples = load_posterior_samples(label=labeliG, config=config)

    # In these cases we need to load parameters from a samples.pkl file
    # after sampling posterior distributions with SBI
    if config.ALL_SAMPLES_LABEL in fitlabel and iF is not None:
        warnings.warn("Setting iF = None (originally given %s), given the fitlabel '%s'!" % (str(iF), fitlabel))
        iF = None
    params_string = ""
    if iF is None or config.ALL_SAMPLES_LABEL in fitlabel:
        if verbosity:
            params_string = "%s PARAMETERS" % joinstr([FUNCMODE, labeliG])
        iF = slice(None)  # Assuming all runs' samples fitting, and not from a specific fitting run
        if config.ALL_SAMPLES_LABEL not in fitlabel and config.ALL_RUNS_LABEL not in fitlabel:
            fitlabel = joinstr([fitlabel, config.ALL_RUNS_LABEL])
    else:
        iF = ensure_list(iF)
        Nf = len(iF)
        if Nf > 1:
            iFstr = "iF_%s" % str(iF)
        else:
            iFstr = "iF%s" % istr(iF[0], Ns=config.N_FIT_RUNS)
        if verbosity:
            params_string = "%s PARAMETERS_%s" % (joinstr([FUNCMODE, labeliG]), iFstr)
        fitlabel = joinstr([fitlabel, iFstr])
    if FUNCMODE == "PPCSIM":
        if iP is None:  # simulations for fitting
            raise ValueError("Parameter sample index iP is None for Post Predictive Check simulations!")
        if verbosity:
            Np = len(ensure_list(iP))
            if Np > 1:
                params_string += "_iP_%s" % str(iP)
            else:
                params_string += "_iP%s" % istr(iP, Ns=config.N_PPC_SIMS)
        try:
            # If iF is None, choose among the posterior samples of all runs:
            #                                                 [Run(s)][param_set(s), params]
            params_vals = np.vstack(np.vstack(samples["samples"])[iF])[iP].squeeze()
        except Exception as e:
            print("\nFailed to get sample for %s\n"
                  "with iF = %s, iP = %s from samples of shape %s!"
                  % (params_string, str(iF), str(iP), str(np.array(samples["samples"]).shape)))
            raise e
    else:
        iP = 0  # MAP or mean is just one parameter set
        if FUNCMODE == "MEANSIM":
            stat = "mean"
        else:
            stat = "map"
        try:
            params_vals = np.vstack(samples[stat])[iF].mean(axis=0)
        except Exception as e:
            print("\nFailed to get %s for %s\n"
                  "with iF = %s from an array of shape %s!"
                  % (stat, params_string, str(iF), str(np.array(samples[stat]).shape)))
            raise e
    params = dict(zip(config.PRIORS_PARAMS_NAMES, params_vals.T))
    return params, iP, fitlabel, params_string


def process_funcmode(FUNCMODE, MODE, config, verbosity=1, iP=None, iR=None, iF=None, iG=None, fitlabel="", **kwargs):
    FUNCMODE = FUNCMODE.upper()
    params = {}
    params_string = ""
    if FUNCMODE in ["TRAINSIM", "PPCSIM", "MEANSIM", "MAPSIM"]:  # this is only for sampling parameters
        # In all these cases we need to load parameters from files
        if "REST" in MODE and iG is None:
            raise ValueError("G parameter index iG is None for REST %s simulations!" % FUNCMODE)
        if FUNCMODE == "TRAINSIM":
            # In this case we need to load parameters from a .pt file after sampling prior distributions with torch
            if iP is None:  # simulations for fitting
                raise ValueError("Parameter sample index iP is None for training simulations!")
            params = dict(zip(config.PRIORS_PARAMS_NAMES,
                              load_train_params_samples_selection(iP, config,
                                                                  # iR=parameters_iR,
                                                                  # label=parameters_label,
                                                                  # filepath=parameters_filepath,
                                                                  # extension=parameters_filepath_ext
                                                                  ).numpy().squeeze()))
            if verbosity:
                params_string = "%s PARAMETERS%s" % (FUNCMODE, istr(iP, Ns=config.N_SIMULATIONS))
        elif FUNCMODE in ["PPCSIM", "MEANSIM", "MAPSIM"]:
            params, iP, fitlabel, params_string = \
                get_stats_params(config, stat=None, FUNCMODE=FUNCMODE, iG=iG, iP=iP, iF=iF,
                                 fitlabel=fitlabel, verbosity=verbosity)
        kwargs.update(params)
        if verbosity:
            print("%s:\n%s" % (params_string, str(params)))
    if FUNCMODE in ["TRAINSIM", "PPCSIM"]:
        kwargs["plot_flag"] = kwargs.get("plot_flag", False)  # too many simulations, we can't plot them
        # Check for noise seed repetitions:
        if iR is None:
            if iP is not None:  # Follow the parameters' index if no iR is given
                iR = iP
    if iR is None:
        iR = 0
        iRpath = None
    else:
        iRpath = iR

    return iRpath, iR, iP, params, params_string, fitlabel, kwargs


def sim_run_plot(iG=None, iP=None, iR=None, FUNCMODE="SIM", label="",
                 config=None, REST_or_TASK=None, **kwargs):
    if config is None:
        if REST_or_TASK is None:
            MODE = kwargs.get("MODE", None)
            if "REST" in MODE:
                REST_or_TASK = "REST"
            elif "TASK" in MODE:
                REST_or_TASK = "TASK"
            else:
                PG = kwargs.get("PATHWAY_GAIN", None)
                if PG is not None:
                    if PG > 0:
                        REST_or_TASK = "TASK"
                    else:
                        REST_or_TASK = "REST"
                else:
                    raise ValueError("No way to determine if it is a REST or TASK simulation"
                                     "since config=%s, REST_or_TASK=%s, MODE=%s and PATHWAY_GAIN=%s!"
                                     % (str(config), str(REST_or_TASK), str(MODE), str(PG)))
        if REST_or_TASK == "REST":
            from rising_net.scripts.rest_run_fit_plot import get_config
        elif REST_or_TASK == "TASK":
            from rising_net.scripts.task_run_fit_plot import get_config
        config, plotter = get_config(iG=iG, iP=iP, iR=iR, FUNCMODE=FUNCMODE, **kwargs)

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
                           results=results, simulator=simulator, plotter=plotter, config=config,
                           write_files=FUNCMODE.upper() == "SIM")
        # if "COSIM" in config.MODE::
        #     plot_write_spiking_network_results(nest_network, connectivity=connectivity,
        #                                        time=None, transient=transient, monitor_period=simulator.monitors[0].period,
        #                                        plot_per_neuron=False, plotter=plotter, writer=None, config=config)
    else:
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=FUNCMODE.upper() == "SIM")
    results["regions"] = simulator.connectivity.region_labels[inds["m1s1brl"]]

    if "REST" in config.MODE:
        # Return only the M1 <-> PSD fitting target
        if "PSD" not in results.keys():
            PSD, PSD_target = compute_PSD_target_and_data(config, results[0], inds, transient,
                                                          write_files=FUNCMODE.upper() == "SIM",
                                                          plotter=None)
            results = {"PSD": PSD, "f": PSD_target['f']}

        if FUNCMODE.upper() != "SIM":
            if FUNCMODE.upper() in ["TRAINSIM", "PPCSIM"]:
                for key in results.keys():
                    if key not in ["PSD", "f", "regions"]:
                        del results[key]
            if config.VERBOSITY:
                print("\nWriting results %s\nto file %s...\n" %
                      (str(results.keys()), simres_filepath(config, iR=iR, label=label)))
            dump_pickled_dict(results, simres_filepath(config, iR=iR, label=label))
    else:
        # Return PSD and COH along the task pathway fitting targets:
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
        results["regions"] = simulator.connectivity.region_labels[taskinds]

        results_keys = list(results.keys())
        if FUNCMODE.upper() != "SIM":
            if FUNCMODE.upper() in ["TRAINSIM", "PPCSIM"]:
                for key in results_keys:
                    if key not in ["PSD", "f", "COH", "pairs", "regions"]:
                        del results[key]
            if config.VERBOSITY:
                print("\nWriting results %s\nto file %s...\n" %
                      (str(results.keys()), simres_filepath(config, iR=iR, label=label)))
            dump_pickled_dict(results, simres_filepath(config, iR=iR, label=label))

    return results, simulator, nest_network, config, inds


def find_all_folders(path, folderstr):
    pathstr = os.path.join(path, folderstr + "_*", "")
    ii = []
    for p in ensure_list(np.sort(glob.glob(pathstr))):
        ii.append(int(p[:-1].split("%s_" % folderstr)[-1]))
    return ii


def load_sims_to_xarrays_for_iR(load_sim_to_xarrays_fun, measures, path=None, config=None, iR=None, average=False,
                                folderstr=NSDSTR, resstr=RESSTR, **kwargs):
    config = assert_config(config, return_plotter=False)
    if path is None:
        path = config.out.FOLDER_RES
    if iR is None:
        iR = find_all_folders(path, folderstr)
    else:
        iR = np.sort(ensure_list(iR)).tolist()
    measures = ensure_list(measures)
    for iM, measure in enumerate(measures):
        measures[iM] = measure.upper()
    if len(iR):
        res = dict(zip(measures, [list() for _ in range(len(measures))]))
        for iiR in iR:
            res_i = load_sim_to_xarrays_fun(os.path.join(path, iRstr(iiR, config.N_SIMS_PER_PARAM)),
                                            config, iR=iiR, resstr=resstr, measures=measures, **kwargs)
            for measure in measures:
                res[measure].append(res_i[measure])
        for measure in measures:
            res[measure] = xr.concat(res[measure], dim=pd.Index(iR, name="Repetitions' index iR"))
            res[measure].name = path + ", Repetitions: %s" % \
                                list(narray_summary_info(np.array(iR), omit_shape=True).values())[0]
            if average:
                res[measure] = res[measure].mean(axis=0).squeeze()
    else:
        iR = None
        res = load_sim_to_xarrays_fun(path, config, iR=iR, resstr=resstr, measures=measures, **kwargs)
    return res, iR


def load_sims_to_xarrays_for_iP(load_sim_to_xarrays_fun, measures, path=None, config=None, iP=None, iR=None,
                                average_repetitions=True, folderstr=NSDSTR, resstr=RESSTR, **kwargs):
    config = assert_config(config, return_plotter=False)
    if path is None:
        path = config.out.FOLDER_RES
    if iP is None:
        iP = find_all_folders(path, RESSTR)
    else:
        iP = np.sort(ensure_list(iP)).tolist()
    measures = ensure_list(measures)
    for iM, measure in enumerate(measures):
        measures[iM] = measure.upper()
    if len(iP):
        res = dict(zip(measures, [list() for _ in range(len(measures))]))
        for iiP in iP:
            res_i = load_sims_to_xarrays_for_iR(load_sim_to_xarrays_fun, measures,
                                                os.path.join(path, iPstr(iiP, Nsims=config.N_SIMULATIONS)),
                                                config, iR=iR, average=average_repetitions,
                                                folderstr=folderstr, resstr=resstr, **kwargs)[0]
            for measure in measures:
                res[measure].append(res_i[measure])
        for measure in measures:
            res[measure] = xr.concat(res[measure], dim=pd.Index(iP, name="Parameters' samples' index iP"))
            res[measure].name = path + ", Parameters' samples: %s" % \
                                list(narray_summary_info(np.array(iP), omit_shape=True).values())[0]
    else:
        iP = None
        res = load_sims_to_xarrays_for_iR(load_sim_to_xarrays_fun, measures,
                                          path, config, iR=iR, average=average_repetitions,
                                          folderstr=folderstr, resstr=resstr, **kwargs)[0]
    return res, iP


def run_fit_plot_args_parser(funname, defargs=DEFAULT_ARGS):

    parser = args_parser(funname, defargs)

    arguments = {'function': ['func', str, 'Function name to run', "cosim_run_plot"],
                 'iG': ['ig', int, "G values' index", None],
                 'iR': ['ir', int, 'Repetition index', None],
                 'iP': ['ip', int, 'Parameter sample index', None],
                 'iF': ['if', int, 'Fitting run index', None],
                 'FUNCMODE': ['fnmd', str, 'Functionality mode name', "SIM"],
                 'label': ['lbl', str, 'Specific label name', ""],
                 'fitlabel': ['flbl', str, 'Specific fitting label name', ""]
                 }
    args = deepcopy(defargs)
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