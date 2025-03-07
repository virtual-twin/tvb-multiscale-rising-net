# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from collections import OrderedDict
import time
import dill

import argparse

import numpy as np
import random

from matplotlib import pyplot as plt

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.simulator.integrators import EulerStochastic

from rising_net.scripts.utils import joinstr


PATHWAY_GAINS = {"TRIG_GAIN": 60.0, "MEDULLA_GAIN": 60.0, "CEREB_GAIN": 60.0,
                 "TRIGS1_GAIN": 1.0, "MEDULLAS1_GAIN": 1.0,
                 "CNS1_GAIN": 20.0, "CNM1_GAIN": 20.0,  "CNM1S1_GAIN": 0.5,
                 "M1S1_GAIN": 1.0,
                 "M1FACIAL_GAIN": 0.5,   # 50.0,
                 "FACIALTRIG_GAIN": 1.0,  # 50.0,
                 "WHISKERS_GAIN": 60.0}


DEFAULT_ARGS = {# TVB model:
                'I_s': 0.1,  # 0.085,
                'I_e': -0.35,
                # "STIMULUS": 0.0,
                # "STIMULUS_BASELINE": 1.0,
                "tau_w": 10.0,
                "I_w": -0.35,
                "G_w": 5.0,
                # TVB network:
                'G': 6.0,
                'FIC': 1.11,  # 2.0,
                'FIC_SPLIT': 0.31,  # 0.0,
                # Pathway gains:
                "PATHWAY_GAIN": 1,
                # TVB <-> NEST Interface:
                # "w_TVB_to_NEST": 35.0, "w_TVB_to_NEST_rest": 35.0,
                # "MAX_RATES": {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
                #               "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0},  # Hz
                # WORKFLOW:
                "NOISE": 1e-4, "NOISE_SEED": 0,
                "SIMULATION_LENGTH": 2 ** 13 + 1.0,
                "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
                "BASENAME": "", 'output_folder': "", 'verbosity': 1, 'plot_flag': True}


for pg, pgdef in PATHWAY_GAINS.items():
    DEFAULT_ARGS[pg] = pgdef


def create_plotter(config):
    from tvb_multiscale.core.plot.plotter import Plotter
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE
    return config, Plotter(config.figures)


def configure(**ARGS):
    from tvb_multiscale.core.config import find_root_dir
    from tvb_multiscale.tvb_nest.config import Config

    args = deepcopy(DEFAULT_ARGS)
    args.update(**ARGS)

    # STIMULUS = defargs.get("STIMULUS", 0)
    G_w = args.get("G_w", 0)
    WHISKERS_GAIN = args.get("WHISKERS_GAIN", 50.0)
    if np.any(G_w * WHISKERS_GAIN > 0.0):
        WHISKERS = 1
    else:
        WHISKERS = 0
        WHISKERS_GAIN = 0
        G_w = 0
    args["WHISKERS_GAIN"] = WHISKERS_GAIN
    PATHWAY_GAIN = args.get("PATHWAY_GAIN", 0)
    TASK = PATHWAY_GAIN * WHISKERS  # (STIMULUS + )

    MODE = args["MODE"]
    if TASK:
        if "TASK" not in MODE:
            MODE = joinstr(["TASK", MODE])
    elif "REST" not in MODE:
        MODE = joinstr(["REST", MODE])

    BASENAME = ARGS.get("BASENAME", "")
    if len(BASENAME) == 0:
        BASENAME = MODE

    # Flags that affect the result's path:
    # Files:
    BRAIN_CONN_FILE = "Connectivity_wavCntrs_TLwav_SummedSubcortical_Thals.h5"
    MAJOR_STRUCTS_LABELS_FILE = "major_structs_labels_SummedSubcortical_Thals.npy"  # "major_structs_labels_Thals.npy" # "major_structs_labels_SummedSubcortical_Thals.npy"
    VOXEL_COUNT_FILE = "voxel_count_SummedSubcortical_Thals.npy"  # "voxel_count_Thals.npy" # "voxel_count_SummedSubcortical_Thals.npy"
    INDS_FILE = "inds_SummedSubcortical_Thals.npy"  # "inds_Thals.npy" # "inds_SummedSubcortical_Thals.npy

    # Construct configuration
    work_path = os.getcwd()
    root_path = find_root_dir()
    rising_net = os.path.join(root_path, "rising_net")
    data_path = os.path.join(rising_net, "data")
    tvb_conn_filepath = os.path.join(data_path, BRAIN_CONN_FILE)
    major_structs_labels_filepath = os.path.join(data_path, MAJOR_STRUCTS_LABELS_FILE)
    voxel_count_filepath = os.path.join(data_path, VOXEL_COUNT_FILE)
    inds_filepath = os.path.join(data_path, INDS_FILE)
    popa_freqs_path = os.path.join(data_path, 'popa2013')
    cereb_scaffold_path = os.path.join(data_path, 'balanced_DCN_IO.hdf5')
    outputs_path = os.path.join(work_path, "outputs")
    if len(args['output_folder']):
        outputs_path = os.path.join(outputs_path, args['output_folder'])
    else:
        outputs_path = os.path.join(outputs_path, BASENAME)

    if args['verbosity']:
        print("Outputs' path: %s" % outputs_path)

    config = Config(output_base=outputs_path)

    config.VERBOSITY = args['verbosity']

    if args['plot_flag']:
        config, plotter = create_plotter(config)
    else:
        plotter = None

    # ------.----- Simulation options ----------------

    # Simulation...
    config.MODE = MODE
    config.TASK = TASK
    config.BASENAME = BASENAME
    config.HEADPATH = os.path.join(config.out.FOLDER_RES.split(BASENAME)[0], BASENAME)
    # Testing: 10: 1025, 11: 2049.0, Fitting: 13: 8193.0, BOLD: 10 mins
    config.TEST_SIMULATION_LENGTH = 2 ** 10 + 1.0
    config.FIT_SIMULATION_LENGTH = 2 ** 13 + 1.0
    config.BOLD_SIMULATION_LENGTH = 10 * 60 * 2**10 + 1.0  # = 10 mins * 60 secs * 2 ** 10 + 1
    config.SIMULATION_LENGTH = args.get("SIMULATION_LENGTH", 2 ** 13 + 1.0)
    config.TRANSIENT_RATIO = args.get("TRANSIENT_RATIO", 0.25)
    config.SOURCE_TS_PATH = os.path.join(config.out.FOLDER_RES, "source_ts.pkl")
    config.AFFERENT_TS_PATH = os.path.join(config.out.FOLDER_RES, "afferent_ts.pkl")
    config.BOLD_TS_PATH = os.path.join(config.out.FOLDER_RES, "bold_ts.pkl")
    # Integration
    config.DEFAULT_DT = 0.1
    config.DEFAULT_NSIG = args.get("NOISE", 1e-4)  # NOISE strength
    config.NOISE_SEED = int(args.get("NOISE_SEED", 0))
    config.DEFAULT_TVB_NOISE_SEED = args.get("DEFAULT_TVB_NOISE_SEED", 42) + config.NOISE_SEED
    config.NEST_MASTER_SEED = args.get("NEST_MASTER_SEED", 143202461) + config.NOISE_SEED
    config.DEFAULT_STOCHASTIC_INTEGRATOR = EulerStochastic
    config.DEFAULT_INTEGRATOR = config.DEFAULT_STOCHASTIC_INTEGRATOR

    # Connectivity
    config.WHISKERS = WHISKERS
    config.CONN_SPEED = 3.0
    config.BRAIN_CONN_FILE = tvb_conn_filepath
    config.MAJOR_STRUCTS_LABELS_FILE = major_structs_labels_filepath
    config.VOXEL_COUNT_FILE = voxel_count_filepath
    config.INDS_FILE = inds_filepath
    config.CEREB_SCAFFOLD_PATH = cereb_scaffold_path
    # Fix Cortex <-> Spec Thal connections according to Griffiths et al model:
    # config.THAL_CRTX_FIX = defargs.get("THAL_CRTX_FIX", "wd")
    config.CONN_NORM_PERCENTILE = 99
    # Task connectivity:
    config.TASK_LATERALITY = -1   # -1: contralatterally, 0: bilaterally, 1: ipsilaterall
    # FIC:
    config.FIC = args['FIC']  # 1.11 for FIC_SPLIT = 0.31
    config.FIC_PARAMS = ["I_e", "w_ie"]
    config.FIC_SPLIT = args.get('FIC_SPLIT', 0.0)  # 0.31 with FIC = 1.11
    # Pathway gains:
    config.PATHWAY_GAIN = args["PATHWAY_GAIN"]
    gain_factor = 1.0
    if config.PATHWAY_GAIN <= 2.0:
        gain_factor = config.PATHWAY_GAIN
    for pg, pgdef in PATHWAY_GAINS.items():
        setattr(config, pg, gain_factor * args[pg])
    # TVB Monitors:
    config.TIME_SERIES_MONITORS = args.get("TIME_SERIES_MONITORS", True)
    config.RAW_PERIOD = 1.0
    config.BOLD_PERIOD = 1024.0  # 1024.0 or None, If None, BOLD will not be computed
    config.AFFERENT_MONITOR = True
    # TVB model parameters
    config.model_params = OrderedDict()
    config.model_params['G'] = args.get('G', 6.0)
    config.model_params['I_s'] = args.get('I_s', 0.1)
    config.model_params['I_e'] = args.get('I_e', -0.35)
    config.model_params['w_ie'] = args.get('w_ie', -3.0)
    config.model_params['w_rs'] = args.get('w_rs', -2.0)
    config.model_params['tau_e'] = args.get('tau_e', 10.0/0.9)
    config.model_params['tau_i'] = args.get('tau_i', 10.0/0.9)
    if WHISKERS > 0:
        config.model_params['tau_w'] = args.get('tau_w', 10.0)
        config.model_params['I_w'] = args.get('I_w', 0.0)
        config.model_params['G_w'] = G_w
    # elif STIMULUS > 0:
    #     config.model_params['STIMULUS'] = STIMULUS  # 0.25
    # config.STIMULUS_RATE = 8.0  # Hz
    # config.STIMULUS_BASELINE = defargs.get('STIMULUS_BASELINE', 1.0)  # 1.0 or 0.0

    # NEST model parameters:
    config.NEST_STIMULUS_RATE = 6.0
    config.NEST_STIMULUS = 50.0  # Hz
    # One of: (a) True. (b) "Input TVB to parrot_medulla". (c) "Input Sinusoidal to mossy_fibers"
    config.NEST_PERIPHERY = "Input to parrot_medulla"  # stimulus towards parrot_medulla in NEST network
    config.NEST_BACKGROUND_FREQ = 4.0  # 4.0 Hz, for NEST only simulations
    config.NEST_MULTIMETER = False

    # TVB - NEST interface parameters:
    config.MAX_RATES = args.get("MAX_RATES",  # Hz
                                {"parrot_medulla": 10.0,
                                 "parrot_ponssens": 10.0, "io_cell": 10.0,
                                 "mossy_fibers": config.NEST_STIMULUS,
                                 "granule_cell": 30.0, "dcn_cell_glut_large": 25.0}
                                )
    config.PONSSENS_INTERFACE = False  # Not part of the latest task pathway -> not in NEST network
    config.ANSILOB_INTERFACE = True    # Existing in NEST only model, although not part of the task pathway
    config.IO_INTERFACE = False        # Existing in NEST only model, although not part of the task pathway
    config.w_TVB_to_NEST_rest = args.get("w_TVB_to_NEST_rest", 35.0)  # Old tuned value = 0.04
    config.w_TVB_to_NEST = {"parrot_medulla": args.get("w_TVB_to_NEST", 35.0)}
    if config.PONSSENS_INTERFACE:
        config.w_TVB_to_NEST["parrot_ponssens"] = config.w_TVB_to_NEST_rest
    if config.IO_INTERFACE:
        config.w_TVB_to_NEST["io_cell"] = config.w_TVB_to_NEST_rest
    if config.ANSILOB_INTERFACE:
        config.w_TVB_to_NEST["mossy_fibers"] = config.w_TVB_to_NEST_rest
    config.INVERSE_SIGMOIDAL_NEST_TO_TVB = True

    config.MAX_GAIN = 99.0
    # Fitting
    config.PRIORS_DEF = \
        {"I_s": {"prior_dist": "normal", "min": -0.25, "max": 0.45, "loc": 0.1, "sc": 0.1},
         # "I_e": {"prior_dist": "normal", "min": -0.7, "max": 0.0, "loc": -0.35, "sc": 0.1},
         "FIC": {"prior_dist": "uniform", "min": 0.0, "max": 2.0, "loc": 1.0, "sc": 0.25},
         "FIC_SPLIT": {"prior_dist": "uniform", "min": 0.0, "max": 0.6, "loc": 0.3, "sc": 0.05},
         # "STIMULUS": {"prior_dist": "normal", "min": 0.0, "max": 0.5, "loc": 0.25, "sc": 0.05},
         # "STIMULUS_BASELINE": {"prior_dist": "normal", "min": 0.0, "max": 1.5, "loc": 1.0, "sc": 0.1},
         "I_w": {"prior_dist": "uniform", "min": -0.7, "max": 0.0, "loc": -0.35, "sc": 0.1},
         "G_w": {"prior_dist": "uniform", "min": 1.0, "max": 9.0, "loc": 5.0, "sc": 1.0},
         "PATHWAY_GAIN": {"prior_dist": "uniform", "min": 30.0, "max": 90.0, "loc": 60.0, "sc": 5.0},
         "M1FACIAL_GAIN": {"prior_dist": "uniform", "min": 0.0, "max": 1.0, "loc": 0.5, "sc": 0.15},
         "CNM1S1_GAIN": {"prior_dist": "uniform", "min": 0.0, "max": 1.0, "loc": 0.5, "sc": 0.15},
         # "WHISKERS_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 50.0, "sc": 10.0},
         # "TRIG_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 50.0, "sc": 10.0},
         # "MEDULLA_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 50.0, "sc": 10.0},
         # "CEREB_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 50.0, "sc": 10.0},
         # "CNM1_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 50.0, "sc": 10.0},
         # "CNS1_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 30.0, "sc": 6.0},
         # "TRIGS1_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 30.0, "loc": 10.0, "sc": 2.0},
         # "MEDULLAS1_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 30.0, "loc": 10.0, "sc": 2.0},
         # "M1S1_GAIN": {"prior_dist": "uniform", "min": 1.0, "max": 90.0, "loc": 10.0, "sc": 2.0}
         }
    config.PRIORS_DIST = args.get('PRIORS_DIST', dict())
    for pname, pd in config.PRIORS_DIST.items():
        config.PRIORS_DEF[pname]["prior_dist"] = pd
    config.SBI_NUM_WORKERS = 1
    config.SBI_ALGORITHM = "SNPE"  # 'SNLE'  # 'SNPE'
    config.SBI_TRAIN_KWARGS = {}
    config.SBI_BUILD_KWARGS = {}  # {"sample_with": "mcmc"}
    config.SBI_SAMPLE_KWARGS = {}  # {"method": "nuts", "num_chains": 4, "warmup_steps": 50}
    config.TARGET_POPA_PATH = popa_freqs_path
    config.PSD_TARGET_PATH = os.path.join(config.out.FOLDER_RES, "PSD_target.npy")
    config.PSD_DATA_PATH = os.path.join(config.out.FOLDER_RES, "PSD_data.npy")
    # config.TASK_TRANSFER_METRICS_PATH = os.path.join(config.out.FOLDER_RES, "task_metrics_data.pkl")
    config.TARGET_FREQS = np.arange(5.0, 48.0, 1.0)  # TODO: Decide about 4 or 5 Hz min frequency!!!
    config.FREQS = np.arange(5.0, 101.0, 1.0)
    config.THETA = np.arange(6.0, 13.0, 1.0)
    config.BETA = np.arange(13.0, 25.0, 1.0)
    config.GAMMA = np.arange(25.0, 61.0, 1.0)
    config.COHERENCE_FISHER_Z_TRANSFORM = True
    config.FREQ_BAND_FITNESS_WEIGHTS = [1.0, 1.0, 1.0]
    config.N_FIT_RUNS = 10
    config.N_SIMULATIONS = 2000
    config.N_SIMS_PER_PARAM = 3
    config.SPLIT_RUN_SAMPLES = 0.8
    # config.N_TRAIN_SAMPLES = 5  #  1000
    # config.TEST_SAMPLES_RATIO = 0.25
    config.N_POSTERIOR_SAMPLES_PER_RUN = 1000
    config.N_PPC_SIMS = 100
    # config.PPT_BATCH_SIM_RES_FILE = "ppt_bsr.npy"  # e.g., ppt_bsr_iG01_iB010.npy
    config.Gs = np.arange(0.0, 11.0)
    config.Gs[config.Gs == 0.0] = 0.1
    config.FILE_FORMAT = "%s_%02d%s"  # "%s_%03d%s"
    config.SIM_RES_FILE = "res.pkl"  # e.g., res_01.pkl
    config.TRAIN_PARAMS_SAMPLES_FILE = "train_params.pt"
    config.TRAIN_SIMS_FOLDER = "train_sims"
    config.FIT_FOLDER = "fit"
    config.INFERENCE_FILE = "inference.pkl"
    config.POSTERIOR_FILE = "posterior.pkl"
    config.PROPOSAL_FILE = "proposal.pkl"
    config.SAMPLES_FILE = "samples.pkl"
    config.PPC_FOLDER = "PPC_sims"
    config.MEAN_FOLDER = "mean_sims"
    config.MAP_FOLDER = "MAP_sims"
    config.BOLD_FOLDER = "BOLD_sims"
    config.ALL_SAMPLES_LABEL = "allsamples"
    config.ALL_RUNS_LABEL = "allruns"
    config.FIT_DIAGNOSTICS = ["map", "mean", "std", "diff", "accuracy", "zscore_prior", "zscore", "shrinkage"]
    if TASK:
        # if config.PATHWAY_GAIN > 2.0:
        DEF_PRIORS_PARAMS_NAMES = ["I_w", "G_w", "PATHWAY_GAIN", "M1FACIAL_GAIN", "CNM1S1_GAIN"]
        # else:
        #     DEF_PRIORS_PARAMS_NAMES = ["I_w", "G_w",
        #                                "M1FACIAL_GAIN", "WHISKERS_GAIN", "TRIG_GAIN",
        #                                "MEDULLA_GAIN", "CEREB_GAIN",
        #                                "CNM1_GAIN", "CNS1_GAIN",
        #                                "TRIGS1_GAIN", "MEDULLAS1_GAIN",
        #                                "M1S1_GAIN"]
    else:
        DEF_PRIORS_PARAMS_NAMES = ["I_s", "FIC", "FIC_SPLIT"]
    config.PRIORS_PARAMS_NAMES = args.get("PRIORS_PARAMS_NAMES", DEF_PRIORS_PARAMS_NAMES)
    config.prior_dist = []
    # Uniform prior:
    config.prior_min = []
    config.prior_max = []  
    # Normal prior:
    config.prior_loc = []  
    config.prior_sc = []
    for pname in config.PRIORS_PARAMS_NAMES:
        config.prior_dist.append(config.PRIORS_DEF[pname]['prior_dist'])
        config.prior_min.append(config.PRIORS_DEF[pname]['min'])
        config.prior_max.append(config.PRIORS_DEF[pname]['max'])
        config.prior_loc.append(config.PRIORS_DEF[pname]['loc'])
        config.prior_sc.append(config.PRIORS_DEF[pname]['sc'])
    config.n_priors = len(config.PRIORS_PARAMS_NAMES)
    config.SBI_FIT_PLOT_PATH = os.path.join(config.figures.FOLDER_FIGURES, "sbi_fit.%s" % config.figures.FIG_FORMAT)
    config.OPT_RES_MODE = "mean"  # or "map"
    config.MIN_ACCURACY = -np.inf

    if config.VERBOSITY:
        print(config)

    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config.__dict__, file, recurse=1)

    return config, plotter


def assert_config(config=None, return_plotter=False, **config_args):
    if config is None:
        if return_plotter:
            # Create a configuration if one is not given
            return configure(**config_args)
        else:
            return configure(**config_args)[0]
    else:
        if return_plotter:
            if config_args.get('plot_flag', DEFAULT_ARGS.get('plot_flag')):
                return create_plotter(config)
            else:
                return config, None
        else:
            return config


def args_parser(funname, defargs=DEFAULT_ARGS):

    def FICtype(FIC):
        if FIC == 'fit':
            return FIC
        return float(FIC)

    args = deepcopy(defargs)
    arguments = {# TVB model:
                 'I_s': ['is', float, 'Thalamic relay excitatory population baseline current'],
                 'I_e': ['ie', float, 'Cortical excitatory population baseline current'],
                 # 'STIMULUS': ['st', float, 'Whisking stimulus amplitude'],
                 # 'STIMULUS_BASELINE': ['sb', float, 'Whisking stimulus baseline'],
                 'G_w': ['gw', float, "Whiskers' scaling"],
                 'tau_w': ['tw', float, "Whiskers' time constant"],
                 'I_w': ['iw', float, "Whiskers'  baseline"],
                  # TVB network:
                 'G': ['g', float, 'Global connectivity scaling'],
                 'FIC': ['fic', FICtype, 'Indegree FIC weight'],
                 'FIC_SPLIT': ['ficsplt', float, 'FIC splitting parameter'],
                 'PATHWAY_GAIN': ['pg', float, "Pathway gain"],
                  # WORKFLOW:
                 'SIMULATION_LENGTH': ['sl', float, "Simulation length"],
                 "NOISE": ['ns', float, "Noise amplitude"],
                 "NOISE_SEED": ['nsd', int, "Noise seed additive"],
                 'MODE': ['md', str, 'Mode name (e.g., TVB, NEST, COSIM)'],
                 'BASENAME': ['bsnm', str, 'Base folder name'],
                 'output_folder': ['o', str, 'Output folder name'],
                 'verbosity': ['v', int,
                             'Integer flag to print output messages (when > 0) or not (when == 0). Default = 1.0'],
                 'plot_flag': ['pf', bool, 'Boolean flag to plot or not']
                 }

    PATHWAY_GAINS_SHORTS = \
        {"TRIG_GAIN": "trg", "MEDULLA_GAIN": "mdg", "CEREB_GAIN": "cbg",
         "TRIGS1_GAIN": "trs1g", "MEDULLAS1_GAIN": "mds1g",
         "CNS1_GAIN": "cnsg", "CNM1_GAIN": "cnmg", "CNM1S1_GAIN": "cnmsg",
         "M1S1_GAIN": "msg",
         "M1FACIAL_GAIN": "mfg",  # 50.0,
         "FACIALTRIG_GAIN": "ftg",  # 50.0,
         "WHISKERS_GAIN": "wsg"}
    for pg in PATHWAY_GAINS:
        arguments[pg] = [PATHWAY_GAINS_SHORTS[pg], float, pg.replace("_", " ").capitalize()]

    parser = argparse.ArgumentParser(description='%s.py' % funname)
    for arg, vals in arguments.items():
        parser.add_argument('--%s' % arg,
                            '-%s' % vals[0],
                            dest=arg, metavar=arg,
                            type=vals[1],
                            # default=args[arg],
                            required=False,  # nargs=1,
                            help=vals[2])
    return parser


def parse_args(parser, argsnames=list(DEFAULT_ARGS.keys())):
    args = dict()
    parser_args = parser.parse_args()
    for arg in argsnames:
        val = getattr(parser_args, arg)
        if val is not None:
            args[arg] = val
    return args, parser_args, parser


def logprint(msg, logger, verbosity):
    msg = "\n" + msg
    try:
        logger.info(msg)
    except:
        pass
    if verbosity:
        print(msg)

