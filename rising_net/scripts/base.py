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


DEFAULT_ARGS = {# TVB model:
                'I_s': 0.1,  # 0.085,
                "STIMULUS": 0.25,
                "WHISKERS": 0,
                "tau_w": 10.0,
                # TVB network:
                'G': 6.0,
                'FIC': 1.11,  # 2.0,
                'FIC_SPLIT': 0.31,  # 2.0,
                # Pathway gains:
                "PATHWAY_GAIN": 1,
                "TRIG_GAIN": 100.0, "MEDULLA_GAIN": 50.0, "CEREB_GAIN": 50.0,
                "TRIGS1_GAIN": 10.0, "MEDULLAS1_GAIN": 10.0, "CNS1_GAIN": 30.0,
                "CNM1_GAIN": 50.0,
                "M1S1_GAIN": 10.0,
                "M1FACIAL_GAIN": 50.0,   # 50.0,
                "FACIALTRIG_GAIN": 1.0,  # 50.0,
                # TVB <-> NEST Interface:
                "w_TVB_to_NEST": 35.0, "w_TVB_to_NEST_rest": 0.15,
                "MAX_RATES": {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
                              "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0},  # Hz
                # WORKFLOW:
                "TASK": True,
                "MODE": "TVB",  # "NEST", "COSIM", + "_CEREBOFF" to turn off Cerebellum
                'output_folder': "", 'verbose': 1, 'plot_flag': True}


def create_plotter(config):
    from tvb_multiscale.core.plot.plotter import Plotter
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE
    return config, Plotter(config.figures)


def configure(**ARGS):
    from tvb_multiscale.tvb_nest.config import Config

    args = deepcopy(DEFAULT_ARGS)
    args.update(**ARGS)
    MODE = args["MODE"]
    TASK = args["TASK"]
    BASENAME = MODE + "_REST" if not TASK else MODE

    # Flags that affect the result's path:
    # Files:
    BRAIN_CONN_FILE = "Connectivity_wavCntrs_TLwav_SummedSubcortical_Thals.h5"
    MAJOR_STRUCTS_LABELS_FILE = "major_structs_labels_SummedSubcortical_Thals.npy"  # "major_structs_labels_Thals.npy" # "major_structs_labels_SummedSubcortical_Thals.npy"
    VOXEL_COUNT_FILE = "voxel_count_SummedSubcortical_Thals.npy"  # "voxel_count_Thals.npy" # "voxel_count_SummedSubcortical_Thals.npy"
    INDS_FILE = "inds_SummedSubcortical_Thals.npy"  # "inds_Thals.npy" # "inds_SummedSubcortical_Thals.npy

    # Construct configuration
    work_path = os.getcwd()
    data_path = os.path.expanduser("~/packages/tvb-multiscale/rising_net/data")
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
    SEED = args.get("SEED", None)
    if SEED is not None:
        SEED = int(SEED)
        outputs_path = os.path.join(outputs_path, "nsd%d" % SEED)
    else:
        SEED = 0
    # # if STIMULUS:
    # #     outputs_path += "_Stim%g" % STIMULUS
    # # outputs_path += '_Is%g' % I_s
    # outputs_path += '_G%g' % G
    # outputs_path += '_FIC%g' % FIC

    if args['verbose']:
        print("Outputs' path: %s" % outputs_path)

    config = Config(output_base=outputs_path)

    config.VERBOSE = args['verbose']

    if args['plot_flag']:
        config, plotter = create_plotter(config)
    else:
        plotter = None

    # ------.----- Simulation options ----------------

    # Simulation...
    config.MODE = MODE
    config.TASK = TASK
    config.BASENAME = BASENAME
    # Testing: 10: 1025, 11: 2049.0, Fitting: 12: 4097.0, BOLD: 16: 65537
    config.SIMULATION_LENGTH = args.get("SIMULATION_LENGTH", 2 ** 11 + 1.0)
    config.TRANSIENT_RATIO = args.get("TRANSIENT_RATIO", 0.25)
    config.SOURCE_TS_PATH = os.path.join(config.out.FOLDER_RES, "source_ts.pkl")
    config.AFFERENT_TS_PATH = os.path.join(config.out.FOLDER_RES, "afferent_ts.pkl")
    config.BOLD_TS_PATH = os.path.join(config.out.FOLDER_RES, "bold_ts.pkl")

    # Integration
    config.DEFAULT_DT = 0.1
    config.DEFAULT_NSIG = args.get("NOISE", 1e-6)  # NOISE strength
    config.DEFAULT_TVB_NOISE_SEED = args.get("DEFAULT_TVB_NOISE_SEED", 42) + SEED
    config.NEST_MASTER_SEED = args.get("NEST_MASTER_SEED", 143202461) + SEED
    config.DEFAULT_STOCHASTIC_INTEGRATOR = EulerStochastic
    config.DEFAULT_INTEGRATOR = config.DEFAULT_STOCHASTIC_INTEGRATOR

    # Connectivity
    config.WHISKERS = args.get("WHISKERS", 0)
    config.CONN_SPEED = 3.0
    config.BRAIN_CONN_FILE = tvb_conn_filepath
    config.MAJOR_STRUCTS_LABELS_FILE = major_structs_labels_filepath
    config.VOXEL_COUNT_FILE = voxel_count_filepath
    config.INDS_FILE = inds_filepath
    config.CEREB_SCAFFOLD_PATH = cereb_scaffold_path
    # Fix Cortex <-> Spec Thal connections according to Griffiths et al model:
    # config.THAL_CRTX_FIX = args.get("THAL_CRTX_FIX", "wd")
    config.CONN_NORM_PERCENTILE = 99
    # Task connectivity:
    config.TASK_LATERALITY = -1   # -1: contralatterally, 0: bilaterally, 1: ipsilaterall
    # FIC:
    config.FIC = args['FIC']  # 1.11 for FIC_SPLIT = 0.31
    config.FIC_PARAMS = ["I_e", "w_ie"]
    config.FIC_SPLIT = args.get('FIC_SPLIT', 0.0)  # 0.31 with FIC = 1.11
    # Pathway gains:
    config.PATHWAY_GAIN = args["PATHWAY_GAIN"]
    config.TRIG_GAIN = args["PATHWAY_GAIN"] * args["TRIG_GAIN"]
    config.MEDULLA_GAIN = args["PATHWAY_GAIN"] * args["MEDULLA_GAIN"]
    config.CEREB_GAIN = args["PATHWAY_GAIN"] * args["CEREB_GAIN"]
    config.TRIGS1_GAIN = args["PATHWAY_GAIN"] * args["TRIGS1_GAIN"]
    config.MEDULLAS1_GAIN = args["PATHWAY_GAIN"] * args["MEDULLAS1_GAIN"]
    config.CNS1_GAIN = args["PATHWAY_GAIN"] * args["CNS1_GAIN"]
    config.CNM1_GAIN = args["PATHWAY_GAIN"] * args["CNM1_GAIN"]
    config.M1S1_GAIN = args["PATHWAY_GAIN"] * args["M1S1_GAIN"]
    config.M1FACIAL_GAIN = args["PATHWAY_GAIN"] * args["M1FACIAL_GAIN"]
    config.FACIALTRIG_GAIN = args["PATHWAY_GAIN"] * args["FACIALTRIG_GAIN"]
    # TVB Monitors:
    config.RAW_PERIOD = 1.0
    config.BOLD_PERIOD = 1024.0  # 1024.0 or None, If None, BOLD will not be computed

    # TVB model parameters
    config.model_params = OrderedDict()
    config.model_params['G'] = args['G']
    config.model_params['I_s'] = args['I_s']
    config.model_params['I_e'] = args.get('I_e', -0.35)
    config.model_params['w_ie'] = args.get('w_ie', -3.0)
    config.model_params['w_rs'] = args.get('w_rs', -2.0)
    config.model_params['tau_w'] = args.get('tau_w', 10.0)
    config.model_params['STIMULUS'] = args.get('STIMULUS', 0.25)  # 0.25
    config.STIMULUS_RATE = 8.0  # Hz
    config.STIMULUS_BASELINE = args.get('STIMULUS_BASELINE', 1.0)  # 1.0 or 0.0

    # NEST model parameters:
    config.NEST_STIMULUS = 15.0  # Hz
    config.NEST_PERIPHERY = True  # "Input TVB to parrot_medulla", "Input Sinusoidal to mossy_fibers"
    config.NEST_PERIPHERY_MANY_NEURONS = False  # True takes for ever in cosimulation
    config.NEST_BACKGROUND_FREQ = 0.0  # 4.0 Hz, for NEST only simulations

    # TVB - NEST interface parameters:
    config.MAX_RATES = args.get("MAX_RATES",  # Hz
                                {"parrot_medulla": 30.0, "parrot_ponssens": 30.0, "io_cell": 30.0,
                                 "mossy_fibers": 3000.0, "granule_cell": 400.0, "dcn_cell_glut_large": 600.0}
                                )
    config.PONSSENS_INTERFACE = True  # Not existing in NEST only model, but part of the task pathway
    config.ANSILOB_INTERFACE = True   # Not existing in NEST only model, but part of the task pathway
    config.IO_INTERFACE = False       # Not existing in NEST only model
    config.w_TVB_to_NEST_rest = args["w_TVB_to_NEST_rest"]  # Old tuned value = 0.04
    config.w_TVB_to_NEST = {"parrot_medulla": args["w_TVB_to_NEST"]}
    if config.PONSSENS_INTERFACE:
        config.w_TVB_to_NEST["parrot_ponssens"] = config.w_TVB_to_NEST_rest
    if config.IO_INTERFACE:
        config.w_TVB_to_NEST["io_cell"] = config.w_TVB_to_NEST_rest
    if config.ANSILOB_INTERFACE:
        config.w_TVB_to_NEST["mossy_fibers"] = config.w_TVB_to_NEST_rest
    config.INVERSE_SIGMOIDAL_NEST_TO_TVB = True

    # Fitting
    config.PRIORS_DIST = args.get('PRIORS_DIST', "normal")  # "normal" or "uniform"
    config.PRIORS_DEF = \
        {# "STIMULUS": {"min": 0.0, "max": 0.5, "loc": 0.25, "sc": 0.05},
         # "STIMULUS_BASELINE": {"min": 0.0, "max": 1.5, "loc": 1.0, "sc": 0.1},
         "I_s": {"min": -0.1, "max": 0.2, "loc": 0.1, "sc": 0.025},
         "FIC": {"min": 0.0, "max": 3.0, "loc": 2.0, "sc": 0.25},
         # "FIC_SPLIT": {"min": 0.0, "max": 0.5, "loc": 0.3, "sc": 0.05}
         }
    config.SBI_NUM_WORKERS = 1
    config.SBI_METHOD = 'SNPE'
    config.TARGET_PSD_POPA_PATH = popa_freqs_path
    config.PSD_TARGET_PATH = os.path.join(config.out.FOLDER_RES, "PSD_target.npy")
    config.PSD_DATA_PATH = os.path.join(config.out.FOLDER_RES, "PSD_data.npy")
    config.TARGET_FREQS = np.arange(5.0, 48.0, 1.0)  # TODO: Decide about 4 or 5 Hz min frequency!!!
    config.POSTERIOR_PATH = os.path.join(config.out.FOLDER_RES, "posterior.pkl")
    config.POSTERIOR_SAMPLES_PATH = os.path.join(config.out.FOLDER_RES, "samples_fit.pkl")
    config.N_FIT_RUNS = 10  # 3 - 10
    config.N_SIMULATIONS = 1200
    config.N_SIM_BATCHES = 30
    config.SPLIT_RUN_SAMPLES = 1
    config.N_TRAIN_SAMPLES = 1200
    config.TEST_SAMPLES_RATIO = 0.25
    config.N_SAMPLES_PER_RUN = 1000
    config.BATCH_FILE_FORMAT = "%s_%03d%s"
    config.BATCH_FILE_FORMAT_G = "%s_iG%02d_%03d%s"
    config.BATCH_PRIORS_SAMPLES_FILE = "bps.pt"  # e.g., bps_iG01_iB010.pt
    config.BATCH_SIM_RES_FILE = "bsr.npy"  # e.g., bsr_iG01_iB010.npy
    config.N_PPT_SIM_BATCHES = 30
    config.N_PPT_SIMS_PER_BATCH = 40
    config.PPT_BATCH_SIM_RES_FILE = "ppt_bsr.npy"  # e.g., ppt_bsr_iG01_iB010.npy
    config.Gs = np.arange(1.0, 11.0)
    config.PRIORS_PARAMS_NAMES = args.get("PRIORS_PARAMS_NAMES",
                                          ['I_s',  "FIC"])  # 'STIMULUS', 'STIMULUS_BASELINE', ..., "FIC_SPLIT"
    # Uniform priors:
    config.prior_min = []
    config.prior_max = []  
    # Normal priors:
    config.prior_loc = []  
    config.prior_sc = []
    for pname in config.PRIORS_PARAMS_NAMES:
        config.prior_min.append(config.PRIORS_DEF[pname]['min'])
        config.prior_max.append(config.PRIORS_DEF[pname]['max'])
        config.prior_loc.append(config.PRIORS_DEF[pname]['loc'])
        config.prior_sc.append(config.PRIORS_DEF[pname]['sc'])
    config.n_priors = len(config.PRIORS_PARAMS_NAMES)
    config.SBI_FIT_PLOT_PATH = os.path.join(config.figures.FOLDER_FIGURES, "sbi_fit.%s" % config.figures.FIG_FORMAT)
    config.OPT_RES_MODE = "map"  # or "mean"
    config.MIN_ACCURACY = -np.inf

    if config.VERBOSE:
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


def args_parser(funname, args=DEFAULT_ARGS):

    def FICtype(FIC):
        if FIC == 'fit':
            return FIC
        return float(FIC)

    arguments = {'G': ['g', float, 'Global connectivity scaling'],
                 'STIMULUS': ['st', float, 'Whisking stimulus amplitude'],
                 'I_s': ['is', float, 'Thalamic relay excitatory population baseline current'],
                 'FIC': ['fic', FICtype, 'Indegree FIC weight'],
                 'output_folder': ['o', str, 'Output folder name'],
                 'verbose': ['v', int,
                             'Integer flag to print output messages (when > 0) or not (when == 0). Default = 1.0'],
                 'plot_flag': ['pf', bool, 'Boolean flag to plot or not']
                 }
    parser = argparse.ArgumentParser(description='%s.py' % funname)
    for arg, vals in arguments.items():
        parser.add_argument('--%s' % arg,
                            '-%s' % vals[0],
                            dest=arg, metavar=arg,
                            type=vals[1],
                            default=args[arg], required=False,  # nargs=1,
                            help=vals[2])
    return parser


def parse_args(parser, def_args=DEFAULT_ARGS):
    args = deepcopy(def_args)
    parser_args = parser.parse_args()
    for arg, val in def_args.items():
        args[arg] = getattr(parser_args, arg)
    return args, parser_args, parser
