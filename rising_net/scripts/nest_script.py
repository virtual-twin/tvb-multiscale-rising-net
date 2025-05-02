# -*- coding: utf-8 -*-
import shutil

import h5py
from rising_net.scripts.base import *
from rising_net.scripts.tvb_script import *


def nest_parameter_settings():
    ########################################## PARAMETERS SETTING ######################################################
    # Reference values: Geminiani et al., Plos Comp Bio, 2024 - https://github.com/AliceGem/mesoscale_simulations_cebc/tree/main/configuration (Z+ configuration for whisking)
    # Synapse parameters: in E-GLIF, 3 synaptic receptors are present: the first is always associated to exc, the second to inh, the third to remaining synapse type
    Erev_exc = 0.0  # [mV]	#[Cavallari et al, 2014]
    Erev_inh = -80.0  # [mV]
    tau_exc = {'golgi': 5.0, 'granule': 1.9, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0,
               'dcnp': 3.64,
               'io': 1.0}  # tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
    tau_inh = {'golgi': 5.0, 'granule': 4.5, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7,
               'dcnp': 1.14, 'io': 60.0}
    tau_exc_cfpc = 0.4
    tau_exc_pfgoc = 1.25
    tau_exc_cfmli = 1.2
    # Single neuron parameters:
    neuron_param = {
        'golgi_cell': {'t_ref': 2.0, 'C_m': 145.0, 'tau_m': 44.0, 'V_th': -55.0, 'V_reset': -75.0, 'Vinit': -62.0,
                       'E_L': -62.0, 'V_min': -150.0,
                       'lambda_0': 1.0, 'tau_V': 0.4, 'I_e': 16.214, 'kadap': 0.217, 'k1': 0.031, 'k2': 0.023,
                       'A1': 259.988, 'A2': 178.01,
                       'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['golgi'],
                       'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
        'granule_cell': {'t_ref': 1.5, 'C_m': 7.0, 'tau_m': 24.15, 'V_th': -41.0, 'V_reset': -70.0, 'Vinit': -62.0,
                         'E_L': -62.0, 'V_min': -150.0,
                         'lambda_0': 1.0, 'tau_V': 0.3, 'I_e': -0.888, 'kadap': 0.022, 'k1': 0.311, 'k2': 0.041,
                         'A1': 0.01, 'A2': -0.94,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['granule'],
                         'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
        'purkinje_cell': {'t_ref': 0.5, 'C_m': 334.0, 'tau_m': 47.0, 'V_th': -43.0, 'V_reset': -69.0, 'Vinit': -59.0,
                          'E_L': -59.0,
                          'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 176.26, 'kadap': 1.492, 'k1': 0.1950, 'k2': 0.041,
                          'A1': 157.622, 'A2': 172.622,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['purkinje'],
                          'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
        'basket_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                        'E_L': -68.0,
                        'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                        'A1': 5.953, 'A2': 5.863,
                        'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                        'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
        'stellate_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                          'E_L': -68.0,
                          'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                          'A1': 5.953, 'A2': 5.863,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                          'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
        'dcn_cell_glut_large': {'t_ref': 1.5, 'C_m': 142.0, 'tau_m': 33.0, 'V_th': -36.0, 'V_reset': -55.0,
                                'Vinit': -45.0, 'E_L': -45.0,
                                'lambda_0': 3.5, 'tau_V': 3.0, 'I_e': 75.385, 'kadap': 0.408, 'k1': 0.697, 'k2': 0.047,
                                'A1': 13.857, 'A2': 3.477,
                                'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcn'],
                                'tau_syn2': tau_inh['dcn']},
        'dcn_cell_GABA': {'t_ref': 3.0, 'C_m': 56.0, 'tau_m': 56.0, 'V_th': -39.0, 'V_reset': -55.0, 'Vinit': -40.0,
                          'E_L': -40.0,
                          'lambda_0': 0.9, 'tau_V': 1.0, 'I_e': 2.384, 'kadap': 0.079, 'k1': 0.041, 'k2': 0.044,
                          'A1': 176.358, 'A2': 176.358,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcnp'],
                          'tau_syn2': tau_inh['dcnp']},
        'io_cell': {'t_ref': 1.0, 'C_m': 189.0, 'tau_m': 11.0, 'V_th': 100.0, 'V_reset': -45.0, 'Vinit': -45.0,
                    'E_L': -45.0,
                    'lambda_0': 1.2, 'tau_V': 0.8, 'I_e': -18.01, 'kadap': 1.928, 'k1': 0.191, 'k2': 0.091,
                    'A1': 1810.923, 'A2': 1358.197,
                    'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['io'],
                    'tau_syn2': tau_inh['io']}}

    # Connection weights
    conn_weights = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 0.822, 'ascending_axon_to_purkinje': 0.882,
                    'basket_to_purkinje': 0.436,
                    'basket_to_basket': 0.006,
                    'glomerulus_to_golgi': 0.240, 'glomerulus_to_granule': 0.232,
                    'golgi_to_granule': 0.148,
                    'golgi_to_golgi': 0.00696,
                    'parallel_fiber_to_basket': 0.1, 'parallel_fiber_to_golgi': 0.054,
                    'parallel_fiber_to_purkinje': 0.136,
                    'parallel_fiber_to_stellate': 0.178,
                    'stellate_to_purkinje': 1.642,
                    'stellate_to_stellate': 0.005,
                    'purkinje_to_dcn_glut_large': 0.297,
                    'mossy_to_dcn_glut_large': 0.554,
                    'purkinje_to_dcn_GABA': 0.072,
                    'io_to_purkinje': 300.0, 'io_to_basket': 3.0, 'io_to_stellate': 11.0, 'io_to_dcn_glut_large': 1.5,
                    'io_to_dcn_GABA': 0.3, 'dcn_GABA_to_io': 0.004}

    # Connection delays
    conn_delays = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 2.0, 'ascending_axon_to_purkinje': 2.0,
                   'basket_to_purkinje': 4.0, 'basket_to_basket': 4.0, \
                   'glomerulus_to_golgi': 1.0, 'glomerulus_to_granule': 1.0, 'golgi_to_granule': 2.0,
                   'golgi_to_golgi': 4.0, \
                   'parallel_fiber_to_basket': 5.0, 'parallel_fiber_to_golgi': 5.0, 'parallel_fiber_to_purkinje': 5.0,
                   'parallel_fiber_to_stellate': 5.0, 'stellate_to_purkinje': 5.0, 'stellate_to_stellate': 4.0, \
                   'purkinje_to_dcn_glut_large': 4.0, 'mossy_to_dcn_glut_large': 4.0, 'purkinje_to_dcn_GABA': 4.0, \
                   'io_to_purkinje': 4.0, 'io_to_basket': 80.0, 'io_to_stellate': 80.0, 'io_to_dcn_glut_large': 4.0,
                   'io_to_dcn_GABA': 5.0, 'dcn_GABA_to_io': 25.0}

    # Connection receptors
    conn_receptors = {'ascending_axon_to_golgi': 3, 'ascending_axon_to_purkinje': 1, 'basket_to_purkinje': 2,
                      'glomerulus_to_golgi': 1, 'glomerulus_to_granule': 1, 'golgi_to_granule': 2, 'golgi_to_golgi': 2,
                      'parallel_fiber_to_basket': 1, 'parallel_fiber_to_golgi': 3, 'parallel_fiber_to_purkinje': 1,
                      'parallel_fiber_to_stellate': 1, 'stellate_to_purkinje': 2, 'stellate_to_stellate': 2,
                      'basket_to_basket': 2, 'purkinje_to_dcn_glut_large': 2, 'mossy_to_dcn_glut_large': 1,
                      'purkinje_to_dcn_GABA': 2, \
                      'io_to_purkinje': 3, 'io_to_basket': 3, 'io_to_stellate': 3, 'io_to_dcn_glut_large': 1,
                      'io_to_dcn_GABA': 1, 'dcn_GABA_to_io': 2}

    # Connection pre and post-synaptic neurons
    conn_pre_post = {'mossy_to_glomerulus': {'pre': 'mossy_fibers', 'post': 'glomerulus'}, \
                     'ascending_axon_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                     'ascending_axon_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                     'basket_to_purkinje': {'pre': 'basket_cell', 'post': 'purkinje_cell'}, \
                     'glomerulus_to_golgi': {'pre': 'glomerulus', 'post': 'golgi_cell'}, \
                     'glomerulus_to_granule': {'pre': 'glomerulus', 'post': 'granule_cell'}, \
                     'golgi_to_granule': {'pre': 'golgi_cell', 'post': 'granule_cell'}, \
                     'golgi_to_golgi': {'pre': 'golgi_cell', 'post': 'golgi_cell'}, \
                     'parallel_fiber_to_basket': {'pre': 'granule_cell', 'post': 'basket_cell'}, \
                     'parallel_fiber_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                     'parallel_fiber_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                     'parallel_fiber_to_stellate': {'pre': 'granule_cell', 'post': 'stellate_cell'}, \
                     'stellate_to_purkinje': {'pre': 'stellate_cell', 'post': 'purkinje_cell'}, \
                     'basket_to_basket': {'pre': 'basket_cell', 'post': 'basket_cell'}, \
                     'stellate_to_stellate': {'pre': 'stellate_cell', 'post': 'stellate_cell'}, \
                     'mossy_to_dcn_glut_large': {'pre': 'mossy_fibers', 'post': 'dcn_cell_glut_large'}, \
                     'purkinje_to_dcn_glut_large': {'pre': 'purkinje_cell', 'post': 'dcn_cell_glut_large'}, \
                     'purkinje_to_dcn_GABA': {'pre': 'purkinje_cell', 'post': 'dcn_cell_GABA'}, \
                     'io_to_purkinje': {'pre': 'io_cell', 'post': 'purkinje_cell'}, \
                     'io_to_basket': {'pre': 'io_cell', 'post': 'basket_cell'}, \
                     'io_to_stellate': {'pre': 'io_cell', 'post': 'stellate_cell'}, \
                     'io_to_dcn_glut_large': {'pre': 'io_cell', 'post': 'dcn_cell_glut_large'},
                     'io_to_dcn_GABA': {'pre': 'io_cell', 'post': 'dcn_cell_GABA'},
                     'dcn_GABA_to_io': {'pre': 'dcn_cell_GABA', 'post': 'io_cell'}}

    neuron_types_to_region = {'golgi_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'granule_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'purkinje_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'basket_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'stellate_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'dcn_cell_glut_large': ['Right Cerebellar Nuclei', 'Left Cerebellar Nuclei'],
                              'dcn_cell_GABA': ['Right Cerebellar Nuclei', 'Left Cerebellar Nuclei'],
                              'io_cell': ['Right Inferior olivary complex', 'Left Inferior olivary complex'],
                              'glomerulus': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'mossy_fibers': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              'parrot_medulla': ['Right Principal sensory nucleus of the trigeminal',
                                                 'Left Principal sensory nucleus of the trigeminal'],
                              'parrot_ponssens': ['Right Pons Sensory', 'Left Pons Sensory']
                              }

    return dict(neuron_param), dict(conn_weights), dict(conn_delays), \
           dict(conn_receptors), dict(conn_pre_post), dict(neuron_types_to_region)


def random_init_vm(neural_pop_ids, neu_param):
    """
    Randomly initializes the membrane potential (Vm) of a population of neurons.
    This function sets the membrane potential (Vm) of every neuron in the 
    provided neural population to a random value between EL-Vreset and EL+half, 
    where 'half' is half the range between EL and Vth.
    Parameters:
    neural_pop_ids (list): List of neuron IDs in the population.
    neu_param (dict): Dictionary containing neuron parameters, including:
        - 'E_L': Resting membrane potential.
        - 'V_reset': Reset potential after a spike.
        - 'V_th': Spike threshold potential.
    Returns:
    None
    """
    # Function for random initialization of Vm
    # (between EL-Vreset and EL+half; being half the half of the range between EL and Vth)
    import nest
    import random

    for x in range(1,len(neural_pop_ids),2):
        nest.SetStatus(neural_pop_ids[x-1:x],
                       {'Vinit':neu_param['E_L'] + random.randint(neu_param['V_reset'] - neu_param['E_L'],
                        int((neu_param['V_th'] - neu_param['E_L'])/2))})


def split_mossy_fibers(start_id_scaffold, n_mfs_groups=1, f=None, config=None):
    """
    Splits mossy fibers into central and border groups based on their positions and specified groups.
    Parameters:
    start_id_scaffold (dict): Dictionary containing the starting scaffold IDs for different cell types.
    n_mfs_groups (int, optional): Number of central mossy fiber groups to split into along the parasagittal direction (z).
                                Default is 1.
    f (h5py.File, optional): HDF5 file object containing the scaffold data. If None,
                             the file is opened using the provided config.
    config (object, optional): Configuration object containing the path to the scaffold file. Used if `f` is None.
    Returns:
    tuple: A tuple containing:
       - border_mfs_id_nest (numpy.ndarray): Array of border mossy fiber IDs in NEST.
       - central_mfs_id (list of lists): List of lists containing central mossy fiber IDs in NEST format for each group.
    """
    if f is None:
        config = assert_config(config, return_plotter=False)
        target_path = os.path.join(config.out.FOLDER_RES, "balanced_DCN_IO.hdf5")
        shutil.copyfile(config.CEREB_SCAFFOLD_PATH, target_path)
        f = h5py.File(target_path, 'r+')
    # We do all this to find the indices of the target mossy fibers!:
    # Localized CS to avoid border effects
    r_x, r_z = 100, 50
    gloms_pos = np.array(f['cells/placement/glomerulus/positions'])
    x_c, z_c = 150., 100.

    # Find glomeruli falling into the selected volume
    central_gloms_bool = np.add(((gloms_pos[:, [0]] - x_c) ** 2) / r_x ** 2,
                                ((gloms_pos[:, [2]] - z_c) ** 2) / r_z ** 2).__lt__(1)  # ellipse equation
    if n_mfs_groups > 1:
        central_gloms_id_scaffold = []  # We reset the central gloms list
        # If we want multiple mfs groups, we split them based on their z coordinate,
        # to follow the parasagittal direction (Apps et al., 2018; Ji and Hawkes, 1994)
        z_split = [z_c * 2 / n_mfs_groups * i for i in range(0, n_mfs_groups)]
        for z in z_split[1:]:
            central_gloms_id_scaffold.append(np.array(
                np.where(central_gloms_bool & (gloms_pos[:, 2] >= z - 1) & (gloms_pos[:, 2] < z))[0] +
                start_id_scaffold['glomerulus']))
    else:
        central_gloms_id_scaffold = [np.array(np.where(central_gloms_bool)[0] + start_id_scaffold['glomerulus'])]
    border_gloms_id_scaffold = [np.array(np.where(np.logical_not(central_gloms_bool))[0]
                                         + start_id_scaffold['glomerulus'])]
    # Select the corresponding original MFs...
    conn_mf_glom = np.array(f['cells/connections/mossy_to_glomerulus'])
    # Central MFs:
    central_mfs_id_scaffold = [conn_mf_glom[np.isin(conn_mf_glom[:, 1], ids_gloms), 0]
                                for ids_gloms in central_gloms_id_scaffold]
    flattened_central_mfs_id_scaffold = np.concatenate(central_mfs_id_scaffold).flatten()
    print("flattened_central_mfs_id_scaffold.size: %d" % flattened_central_mfs_id_scaffold.size)
    # Remove any duplicates for central mfs:
    for il, lst in enumerate(central_mfs_id_scaffold):
        central_mfs_id_scaffold[il] = np.sort(np.unique(lst))
    # Border MFs ...and remove duplicates for border mfs:
    border_mfs_id_scaffold = np.sort(
                                np.unique(
                                    conn_mf_glom[np.isin(conn_mf_glom[:, 1], border_gloms_id_scaffold), 0]))
    print("border_mfs_id_scaffold.size including some central ones: %d" % border_mfs_id_scaffold.size)
    # Still there are a few mossy fibers that project to both central and border glomeruli:
    bool_inds = ~np.isin(border_mfs_id_scaffold, flattened_central_mfs_id_scaffold)
    print("bool_inds.sum(): %d" % bool_inds.sum())
    border_mfs_id_scaffold = border_mfs_id_scaffold[bool_inds]
    print("Only border_mfs_id_scaffold.size: %d" % border_mfs_id_scaffold.size)
    if config.VERBOSITY > 1:
        print("\nBorder mossy fibers scaffold ids size: %d" % border_mfs_id_scaffold.size)
        for ii in range(len(central_mfs_id_scaffold)):
            print("Central mossy fibers subpopulation %d scaffold ids size: %d"
                  % (ii+1, central_mfs_id_scaffold[ii].size))
        print("\n")
    return border_mfs_id_scaffold, central_mfs_id_scaffold


def get_mossy_targets(region_mf, neuron_models, start_id_scaffold, target_mfs_id_scaffold):
    # translate to NEST ids
    target_mfs_id_nest = target_mfs_id_scaffold - start_id_scaffold['mossy_fibers'] + \
                                neuron_models['mossy_fibers'][region_mf][0]
    target_mfs_id_nest = target_mfs_id_nest.astype(int)
    # Obtain an ordered list of non-duplicates
    return sorted(list(set(target_mfs_id_nest)))  # Medulla or PONS Sensory


def build_NEST_network(config=None):

    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
    from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
    from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
    from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
    from tvb_multiscale.core.spiking_models.devices import DeviceSet
    from tvb_multiscale.tvb_nest.nest_models.devices import NESTSpikeRecorder  # , NESTMultimeter
    from tvb_multiscale.tvb_nest.nest_models.devices import \
        NESTPoissonGenerator, NESTInhomogeneousPoissonGenerator, NESTSinusoidalPoissonGenerator
    from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel

    config = assert_config(config, return_plotter=False)

    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = load_pickled_dict(sim_serial_filepath)

    neuron_param, conn_weights, conn_delays, conn_receptors, conn_pre_post, neuron_types_to_region = \
        nest_parameter_settings()


    # Load NEST and use defaults to configure its kernel:
    nest = configure_nest_kernel(load_nest(config=config), config)
    nest.rng_seed = config.NEST_MASTER_SEED

    if 'eglif_cond_alpha_multisyn' not in nest.Models():
        try:
            if config.VERBOSITY > 1:
                print("Installing cereb module...")
            nest.Install('cerebmodule')
            assert 'eglif_cond_alpha_multisyn' in nest.Models()
        except Exception as e:
            warnings.warn(str(e))
            try:
                if config.VERBOSITY > 1:
                    print("FAILED! Needing to compile it first!")
                import subprocess
                cwd = os.getcwd()
                pwd = __file__
                tvb_multiscale_base_path = pwd.split("rising_net")[0]
                cereb_path = os.path.join(tvb_multiscale_base_path, "tvb_multiscale/tvb_nest/nest/modules/cereb")
                os.chdir(os.path.join(cereb_path, 'build'))
                # This is our shell command, executed by Popen.
                if config.VERBOSITY > 1:
                    print("Compiling cereb module...")
                p = subprocess.Popen("cmake -Dwith-nest=/home/docker/build/nest/bin/nest-config ..; make; make install",
                                     stdout=subprocess.PIPE, shell=True)
                if config.VERBOSITY > 1:
                    print(p.communicate())
                    print("Installing cereb module...")
                nest.Install('cerebmodule')
                os.chdir(cwd)
                assert 'eglif_cond_alpha_multisyn' in nest.Models()
            except Exception as e:
                warnings.warn(str(e))

    ###################### NEST simulation parameters #########################################
    TOT_DURATION = config.SIMULATION_LENGTH * (1 + config.TRANSIENT_RATIO)  # ms
    BACKGROUND_FREQ = config.NEST_BACKGROUND_FREQ
    STIM_FREQ = config.NEST_STIMULUS_RATE
    STIM_AMPLITUDE = config.NEST_STIMULUS
    STIM_RATE = 0.

    high_iomli = 120.0  # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
    min_iomli = 40.0

    ######################## NEST simulation setup ##########################################
    # First configure NEST kernel:
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({"overwrite_files": True, "data_path": "sim_data/", "resolution": 0.05})

    if config.VERBOSITY:
        print("Building NESTNetwork...")

    # Create NEST network...
    nest_network = NESTNetwork(nest)

    # Load file with positions and connections data
    target_path = os.path.join(config.out.FOLDER_RES, "balanced_DCN_IO.hdf5")
    shutil.copyfile(config.CEREB_SCAFFOLD_PATH, target_path)
    f = h5py.File(target_path, 'r+')

    neuron_types = list(f['cells/placement'].keys())
    if config.VERBOSITY > 1:
        print(neuron_types)

    neuron_number = dict()
    start_id_scaffold = dict()

    # Create a dictionary; keys = cell names, values = lists to store neuron models
    neuron_models = {key: [] for key in neuron_types}

    # ...starting from neuronal populations located at specific brain regions...
    nest_network.brain_regions = NESTBrain()

    nest_nodes_inds = []

    input_populations = []
    PARROT_MEDULLA = False
    PARROT_PONSSENS = False
    if config.NEST_PERIPHERY is True:
        PARROT_MEDULLA = True
        PARROT_PONSSENS = True
    else:
        if 'parrot_medulla' in str(config.NEST_PERIPHERY):
            PARROT_MEDULLA = True
        if 'parrot_ponssens' in str(config.NEST_PERIPHERY):
            PARROT_PONSSENS = True
    if not(PARROT_MEDULLA):
        del neuron_types_to_region['parrot_medulla']
    else:
        input_populations.append('parrot_medulla')
    if not(PARROT_PONSSENS):
        del neuron_types_to_region['parrot_ponssens']
    else:
        input_populations.append('parrot_ponssens')
    n_input_populations = len(input_populations)

    # All cells are modelled as E-GLIF models;
    # with the only exception of Glomeruli and Mossy Fibers (not cells, just modeled as
    # relays; i.e., parrot neurons)
    neuron_types.remove('dcn_cell_Gly-I')
    for neuron_name in neuron_types:
        pop = neuron_name
        if neuron_name != 'glomerulus' and neuron_name != 'mossy_fibers':
            if neuron_name not in nest.Models():
                nest.CopyModel('eglif_cond_alpha_multisyn', neuron_name)
                nest.SetDefaults(neuron_name, neuron_param[neuron_name])
        else:
            if neuron_name not in nest.Models():
                nest.CopyModel('parrot_neuron', neuron_name)

        neuron_number[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[1]
        start_id_scaffold[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[0]

        neuron_models[neuron_name] = dict()
        region_names = neuron_types_to_region[neuron_name]
        nodes_inds = []
        for region in region_names:
            neuron_models[neuron_name][region] = nest.Create(neuron_name, neuron_number[neuron_name])
            if neuron_name != 'glomerulus' and neuron_name != 'mossy_fibers':
               random_init_vm(neuron_models[neuron_name][region], neuron_param[neuron_name])
            if region not in nest_network.brain_regions:
                nest_network.brain_regions[region] = NESTRegionNode(label=region)
                nodes_inds.append(np.where(sim_serial['connectivity.region_labels'] == region)[0][0])
            nest_network.brain_regions[region][pop] = \
                NESTPopulation(neuron_models[neuron_name][region],  # possible NEST model params as well here
                               nest, label=pop, brain_region=region)
            if config.VERBOSITY > 1:
                print("\n...created: %s..." % nest_network.brain_regions[region][pop].summary_info())
        nest_nodes_inds += nodes_inds

    # Split mossy fibers between central and border ones, and the central ones among N input populations, if any:
    target_border_mfs, target_central_mfs = split_mossy_fibers(start_id_scaffold,
                                                               n_mfs_groups=np.maximum(1, n_input_populations),
                                                               f=f, config=config)

    if n_input_populations:
        # If input comes via some parrot neuron population, MEDULLA and/or PONSSENS
        # create one parrot neuron for each mossy fiber target for the input population
        # (for each hemisphere's region):
        for pop, target in zip(input_populations, target_central_mfs):
            nodes_inds = []
            neuron_models[pop] = dict()
            for region in neuron_types_to_region[pop]:
                if region not in nest_network.brain_regions:
                    nest_network.brain_regions[region] = NESTRegionNode(label=region)
                    nodes_inds.append(np.where(sim_serial['connectivity.region_labels'] == region)[0][0])
                neuron_models[pop][region] = nest.Create("parrot_neuron", target.size)
                nest_network.brain_regions[region][pop] = \
                    NESTPopulation(neuron_models[pop][region], nest, label=pop, brain_region=region)
                if config.VERBOSITY > 1:
                    print("\n...created: %s..." % nest_network.brain_regions[region][pop].summary_info())
            nest_nodes_inds += nodes_inds

    ### Load connections from hdf5 file and create them in NEST:
    for conn_name in conn_weights.keys():
        conn = np.array(f['cells/connections/' + conn_name])
        pre_name = conn_pre_post[conn_name]["pre"]
        post_name = conn_pre_post[conn_name]["post"]
        for pre_region, post_region in zip(neuron_models[pre_name].keys(), neuron_models[post_name].keys()):
            source = np.array(conn[:, 0] - start_id_scaffold[pre_name] + neuron_models[pre_name][pre_region][0])
            target = np.array(conn[:, 1] - start_id_scaffold[post_name] + neuron_models[post_name][post_region][0])
            pre = list(source.astype(int))
            post = list(target.astype(int))
            if config.VERBOSITY > 1:
                print("Connecting  ", conn_name, "!")
                print("%s - %s -> %s -> %s" % (pre_name, pre_region, post_name, post_region))
            if conn_name == "mossy_to_glomerulus":
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * [conn_weights[conn_name]],
                             "delay": np.ones(len(pre)) * conn_delays[conn_name]}
            elif conn_name == "io_bc" or conn_name == "io_sc":
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * conn_weights[conn_name], \
                             "delay": {'distribution': 'exponential_clipped_to_boundary', 'low': min_iomli,
                                       'high': high_iomli, 'lambda': conn_delays[conn]},
                             "receptor_type": conn_receptors[conn_name]}
            else:
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * [conn_weights[conn_name]],
                             "delay": np.ones(len(pre)) * conn_delays[conn_name],
                             "receptor_type": conn_receptors[conn_name]}
            nest.Connect(pre, post, {"rule": "one_to_one"}, syn_param)

    if n_input_populations:
        # If there are such input populations, connect them to their target central mossy fibers:
        for pop, target in zip(input_populations, target_central_mfs):
            for region, region_mf in zip(neuron_types_to_region[pop],
                                         ['Right Ansiform lobule', 'Left Ansiform lobule']):
                if config.VERBOSITY > 1:
                    print("Connecting! %s - %s -> %s -> %s" % (pop, region, "mossy_fibers", region_mf))
                # translate to NEST ids
                nest.Connect(nest_network.brain_regions[region][pop].nodes,
                             get_mossy_targets(region_mf, neuron_models, start_id_scaffold, target),
                             conn_spec={"allow_autapses": False, 'allow_multapses': False, "rule": "one_to_one"})

    # Background noise input device as Poisson process
    if BACKGROUND_FREQ:
        nest_network.input_devices["Background"] = DeviceSet(label="Background", model="poisson_generator")
        if n_input_populations:
            # If there are such input populations, they should also receive the background noise input:
            noise_input_populations = list(input_populations)
            target_funs = [lambda pop, region: neuron_models[pop][region]] * n_input_populations
            # However, the background noise input should also go to the remaining border mossy fibers:
            noise_input_populations.append("mossy_fibers")
            # Border mossy fibers' indices:
            target_funs.append(lambda pop, region:
                                         get_mossy_targets(region, neuron_models, start_id_scaffold, target_border_mfs))
        else:
            # Otherwise, the whole mossy_fibers are targeted by this input:
            noise_input_populations = ["mossy_fibers"]
            target_funs = [lambda pop, region: neuron_models[pop][region]]
        for pop, target_fun in zip(noise_input_populations, target_funs):
            for region in neuron_types_to_region[pop]:
                nest_network.input_devices["Background"][region] = \
                    NESTPoissonGenerator(nest.Create('poisson_generator',
                                                     params={'rate': BACKGROUND_FREQ,
                                                             'start': 0.0, 'stop': TOT_DURATION}),
                                         nest, model="poisson_generator", label="Background", brain_region=region)
                nest.Connect(nest_network.input_devices["Background"][region].device, target_fun(pop, region),
                             conn_spec={"allow_autapses": False, 'allow_multapses': False, "rule": "all_to_all"})
                if config.VERBOSITY > 1:
                    print("Connected!  %s - %s -> %s -> %s" % ("Background", region, pop, region))

    if "input" in str(config.NEST_PERIPHERY).lower() or "TVB" in str(config.NEST_PERIPHERY):
        if "TVB" in str(config.NEST_PERIPHERY):
            # Whisking stimulus input device as TVB input signal from file
            dev_model = "inhomogeneous_poisson_generator"
            dev_model_class = NESTInhomogeneousPoissonGenerator
            npzfiles = np.load(config.NEST_STIMULUS_FILE)
            params = lambda iR: {"rate_values": npzfiles["ansilob_affts_trans_dt"][:, iR],
                                 "rate_times": npzfiles["time_dt"]}
        else:
            # Whisking stimulus input device as sinusoidally modulated Poisson process
            dev_model = "sinusoidal_poisson_generator"
            dev_model_class = NESTSinusoidalPoissonGenerator
            params = lambda iR: {"rate": STIM_RATE, "amplitude": STIM_AMPLITUDE,
                                 "frequency": STIM_FREQ, "phase": 0.0}
        conn_spec = {"allow_autapses": False, 'allow_multapses': False, "rule": "all_to_all"}
        nest_network.input_devices["Stimulus"] = DeviceSet(label="Stimulus", model=dev_model)

        if n_input_populations:
            # If there are specific input populations...
            # the Stimulus will target them:
            target_funs = [lambda pop, region: neuron_models[pop][region]] * n_input_populations
        else:
            # If there are no specific input populations...
            # ...the input stimulus should go directly to all central mossy_fibers:
            input_populations = ["mossy_fibers"]
            target_funs = [lambda pop, region:
                                    get_mossy_targets(region, neuron_models, start_id_scaffold, target_central_mfs[0])]
        for pop, target_fun in zip(input_populations, target_funs):
            for iR, region in enumerate(neuron_types_to_region[pop]):
                nest_network.input_devices["Stimulus"][region] = \
                    dev_model_class(nest.Create(dev_model, 1, params=params(iR)),
                                    nest, model=dev_model, label="Stimulus", brain_region=region)
                nest.Connect(nest_network.input_devices["Stimulus"][region].device, target_fun(pop, region),
                             conn_spec=conn_spec)
            if config.VERBOSITY > 1:
                print("Connected!  %s - %s -> %s -> %s" % ("Stimulus", region, pop, region))
    # Create output, measuring devices, spike_recorders and multimeters measuring V_m:
    params_spike_recorder = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
    params_spike_recorder["record_to"] = "ascii"
    # params_multimeter = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"].copy()
    # params_multimeter["record_to"] = "ascii"
    # params_multimeter["interval"] = 1.0
    for pop, regions in neuron_types_to_region.items():
        # pop_ts = "%s_ts" % pop
        nest_network.output_devices[pop] = DeviceSet(label=pop, model="spike_recorder")

        for region in regions:
            nest_network.output_devices[pop][region] = \
                NESTSpikeRecorder(nest.Create("spike_recorder", 1, params=params_spike_recorder),
                                  nest, model="spike_recorder", label=pop, brain_region=region)
            if pop == "granule_cell":
                nodes = nest_network.brain_regions[region][pop].nodes[0::10]
            else:
                nodes = nest_network.brain_regions[region][pop].nodes
            nest.Connect(nodes, nest_network.output_devices[pop][region].device)
            nest_network.output_devices[pop].update()  # update DeviceSet after the new NESTDevice entry
            if config.VERBOSITY > 1:
                print("\n...created spike_recorder device for population %s in brain region %s..." % (pop, region))

        if config.NEST_MULTIMETER:
            if pop not in ['mossy_fibers', "parrot_medulla", "parrot_ponssens", "whisking_stimulus"]:
                nest_network.output_devices[pop_ts] = DeviceSet(label=pop_ts, model="multimeter")
                # Create and connect population multimeter for this region:
                nest_network.output_devices[pop_ts][region] = \
                    NESTMultimeter(nest.Create("multimeter", 1, params=params_multimeter),
                                   nest, model="multimeter", label=pop_ts, brain_region=region)
                nest.Connect(nest_network.output_devices[pop_ts][region].device,
                             nest_network.brain_regions[region][pop].nodes)
                nest_network.output_devices[pop_ts].update()  # update DeviceSet after the new NESTDevice entry
                if config.VERBOSITY > 1:
                    print("\n...created multimeter device for population %s in brain region %s..." % (pop_ts, region))

    nest_network.configure()
    if config.VERBOSITY > 1:
        nest_network.print_summary_info_details(recursive=1, connectivity=False)

    return nest_network, nest_nodes_inds, neuron_models, neuron_number, start_id_scaffold


def plot_nest_results_raster(nest_network, neuron_models, neuron_number, config):

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    goc_events = nest_network.output_devices['golgi_cell']['Left Ansiform lobule'].events
    goc_evs = goc_events['senders']
    goc_times = goc_events['times']

    grc_events = nest_network.output_devices['granule_cell']['Left Ansiform lobule'].events
    grc_evs = grc_events['senders']
    grc_times = grc_events['times']

    glom_events = nest_network.output_devices['glomerulus']['Left Ansiform lobule'].events
    glom_evs = glom_events['senders']
    glom_times = glom_events['times']

    pc_events = nest_network.output_devices['purkinje_cell']['Left Ansiform lobule'].events
    pc_evs = pc_events['senders']
    pc_times = pc_events['times']

    sc_events = nest_network.output_devices['stellate_cell']['Left Ansiform lobule'].events
    sc_evs = sc_events['senders']
    sc_times = sc_events['times']

    bc_events = nest_network.output_devices['basket_cell']['Left Ansiform lobule'].events
    bc_evs = bc_events['senders']
    bc_times = bc_events['times']

    io_events = nest_network.output_devices['io_cell']['Left Inferior olivary complex'].events
    io_evs = io_events['senders']
    io_times = io_events['times']

    dcng_events = nest_network.output_devices['dcn_cell_GABA']['Left Cerebellar Nuclei'].events
    dcng_evs = dcng_events['senders']
    dcng_times = dcng_events['times']

    dcn_events = nest_network.output_devices['dcn_cell_glut_large']['Left Cerebellar Nuclei'].events
    dcn_evs = dcn_events['senders']
    dcn_times = dcn_events['times']


    # ######################### PLOTTING PSTH AND RASTER PLOTS ########################

    CELL_TO_PLOT = ['glomerulus', 'granule_cell', 'basket_cell', 'stellate_cell', 'purkinje_cell',
                    'io_cell', 'dcn_cell_GABA',  'dcn_cell_glut_large']

    cells = {'granule_cell': [grc_times, grc_evs],
             'golgi_cell': [goc_times, goc_evs],
             'glomerulus': [glom_times, glom_evs],
             'purkinje_cell': [pc_times, pc_evs],
             'stellate_cell': [sc_times, sc_evs],
             'basket_cell': [bc_times, bc_evs],
             'io_cell': [io_times, io_evs],
             'dcn_cell_GABA': [dcng_times, dcng_evs],
             'dcn_cell_glut_large': [dcn_times, dcn_evs]}

    color = {'granule_cell': '#E62214',  # 'rgba(255, 0, 0, .8)',
             'golgi_cell': '#332EBC',  # 'rgba(0, 255, 0, .8)',
             'glomerulus': '#0E1030',  # rgba(0, 0, 0, .8)',
             'purkinje_cell': '#0F8944',  # 'rgba(64, 224, 208, .8)',
             'stellate_cell': '#FFC425',  # 'rgba(234, 10, 142, .8)',
             'basket_cell': '#F37735',
             'io_cell': 'rgba(75, 75, 75, .8)',
             'dcn_cell_GABA': 'rgba(100, 100, 100, .8)',
             'dcn_cell_glut_large': '#080808'}  # 'rgba(234, 10, 142, .8)'}

    # PSTH

    def metrics(spikeData, TrialDuration, cell, figure_handle, sel_row):
        id_spikes = np.sort(np.unique(spikeData, return_index=True))
        bin_size = 5  # [ms]
        n_bins = int(TrialDuration / bin_size) + 1
        psth, tms = np.histogram(spikeData, bins=n_bins, range=(0, TrialDuration))

        # absolute frequency
        abs_freq = np.zeros(id_spikes[0].shape[0])
        for idx, i in enumerate(id_spikes[0]):
            count = np.where(spikeData == i)[0]
            abs_freq[idx] = count.shape[0]

        # mean frequency
        m_f = (id_spikes[0].shape[0]) / ((TrialDuration / 1000) * len(neuron_models[cell]))

        layout = go.Layout(
            scene=dict(aspectmode='data'),
            xaxis={'title': 'time (ms)'},
            yaxis={'title': 'number of spikes'}
        )

        n_neurons = neuron_number[cell]
        if cell == "granule_cell":
            n_neurons = int(np.round(n_neurons/10))
        figure_handle.add_trace(go.Bar(
            x=tms[0:len(tms) - 1],
            y=psth / ((bin_size * 0.001) * n_neurons),
            width=4.0,
            marker=dict(
                color=color[cell])
        ), row=sel_row, col=1)

        if config.VERBOSITY > 1:
            print("mean frequency: ", int(m_f))

        return tms

    # RASTER
    def raster(times, cell_ids, cell, fig_handle, sel_row):
        trace0 = go.Scatter(
            x=times,
            y=cell_ids,
            name='',
            mode='markers',
            marker=dict(
                size=4,
                color=color[cell],
                line=dict(
                    width=.2,
                    color='rgb(0, 0, 0)'
                )
            )
        )
        fig_handle.add_trace(trace0, row=sel_row, col=1)

    fig_psth = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                             y_title='Frequency [Hz]')
    fig_raster = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                               y_title='# cells')
    num = 1
    for c in CELL_TO_PLOT:
        times = cells[c][0]
        cell_ids = cells[c][1]
        metrics(times, config.SIMULATION_LENGTH, c, fig_psth, num)
        raster(times, cell_ids, c, fig_raster, num)
        num += 1
    fig_psth.update_xaxes(range=[0, config.SIMULATION_LENGTH * 1.1])
    fig_raster.update_xaxes(range=[0, config.SIMULATION_LENGTH * 1.1])
    fig_psth.update_layout(showlegend=False)
    fig_raster.update_layout(showlegend=False)
    if config.figures.SAVE_FLAG:
        try:
            fig_psth.write_image(os.path.join(config.figures.FOLDER_FIGURES, "NESTpsth.%s" % config.figures.FIG_FORMAT))
            fig_raster.write_image(os.path.join(config.figures.FOLDER_FIGURES, "NESTraster.%s" % config.figures.FIG_FORMAT))
        except Exception as e:
            warnings.warn("Failed to write_image for plotly figures with error:\n%s" % str(e))
    if config.figures.SHOW_FLAG:
        fig_psth.show()
        fig_raster.show()
    else:
        # TODO: find a better way to delete plotly figures
        # The current one is taken from here: https://community.plotly.com/t/remove-all-traces/13469
        # There might not be a better one yet...: https://github.com/plotly/plotly.py/issues/2725
        fig_psth.data = []
        fig_raster.data = []
        fig_psth.layout = dict()
        fig_raster.layout = dict()
        fig_psth = None
        fig_raster = None
    return fig_psth, fig_raster


def simulate_nest_network(nest_network, config, neuron_models=dict(), neuron_number=dict()):
    simulation_length, transient = configure_simulation_length_with_transient(config)
    tic = time.time()
    # Simulate:
    if config.VERBOSITY:
        print("\nSimulating NEST network...")
    nest_network.nest_instance.Simulate(simulation_length)
    if config.VERBOSITY:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    return nest_network, transient


def run_nest_workflow(PSD_target=None, model_params=dict(), config=None, **config_args):
    tic = time.time()
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    if config.VERBOSITY:
        print("\n\n------------------------------------------------\n\n"+
              "Running NEST workflow for plot_flag=%s, \nand model_params=\n%s...\n" 
              % (str(plot_flag), str(config.model_params)))
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Build the NEST network
    nest_network, nest_nodes_inds, neuron_models, neuron_number, start_id_scaffold = build_NEST_network(config)

    if "OFF" in config.MODE:
        inds_off = np.sort(inds['cereb_crtx'].tolist() +
                           inds['cereb_nuclei'].tolist() +
                           inds['ansilob'].tolist())
        simulator.connectivity.weights[inds_off, :] = 0
        simulator.connectivity.weights[:, inds_off] = 0
        if config.VERBOSITY:
            print("\n")
            print("-"*25)
            print("-"*25)
            print("Setting to 0.0 connections in and out of cerebellum\n"
                  "['Left/Right Cerebellar Cortex'\n"
                  "'Left/Right Cerebellar Nuclei'\n"
                  "'Left Ansiform lobule']!!!:\n"
                  "IN: %s\n"
                  "OUT: %s" % (str(simulator.connectivity.weights[inds_off, :]),
                               str(simulator.connectivity.weights[:, inds_off])))
        simulator.connectivity.configure()
        simulator.configure()
        for hemi in ["Right", "Left"]:
            nest_network.brain_regions['%s Cerebellar Nuclei' % hemi]['dcn_cell_glut_large'].Set({"V_th": 35.0})
            print('%s Cerebellar Nuclei - dcn_cell_glut_large' % hemi)
            print(nest_network.brain_regions['%s Cerebellar Nuclei' % hemi]['dcn_cell_glut_large'].Get("V_th"))

    # Simulate the NEST network
    nest_network, transient = simulate_nest_network(nest_network, config, neuron_models, neuron_number)
    # Plot results
    if plotter is not None:
        try:
            if config.SIMULATION_LENGTH <= 2000.0:
                plot_nest_results_raster(nest_network, neuron_models, neuron_number, config)
            from examples.plot_write_results import plot_write_spiking_network_results
            plot_write_spiking_network_results(nest_network, connectivity=connectivity,
                                               time=None, transient=transient,
                                               monitor_period=simulator.monitors[0].period,
                                               plot_per_neuron=False, plotter=plotter, writer=None, config=config)
        except Exception as e:
            warnings.warn(
                "Failed to plot and/or write at least some of the NEST simulation results with error:\n%s"
                % str(e))
    if config.VERBOSITY:
        print("\nFinished NEST workflow in %g sec!\n" % (time.time() - tic))
    results = {"nest_network": nest_network, "simulator": simulator, "config": config}
    return results


if __name__ == "__main__":
    parser = args_parser("nest_script")
    args, parser_args, parser = parse_args(parser, argsnames=list(DEFAULT_ARGS.keys()))
    verbosity = args.get('verbosity', DEFAULT_ARGS['verbosity'])
    if verbosity:
        print("Running %s with user provided arguments:\n" % parser.description)
        print(args, "\n")
    run_nest_workflow(**args)
