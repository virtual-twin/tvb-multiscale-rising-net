# -*- coding: utf-8 -*-

import warnings

import numpy
from scipy.signal import welch
import matplotlib.pyplot as plt

from rising_net.scripts.base import *
from rising_net.scripts.utils import get_regions_indices, dump_pickled_time_series, \
    compute_data_PSDs, compute_data_PSDs_from_raw  # , compute_task_transfer_metrics
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict

# Put the results in a Timeseries instance
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def load_connectome(config):
    import h5py
    connectome = {}
    f = h5py.File(config.BRAIN_CONN_FILE)
    connectome['region_labels'] = np.array(f["region_labels"][()]).astype("<U128")
    connectome['centres'] = np.array(f["centres"][()])
    # connectome['hemispheres'] = np.array(f["hemispheres"][()]),
    connectome['weights'] = np.array(f["weights"][()])
    connectome['tract_lengths'] = np.array(f["tract_lengths"][()])
    f.close()

    major_structs_labels = np.load(config.MAJOR_STRUCTS_LABELS_FILE)
    voxel_count = np.load(config.VOXEL_COUNT_FILE)
    inds = np.load(config.INDS_FILE, allow_pickle=True).item()
    inds["ponssens"] = inds["ponssens"][[0, 2]]
    if config.VERBOSITY > 1:
        print("major_structs_labels:\n", np.unique(major_structs_labels))
        print("ROI inds:\n", inds)

    return connectome, major_structs_labels, voxel_count, inds


def insert_whiskers_to_connectome(connectome, major_structs_labels, voxel_count, inds, config):

    def insert_along_axis(arr, vals, N, axis=0):
        new_arr = np.insert(arr, int(N / 2), vals[0], axis=axis)
        return np.insert(new_arr, 212 + 1, vals[1], axis=axis)

    def insert_2D(arr, vals, N):
        new_arr = insert_along_axis(arr, vals, N, axis=0)
        return insert_along_axis(new_arr, vals, N, axis=1)

    connectome = dict(connectome)
    N = connectome["region_labels"].shape[0]

    connectome["region_labels"] = insert_along_axis(connectome["region_labels"],
                                                    ['Right Whiskers', 'Left Whiskers'],
                                                    N, axis=0)
    #     whiskers_cntrs = (connectome["centres"][inds["facial"]] - connectome["centres"][inds["trigeminal"]])/2
    #     connectome["centres"] = insert_along_axis(connectome["centres"],
    #                                               [whiskers_cntrs[0], whiskers_cntrs[1]],
    #                                               N, axis=0)

    connectome["weights"] = insert_2D(connectome["weights"], [0, 0], N)
    connectome["tract_lengths"] = insert_2D(connectome["tract_lengths"], [0, 0], N)

    major_structs_labels = insert_along_axis(major_structs_labels,
                                             ['Right Whiskers', 'Left Whiskers'],
                                             N, axis=0)

    voxel_count = insert_along_axis(voxel_count, [0, 0], N, axis=0)

    for key, val in inds.items():
        val[val >= 106] = val[val >= 106] + 1

    inds["whiskers"] = np.array([int(N / 2), int(N + 1)])

    connectome["weights"][inds["whiskers"], inds["facial"]] = config.WHISKERS

    return connectome, major_structs_labels, voxel_count, inds


def construct_extra_inds_and_maps(connectome, inds, config):
    whiskinds = list(inds.get("whiskers", []))
    maps = {}
    region_labels = connectome['region_labels']
    inds["subcrtx"] = np.arange(len(region_labels)).astype('i')
    inds["subcrtx"] = np.delete(inds["subcrtx"], inds["crtx"].tolist() + whiskinds)
    maps["is_subcortical"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_subcortical"][inds["subcrtx"]] = True
    maps["is_cortical"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_cortical"][inds["crtx"]] = True
    maps["is_thalamic"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_thalamic"][inds["thalspec"]] = True
    maps["is_whiskers"] = np.array([False] * region_labels.shape[0]).astype("bool")
    if len(whiskinds):
        maps["is_whiskers"][whiskinds] = True
    maps["not_thalamic"] = np.logical_not(np.logical_or(maps["is_thalamic"], maps["is_whiskers"]))
    maps["is_subcortical_not_thalspec"] = np.logical_and(maps["is_subcortical"], np.logical_not(maps["is_thalamic"]))
    inds["subcrtx_not_thalspec"] = np.where(maps["is_subcortical_not_thalspec"])[0]
    inds['crtx_and_subcrtx'] = np.sort(np.concatenate([inds['crtx'], inds["subcrtx_not_thalspec"]]))
    # Indices of cortical and subcortical regions excluding specific thalami
    inds["non_thalamic"] = np.unique(inds['crtx'].tolist() + inds["subcrtx_not_thalspec"].tolist())
    # Task related regions' indices:
    inds['medulla'] = inds['ponssens_trigeminal']
    config.TASKREGS = ["m1", 'facial',
                       "trigeminal", 'ponssens_trigeminal',
                       'ansilob', 'cereb_nuclei',
                       "m1thal", "s1brlthal", "s1brl"]
    if len(whiskinds):
        config.TASKREGS.insert(2, 'whiskers',)
    config.TASKINDS = []
    for reg in config.TASKREGS:
        config.TASKINDS += list(inds[reg])

    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config.__dict__, file, recurse=1)

    return inds, maps, config


def plot_norm_w_hist(w, wp, inds, plotter_config, title_string=""):
    h = w[wp].flatten()
    # print('number of all connections > 0: %d' % h.size)
    h, bins = np.histogram(h, range=(1.0, 31), bins=100)

    w_within_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_from_sub = w[inds["crtx"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_to_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["crtx"][None, :]]
    h_sub = np.array(w_within_sub.flatten().tolist() +
                     w_from_sub.flatten().tolist() +
                     w_to_sub.flatten().tolist())
    h_sub = h_sub[h_sub > 0].flatten()
    # print('number of h_sub > 0: %d' % h_sub.size)
    h_sub, bins_sub = np.histogram(h_sub, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_sub)

    h_crtx = np.array(w[inds["crtx"][:, None], inds["crtx"][None, :]].flatten().tolist())
    h_crtx = h_crtx[h_crtx > 0]
    # print('number of h_crtx > 0: %d' % h_crtx.size)
    h_crtx, bins_crtx = np.histogram(h_crtx, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_crtx)

    h2 = h_crtx + h_sub
    # print('number of total > 0: %d' % np.sum(h2))

    x = bins[:-1] + np.diff(bins) / 2
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, h, 'b', label='All connections')
    plt.plot(x, h_crtx, 'g', label='Isocortical connections')
    plt.plot(x, h_sub, 'r', label='Non-isocortical connections')
    # plt.plot(x, h-h_sub, 'r--', label='All - Subcortical connections')
    # plt.plot(x, h-h_crtx, 'g--', label='All - Non Subcortical connections')
    # plt.plot(x, h2, 'k--', label='Total connections')
    plt.title("Histogram of %s connectome weights" % title_string)
    plt.legend()
    plt.ylim([0.0, h.max()])
    plt.tight_layout()
    if plotter_config.SAVE_FLAG:
        plt.savefig(os.path.join(plotter_config.FOLDER_FIGURES, "%sWeightsHistogram.png" % title_string))
    if plotter_config.SHOW_FLAG:
        fig.show()
    else:
        plt.close(fig)
    return fig


def logprocess_weights(connectome, inds, verbosity=1, plotter=None):
    w = connectome['weights'].copy()
    w[np.isnan(w)] = 0.0  # zero nans
    w0 = w <= 0  # zero weights
    wp = w > 0  # positive weights
    if plotter:
        plot_norm_w_hist(w, wp, inds, plotter.config)
    w /= w[wp].min()  # divide by the minimum to have a minimum of 1.0
    w *= np.exp(1)  # multiply by e to have a minimum of e
    w[wp] = np.log(w[wp])  # log positive values
    w[w0] = 0.0  # zero zero values (redundant)
    connectome['weights'] = w
    if verbosity > 1:
        print('\nnormalized weights [min, max] = \n', [w[wp].min(), w[wp].max()])
    if plotter:
        plot_norm_w_hist(w, wp, inds, plotter.config, title_string="logtransformed ")
    return connectome


def prepare_connectome(config, plotter=None):
    # Load connectome and other structural files
    connectome, major_structs_labels, voxel_count, inds = load_connectome(config)
    if config.WHISKERS:
        connectome, major_structs_labels, voxel_count, inds = \
            insert_whiskers_to_connectome(connectome, major_structs_labels, voxel_count, inds, config)
        # Construct some more indices and maps
    inds, maps, config = construct_extra_inds_and_maps(connectome, inds, config)
    # if config.CONN_LOG:
    if config.VERBOSITY:
        print("Logtransforming connectivity weights!")
    # Logprocess connectome
    connectome = logprocess_weights(connectome, inds, verbosity=config.VERBOSITY, plotter=plotter)
    # Prepare connectivity with all possible normalizations
    return connectome, major_structs_labels, voxel_count, inds, maps, config


def build_connectivity(connectome, inds, config):
    from tvb.datatypes.connectivity import Connectivity

    connectivity = Connectivity(**connectome)

    # Normalize connectivity weights
    # Set all NaN and Inf weights to 0.0, if any:
    connectivity.weights[np.logical_or(np.isnan(connectivity.weights), np.isinf(connectivity.weights))] = 0.0
    if config.CONN_NORM_PERCENTILE:
        if config.VERBOSITY:
            print("Normalizing connectivity weights with %g percentile!" % config.CONN_NORM_PERCENTILE)
        connectivity.weights /= np.percentile(connectivity.weights, config.CONN_NORM_PERCENTILE)
    # Set maximum tract length so that we have a minimum time delay of one TVB integration time step:
    connectivity.speed = np.array([config.CONN_SPEED])
    connectivity.tract_lengths = np.maximum(connectivity.speed * config.DEFAULT_DT,
                                            connectivity.tract_lengths)
    if config.WHISKERS:
        connectivity.weights[inds["whiskers"], inds["facial"]] = 1.0
        connectivity.weights[inds["trigeminal"], inds["whiskers"]] = 1.0
    connectivity.configure()
    if config.WHISKERS * config.VERBOSITY:
        print("Facial -> Whiskers weight!:\n%s" % str(connectivity.weights[inds["whiskers"], inds["facial"]]))
        print("Facial -> Whiskers delay!:\n%s" % str(connectivity.delays[inds["whiskers"], inds["facial"]]))
        print("Whiskers -> Trigeminal weight!:\n%s" % str(connectivity.weights[inds["trigeminal"], inds["whiskers"]]))
        print("Whiskers -> Trigeminal delay!:\n%s" % str(connectivity.delays[inds["trigeminal"], inds["whiskers"]]))

    #if "w" in config.THAL_CRTX_FIX:
    # Fix the thalamocortical weights to 1.0:
    if config.VERBOSITY:
        print("Fixing thalamocortical weights!")
    # Fix structural connectivity (specific) thalamo-cortical weights to 1,
    # such that all thalamo-cortical weights are equal to the parameters
    # w_er, w_es, w_se, w_si
    connectivity.weights[inds["crtx"], inds["thalspec"]] = 1.0
    connectivity.weights[inds["thalspec"], inds["crtx"]] = 1.0

    # Remove connections between specific thalami and the rest of the subcortex:
    # Keep only the following connections:
    # TRIGEMINAL, MEDULLA -> S1 Brl field Spec thalami
    # Cereb Nuclei -> M1 & S1 Brl field Spec thalami
    trigeminal_inds = np.copy(inds["trigeminal"])
    senstrig_inds = np.copy(inds["ponssens_trigeminal"])
    cereb_nuclei_inds = np.copy(inds["cereb_nuclei"])
    if config.TASK_LATERALITY == -1:    # -1: contralatterally, 0: bilaterally, 1: ipsilaterally
        trigeminal_inds = trigeminal_inds[::-1]
        senstrig_inds = senstrig_inds[::-1]
        cereb_nuclei_inds = cereb_nuclei_inds[::-1]
    if config.TASK_LATERALITY == 0:
        w_s1brlthal_trigeminal = connectivity.weights[inds["s1brlthal"]][:, trigeminal_inds].copy()
        w_s1brlthal_senstrig = connectivity.weights[inds["s1brlthal"]][:, senstrig_inds].copy()
        w_m1thal_cerebnuclei = connectivity.weights[inds["m1thal"]][:, inds["cereb_nuclei"]].copy()
        w_s1brlthal_cerebnuclei = connectivity.weights[inds["s1brlthal"]][:, inds["cereb_nuclei"]].copy()
    else:
        w_s1brlthal_trigeminal = connectivity.weights[inds["s1brlthal"], trigeminal_inds].copy()
        w_s1brlthal_senstrig = connectivity.weights[inds["s1brlthal"], senstrig_inds].copy()
        w_m1thal_cerebnuclei = connectivity.weights[inds["m1thal"], inds["cereb_nuclei"]].copy()
        w_s1brlthal_cerebnuclei = connectivity.weights[inds["s1brlthal"], inds["cereb_nuclei"]].copy()
    # Zero all Spec Thal <-> Subcortex connections:
    connectivity.weights[inds["subcrtx_not_thalspec"][:, None], inds["thalspec"][None, :]] = 0.0
    connectivity.weights[inds["thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]] = 0.0
    # Recover the stored connections:
    if config.TASK_LATERALITY == 0:
        # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
        connectivity.weights[inds["s1brlthal"][:, None], trigeminal_inds[None, :]] = w_s1brlthal_trigeminal
        connectivity.weights[inds["s1brlthal"][:, None], senstrig_inds[None, :]] = w_s1brlthal_senstrig
        # from merged Cerebellar Nuclei to M1:
        connectivity.weights[inds["m1thal"][:, None], cereb_nuclei_inds[None, :]] = w_m1thal_cerebnuclei
        connectivity.weights[inds["s1brlthal"][:, None], cereb_nuclei_inds[None, :]] = w_s1brlthal_cerebnuclei
    else:
        # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
        connectivity.weights[inds["s1brlthal"], trigeminal_inds] = w_s1brlthal_trigeminal
        connectivity.weights[inds["s1brlthal"], senstrig_inds] = w_s1brlthal_senstrig
        # from merged Cerebellar Nuclei to M1:
        connectivity.weights[inds["m1thal"], cereb_nuclei_inds] = w_m1thal_cerebnuclei
        connectivity.weights[inds["s1brlthal"], cereb_nuclei_inds] = w_s1brlthal_cerebnuclei

    return connectivity


def build_model(number_of_regions, inds, maps, config):
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical

    dummy = np.ones((number_of_regions,1))

    if config.VERBOSITY:
        print("Configuring model with parameters:\n%s" % str(config.model_params))

    # STIMULUS = config.model_params.get("STIMULUS", None)

    model_params = {}
    for p, pval in config.model_params.items():
        # if p != "STIMULUS":
        if pval is not None:
            pval = np.array([pval]).flatten()
            if p == 'G':
                # G normalized by the number of regions as in Griffiths et al paper
                # Geff = G /(number_of_regions - inds['thalspec'].size)
                pval = pval / (number_of_regions - inds['thalspec'].size)
            model_params[p] = pval

    # if STIMULUS:
    #     if model_params.get("G", WilsonCowanThalamoCortical.G.default)[0].item() > 0.0:
    #         # Stimulus to M1 and S1 barrel field
    #         # inds_stim = np.concatenate((inds["motor"][:2], inds["sens"][-2:])
    #         # if config.NEST_PERIPHERY:
    #         #     inds_stim = np.array(inds["facial"])
    #         # else:
    #         inds_stim = np.concatenate((inds["facial"], inds["trigeminal"]))
    #     else:
    #         # Stimulus directly to all specific thalami:
    #         inds_stim = inds['thalspec']
    #     # Stimuli:
    #     A_st = 0 * dummy.astype("f")
    #     B_st = 0 * dummy.astype("f")
    #     f_st = 0 * dummy.astype("f")
    #     # Stimulus to trigeminal
    #     A_st[inds_stim] = STIMULUS
    #     B_st[inds_stim] = config.STIMULUS_BASELINE
    #     f_st[inds_stim] = config.STIMULUS_RATE  # Hz
    #     model_params.update({"A_st": A_st, "B_st": B_st, "f_st": f_st})

    model = WilsonCowanThalamoCortical(is_cortical=maps['is_cortical'][:, np.newaxis],
                                       is_thalamic=maps['is_thalamic'][:, np.newaxis],
                                       is_whiskers=maps["is_whiskers"][:, np.newaxis],
                                       **model_params)



    model.dt = config.DEFAULT_DT

    # Remove Specific thalamic relay -> nonspecific subcortical structures connections!
    w_se = model.w_se * dummy
    w_se[inds['subcrtx']] = 0.0
    model.w_se = w_se
    # Remove specific thalamic relay -> inhibitory nonspecific subcortical structures connections
    w_si = model.w_si * dummy
    w_si[inds['subcrtx']] = 0.0
    model.w_si = w_si

    # Long range connections to specific thalamic relay and reticular structures connections' weights:
    model.G = model.G * dummy
    model.G[inds["thalspec"]] = 0.0  # Zero all long range connections' inputs to Specific Thalami
    # Keep only the connections:
    # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
    model.G[inds["s1brlthal"]] = model.G[inds["crtx"][0]]
    # from Cerebellar Nuclei to M1:
    model.G[inds["m1thal"]] = model.G[inds["crtx"][0]]

    return model


# An approximate automatic FIC:

def fic(param, p_orig, weights, trg_inds=None, src_inds=None, FIC=1.0, G=None, dummy=None, subtitle="", plotter=None):
    number_of_regions = weights.shape[0]
    # This function will adjust inhibitory weights based on total indegree and some scaling
    if trg_inds is None:
        trg_inds = np.arange(number_of_regions).astype('i')

    if src_inds is None:
        src_inds = np.arange(number_of_regions).astype('i')

    # Scale w_ie or I_e to grow to greater negative values from the defaults
    p_orig = np.array(p_orig)
    if p_orig.size == 1:
        if dummy is None:
            dummy = np.ones((number_of_regions,))
            p_orig = p_orig.item() * dummy
    p = p_orig.copy()
    pscalar = p_orig[trg_inds].mean().item()
    # Move them to have a maximum of p_orig:
    # FICindegree = (indegree - indegree_min) / indegree_max
    indegree = weights[trg_inds][:, src_inds].sum(axis=1)
    indgree_min = indegree.min()
    indgree_max = indegree.max()
    FICindegree = np.maximum(0.0, FIC * (indegree - indgree_min) / (indgree_max - indgree_min))
    # p_fic = p * (1 + FIC * FICindegree) = p * (1 + FIC * (indegree - indegree_min) / (indegree_max - indegree_min))
    # assuming p < 0.0, and FIC >= 0.0
    if G is not None:
        FICindegree *= G
    p[trg_inds] = pscalar * (1.0 + FICindegree)

    try:
        assert np.all(np.argsort(indegree) == np.argsort(-p[trg_inds]))  # the orderings should reverse
    except Exception as e:
        if plotter:
            fig = plt.figure()
            plt.plot(indegree, p[trg_inds], "o")
            if G is None:
                plt.xlabel("%g*indegree" % FIC)
            else:
                plt.xlabel("%g*%g*indegree" % (G, FIC))
            plt.ylabel("%s scaled" % param)
            plt.title("Testing indegree and parameter anti-correlation")
            plt.tight_layout()
        warnings.warn(str(e))
        # raise e

    # Plot and confirm:
    if plotter:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[1].hist(FICindegree, 30)
        axes[1].set_xlabel("Indegree Scaler values")
        axes[1].set_ylabel("Histogram of region counts")
        if G is None:
            axes[1].set_title("Indegree scaler = %g*(indegree - indegree_min) / (indegree_max - indegree_min)" % FIC)
        else:
            axes[1].set_title("Indegree scaler = %g*%g*(indegree - indegree_min) / (indegree_max - indegree_min)"
                              % (G, FIC))
        axes[0].hist(p[trg_inds], 30)
        axes[0].set_xlabel("Parameter values")
        axes[0].set_ylabel("Histogram of region counts")
        axes[0].set_title("FICed parameter %s%s = %g * (1 + Indegree scaler)" % (param, subtitle, pscalar))
        fig.tight_layout()
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "FIC.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return p


def apply_fic(simulator, inds, config, plotter=None):
    n_non_thalamic_regions = (simulator.connectivity.weights.shape[0] - inds['thalspec'].size)
    G = simulator.model.G[0].item() * n_non_thalamic_regions
    for fp, fv, split_string in zip(config.FIC_PARAMS,
                                    [config.FIC_SPLIT, 1.0-config.FIC_SPLIT],
                                    ["FIC_SPLIT", "(1.0-FIC_SPLIT)"]):
        ficsplit = config.FIC * fv
        if ficsplit > 0:
            if config.VERBOSITY:
                print("Applying FIC for parameter %s: G * FIC * %s = %g * %g * %g = %g!" %
                      (fp, split_string, G, config.FIC, fv,  G * ficsplit))
            # We will modify the w_ie and w_rs parameters a bit based on indegree:
            setattr(simulator.model, fp,
                    fic(fp, getattr(simulator.model, fp), simulator.connectivity.weights,
                        inds["non_thalamic"], inds["non_thalamic"], FIC=ficsplit, G=G, dummy=None, subtitle="",
                        plotter=plotter))
    return simulator


def apply_pathway_gain_to_target(src_inds, trg_inds, pathway_gain, weights, task_laterality=-1,
                                 fix_inds=[], indegree_gain=None, verbosity=1):
    # Determine source and fixed regions' indices:
    FIXflag = False
    if len(fix_inds):
        FIXflag = True
    if src_inds is None:
        src_inds = np.arange(weights.shape[1]).astype('i')
        if FIXflag:
            src_inds = np.delete(src_inds, fix_inds)
    nsrc = len(src_inds)
    # Prepare pathway gains depending on the number of source indices:
    try:
        pathway_gains = list(pathway_gain)
    except:
        pathway_gains = [pathway_gain]
    ngains = len(pathway_gains)
    if ngains == 1:
        if task_laterality == 0:  # assuming the same pathway gain for all source regions was given
            pathway_gains *= nsrc
        else:
            pathway_gains *= int(nsrc * 0.5)
    elif ngains == nsrc/2:
        if task_laterality == 0:
            # assuming one pathway gain per region for both hemispheres was given...
            # ...double pathway gains for the two hemispheres if we apply gain bilaterally:
            pathway_gains = [pg for pg in pathway_gains for _ in range(2)]
    elif ngains != nsrc:
        raise ValueError("Pathway gains %s of length %d do not match with target regions inds %s of length %d!"
                         % (str(pathway_gains), len(pathway_gains), str(src_inds), len(src_inds)))
    pathway_gains = np.array(pathway_gains)
    if verbosity: print("pathway_gains = ", pathway_gains)
    indegree_ratio = []
    for iT, trg in enumerate(trg_inds):
        if verbosity: print("trg = ", trg)
        indegree = weights[trg].sum()                  # initial total indegree
        if verbosity: print("indegree = ", indegree)
        if indegree_gain is not None:
            new_indegree = indegree / indegree_gain
            print("new_indegree to fix = ", new_indegree)
        if task_laterality > 0:
            hemi_src_inds = src_inds[slice(np.mod(iT, 2), None, 2)]
        elif task_laterality < 0:
            hemi_src_inds = src_inds[slice(np.abs(np.mod(iT, 2)-1), None, 2)]
        else:
            hemi_src_inds = src_inds
        if verbosity: print("hemi_src_inds = ", hemi_src_inds)
        orig = weights[trg, hemi_src_inds]
        if verbosity: print("orig = ", orig)
        origsum = orig.sum()
        if verbosity: print("origsum = ", origsum)
        if FIXflag:
            hemi_fix_inds = fix_inds[slice(np.mod(iT, 2), None, 2)]  # Only ipsilaterally
            if verbosity: print("hemi_fix_inds = ", hemi_fix_inds)
            fix = weights[trg, hemi_fix_inds]
            if verbosity: print("fix = ", fix)
            fixsum = fix.sum()
            if verbosity: print("fixsum = ", fixsum)
        else:
            fixsum = 0.0
        pathway_gains_corr = pathway_gains
        if indegree_gain is not None:
            maxnewsum = 0.99*new_indegree - fixsum
            newsum = (weights[trg, hemi_src_inds] * pathway_gains_corr).sum()
            if newsum > maxnewsum:
                pathway_gains_corr *= (maxnewsum/newsum)
        if verbosity: print("pathway_gains_corr = ", pathway_gains_corr)
        weights[trg, hemi_src_inds] *= pathway_gains_corr  # increase pathway
        if verbosity: print("w = \n", weights[trg, hemi_src_inds])
        newsum = weights[trg, hemi_src_inds].sum()
        if verbosity: print("wsum = \n", newsum)
        if indegree_gain is not None:
            nornom = new_indegree - newsum - fixsum
            if nornom < 0.0:
                new_indegree = newsum + fixsum + 0.01 * indegree
                if verbosity: print("new_indegree_corr = ", new_indegree)
                indegree_gain = indegree / new_indegree
                if verbosity: print("indegree_gain_corr = ", indegree_gain)
                nornom = new_indegree - newsum - fixsum
            norm = nornom / (indegree - origsum - fixsum)
            if verbosity: print("norm = ", norm)
            weights[trg] *= norm
            weights[trg, hemi_src_inds] /= norm
            if FIXflag:
                weights[trg, hemi_fix_inds] /= norm  # set fixed connections
                if verbosity: print("wfix = \n", weights[trg, hemi_fix_inds])
        final_indegree = weights[trg].sum()
        if verbosity: print("final indegree = ", final_indegree)
        try:
            assert np.all(final_indegree >= 0.0)
        except Exception as e:
            print(weights[trg])
            raise e
        indegree_ratio.append(indegree / final_indegree)
        if indegree_gain is not None:
            try:
                assert np.isclose(indegree_ratio[-1], indegree_gain, rtol=1e-03, atol=1e-03)
            except Exception as e:
                print(indegree)
                print(final_indegree)
                raise e
    return np.abs(weights), indegree_ratio


def apply_pathway_gains(weights, inds, config):

    task_laterality = config.TASK_LATERALITY  # -1: contralatterally, 0: bilaterally, 1: ipsilaterally
    if config.VERBOSITY:
        print("\n" + "-" * 50)
        print("Applying pathway gain...")
        print("-" * 50 + "\n")

    indegree_ratios = {}

    # WHISKERS:
    if config.WHISKERS_GAIN > 1.0:
        weights[inds["whiskers"], inds["facial"]] = 1.0
        weights[inds["trigeminal"], inds["whiskers"]] = config.WHISKERS_GAIN
        if config.VERBOSITY:
            print("Whiskers -> Trigeminal weight!:\n%s" % str(weights[inds["trigeminal"], inds["whiskers"]]))
            print("-" * 50 + "\n")
    # A. INPUT CEREB PATHWAY:

    # 1. PosSens Trigeminal (Medulla) <- Trigeminal (stimulus)

    if config.TRIG_GAIN > 1.0:
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("trigeminal -> ponssens_trigeminal (Medulla)")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["trigeminal"],
                                         inds["ponssens_trigeminal"],
                                         config.TRIG_GAIN, weights,
                                         task_laterality=-task_laterality,  # ipsilaterally
                                         indegree_gain=1.0,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["ponssens_trigeminal"] = indegree_ratio

    # 2. AnsiLob <- PosSens Trigeminal (Medulla)
    if config.MEDULLA_GAIN > 1.0:
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("trigeminal ponssens_trigeminal (Medulla) -> ansilob")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["ponssens_trigeminal"],
                                         inds["ansilob"],
                                         config.MEDULLA_GAIN, weights,
                                         task_laterality=-task_laterality,  # ipsilaterally
                                         indegree_gain=1.0,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["ansilob"] = indegree_ratio

    if config.CEREB_GAIN > 1.0:
        # 5. Cereb nuclei <- AnsiLob
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("ansilob -> CerebNuclei")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["ansilob"],
                                         inds["cereb_nuclei"],
                                         config.CEREB_GAIN, weights,
                                         task_laterality=-task_laterality,  # ipsilaterally
                                         indegree_gain=1.0,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["cereb_nuclei"] = indegree_ratio

    # B. INPUT SENSORY PATHWAY:
    # 4. S1 brl thal <- [Trigeminal (stimulus), PonsSens Trigeminal, CerebNuclei]
    sources = []
    sourcenames = []
    pathway_gains = []
    for gain, source, name in zip([config.TRIGS1_GAIN, config.MEDULLAS1_GAIN, config.CNS1_GAIN],
                                  [inds["trigeminal"], inds["ponssens_trigeminal"], inds["cereb_nuclei"]],
                                  ["Trigeminal (stimulus)", "PonsSens Trigeminal (Medulla)", "CerebNuclei"]):
        if gain > 1.0:
            pathway_gains.append(gain)
            sources.append(source)
            sourcenames.append(name)
    if len(sources):
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("[%s] -> s1brlthal" % (", ".join(sourcenames)))
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(np.concatenate(sources),
                                         inds["s1brlthal"],
                                         pathway_gains, weights,  # 30.0
                                         task_laterality=task_laterality,  # contralaterally
                                         fix_inds=inds["s1brl"],
                                         indegree_gain=None,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["s1brlthal"] = indegree_ratio


    # C. INPUT MOTOR PATHWAY
    if config.CNM1_GAIN > 1.0:
        # 1.  M1 thal <- CerebNuclei
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("CerebNuclei -> m1thal")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["cereb_nuclei"],
                                         inds["m1thal"],
                                         config.CNM1_GAIN, weights,  # 30.0
                                         task_laterality=task_laterality,  # contralaterally
                                         fix_inds=inds["m1"],
                                         indegree_gain=None,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["m1thal"] = indegree_ratio

    # C. OUTPUT MOTOR PATHWAY
    if config.M1FACIAL_GAIN > 1.0:
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("M1 -> facial motor nucleus")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["m1"],
                                         inds["facial"],
                                         config.M1FACIAL_GAIN, weights,
                                         task_laterality=task_laterality,  # contralaterally
                                         indegree_gain=1.0,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["facial"] = indegree_ratio
    if config.FACIALTRIG_GAIN > 1.0:
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("facial motor nucleus -> trigeminal")
        weights, indegree_ratio = \
            apply_pathway_gain_to_target(inds["facial"],
                                         inds["trigeminal"],
                                         config.FACIALTRIG_GAIN, weights,
                                         task_laterality=-task_laterality,  # ipsilaterally
                                         indegree_gain=1.0,
                                         verbosity=config.VERBOSITY)
        indegree_ratios["facial"] = indegree_ratio

    # E. M1 <-> S1
    if config.M1S1_GAIN > 1.0:
        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("M1 -> S1")
        weights, indegree_ratio = apply_pathway_gain_to_target(
            inds["s1brl"], inds["m1"],
            1.0,
            weights, task_laterality=0,
            fix_inds=inds["m1thal"].tolist(),
            indegree_gain=config.M1S1_GAIN,
            verbosity=config.VERBOSITY
        )
        indegree_ratios["m1"] = indegree_ratio

        if config.VERBOSITY:
            print("-" * 25 + "\n")
            print("S1 -> M1")
        weights, indegree_ratio = apply_pathway_gain_to_target(
            inds["m1"], inds["s1brl"],
            1.0,
            weights, task_laterality=0,
            fix_inds=inds["s1brlthal"].tolist(),
            indegree_gain=config.M1S1_GAIN,
            verbosity=config.VERBOSITY
            )
        indegree_ratios["s1brl"] = indegree_ratio

    return weights, indegree_ratios


def adjust_ficed_params(simulator, indegree_ratios, inds, FIC_SPLIT, verbosity=1):

    def adjust_fic(weights, indegree_ratios, inds, fie, fwie, I_e, w_ie):  #  ...,
        for ind, indegree_ratio in zip(inds, indegree_ratios):
            if verbosity: print("region ind = %d" % ind)
            new_indegree = weights[ind].sum()
            if verbosity: print("new_indegree = %g" % new_indegree)
            if verbosity: print("indegree_ratio = %g" % indegree_ratio)
            indegree = new_indegree * indegree_ratio
            if fie is not None:
                if verbosity: print("indegree = %g" % indegree)
                if verbosity: print("I_e[%d]_old = %g" % (ind, I_e[ind]))
                I_e[ind] = fie(I_e[ind], indegree, new_indegree)
                if verbosity: print("I_e[%d]_adj = %g" % (ind, I_e[ind]))
            if fwie is not None:
                if verbosity: print("indegree = %g" % indegree)
                if verbosity: print("w_ie[%d]_old = %g" % (ind, w_ie[ind]))
                w_ie[ind] = fwie(w_ie[ind], indegree, new_indegree)
                if verbosity: print("w_ie[%d]_adj = %g" % (ind, w_ie[ind]))
        return I_e, w_ie

    if verbosity:
        print("-" * 25)
        print("-" * 25)
        print("Adjusting FICed parameters for (non thalamic) regions with modified indegree...")
    # Necessary functions for adjustment of FICed paremeters
    if FIC_SPLIT > 0.0:
        indmax_Ie = inds["crtx_and_subcrtx"][[np.argmax(simulator.model.I_e[inds["crtx_and_subcrtx"]])]][0].item()
        iemax = simulator.model.I_e[indmax_Ie]

        if verbosity:  print("Ie[%d]_max = %g" % (indmax_Ie, iemax))
        fie = lambda ie, indegree, new_indegree: np.minimum(iemax,
                                                            iemax + (new_indegree - indegree_min) * np.minimum(0.0, (
                                                                        ie - iemax) / (indegree - indegree_min)))
        indegree_min = simulator.connectivity.weights[indmax_Ie].sum()
    else:
        fie = None
    if FIC_SPLIT < 1.0:
        indmax_wie = inds["crtx_and_subcrtx"][[np.argmax(simulator.model.w_ie[inds["crtx_and_subcrtx"]])]][0].item()
        wiemax = simulator.model.w_ie[indmax_wie]
        if verbosity: print("w_ie[%d]_max = %g" % (indmax_wie, wiemax))
        #  i.e., ymax  + (     x       -     x0     ) *            (yold - ymax)  / (   xold  -    xmin)
        fwie = lambda wie, indegree, new_indegree: np.minimum(wiemax,
                                                              wiemax + (new_indegree - indegree_min) * np.minimum(0.0, (
                                                                          wie - wiemax) / (indegree - indegree_min)))
        indegree_min = simulator.connectivity.weights[indmax_wie].sum()
    else:
        fwie = None
    if FIC_SPLIT > 0.0 and FIC_SPLIT < 1.0:
        assert indmax_Ie == indmax_wie
    if verbosity: print("indegree_min = %g" % indegree_min)
    for reg, indratios in indegree_ratios.items():
        if "thal" not in reg and np.any(np.logical_not(np.isclose(indratios, 1.0, rtol=1e-03, atol=1e-03))):
            if verbosity: print("...adjusting regions %s..." % reg)
            simulator.model.I_e, simulator.model.w_ie = \
                adjust_fic(simulator.connectivity.weights, indratios, inds[reg],
                           fie, fwie, simulator.model.I_e, simulator.model.w_ie)
    return simulator


def print_weight_to_indegree(src, trg, inds, w, task_laterality=-1):
    print("\n" + "-"*25)
    print("%s -> %s" % (src, trg))
    print(w[inds[trg]][:, inds[src]])
    print("%:")
    print(w[inds[trg]][:, inds[src]] / w[inds[trg]].sum() * 200 / (2 - np.abs(task_laterality)))


def apply_pathway_gains_and_adjust_FIC(simulator, inds, config, plotter=None):

    simulator.connectivity.weights, indegree_ratios = apply_pathway_gains(simulator.connectivity.weights, inds, config)

    if config.VERBOSITY:
        print("-" * 25)
        print("Indegree ratios:")
        print(indegree_ratios)
        print("-" * 25)
    if config.FIC > 0.0:
        simulator = adjust_ficed_params(simulator, indegree_ratios, inds, config.FIC_SPLIT, config.VERBOSITY)

    if config.VERBOSITY:
        print("\n" + "-" * 50)
        print("Pathway connections to indegree %:")

        for conns in [
            ["s1brl", "m1"], ["m1", "s1brl"],
            ["s1brlthal", "s1brl"], ["s1brl", "s1brlthal"],
            ["m1thal", "m1"], ["m1", "m1thal"],
            ["trigeminal", "ponssens_trigeminal"],
            ["ponssens_trigeminal", "ansilob"],
            ["ansilob", "cereb_nuclei"],
            ["trigeminal", "s1brlthal"], ["ponssens_trigeminal", "s1brlthal"], ["cereb_nuclei", "s1brlthal"],
            ["cereb_nuclei", "m1thal"],
            ["m1", "facial"], ["facial", "trigeminal"]
        ]:
            print_weight_to_indegree(conns[0], conns[1], inds, simulator.connectivity.weights,
                                     task_laterality=config.TASK_LATERALITY)

    if plotter:
        # Plot task network:
        from rising_net.scripts.plot_utils import shorten_region_name, matrix_plot
        config.TASK_SHORT_REG_LABELS = [shorten_region_name(reg, exclude=["of", "the", "to"])
                                        for reg in simulator.connectivity.region_labels[config.TASKINDS]]
        fig, ax = plt.subplots()
        ax = matrix_plot(simulator.connectivity.weights[config.TASKINDS][:, config.TASKINDS].copy(),
                         labels=config.TASK_SHORT_REG_LABELS,
                         label="SC", ax=ax, colorbar=True, fontsize=10)
        fig.tight_layout()
        plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, "taskSC.png"), format="png")

    return simulator


def distribute_pathway_gain(config):
    if config.VERBOSITY:
        print("\n")
        print("-"*50)
        print("-"*50)
        print("Distributing pathway gains with config.PATHWAY_GAIN = %g:" % config.PATHWAY_GAIN)
        print("-" * 50)
    # Main pathway gets PATHWAY_GAIN
    for gain in ["M1FACIAL_GAIN"]:
        setattr(config, gain, 1.5*float(config.PATHWAY_GAIN))
        if config.VERBOSITY:
            print("config.%s = %g" % (gain, getattr(config, gain)))
    for gain in ["WHISKERS_GAIN",
                 "TRIG_GAIN", "MEDULLA_GAIN",
                 "CEREB_GAIN"]:
        setattr(config, gain, float(config.PATHWAY_GAIN))
        if config.VERBOSITY:
            print("config.%s = %g" % (gain, getattr(config, gain)))
    for gain in ["CNM1_GAIN", "CNS1_GAIN"]:
        setattr(config, gain, float(config.PATHWAY_GAIN)/3)
        if config.VERBOSITY:
            print("config.%s = %g" % (gain, getattr(config, gain)))
    # All other connections get 1.0
    for gain in ["TRIGS1_GAIN", "MEDULLAS1_GAIN",
                 "FACIALTRIG_GAIN",
                 "M1S1_GAIN"]:
        setattr(config, gain, 1.0)
        if config.VERBOSITY:
            print("config.%s = %g" % (gain, getattr(config, gain)))
    config.PATHWAY_GAIN = 1.0
    if config.VERBOSITY:
        print("-" * 50)
        print("Resetting now config.PATHWAY_GAIN = %g:" % config.PATHWAY_GAIN)
        print("-" * 50)
        print("-" * 50)
        print("\n")
    return config


def build_simulator(connectivity, model, inds, maps, config, plotter=None):
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import SigmoidalPreThalamoCortical
    from tvb.simulator.monitors import Raw, Bold, TemporalAverage, AfferentCoupling, AfferentCouplingTemporalAverage

    simulator = CoSimulatorSerial()

    simulator.model = model
    simulator.connectivity = connectivity

    dummy = np.ones((simulator.connectivity.number_of_regions,))

    # Variability to thalamocortical connections:
    # if config.THAL_CRTX_FIX:
    #     if "d" in config.THAL_CRTX_FIX:
    if config.VERBOSITY:
        print("Fixing thalamocortical delays!")
    # Fix structural connectivity (specific) thalamo-cortical tracts length to a value,
    # such that all thalamo-cortical delays are equal to the parameter tau_ct,
    # given connectivity's speed.
    ct_lengths = simulator.connectivity.speed * \
                 simulator.model.tau_ct * dummy[inds["crtx"]]
    simulator.connectivity.tract_lengths[inds["crtx"], inds["thalspec"]] = ct_lengths
    simulator.connectivity.tract_lengths[inds["thalspec"], inds["crtx"]] = ct_lengths
    simulator.connectivity.configure()

    # if not config.THAL_CRTX_FIX or "d" not in config.THAL_CRTX_FIX:
    #     tau_ct = simulator.model.tau_ct * dummy
    #     tau_ct[inds['crtx']] = simulator.connectivity.delays[inds["thalspec"], inds["crtx"]]
    #     tau_ct[inds['thalspec']] = simulator.connectivity.delays[inds["crtx"], inds["thalspec"]]
    #     simulator.model.tau_ct = tau_ct

    # Set the sigmoidal coupling function:
    simulator.coupling = SigmoidalPreThalamoCortical(
        is_thalamic=maps['is_thalamic'],
        is_subcortical=np.logical_or(maps['is_subcortical'], maps['is_whiskers']),
        sigma=np.array([1.0]),
        midpoint=simulator.model.sigma,
        cmin=np.array([0.0]),
        cmax=np.array([1.0]),
        a=simulator.model.beta)

    # Set integrator and noise
    simulator.integrator = EulerStochastic()
    simulator.integrator.dt = config.DEFAULT_DT
    simulator.integrator.noise.nsig = np.array(
        [config.DEFAULT_NSIG] * (simulator.model.nvar - 1) + [0.0])  # No Noise for state variabla A for BOLD monitor
    simulator.integrator.noise.noise_seed = config.DEFAULT_TVB_NOISE_SEED

    # Set initial conditions around baseline currents of each kind of population for a shorter transient:
    simulator.initial_conditions = np.zeros((1000, simulator.model.nvar, connectivity.number_of_regions, 1))
    n_crtx_subcrtx = len(inds['crtx_and_subcrtx'])
    simulator.initial_conditions[:, [[0]], inds['crtx_and_subcrtx']] =\
        simulator.model.I_e.mean().item()*(1.0 + np.random.normal(size=(1000, 1, n_crtx_subcrtx, 1)))
    simulator.initial_conditions[:, [[1]], inds['crtx_and_subcrtx']] = \
        simulator.model.I_i.mean().item()*(1.0 + np.random.normal(size=(1000, 1, n_crtx_subcrtx, 1)))
    n_thalspec = len(inds['thalspec'])
    simulator.initial_conditions[:, [[0]], inds['thalspec']] = \
        simulator.model.I_s.mean().item() * (1.0 + np.random.normal(size=(1000, 1, n_thalspec, 1)))
    simulator.initial_conditions[:, [[1]], inds['thalspec']] = \
        simulator.model.I_r.mean().item() * (1.0 + np.random.normal(size=(1000, 1, n_thalspec, 1)))
    if config.WHISKERS:
        simulator.initial_conditions[:, :, inds['whiskers']] = 0.0

    # Apply FIC if required:
    if config.FIC and simulator.model.G[0].item():
        simulator = apply_fic(simulator, inds, config, plotter)

    # Apply pathway gain and adjust FIC for changed indegrees:
    if config.PATHWAY_GAIN >= 1:
        if config.PATHWAY_GAIN > 2.0:
            config = distribute_pathway_gain(config)
        simulator = apply_pathway_gains_and_adjust_FIC(simulator, inds, config, plotter)

    # for regs in ["facial", "trigeminal", "medulla", "ansilob"]:  # , "cereb_nuclei"
    #     for p, pval in zip(["I_e", "tau_e", "tau_i"],
    #                        [-0.15, 10.0/0.9, 100.0/0.9]):
    #         pvec = getattr(simulator.model, p) * np.ones((simulator.connectivity.number_of_regions, 1))
    #         if p == "I_e":
    #             pvec = pvec[:, 0]
    #         pvec[inds[regs]] = pval
    #         setattr(simulator.model, p, pvec)
    #         print("\n" + "-" * 50)
    #         print("%s %s = " % (regs, p), getattr(simulator.model, p)[inds[regs]].mean())
    #         print("-" * 50 + "\n")
    #
    # simulator.model.I_e[inds["cereb_nuclei"]] = -0.7
    # simulator.model.tau_e[inds["cereb_nuclei"]] = 50.0/0.9

    # Set monitors:
    monitors = ()
    if config.TIME_SERIES_MONITORS:
        if config.RAW_PERIOD > config.DEFAULT_DT:
            monitors += (TemporalAverage(period=config.RAW_PERIOD), )  # ms
            if config.AFFERENT_MONITOR:
                monitors += (AfferentCouplingTemporalAverage(period=config.RAW_PERIOD,
                                                             variables_of_interest=np.array([0, 1, 2])), )
        else:
            monitors += (Raw(), )
            if config.AFFERENT_MONITOR:
                monitors += (AfferentCoupling(variables_of_interest=np.array([0, 1, 2])), )
    if config.BOLD_PERIOD:
        monitors += (Bold(period=config.BOLD_PERIOD, variables_of_interest=np.array([2])), )
    simulator.monitors = monitors

    simulator.configure()

    simulator.integrate_next_step = simulator.integrator.integrate_with_update

    if config.VERBOSITY > 1:
        simulator.print_summary_info_details(recursive=config.VERBOSITY)

    # Serializing TVB cosimulator is necessary for parallel cosimulation:
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = serialize_tvb_cosimulator(simulator)

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    # if plotter:
    #     # Plot TVB connectome:
    #     plotter.plot_tvb_connectivity(simulator.connectivity);

    return simulator


def configure_simulation_length_with_transient(config):
    # Compute transient as a percentage of the total simulation length, and add it to the simulation length:
    simulation_length = float(config.SIMULATION_LENGTH)
    transient = config.TRANSIENT_RATIO * simulation_length
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2
    simulation_length += transient
    return simulation_length, transient


def simulate(simulator, config):
    simulator.simulation_length, transient = configure_simulation_length_with_transient(config)
    # Simulate and return results
    tic = time.time()
    results = simulator.run()
    if config.VERBOSITY:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    return results, transient


def compute_target_PSDs(config):
    # Load Popa 2013 files:
    psd_m1 = np.load(os.path.join(config.TARGET_POPA_PATH, "PSD_M1.npy"))
    psd_s1 = np.load(os.path.join(config.TARGET_POPA_PATH, "PSD_S1.npy"))

    # Interpolate to the desired frequency range:
    psd_m1_target = np.interp(config.TARGET_FREQS, psd_m1[:, 0], psd_m1[:, 1])
    psd_s1_target = np.interp(config.TARGET_FREQS, psd_s1[:, 0], psd_s1[:, 1])

    # Normalize to generate a PSD:
    psd_m1_target = psd_m1_target / psd_m1_target.sum()
    psd_s1_target = psd_s1_target / psd_s1_target.sum()

    return psd_m1_target, psd_s1_target


def compute_target_PSDs_1D(config, write_files=True, plotter=None):
    # Load, interpolate and normalize Popa 2013 m1 and s1 power spectra:
    psd_m1_target, psd_s1_target = compute_target_PSDs(config)

    psd_target = (psd_m1_target + psd_s1_target)/2

    PSD_target = {"f": config.TARGET_FREQS, "PSD_target": psd_target}
    if write_files:
        np.save(config.PSD_TARGET_PATH, PSD_target)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(config.TARGET_FREQS, psd_target, "k")
        axes[0].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[0].grid(True, axis="x")
        axes[0].set_ylabel('PS')
        axes[0].set_title('Target average of M1 and S1 PS')
        axes[1].semilogy(config.TARGET_FREQS, psd_target, "k")
        axes[1].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[1].grid(True, axis="x")
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "TargetPSD1D.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig) 
    return PSD_target


def compute_target_PSDs_m1s1brl(config, write_files=True, plotter=None):
    # Load, interpolate and normalize Popa 2013 m1 and s1 power spectra:
    psd_m1_target, psd_s1_target = compute_target_PSDs(config)

    PSD_target = {"f": config.TARGET_FREQS, "PSD_M1_target": psd_m1_target, "PSD_S1_target": psd_s1_target}
    if write_files:
        np.save(config.PSD_TARGET_PATH, PSD_target)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(config.TARGET_FREQS, psd_m1_target, "b", label='M1')
        axes[0].plot(config.TARGET_FREQS, psd_s1_target, "g", label='S1')
        axes[0].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[0].grid(True, axis="x")
        axes[0].set_ylabel('PS')
        axes[0].set_title('Target M1 and S1 PS')
        axes[0].legend()
        axes[1].semilogy(config.TARGET_FREQS, psd_m1_target, "b", label='M1')
        axes[1].semilogy(config.TARGET_FREQS, psd_s1_target, "g", label='S1')
        axes[1].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[1].grid(True, axis="x")
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "TargetPSDm1s1brl.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return PSD_target


def compute_data_PSDs_1D(raw_results, PSD_target, inds,
                         transient=None, write_files=True, psd_data_path='./', plotter=None):

    # Select regions' data, compute PSDs, average them across region,
    # interpolate them to the target frequencies, and normalize them to sum up to 1.0:
    ftarg = PSD_target['f']
    Pxx_den = compute_data_PSDs_from_raw(raw_results, ftarg, inds['crtx'],
                                         transient=transient, average_region_ps=True)
    Pxx_den = Pxx_den.flatten()
    if write_files:
        np.save(psd_data_path, Pxx_den)
    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(ftarg, PSD_target['PSD_target'], "k", label='Target')
        axes[0].plot(ftarg, Pxx_den, "r", label='Cortical average')
        axes[0].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[0].grid(True, axis="x")
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(ftarg, PSD_target['PSD_target'], "k", label='Target')
        axes[1].semilogy(ftarg, Pxx_den, "r", label='Cortical average')
        axes[1].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[1].grid(True, axis="x")
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "DataVSTargetPSD1D.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return Pxx_den


def compute_data_PSDs_m1s1brl(raw_results, PSD_target, inds,
                              transient=None, write_files=True, psd_data_path='./', plotter=None):

    # Select regions' data, compute PSDs, interpolate them to the target frequencies, 
    # and normalize them to sum up to 1.0:
    ftarg = PSD_target['f']
    Pxx_den = compute_data_PSDs_from_raw(raw_results, ftarg, inds['m1s1brl'],
                                         transient=transient, average_region_ps=False)
    if write_files:
        np.save(psd_data_path, Pxx_den)
    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 target')
        axes[0].plot(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 target')
        axes[0].plot(ftarg, Pxx_den[0], "b--", label='M1 right')
        axes[0].plot(ftarg, Pxx_den[1], "b-.", label='M1 left')
        axes[0].plot(ftarg, Pxx_den[2], "g--", label='S1 right')
        axes[0].plot(ftarg, Pxx_den[3], "g-.", label='S1 left')
        axes[0].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[0].grid(True, axis="x")
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 target')
        axes[1].semilogy(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 target')
        axes[1].semilogy(ftarg, Pxx_den[0], "b--", label='M1 right')
        axes[1].semilogy(ftarg, Pxx_den[1], "b-.", label='M1 left')
        axes[1].semilogy(ftarg, Pxx_den[2], "g--", label='S1 right')
        axes[1].semilogy(ftarg, Pxx_den[3], "g-.", label='S1 left')
        axes[1].set_xticks([6.0, 8.0, 10.0, 25.0, 35.0, 45.0])
        axes[1].grid(True, axis="x")
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "DataVSTargetPSDm1s1brl.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)

    return Pxx_den.flatten()


def tvb_res_to_bold_time_series(results, simulator, config, write_files=True):
    config = assert_config(config, return_plotter=False)
    bold_ts = None
    outputs = {}
    try:
        bold_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
            data=results[1], time=results[0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": ["BOLD"],
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.monitors[-1].period)
        bold_ts.configure()

        if config.VERBOSITY > 1:
            print("BOLD ts:\n%s" % str(bold_ts))

        outputs["bold_ts"] = bold_ts

    except Exception as e:
        outputs["bold_ts"] = results
        warnings.warn("Failed to construct BOLD time series with error!:\n%s" % str(e))
        if write_files:
            if config.VERBOSITY:
                print("Pickle-dumping BOLD TVB monitor output to %s!" % config.BOLD_TS_PATH)
            try:
                dump_pickled_dict({"bold_ts": results[1],
                                   "bold_t": results[0],
                                   "regions": simulator.connectivity.region_labels},
                                  config.BOLD_TS_PATH)
            except Exception as e:
                warnings.warn("Failed to pickle dump BOLD TVB monitor output with error!:\n%s" % str(e))

    if bold_ts is not None:
        if write_files:
            if config.VERBOSITY:
                print("Pickle-dumping bold_ts to %s!" % config.BOLD_TS_PATH)
            dump_pickled_time_series(bold_ts, config.BOLD_TS_PATH)

            # # Write to file
            # if writer:
            #     try:
            #         write_RegionTimeSeriesXarray_to_h5(bold_ts, writer,
            #                                            os.path.join(config.out.FOLDER_RES,
            #                                                         bold_ts.title) + ".h5")
            #     except Exception as e:
            #         warnings.warn("Failed to to write BOLD time series to file with error!:\n%s" % str(e))

    return outputs


def tvb_res_to_time_series(results, simulator, config=None, write_files=True):

    config = assert_config(config, return_plotter=False)

    # writer = False
    # if write_files:
    #     # If you want to see what the function above does, take the steps, one by one
    #     try:
    #         # We need framework_tvb for writing and reading from HDF5 files
    #         from tvb_multiscale.core.tvb.io.h5_writer import H5Writer
    #         from examples.plot_write_results import write_RegionTimeSeriesXarray_to_h5
    #         writer = H5Writer()
    #     except:
    #         warnings.warn("H5Writer cannot be imported! Probably you haven't installed tvb_framework.")

    source_ts = None
    bold_ts = None
    afferent_ts = None

    outputs = {}
    if results is not None:
        source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion for TVB like functionality
            data=results[0][1], time=results[0][0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.monitors[0].period)

        source_ts.configure()
        outputs["source_ts"] = source_ts

        if config.AFFERENT_MONITOR:
            afferent_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
                data=results[1][1], time=results[1][0],
                connectivity=simulator.connectivity,
                labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                labels_dimensions={"State Variable": ["cortical coupling", "subcortical coupling", "thalamic coupling"],
                                   "Region": simulator.connectivity.region_labels.tolist()},
                sample_period=simulator.monitors[1].period)

            afferent_ts.configure()
            outputs["afferent_ts"] = afferent_ts

        if write_files:
            if source_ts is not None:
                if config.VERBOSITY:
                    print("Pickle-dumping source_ts to %s!" % config.SOURCE_TS_PATH)
                dump_pickled_time_series(source_ts, config.SOURCE_TS_PATH)
            if config.AFFERENT_MONITOR:
                if config.VERBOSITY:
                    print("Pickle-dumping afferent_ts to %s!" % config.AFFERENT_TS_PATH)
                dump_pickled_time_series(afferent_ts, config.AFFERENT_TS_PATH)

            # # Write to file
            # if writer:
            #     try:
            #         write_RegionTimeSeriesXarray_to_h5(source_ts, writer,
            #                                            os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
            #     except Exception as e:
            #             warnings.warn("Failed to to write source time series to file with error!:\n%s" % str(e))
            #
            # if config.VERBOSITY > 1:
            #     print("Raw ts:\n%s" % str(source_ts))

        if len(results) > 2:
            outputs.update(
                tvb_res_to_bold_time_series(results[2], simulator, config, write_files=write_files))

    return outputs


def compute_PSD_target_and_data(config, results, inds, transient, plotter=None, write_files=True):
    # This is the PSD target we are trying to fit...
    if config.model_params['G']:
        # ...for a connected brain, i.e., PS of bilateral M1 and S1:
        PSD_target = compute_target_PSDs_m1s1brl(config, write_files=write_files, plotter=plotter)
        # ...for a connected brain, i.e., PS of bilateral M1 and S1:
        PSD = compute_data_PSDs_m1s1brl(results, PSD_target, inds, transient,
                                        write_files=write_files, psd_data_path=config.PSD_DATA_PATH, plotter=plotter)
    else:
        # ...for a disconnected brain, average PS of all regions:
        PSD_target = compute_target_PSDs_1D(config, write_files=write_files, plotter=plotter)
        # ...for a disconnected brain, average PS of all regions:
        PSD = compute_data_PSDs_1D(results, PSD_target, inds, transient,
                                   write_files=write_files, psd_data_path=config.PSD_DATA_PATH, plotter=plotter)
    return PSD, PSD_target


def plot_tvb(transient, inds, results, simulator=None, plotter=None, config=None, write_files=True):
    from rising_net.scripts.utils import \
        compute_plot_selected_spectra_coherence  # , compute_plot_ica

    if plotter is None:
        config, plotter = assert_config(config, return_plotter=True)
    else:
        config = assert_config(config, return_plotter=False)
    MAX_VARS_IN_COLS = 2
    MAX_VARS_IN_COLS_AFF = 3
    MAX_REGIONS_IN_ROWS = 10
    MIN_REGIONS_FOR_RASTER_PLOT = 9
    FIGSIZE = config.figures.DEFAULT_SIZE

    PSD, PSD_target = compute_PSD_target_and_data(config, results[0],  inds, transient,
                                                  write_files=write_files, plotter=plotter)

    if isinstance(results, (list, tuple)):
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
    results["PSD_target"] = PSD_target
    results["PSD"] = PSD
    results["f"] = PSD_target["f"]

    source_ts = results.get("source_ts", None)
    bold_ts = results.get("bold_ts", None)
    afferent_ts = results.get("afferent_ts", None)

    if isinstance(source_ts, TimeSeriesXarray):
        dt = source_ts.time[1] - source_ts.time[0]
        n_time_len = source_ts.shape[0]
    elif simulator is not None:
        dt = simulator.integrator.dt
        n_time_len = int(simulator.simulation_length / dt)
    else:
        dt = config.DEFAULT_DT
    TIME_PLOT = np.minimum(1000.0,  np.maximum(0.0, n_time_len * dt - 100.0))  # ms
    N_TIME_PLOT = int(np.round(TIME_PLOT / dt))
    n_time_len -= int(np.round(transient / dt))

    # TaskMetrics = compute_task_transfer_metrics(source_ts, transient, simulator.connectivity.region_labels,
    #                                             config.TASKINDS, config.THETA, config.GAMMA, config.FREQS,
    #                                             methods=(5, 2, 3), plot_flag=True,
    #                                             figpath=config.figures.FOLDER_FIGURES)
    # dump_pickled_dict(TaskMetrics.to_dict(), config.TASK_TRANSFER_METRICS_PATH)
    # results["TaskMetrics"] = TaskMetrics

    # Plot TVB time series
    if isinstance(source_ts, TimeSeriesXarray):
        source_ts[:, :, :, :].plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if source_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                                              figsize=FIGSIZE)
        # Focus on the m1 and s1 barrel field nodes:
        source_ts_m1s1brl = source_ts[-N_TIME_PLOT:, :, inds["m1s1brl"]]
        source_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                          hue="Region" if source_ts_m1s1brl.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                          per_variable=source_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                          figsize=FIGSIZE, figname="M1 and S1 barrel field nodes TVB Time Series")

        # Focus on the motor pathway:
        if len(inds.get("motor", [])):
            source_ts_motor = source_ts[-N_TIME_PLOT:, :, inds["motor"]]
            source_ts_motor.plot_timeseries(plotter_config=plotter.config,
                                            hue="Region" if source_ts_motor.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                            per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                            figsize=FIGSIZE, figname="Motor pathway TVB Time Series")
        if config.WHISKERS:
            # Focus on the m1 and s1 barrel field nodes:
            source_ts_w = source_ts[-N_TIME_PLOT:, 0, inds["whiskers"]]
            source_ts_w.plot_timeseries(plotter_config=plotter.config,
                                        hue="Region" if source_ts_w.shape[
                                                            2] > MAX_REGIONS_IN_ROWS else None,
                                        per_variable=source_ts_w.shape[1] > MAX_VARS_IN_COLS,
                                        figsize=FIGSIZE, figname="Whiskers' TVB Time Series")
        # Focus on the sensory pathway:
        if len(inds.get("sens", [])):
            source_ts_sens = source_ts[-N_TIME_PLOT:, :, inds["sens"]]
            source_ts_sens.plot_timeseries(plotter_config=plotter.config,
                                           hue="Region" if source_ts_sens.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                           per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                           figsize=FIGSIZE, figname="Sensory pathway TVB Time Series")
        if len(inds.get("cereb", [])):
            # Focus on regions potentially modelled in NEST (ansiform lobule, Cerebellar Nuclei, inferior olive):
            source_ts_cereb = source_ts[-N_TIME_PLOT:, :, inds["cereb"]]
            source_ts_cereb.plot_timeseries(plotter_config=plotter.config,
                                            hue="Region" if source_ts_cereb.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                            per_variable=source_ts_cereb.shape[1] > MAX_VARS_IN_COLS,
                                            figsize=FIGSIZE, figname="Cerebellum TVB Time Series")

        # Power Spectra and Coherence for M1 - S1 barrel field
        Pxx_den, f, CxyR, fR, CxyL, fL = \
            compute_plot_selected_spectra_coherence(source_ts, inds["m1s1brl"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES,
                                                    figname="M1_S1brl", figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)
        results["PSD_M1_S1"] = Pxx_den
        results["PSD_f"] = f
        results["CxyR_M1_S1"] = CxyR
        results["fR"] = fR
        results["Cxyl_M1_S1"] = CxyL
        results["fL"] = fL

        if write_files:
            import pickle
            with open('coherence_MF_cerebON_2sec.pickle', 'wb') as handle:
                pickle.dump([CxyR, fR, fL, CxyL], handle)

        # Power Spectra and Coherence along the motor pathway:
        if len(inds.get("motor", [])):
            compute_plot_selected_spectra_coherence(source_ts, inds["motor"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES,
                                                    figname="Motor", figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)

        if config.WHISKERS:
            compute_plot_selected_spectra_coherence(source_ts, inds["whiskers"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES,
                                                    figname="Whiskers", figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)

        # Power Spectra and Coherence along the sensory pathway:
        # for Medulla SPV, Sensory PONS
        if len(inds.get("sens", [])):
            compute_plot_selected_spectra_coherence(source_ts, inds["sens"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES,
                                                    figname="SPV_PonsSens", figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)

        if len(inds.get("cereb", [])):
            compute_plot_selected_spectra_coherence(source_ts, inds["cereb"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES, figname="Cereb",
                                                    figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)
        elif len(inds.get("ansilob", [])):
            print("psd input cereb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Power Spectra and Coherence at cerebellar input - ansiform lobule:
            print("inds ansilob", inds["ansilob"])
            print("Ansiform lobule source_ts PSD, with compute_plot_selected_spectra_coherence")
            compute_plot_selected_spectra_coherence(source_ts, inds["ansilob"],
                                                    transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                                                    figures_path=config.figures.FOLDER_FIGURES, figname="AnsiLob",
                                                    figformat="png",
                                                    show_flag=plotter.config.SHOW_FLAG,
                                                    save_flag=plotter.config.SAVE_FLAG)

        # Better summary figure:

        data = source_ts.data
        time = source_ts.time

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for iT, regs in enumerate(["crtx", "subcrtx_not_thalspec", "thalspec"]):
            # transient_in_points = int((transient + 0.5) / simulator.monitors[0].period)
            dat = data[-N_TIME_PLOT:, 0, inds[regs]].squeeze()
            axes[iT].plot(time[-N_TIME_PLOT:], dat, alpha=0.25)
            if iT == 0:
                axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["m1"]].squeeze(),
                              'b--', linewidth=3, label='M1')
                axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["s1brl"]].squeeze(),
                              'g--', linewidth=3, label='S1 barrel field')
            elif iT == 1:
                if len(inds.get("facial", [])):
                    axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["facial"]].squeeze(),
                                  'b--', linewidth=3, label='Facial motor nucleus')
                if len(inds.get("trigeminal", [])):
                    axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["trigeminal"]].squeeze(),
                                  'g--', linewidth=3, label='Spinal trigeminal nuclei')
            else:
                axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["m1thal"]].squeeze(),
                              'b--', linewidth=3, label='M1 specific thalami')
                axes[iT].plot(time[-N_TIME_PLOT:], data[-N_TIME_PLOT:, 0, inds["s1brlthal"]].squeeze(),
                              'g--', linewidth=3, label='S1 barrel field specific thalami')
                axes[iT].set_xlabel('Time (ms)')
            axes[iT].plot(time[-N_TIME_PLOT:], dat.mean(axis=1), 'k--', linewidth=3, label='Total mean')
            axes[iT].legend()
            axes[iT].set_title("%s range=[%g, %g, %g, %g, %g] " %
                               (regs, dat.min(), np.percentile(dat, 5), dat.mean(), np.percentile(dat, 95), dat.max()))
        fig.tight_layout()
        if config.figures.SAVE_FLAG:
            plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, "SummaryTimeSeries." + config.figures.FIG_FORMAT))
        if config.figures.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)

    # Focus on the s1 barrel field nodes:
    if isinstance(afferent_ts, TimeSeriesXarray):
        afferent_ts_m1s1brl = afferent_ts[-N_TIME_PLOT:, :, inds["m1s1brl"]]
        afferent_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                            hue="Region" if afferent_ts_m1s1brl.shape[
                                                                2] > MAX_REGIONS_IN_ROWS else None,
                                            per_variable=afferent_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS_AFF,
                                            figsize=FIGSIZE, figname="M1 and S1 barrel Afferent TVB Time Series")
        afferent_ts_m1s1brlthal = afferent_ts[-N_TIME_PLOT:, :, np.concatenate([inds["m1thal"], inds["s1brlthal"]])]
        afferent_ts_m1s1brlthal.plot_timeseries(
            plotter_config=plotter.config,
            hue="Region" if afferent_ts_m1s1brlthal.shape[2] > MAX_REGIONS_IN_ROWS else None,
            per_variable=afferent_ts_m1s1brlthal.shape[1] > MAX_VARS_IN_COLS_AFF,
            figsize=FIGSIZE, figname="M1 and S1 barrel specific thalami Afferent TVB Time Series")

        if len(inds.get("facial", [])):
            afferent_ts_facial = afferent_ts[-N_TIME_PLOT:, :, inds["facial"]]
            afferent_ts_facial.plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if afferent_ts_facial.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=afferent_ts_facial.shape[1] > MAX_VARS_IN_COLS_AFF,
                                              figsize=FIGSIZE, figname="`Facial motor nucleus TVB Afferent Time Series")

        if config.WHISKERS:
            afferent_ts_w = afferent_ts[-N_TIME_PLOT:, :, inds["whiskers"]]
            afferent_ts_w.plot_timeseries(plotter_config=plotter.config,
                                          hue="Region" if afferent_ts_w.shape[
                                                                   2] > MAX_REGIONS_IN_ROWS else None,
                                          per_variable=afferent_ts_w.shape[1] > MAX_VARS_IN_COLS_AFF,
                                          figsize=FIGSIZE, figname="`Whiskers TVB Afferent Time Series")

        if len(inds.get("trigeminal", [])):
            afferent_ts_trig = afferent_ts[-N_TIME_PLOT:, :, inds["trigeminal"]]
            afferent_ts_trig.plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if afferent_ts_trig.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=afferent_ts_trig.shape[1] > MAX_VARS_IN_COLS_AFF,
                                              figsize=FIGSIZE, figname="`Trigeminal TVB Afferent Time Series")

            if len(inds.get("ponssens_trigeminal", [])):
                afferent_ts_senstrig = afferent_ts[-N_TIME_PLOT:, :, inds["ponssens_trigeminal"]]
                afferent_ts_senstrig.plot_timeseries(
                    plotter_config=plotter.config,
                    hue="Region" if afferent_ts_senstrig.shape[2] > MAX_REGIONS_IN_ROWS else None,
                    per_variable=afferent_ts_senstrig.shape[1] > MAX_VARS_IN_COLS_AFF,
                    figsize=FIGSIZE, figname="Princ. Sens. Trigeminal TVB Afferent Time Series")

       # Focus on regions potentially modelled in NEST (ansiform lobule, interposed nucleus, inferior olive):
        if len(inds.get("ansilob", [])):
            afferent_ts_cereb = afferent_ts[-N_TIME_PLOT:, :, inds["ansilob"]]
            afferent_ts_cereb.plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if afferent_ts_cereb.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=afferent_ts_cereb.shape[1] > MAX_VARS_IN_COLS_AFF,
                                              figsize=FIGSIZE, figname="Ansiform Lobule TVB Afferent Time Series")

            if len(inds.get("cereb_nuclei", [])):
                afferent_ts_cn = afferent_ts[-N_TIME_PLOT:, :, inds["cereb_nuclei"]]
                afferent_ts_cn.plot_timeseries(
                    plotter_config=plotter.config,
                    hue="Region" if afferent_ts_cn.shape[2] > MAX_REGIONS_IN_ROWS else None,
                    per_variable=afferent_ts_cn.shape[1] > MAX_VARS_IN_COLS_AFF,
                    figsize=FIGSIZE, figname="Cerebellar Nuclei TVB Afferent Time Series")

            try:
                # Power Spectra and Coherence of cerebellar input - afferent to ansiform lobule:
                print("Ansiform lobule afferent PSD, with compute_plot_selected_spectra_coherence")
                Pxx_den_ansilob = []
                f_ansilob = []
                for iC, coupl in enumerate(["cortical", "subcortical"]):
                    print("%s coupling:" % coupl)
                    Pxx_den_ansilob_temp, f_ansilob, CxyR_ansilob, fR_ansilob, CxyL_ansilob, fL_ansilob = \
                        compute_plot_selected_spectra_coherence(
                            afferent_ts[:, iC], inds["ansilob"],
                            transient=transient, nperseg=None, fmin=0.0, fmax=100.0,
                            figures_path=config.figures.FOLDER_FIGURES, figname="AnsiLob %s afferent" % coupl,
                            figformat="png", show_flag=plotter.config.SHOW_FLAG, save_flag=plotter.config.SAVE_FLAG)
                    Pxx_den_ansilob.append(Pxx_den_ansilob_temp)
                results["PSD_ansilob"] = Pxx_den_ansilob
                results["PSD_ansilob_f"] = f_ansilob
                # results["CxyR_M1_S1"] = CxyR_ansilob
                # results["fR"] = fR_ansilob
                # results["Cxyl_M1_S1"] = CxyL_ansilob
                # results["fL"] = fL_ansilob
            except Exception as e:
                warnings.warn(str(e))

    # bold_ts TVB time series
    if isinstance(bold_ts, TimeSeriesXarray):
        bold_ts.plot_timeseries(plotter_config=plotter.config,
                                hue="Region" if bold_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                per_variable=bold_ts.shape[1] > MAX_VARS_IN_COLS,
                                figsize=FIGSIZE)

    return results


# def ansilob_affrerent_coupling_psd_rmse(ref_mossy_firing, afferent_ts, ftarg=None, transient=None):
#     if ftarg is None:
#         # TODO: confirm that we like this ftarg!
#         ftarg = np.arange(2.0, 51.0, 1.0)
#     # Adding the time vector to ref_mossy_firing - for a sim duration of 10s and 2.5-ms time bins
#     # TODO: Confirm that dt = 5.0 ms!
#     Pxx_den_ref = compute_data_PSDs(ref_mossy_firing, 5.0, ftarg,
#                                     transient=None, average_region_ps=False)
#     # First sum up the (non)isocortical afferent couplings!
#     #                                       iscortical                        non-isocortical
#     total_afferent_ts_ansilob = afferent_ts[1][:, 0, inds["ansilob"]] + afferent_ts[1][:, 1, inds["ansilob"]]
#     Pxx_den_ansilob = compute_data_PSDs(total_afferent_ts_ansilob.squeeze(),
#                                         np.mean(np.diff(afferent_ts[0])), ftarg,
#                                         transient=transient, average_region_ps=False)
#     MSE = np.square(np.subtract(Pxx_den_ansilob, Pxx_den_ref)).mean()
#     RMSE = math.sqrt(MSE)
#     print("RMSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE with pathway gain = ", pathway_gain,
#           " is ", RMSE)
#     return RMSE


def run_workflow(PSD_target=None, model_params={}, config=None, write_files=True, **config_args):
    tic = time.time()
    # Get configuration
    plot_flag = config_args.get('plot_flag', DEFAULT_ARGS.get('plot_flag'))
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    if config.VERBOSITY:
        print("\n\n------------------------------------------------\n\n"+
              "Running TVB workflow for plot_flag=%s, write_files=%s,\nand model_params=\n%s...\n" 
              % (str(plot_flag), str(write_files), str(config.model_params)))
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps, config = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

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

    # Run simulation and get results
    results, transient = simulate(simulator, config)

    if plotter is not None:
        results = plot_tvb(transient, inds, results, simulator=simulator, plotter=plotter,
                           config=config, write_files=write_files)
    else:
        if PSD_target is None:
            # This is the PSD target we are trying to fit...
            if config.model_params['G']:
                # ...for a connected brain, i.e., PS of bilateral M1 and S1:
                PSD_target = compute_target_PSDs_m1s1brl(config, write_files=write_files, plotter=plotter)
            else:
                # ...for a disconnected brain, average PS of all regions:
                PSD_target = compute_target_PSDs_1D(config, write_files=write_files, plotter=plotter)
            # This is the PSD computed from our simulation results...
        if config.model_params['G']:
            # ...for a connected brain, i.e., PS of bilateral M1 and S1:
            PSD = compute_data_PSDs_m1s1brl(results[0], PSD_target, inds, transient,
                                            write_files=write_files, psd_data_path=config.PSD_DATA_PATH,
                                            plotter=plotter)
        else:
            # ...for a disconnected brain, average PS of all regions:
            PSD = compute_data_PSDs_1D(results[0], PSD_target, inds, transient,
                                       write_files=write_files, psd_data_path=config.PSD_DATA_PATH, plotter=plotter)
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
        results.update({"PSD": PSD, "PSD_target": PSD_target})
    results.update({"transient": transient, "simulator": simulator, "inds": inds, "config": config})
    if config.VERBOSITY:
        print("\nFinished TVB workflow in %g sec!\n" % (time.time() - tic))
    return results


if __name__ == "__main__":
    parser = args_parser("tvb_script")
    args, parser_args, parser = parse_args(parser, argsnames=list(DEFAULT_ARGS.keys()))
    verbosity = args.get('verbosity', DEFAULT_ARGS['verbosity'])
    if verbosity:
        print("Running run_workflow with user provided arguments:\n" % parser.description)
        print(args, "\n")
    run_workflow(**args)
