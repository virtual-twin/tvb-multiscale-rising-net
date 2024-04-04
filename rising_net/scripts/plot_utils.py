# -*- coding: utf-8 -*-
import warnings
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def percent_plot(x, data, percentile_min=10, percentile_max=90, n=5,
                 plot_mean=True, plot_median=False,
                 color='b', alpha=0.5, ax=None, mode="linear",
                 **line_kwargs):

    # TEST:
    # data
    # t = np.linspace(0, 100, 100)
    # y = 5 * np.sin(t / 10) + 4 * np.random.randn(100 * 150).reshape(150, 100)
    # y_ = 5 * np.sin(t / 10) + 4 * np.random.randn(100 * 4000).reshape(4000, 100)
    #
    # t__ = np.linspace(0, 100, 6)
    # y__ = 5 * np.sin(t__ / 10) + 4 * np.random.randn(6 * 4000).reshape(4000, 6)
    # ax = plt.axes()
    # ax = ts_percent_plot(t, y, percentile_min=1, percentile_max=99, n=100, ax=ax)

    if mode == "semilog":
        data = np.log(data)
    if data.ndim < 2:
        data = data[np.newaxis]

    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(data, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)[::-1]
    perc2 = np.percentile(data, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if ax is None:
        ax = plt.axes()

    # fill lower and upper percentile groups
    for ii, (p1, p2) in enumerate(zip(perc1, perc2)):
        ax.fill_between(x, p1, p2, alpha=alpha / (ii + 1), color=color, edgecolor=None)

    if plot_mean or plot_median:
        line_color = line_kwargs.pop("color", color)
        if plot_mean:
            ax.plot(x, np.mean(data, axis=0), color=line_color, **line_kwargs)
        if plot_median:
            ax.plot(x, np.median(data, axis=0), color=line_color, **line_kwargs)

    return ax


def psd_percent_plot(results,
                     inds=None,
                     tests=["cosim", "tvb-only", "cerebOFF"], colors=["b", "g", "r"],
                     percentile_min=10, percentile_max=90, n=5,
                     plot_mean=False, plot_median=True,
                     alpha=0.5, figsize=(20, 10), fontsize=16, **line_kwargs):

    if inds is None:
        inds = results["inds"]
    nR = inds.shape[0]
    fig, axes = plt.subplots(nR, 2, figsize=(figsize[0], figsize[1]*nR/2))
    FONTSIZE = 16
    for ind in inds:
        iR = np.where(results["inds"] == ind)[0].item()
        for test_name, col in zip(tests, colors):
            axes[iR, 0] = percent_plot(results["f"], results[test_name]['PSD'][:, iR, :].squeeze(),
                              percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                              plot_mean=plot_mean, plot_median=plot_median,
                              color=col, alpha=alpha, mode="linear",
                              ax=axes[iR, 0], label=test_name, **line_kwargs)
            axes[iR, 0].set_ylabel("PSD %s" % results["short_labels"][iR], fontsize=FONTSIZE)
            axes[iR, 1] = percent_plot(results["f"], results[test_name]['PSD'][:, iR, :].squeeze(),
                              percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                              plot_mean=plot_mean, plot_median=plot_median,
                              color=col, alpha=alpha, mode="semilog",
                              ax=axes[iR, 1], label=test_name, **line_kwargs)
            if iR == 0:
                axes[iR, 1].legend(fontsize=fontsize)
                axes[iR, 0].set_title("Linear PSD", fontsize=fontsize)
                axes[iR, 1].set_title("Log PSD", fontsize=fontsize)
            if iR == nR-1:
                axes[iR, 0].set_xlabel("f (Hz)", fontsize=fontsize)
                axes[iR, 1].set_xlabel("f (Hz)", fontsize=fontsize)
    fig.tight_layout()
    return fig, axes


def plot_pathway_psd_coh_old(results, inds,
                         anslob_psd=None, tests=["tvb-only", "cerebOFF"], colors=["g", "r"],
                         percentile_min=1, percentile_max=99, n=1,
                         plot_mean=False, plot_median=True, mode="semilog",
                         alpha=0.5, figsize=(20, 20), fontsize=16, **line_kwargs):
    REGIONS = ["s1brl", "m1",
               "s1brlthal", "m1thal",
               "ponssens", "ponsmotor",
               "ansilob", "cereb_nuclei",
               "trigeminal", "ponssens_trigeminal"]

    REGPAIRS = [["s1brl", "m1"],
                ["s1brl", "s1brlthal"], ["m1", "m1thal"],
                ["s1brl", "ponssens"], ["m1", "ponsmotor"],
                ["ponssens", "ansilob"], ["ponsmotor", "ansilob"],
                ["trigeminal", "s1brlthal"],
                ["trigeminal", "ansilob"],
                ["trigeminal", "ponssens_trigeminal"],
                ["ponssens_trigeminal", "ansilob"],
                ["ansilob", "cereb_nuclei"],
                ["cereb_nuclei", "s1brlthal"], ["cereb_nuclei", "m1thal"]]

    mosaic = np.tile(["."], (7, 7)).astype('O')
    # PSD plots:
    for ax, reg in zip([[0, 1], [0, 6],
                        [2, 1], [2, 6],
                        [2, 2], [2, 5],
                        [4, [3, 4]], [6, [3, 4]],
                        [6, 0], [4, 1]],
                       REGIONS):
        mosaic[ax[0], ax[1]] = reg
    # COH plots:
    for ax, regs, in zip([[0, [3, 4]],
                          [1, 1], [1, 6],
                          [1, 2], [1, 5],
                          [2, 3], [2, 4],
                          [3, 0], [4, 2], [5, 1], [5, 2],
                          [5, [3, 4]],
                          [6, [1, 2]], [6, [5, 6]]],
                         REGPAIRS):
        mosaic[ax[0], ax[1]] = "-".join(regs)

    figR, axR = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    figL, axL = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)

    # PSD plots:
    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for reg, hemiI in zip(REGIONS,
                              [True, True,
                               True, True,
                               True, True,
                               False, False,
                               False, False]):
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
        for regs, hemiI in zip(REGPAIRS,
                               [[True, True],
                                [True, True], [True, True],
                                [True, True], [True, True],  # ??
                                [True, False], [True, False],
                                [False, True], [False, False], [False, False], [False, False],
                                [False, False],
                                [False, True], [False, True]]):
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


def group_percent_barplot(data, errlows, errhighs,
                          index, legend,
                          data_label, index_label, legend_label,
                          **kwargs):  # ax in kwargs!!!

    # TEST:
    # data = np.random.normal(loc=1.0, scale=0.1, size=(3,3))
    # data_label = "COH"
    # errlows = 0.1*np.ones((3,3));
    # errhighs = 0.2*np.ones((3,3));
    # legend_label = "tests"
    # legend = ["cosim", "tvb-only", "cerebOFF"]
    # index_label = "regions"
    # index = ["m1s1", "m1c", "s1c"]
    # ax = plt.axes()
    # ax = group_percent_barplot(data, errlows, errhighs,
    #                            index, legend,  # cols, rows, respectively
    #                            data_label, index_title, legend_title, colormap="jet", ax=ax)

    data = np.array(data)

    nL = data.shape[0]  # number of legends, rows
    nI = data.shape[1]  # number of indexes, cols

    df = pd.DataFrame({
        data_label: data.flatten(),
        legend_label: np.array(nI * ensure_list(legend)).flatten(),
        index_label: np.array([[ind] * nL for ind in ensure_list(index)]).flatten(),
        "errlows": errlows.flatten(),
        "errhighs": errhighs.flatten()
    })

    errLo = df.pivot_table(index=index_label, columns=legend_label, values="errlows", sort=False)
    errHi = df.pivot_table(index=index_label, columns=legend_label, values="errhighs", sort=False)
    err = []
    for col in errLo:  # Iterate over bar legend (represented as columns)
        err.append([errLo[col].values, errHi[col].values])

    df = df.pivot_table(index=index_label, columns=legend_label, values=data_label, sort=False)

    return df.plot(kind='bar', yerr=err, ylabel=data_label, **kwargs)


def get_coherence(ii, jj, COH, ij, taskinds):
    return COH[:, np.where(np.logical_and(taskinds[ij[:, 0]] == ii, taskinds[ij[:, 1]] == jj))[0]]


def barplots(inds, resname, results, tests=["cosim", "tvb-only", "cerebOFF"], **kwargs):
    nR = inds.size
    data = []
    for iT, test_name in enumerate(tests):
        dataT = []
        pairs = []
        for i1 in range(nR):
            for i2 in range(i1 + 1, nR):
                pair = [inds[i1], inds[i2]]
                coh = get_coherence(pair[0], pair[1], results[test_name][resname], results['ij'], results["inds"])
                if coh.size == 0:
                    pair = pair[::-1]
                    coh = get_coherence(pair[0], pair[1], results[test_name][resname], results['ij'], results["inds"])
                pairs.append(list(pair))
                dataT.append(coh)

        data.append(np.array(dataT).copy())

    data = np.array(data).squeeze()
    pairs = np.array(pairs)

    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    mean = data.mean(axis=2)
    errlows = mean - np.percentile(data, 10, axis=2)
    errhighs = np.percentile(data, 90, axis=2) - mean

    data_label = resname
    legend_label = "Test modes"
    legend = np.array(tests)
    index_label = "Regions ij"
    index = []
    for iP, pair in enumerate(pairs):
        index.append("%s - %s" % (results["short_labels"][np.where(results["inds"] == pair[0])[0].item()],
                                  results["short_labels"][np.where(results["inds"] == pair[1])[0].item()]))
    index = np.array(index)
    ax = group_percent_barplot(mean, errlows, errhighs,
                               index, legend,
                               data_label, index_label, legend_label, **kwargs)
    ax.set_ylim(np.percentile(data, 10, axis=2).min() * 0.9)

    return ax


def matrix_plot(data, labels=None, label=None, ax=None, colorbar=True, fontsize=16, **kwargs):
    if ax is None:
        ax = plt.axes()
    im = ax.imshow(data, interpolation="nearest", **kwargs)
    labels = ["%d. %s" % (iL, lbl) for iL, lbl in enumerate(labels)]
    ticks = np.linspace(0, data.shape[0] - 1, data.shape[0])
    ax.set_xticks(ticks, labels, rotation=90, fontsize=fontsize)
    ax.set_yticks(ticks, labels, fontsize=fontsize)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(label=label, size=fontsize)
    return ax


def half_matrix_plot(data, labels=None, label=None, ax=None, colorbar=True, fontsize=16, **kwargs):

    # TEST:
    # data = np.random.rand(5, 5)
    # labels = ["a", "b", "c", "d", "e"]
    # half_matrix_plot(data, ax=None, labels=labels, label="COH")

    if ax is None:
        ax = plt.axes()
    mask = 1 - np.tri(data.shape[0], k=0)
    data = np.ma.array(data, mask=mask)
    np.fill_diagonal(data, np.nan)
    im = ax.imshow(data, interpolation="nearest", **kwargs)
    ticks = np.linspace(0, data.shape[0]-1, data.shape[0])
    if labels is not None:
        labels = ["%d. %s" % (iL, lbl) for iL, lbl in enumerate(labels)]
        ax.set_xticks(ticks, labels, rotation=90, fontsize=fontsize)
        ax.set_yticks(ticks, labels, fontsize=fontsize)
    else:
        ax.set_xticks(ticks, rotation=90, fontsize=fontsize)
        ax.set_yticks(ticks, fontsize=fontsize)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(label=label, size=fontsize)
    plt.box(False)
    return ax


def shorten_region_name(region_name, exclude=["of", "the", "to"]):
    return "".join([word[0] for word in region_name.split(" ") if word not in exclude])


def coherence_networks_plot(results,
                            tests=["cosim", "tvb-only", "cerebOFF"],
                            resnames=['COHth', 'COHgm'],
                            bands=["theta", "gamma"],
                            figsize=(20, 10), fontsize=16):
    nR = results["inds"].shape[0]

    vmins = []
    vmaxs = []
    data = []
    for iB, (band, res) in enumerate(zip(bands, resnames)):
        dataB = []
        for iT, test_name in enumerate(tests):
            dataT = np.zeros((nR, nR))
            dataT[results["ij"][:, 0], results["ij"][:, 1]] = results[test_name][res].mean(axis=0)
            dataB.append(dataT)
        dataB = np.array(dataB)
        vmins.append(np.percentile(dataB, 5))
        vmaxs.append(np.percentile(dataB, 95))
        data.append(dataB)
    data = np.array(data)
    fig, axes = plt.subplots(len(resnames), len(tests), figsize=(figsize[0], 1.5*figsize[1]))
    for iB, (band, res) in enumerate(zip(bands, resnames)):
        for iT, test_name in enumerate(tests):
            axes[iB, iT] = half_matrix_plot(data[iB, iT].T, labels=results["short_labels"],
                                            label="Average COH in %s band" % band, ax=axes[iB, iT],
                                            vmin=vmins[iB], vmax=vmaxs[iB],
                                            colorbar=True if iT==2 else False,
                                            fontsize=fontsize)

    fig.tight_layout()
    return fig, axes


def prepare_plot_pathway():

    """
         0000 1111 2222 3333 4444 5555
    0000     S1       S1M1      M1
    1111    S1Th 	       M1Th
    2222    S1Th           M1Th
    3333 TRS1 MDS1       AL     M1FC
    4444  MD    MDAL      ALCN     FC
    5555    TRMD        CN     CNM1
    6666      TR      CNS1     FCTR
    """
    mosaic = np.tile(["."], (7, 6)).astype('O')

    REGIONS = ["s1brl", "m1",
               "s1brlthal", "m1thal",
               "facial",
               "medulla",
               "ansilob", "cereb_nuclei",
               "trigeminal"]

    # PSD plots:
    subplotsPSD = [[0, [0, 1]], [0, [4, 5]],  # S1, M1
                   [2, [0, 1]], [2, [3, 4]],   # S1Th, M1Th
                   [4,  5],                   # Facial
                   [4, 0],                    # Medulla
                   [3, 3], [5, 3],            # Ansilob, CerebNuclei
                   [6, 1]]                    # Trigeminal

    ipsiPSD = [True, True,
               True, True,
               False,
               False,
               False, False,
               False]

    REGPAIRS = [["s1brl", "m1"],
                ["s1brl", "s1brlthal"], ["m1", "m1thal"],
                ["m1", "facial"],
                ["trigeminal", "s1brlthal"], ["ponssens_trigeminal", "s1brlthal"],
                ["trigeminal", "ponssens_trigeminal"],
                ["ponssens_trigeminal", "ansilob"],
                ["ansilob", "cereb_nuclei"],
                ["cereb_nuclei", "m1thal"], ["cereb_nuclei", "s1brlthal"],
                ["facial", "trigeminal"]]

    subplotsCOH = [[0, [2, 3]],               # S1 <-> M1
                   [1, [0, 1]], [1, [3, 4]],  # Crtx <-> Thal
                   [3, [4, 5]],                # M1 -> Facial
                   [3, 0], [3, 1],            # [Trigeminal, Medulla] -> S1 thal
                   [5, [0, 1]],               # Trigeminal -> Medulla
                   [4, [1, 2]],               # Medulla -> Ansilob
                   [4, [3, 4]],               # Ansilob -> CerebNuclei
                   [5, [4, 5]], [6, [2, 3]],  # CerebNuclei -> [M1thal, S1thal]
                   [6, [4, 5]]]               # Facial -> Trigeminal

    ipsiCOH = [[True, True],                  # S1 <-> M1
               [True, True],  [True, True],   # Crtx <-> Thal
               [True, False],                 # M1 -> Facial
               [False, True], [False, True],  # [Trigeminal, Medulla] -> S1 thal
               [False, False],                # Trigeminal -> Medulla
               [False, False],                # Medulla -> Ansilob
               [False, False],                # Ansilob -> CerebNuclei
               [False, True], [False, True],  # CerebNuclei -> [M1thal, S1thal]
               [False, False]]                # Facial -> Trigeminal

    # PSD plots:
    for ax, reg in zip(subplotsPSD, REGIONS):
        mosaic[ax[0], ax[1]] = reg
    # COH plots:
    for ax, regs, in zip(subplotsCOH, REGPAIRS):
        mosaic[ax[0], ax[1]] = "-".join(regs)

    return mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH


def plot_pathway_psd_coh(results, inds, tests=["TVB", "TVB_CEREBOFF"], colors=["g", "r"],
                         percentile_min=1, percentile_max=99, n=1,
                         plot_mean=False, plot_median=True, mode="semilog",
                         alpha=0.5, figsize=(10, 10), fontsize=16, **line_kwargs):

    mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH = prepare_plot_pathway()

    figR, axR = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    figL, axL = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)

    print(axR)
    print(axL)
    # PSD plots:

    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for reg, hemiI in zip(REGIONS, ipsiPSD):
            indhs = inds.get(reg, [])
            if len(indhs):
                if hemiI:
                    ind = indhs[hemi]
                else:
                    ind = indhs[1 - hemi]
                try:
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
                except Exception as e:
                    warnings.warn(str(e))
                    axH[reg].set_axis_off()
            else:
                axH[reg].set_axis_off()

    # COH plots:
    for figH, axH, hemi in zip([figR, figL], [axR, axL], [0, 1]):
        for regs, hemiIs in zip(REGPAIRS, ipsiCOH):
            try:
                ax = axH["-".join(regs)]
            except:
                try:
                    ax = axH["-".join(regs[::-1])]
                except Exception as e:
                    print(axH)
                    raise e

            pair = []
            for reg, hemiI in zip(regs, hemiIs):
                indhs = inds.get(reg, [])
                if len(indhs):
                    if hemiI:
                        ind = inds[reg][hemi]
                    else:
                        ind = inds[reg][1 - hemi]
                    try:
                        pair.append(np.where(results["inds"] == ind)[0].item())
                    except Exception as e:
                        warnings.warn(str(e))

            if len(pair) == 2:
                try:
                    iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                                 results["ij"][:, 1].flatten() == pair[1]))[0].item()
                except:
                    pair = pair[::-1]
                    iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
                                                 results["ij"][:, 1].flatten() == pair[1]))[0].item()
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
            else:
                ax.set_axis_off()
        figH.tight_layout()

    return figR, axR, figL, axL
