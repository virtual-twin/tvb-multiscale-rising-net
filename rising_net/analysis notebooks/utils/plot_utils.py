# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import pyplot


# Mapping from internal data condition names to display labels
_CONDITION_DISPLAY_LABELS = {
    'PKJtoDCN': 'PCtoCN',
    'MOStoDCN': 'MOStoCN',
    'INHtoPKJ': 'MLItoPC',
}


def remap_condition_label(condition):
    """Remap internal condition name to display label."""
    return _CONDITION_DISPLAY_LABELS.get(condition, condition)


def ensure_list(x):
    """Ensure the input is a list. If not, wrap it in a list."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


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
    ax.grid(True, axis="x")
    return ax


def psd_percent_plot(results,
                     inds=None,
                     tests=["cosim", "tvb-only", "cerebOFF"], colors=["b", "g", "r"],
                     percentile_min=10, percentile_max=90, n=5,
                     plot_mean=False, plot_median=True,
                     alpha=0.5, figsize=(20, 10), fontsize=16, **line_kwargs):

    if inds is None:
        inds = results["inds"]
    nR = len(inds)
    nR2 = int(nR / 2)
    fig, axes = plt.subplots(nR2, 4, figsize=(figsize[0], figsize[1]*np.ceil(nR/4)))
    FONTSIZE = 16
    for ind in inds:
        iR = np.where(results["inds"] == ind)[0].item()
        iR0 = int(iR/2)
        iC0 = 2*np.mod(iR, 2)
        iC1 = iC0 + 1
        for test_name, col in zip(tests, colors):
            axes[iR0, iC0] = percent_plot(results["f"], results[test_name]['PSD'][:, iR, :].squeeze(),
                              percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                              plot_mean=plot_mean, plot_median=plot_median,
                              color=col, alpha=alpha, mode="linear",
                              ax=axes[iR0, iC0], label=test_name, **line_kwargs)
            axes[iR0, iC0].set_ylabel("PSD %s" % results["short_labels"][iR], fontsize=FONTSIZE)
            axes[iR0, iC1] = percent_plot(results["f"], results[test_name]['PSD'][:, iR, :].squeeze(),
                              percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                              plot_mean=plot_mean, plot_median=plot_median,
                              color=col, alpha=alpha, mode="semilog",
                              ax=axes[iR0, iC1], label=test_name, **line_kwargs)
            if iR0 == 0:
                axes[iR0, 3].legend(fontsize=fontsize)
                axes[iR0, iC0].set_title("Linear PSD", fontsize=fontsize)
                axes[iR0, iC1].set_title("Log PSD", fontsize=fontsize)
            if iR0 == nR2-1:
                axes[iR0, iC0].set_xlabel("f (Hz)", fontsize=fontsize)
                axes[iR0, iC1].set_xlabel("f (Hz)", fontsize=fontsize)
    fig.tight_layout()
    return fig, axes


def plot_pathway_psd_coh_old(results, inds, tests=["tvb-only", "cerebOFF"],
                             colors=["g", "r"], percentile_min=1, percentile_max=99, n=1,
                             plot_mean=False, plot_median=True, mode="linear",
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
            # try:
            #     iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
            #                                  results["ij"][:, 1].flatten() == pair[1]))[0].item()
            # except:
            #     pair = pair[::-1]
            #     iR = np.where(np.logical_and(results["ij"][:, 0].flatten() == pair[0],
            #                                  results["ij"][:, 1].flatten() == pair[1]))[0].item()
            ax = axH["-".join(regs)]
            for col, test in zip(colors, tests):
                percent_plot(results["f"], results[test]['COH'][:, pair[0], pair[1], :].squeeze(),
                             percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                             plot_mean=plot_mean, plot_median=plot_median,
                             color=col, alpha=alpha, ax=ax, mode=mode,
                             **line_kwargs)
                for band, COH, f in zip(["theta", "gamma"],
                                        ["COHth", "COHgm"],
                                        ["fth", "fgm"]):
                    if mode == "semilog":
                        mean = np.log(results[test]['COH'][:,  pair[0], pair[1], results[f]]).mean()
                    else:
                        mean = results[test]['COH'][:,  pair[0], pair[1], results[f]].mean()
                    ax.plot(results[band], [mean] * 2, color=col, linewidth=2.0)
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


def barplots(inds, resname, results, tests=["TVB", "TVB_CEREBOFF"], **kwargs):
    nR = inds.size
    data = []
    for iT, test_name in enumerate(tests):
        dataT = []
        pairs = []
        for i1 in range(nR):
            for i2 in range(i1 + 1, nR):
                pair = [inds[i1], inds[i2]]
                coh = results[test_name][resname][:, pair[0], pair[0]]
                if coh.size == 0:
                    pair = pair[::-1]
                    coh = results[test_name][resname][:, pair[0], pair[0]]
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
    nR = len(results["inds"])
    vmins = []
    vmaxs = []
    data = []
    for iB, (band, res) in enumerate(zip(bands, resnames)):
        dataB = []
        for iT, test_name in enumerate(tests):
            dataT = np.nan * np.ones((nR, nR))
            dataT[results["ij"][:, 0], results["ij"][:, 1]] = \
                results[test_name][res][:, results["ij"][:, 0], results["ij"][:, 1]].mean(axis=0)
            dataB.append(dataT)
        dataB = np.array(dataB)
        vmins.append(np.nanpercentile(dataB, 5))
        vmaxs.append(np.nanpercentile(dataB, 95))
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
                         plot_mean=False, plot_median=True, modePSD="semilog", modeCOH="linear",
                         alpha=0.5, figsize=(10, 10), fontsize=16, **line_kwargs):

    mosaic, REGIONS, subplotsPSD, ipsiPSD, REGPAIRS, subplotsCOH, ipsiCOH = prepare_plot_pathway()

    figR, axR = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    figL, axL = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)

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
                    iR = np.where(results["inds"] == ind)[0]
                    if len(iR):
                        iR = iR[0].item()
                    else:
                        raise ValueError("No region index iR for index %s and results['inds']=\n%s"
                                         % (str(ind), str(results["inds"])))
                    for col, test in zip(colors, tests):
                        percent_plot(results["f"], results[test]['PSD'][:, iR, :].squeeze(),
                                     percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                                     plot_mean=plot_mean, plot_median=plot_median,
                                     color=col, alpha=alpha, ax=axH[reg], mode=modePSD,
                                     **line_kwargs)
                    axH[reg].set_title(results['short_labels'][iR])
                    if modePSD == "semilog":
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
                for col, test in zip(colors, tests):
                    COH = results[test]['COH'][:, pair[0], pair[1], :].squeeze()
                    percent_plot(results["f"], COH,
                                 percentile_min=percentile_min, percentile_max=percentile_max, n=n,
                                 plot_mean=plot_mean, plot_median=plot_median,
                                 color=col, alpha=alpha, ax=ax, mode=modeCOH,
                                 **line_kwargs)
                    for band in ["theta", "gamma"]:
                        if modeCOH == "semilog":
                            mean = np.log(COH).mean()
                        else:
                            mean = COH.mean()
                        ax.plot(results[band], [mean] * 2, color=col, linewidth=2.0)
                ax.set_title("%s - %s" % (results['short_labels'][pair[0]],
                                          results['short_labels'][pair[1]]), fontsize=fontsize)
                if modeCOH == "semilog":
                    ax.set_ylabel('log(COH)', fontsize=fontsize)
                else:
                    ax.set_ylabel('COH', fontsize=fontsize)
            else:
                ax.set_axis_off()
        figH.tight_layout()

    return figR, axR, figL, axL


def sbi_pairplot(samples, figpath=None, save_flag=True, show_flag=True, **kwargs):
    if kwargs.get("points", None) is not None:
        if kwargs.get("points_colors", None) is None:
            kwargs["points_colors"] = ['r'] * len(kwargs["points"])
        if kwargs.get("points_offdiag", None) is None:
            kwargs["points_offdiag"] = {'markersize': 6}
    if kwargs.get("limits", None) is None:
        kwargs["limits"] = np.array([np.min(samples, axis=0), np.max(samples, axis=0)]).T.tolist()
    if np.array(kwargs["limits"]).ndim == 1:
        kwargs["limits"] = np.array([kwargs["limits"]]).tolist()
    if samples.ndim == 1:
        nParams = 1
    else:
        nParams = np.maximum(1, samples.shape[1])
    if kwargs.get("ticks", None) is None:
        if kwargs.get("limits", None) is not None:
            kwargs["ticks"] = deepcopy(kwargs["limits"])
        if kwargs.get("points") is not None:
            if kwargs.get("ticks", None) is None:
                kwargs["ticks"] = [[]] * nParams
            for iT, point in enumerate(kwargs["points"]):
                kwargs["ticks"][iT].append(point)
                kwargs["ticks"][iT] = np.sort(kwargs["ticks"][iT]).tolist()
    elif np.array(kwargs["ticks"]).ndim == 1:
        kwargs["ticks"] = np.array([kwargs["ticks"]]).tolist()
    if kwargs.get("figsize", None) is None:
        kwargs["figsize"] = (20, 20)
    fig, axes = analysis.pairplot(samples, **kwargs)
    if save_flag:
        plt.savefig(figpath)
    if show_flag:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def define_colors(N):
    from cycler import cycler
    import matplotlib as mpl

    cmap = mpl.colormaps['jet']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, N))

    custom_cycler = cycler(color=colors)  # or simply color=colorlist
    plt.rc('axes', prop_cycle=custom_cycler)

    return custom_cycler


# Function for statistical annotations
def add_stat_annotation(ax, x1, x2, y, p_value, h=0.02, significance_threshold=0.05):
    """Add statistical annotation to plot"""
    y_pos = y + h
    
    # Add text with significance marker
    # Color is black for significant (p < 0.05), gray for non-significant
    if p_value < 0.001:
        p_text = '***'  # Very highly significant
        color = 'black'
    elif p_value < 0.01:
        p_text = '**'  # Highly significant
        color = 'black'
    elif p_value < 0.05:
        p_text = '*'  # Significant
        color = 'black'
    else:
        p_text = f'p = {p_value:.3f}'  # Non-significant
        color = 'gray'
    
    # Draw the horizontal line (clip_on=False so it renders outside axes limits)
    ax.plot([x1, x2], [y_pos, y_pos], '-', color=color, linewidth=1, clip_on=False)
    
    ax.text((x1 + x2) / 2, y_pos, p_text, ha='center', va='bottom', 
            fontsize=12, color=color, clip_on=False)


def set_default_plot_params(font_size=14, axes_label_size=14, axes_title_size=16,
                            xtick_size=12, ytick_size=12):
    """
    Set default matplotlib rcParams for consistent plot styling.
    
    Parameters
    ----------
    font_size : int
        Base font size
    axes_label_size : int
        Font size for axis labels
    axes_title_size : int
        Font size for subplot titles
    xtick_size : int
        Font size for x-axis tick labels
    ytick_size : int
        Font size for y-axis tick labels
    """
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': axes_label_size,
        'axes.titlesize': axes_title_size,
        'xtick.labelsize': xtick_size,
        'ytick.labelsize': ytick_size
    })


def save_figure_multi_format(fig, output_path, dpi=300, formats=('png', 'eps', 'svg')):
    """
    Save a matplotlib figure in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str
        Base path for saving (without extension)
    dpi : int
        Resolution for raster formats
    formats : tuple
        Tuple of format extensions to save
    """
    for fmt in formats:
        fig.savefig(f"{output_path}.{fmt}", format=fmt, dpi=dpi, bbox_inches='tight')


def compute_condition_stats(data_array, conditions, flatten=True):
    """
    Compute mean, std, and individual points for each condition.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Data array with a 'Lesion Condition' coordinate
    conditions : list
        List of condition names to process
    flatten : bool
        Whether to flatten the data for each condition
        
    Returns
    -------
    means : numpy.ndarray
        Mean values for each condition
    stds : numpy.ndarray
        Standard deviation for each condition
    individual_points : list
        List of arrays containing individual data points for each condition
    """
    means = []
    stds = []
    individual_points = []
    
    for cond in conditions:
        cond_data = data_array.sel({"Lesion Condition": cond}).values
        if flatten:
            flat_data = cond_data.flatten()
        else:
            flat_data = cond_data
        means.append(np.mean(flat_data))
        stds.append(np.std(flat_data))
        individual_points.append(flat_data)
    
    return np.array(means), np.array(stds), individual_points


def compute_condition_stats_hemisphere_averaged(data_array, conditions):
    """
    Compute mean, std, and individual points for each condition,
    after first averaging across hemispheres (Connections dimension) for each simulation.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Data array with 'Lesion Condition' and 'Connections' (for hemispheres) coordinates.
        Expected dimensions: (Repetitions, Lesion Condition, Connections, ...)
    conditions : list
        List of condition names to process
        
    Returns
    -------
    means : numpy.ndarray
        Mean values for each condition (averaged across simulations)
    stds : numpy.ndarray
        Standard deviation for each condition
    individual_points : list
        List of arrays containing hemisphere-averaged data points for each condition
        (one value per simulation, not per hemisphere)
    """
    means = []
    stds = []
    individual_points = []
    
    for cond in conditions:
        cond_data = data_array.sel({"Lesion Condition": cond})
        
        # Average across hemispheres (Connections dimension) first
        # This gives one value per simulation/repetition
        if 'Connections' in cond_data.dims:
            hemisphere_averaged = cond_data.mean(dim='Connections').values
        else:
            # If no Connections dimension, just use the data as is
            hemisphere_averaged = cond_data.values
        
        # Flatten any remaining dimensions (e.g., if there are multiple repetition dimensions)
        flat_data = hemisphere_averaged.flatten()
        
        means.append(np.mean(flat_data))
        stds.append(np.std(flat_data))
        individual_points.append(flat_data)
    
    return np.array(means), np.array(stds), individual_points


def bar_plot_with_stats(ax, means, stds, individual_points, conditions, 
                        bar_color='skyblue', point_color='royalblue',
                        bar_width=0.4, bar_alpha=0.7, point_alpha=1, point_size=18,
                        point_edgecolor='darkgray', point_linewidth=0.5,
                        x_rotation=45, ylabel=None, title=None,
                        add_value_labels=True, value_label_fontsize=12,
                        add_stat_annotations=True, reference_idx=0,
                        significance_threshold=0.01, ylim_bottom=None, ylim_top=None,
                        p_values=None):
    """
    Create a bar plot with error bars, scatter points, value labels, and statistical annotations.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    means : array-like
        Mean values for each condition
    stds : array-like
        Standard deviations for each condition
    individual_points : list
        List of arrays containing individual data points for each condition
    conditions : list
        List of condition names
    bar_color : str
        Color for the bars
    point_color : str
        Color for the scatter points
    bar_width : float
        Width of the bars
    bar_alpha : float
        Transparency for bars
    point_alpha : float
        Transparency for scatter points
    point_size : int
        Size of scatter points
    point_edgecolor : str
        Edge color for the scatter points
    point_linewidth : float
        Line width for the scatter point edges
    x_rotation : int
        Rotation angle for x-axis labels
    ylabel : str
        Label for y-axis
    title : str
        Title for the subplot
    add_value_labels : bool
        Whether to add value labels on top of bars
    value_label_fontsize : int
        Font size for value labels
    add_stat_annotations : bool
        Whether to add statistical annotations
    reference_idx : int
        Index of the reference condition for statistical comparisons
    significance_threshold : float
        P-value threshold for significance
    ylim_bottom : float
        Bottom limit for y-axis (None for auto)
    p_values : list, optional
        Pre-computed p-values for each comparison (e.g., FDR-corrected).
        If provided, these are used instead of computing t-tests.
        Should have length equal to number of comparisons (len(conditions) - 1).
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes
    """
    from scipy import stats
    import matplotlib.colors as mcolors
    
    # Function to darken a color
    def darken_color(color, factor=0.6):
        """Darken a color by multiplying RGB values by factor (0-1)."""
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)
    
    # Derive dark point color from bar color
    dark_point_color = darken_color(bar_color, factor=0.6)
    
    # Plot bars with error bars
    bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
                  color=bar_color, alpha=bar_alpha, width=bar_width)
    
    # Add individual points spread evenly across bar width
    for i, points in enumerate(individual_points):
        # Spread points evenly across bar width
        n_points = len(points)
        if n_points > 1:
            x_positions = np.linspace(-bar_width/2 * 0.8, bar_width/2 * 0.8, n_points)
        else:
            x_positions = np.array([0])
        ax.scatter(x_positions + i, points, color=dark_point_color, 
                   alpha=point_alpha, s=point_size,
                   edgecolor=point_edgecolor, linewidth=point_linewidth)
    
    # Customize x-axis
    ax.set_xticks(list(range(len(means))))
    display_labels = [remap_condition_label(c) for c in conditions]
    ax.set_xticklabels(display_labels, rotation=x_rotation, fontsize=12)
    
    # Set labels and title
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    
    # Add value labels
    if add_value_labels:
        for i, (mean, std) in enumerate(zip(means, stds)):
            # Place label above bar top (positive) or below bar bottom (negative)
            if mean >= 0:
                ax.text(i, mean + std + 0.02, f'{mean:.2f}±{std:.2f}', 
                        ha='center', va='bottom', fontsize=value_label_fontsize)
            else:
                ax.text(i, mean - std - 0.02, f'{mean:.2f}±{std:.2f}', 
                        ha='center', va='top', fontsize=value_label_fontsize)
    
    # Add statistical annotations
    if add_stat_annotations and len(conditions) > 1:
        comparisons = [(reference_idx, i) for i in range(len(conditions)) if i != reference_idx]
        n_comparisons = len(comparisons)
        max_bar_top = max(m + s for m, s in zip(means, stds))

        if ylim_top is not None:
            # Spread annotations evenly between max data and ylim_top
            available = ylim_top - max_bar_top
            spacing = available / (n_comparisons + 2)
            annotation_base = max_bar_top
        else:
            min_bar_bottom = min(m - s for m, s in zip(means, stds))
            data_range = max_bar_top - min_bar_bottom
            spacing = max(0.05, 0.18 * data_range)
            annotation_base = max_bar_top + spacing
        
        for i, (c1, c2) in enumerate(comparisons):
            if p_values is not None:
                p_val = p_values[i]
            else:
                t_stat, p_val = stats.ttest_ind(individual_points[c1], individual_points[c2])
            add_stat_annotation(ax, c1, c2, annotation_base, p_val, h=spacing*(i+1),
                                significance_threshold=significance_threshold)
        
        if ylim_top is not None:
            ax.set_ylim(top=ylim_top)
        else:
            ax.set_ylim(top=annotation_base + spacing * (n_comparisons + 3))
    
    # Set bottom y-limit if specified
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    
    return ax


def correlation_plot(ax, x_data, y_data, conditions=None, condition_colors=None,
                     xlabel=None, ylabel=None, title=None,
                     marker_size=50, marker_alpha=0.6,
                     add_regression_line=True, regression_color='r',
                     show_stats_in_title=True):
    """
    Create a correlation scatter plot with optional regression line and Pearson correlation.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    x_data : dict or array-like
        X-axis data. If dict, organized by condition and hemisphere
    y_data : dict or array-like
        Y-axis data. If dict, organized by condition and hemisphere
    conditions : list, optional
        List of conditions for coloring points
    condition_colors : list, optional
        Colors for each condition
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str
        Base title for the plot
    marker_size : int
        Size of scatter markers
    marker_alpha : float
        Transparency of markers
    add_regression_line : bool
        Whether to add a linear regression line
    regression_color : str
        Color for the regression line
    show_stats_in_title : bool
        Whether to show r and p-value in the title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes
    r : float
        Pearson correlation coefficient
    p_val : float
        P-value for the correlation
    """
    from scipy import stats
    
    x_vals = []
    y_vals = []
    
    # Handle dictionary-organized data (by condition and hemisphere)
    if isinstance(x_data, dict) and isinstance(y_data, dict):
        if conditions is None:
            conditions = list(x_data.keys())
        if condition_colors is None:
            condition_colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
        
        for cond_idx, condition in enumerate(conditions):
            cond_color = condition_colors[cond_idx] if len(condition_colors) > cond_idx else 'blue'
            
            for hemisphere in x_data[condition].keys():
                x_vals_cond = x_data[condition][hemisphere]
                y_vals_cond = y_data[condition][hemisphere]
                
                n_points = min(len(x_vals_cond), len(y_vals_cond))
                if n_points > 0:
                    ax.scatter(x_vals_cond[:n_points], y_vals_cond[:n_points],
                               color=cond_color, alpha=marker_alpha, s=marker_size,
                               edgecolor='darkgray', linewidth=1,
                               label=f"{condition}" if hemisphere == 'Right' else "_nolegend_")
                    
                    x_vals.extend(x_vals_cond[:n_points])
                    y_vals.extend(y_vals_cond[:n_points])
        
        ax.legend(fontsize=12)
    else:
        # Handle simple array data
        x_vals = np.array(x_data).flatten()
        y_vals = np.array(y_data).flatten()
        ax.scatter(x_vals, y_vals, alpha=marker_alpha, s=marker_size,
                   edgecolor='darkgray', linewidth=1)
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    r, p_val = 0, 1
    if len(x_vals) > 0:
        r, p_val = stats.pearsonr(x_vals, y_vals)
        
        if add_regression_line:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot([min(x_vals), max(x_vals)], 
                    p([min(x_vals), max(x_vals)]), 
                    f"{regression_color}--", alpha=0.8, linewidth=2)
        
        if show_stats_in_title and title:
            ax.set_title(f'{title}\nr={r:.2f}, p={p_val:.3e}', fontsize=16)
        elif title:
            ax.set_title(title, fontsize=16)
    else:
        if title:
            ax.set_title(f'{title}\nNo matching data points found', fontsize=16)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return ax, r, p_val


def print_condition_stats(conn_set_name, freq_band_name, freq_range, 
                          conditions, means, stds, individual_points):
    """
    Print numerical statistics for each condition.
    
    Parameters
    ----------
    conn_set_name : str
        Name of the connection set
    freq_band_name : str
        Name of the frequency band
    freq_range : tuple
        Frequency range (min, max)
    conditions : list
        List of condition names
    means : array-like
        Mean values
    stds : array-like
        Standard deviation values
    individual_points : list
        Individual data points for each condition
    """
    print(f"\nResults for {conn_set_name} - {freq_band_name} ({freq_range[0]}-{freq_range[1]} Hz):")
    for i, condition in enumerate(conditions):
        print(f"{condition}:")
        print(f"  Mean: {means[i]:.3f}")
        print(f"  STD: {stds[i]:.3f}")
        points = individual_points[i]
        print(f"  N: {len(points)}")
        print(f"  Min: {np.min(points):.3f}")
        print(f"  Max: {np.max(points):.3f}")
        print(f"  Range: {np.max(points) - np.min(points):.3f}")


def print_statistical_tests(conn_set_name, freq_band_name, conditions, 
                            individual_points, reference_idx=0):
    """
    Print results of statistical tests (t-tests and Mann-Whitney U tests) with Cohen's d effect size.
    
    Parameters
    ----------
    conn_set_name : str
        Name of the connection set
    freq_band_name : str
        Name of the frequency band
    conditions : list
        List of condition names
    individual_points : list
        Individual data points for each condition
    reference_idx : int
        Index of reference condition for comparisons
        
    Returns
    -------
    results : list
        List of dictionaries containing comparison results
    """
    from scipy import stats
    from scipy.stats import mannwhitneyu
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size for two independent groups."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def interpret_cohens_d(d):
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    comparisons = [(reference_idx, i) for i in range(len(conditions)) if i != reference_idx]
    results = []
    
    print(f"\nStatistical Tests for {conn_set_name} - {freq_band_name}:")
    print("=" * 70)
    
    for c1, c2 in comparisons:
        # T-test
        t_stat, p_val = stats.ttest_ind(individual_points[c1], individual_points[c2])
        
        # Mann-Whitney U test
        u_stat, p_val_mw = mannwhitneyu(individual_points[c1], individual_points[c2], alternative='two-sided')
        
        # Cohen's d
        d = cohens_d(individual_points[c1], individual_points[c2])
        d_interpretation = interpret_cohens_d(d)
        
        # Significance markers
        if p_val < 0.001:
            sig_marker = "***"
        elif p_val < 0.01:
            sig_marker = "**"
        elif p_val < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"
        
        label_c1 = remap_condition_label(conditions[c1])
        label_c2 = remap_condition_label(conditions[c2])
        print(f"\n{label_c1} vs {label_c2}:")
        print(f"  T-test: t={t_stat:.3f}, p={p_val:.6f} [{sig_marker}]")
        print(f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_mw:.6f}")
        print(f"  Cohen's d: {d:.3f} ({d_interpretation} effect)")
        
        # Store results
        results.append({
            'analysis': conn_set_name,
            'measure': freq_band_name,
            'condition_1': label_c1,
            'condition_2': label_c2,
            'n_1': len(individual_points[c1]),
            'n_2': len(individual_points[c2]),
            'mean_1': np.mean(individual_points[c1]),
            'mean_2': np.mean(individual_points[c2]),
            'std_1': np.std(individual_points[c1]),
            'std_2': np.std(individual_points[c2]),
            't_statistic': t_stat,
            'p_value_ttest': p_val,
            'u_statistic': u_stat,
            'p_value_mannwhitney': p_val_mw,
            'cohens_d': d,
            'effect_size_interpretation': d_interpretation,
            'significance': sig_marker
        })
    
    return results