# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def percent_plot(x, data, percentile_min=10, percentile_max=90, n=5,
                 plot_mean=False, plot_median=True,
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


def barplots(inds, resname, results, **kwargs):
    nR = inds.size
    pairs = []
    for i1 in range(nR):
        for i2 in range(i1 + 1, nR):
            pairs.append([inds[i1], inds[i2]])
    pairs = np.array(pairs)

    data = []
    for iT, test_name in enumerate(["cosim", "tvb-only", "cerebOFF"]):

        dataT = []
        for iP, pair in enumerate(pairs):
            coh = get_coherence(pair[0], pair[1], results[test_name][resname], results['ij'], results["inds"])
            if coh.size == 0:
                coh = get_coherence(pair[1], pair[0], results[test_name][resname], results['ij'], results["inds"])
            dataT.append(coh)

        data.append(np.array(dataT).copy())

    data = np.array(data).squeeze()

    mean = data.mean(axis=2)
    errlows = mean - np.percentile(data, 10, axis=2)
    errhighs = np.percentile(data, 90, axis=2) - mean

    data_label = resname
    legend_label = "Test modes"
    legend = np.array(["cosim", "tvb-only", "cerebOFF"])
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


def half_matrix_plot(data, labels=None, label=None, ax=None, colorbar=True, fontsize=16, **kwargs):

    # TEST:
    # data = np.random.rand(5, 5)
    # labels = ["a", "b", "c", "d", "e"]
    # half_matrix_plot(data, ax=None, labels=labels, label="COH")

    if ax is None:
        ax = plt.axes()
    mask = 1 - np.tri(data.shape[0], k=-1)
    data = np.ma.array(data, mask=mask)
    im = ax.imshow(data, interpolation="nearest", **kwargs)
    ticks = np.linspace(0, data.shape[0]-2, data.shape[0]-1)
    if labels is not None:
        labels = ["%d. %s" % (iL, lbl) for iL, lbl in enumerate(labels)]
        ax.set_xticks(ticks, labels[:-1], rotation=90, fontsize=fontsize)
        ax.set_yticks(1+ticks, labels[1:], fontsize=fontsize)
    else:
        ax.set_xticks(ticks, rotation=90, fontsize=fontsize)
        ax.set_yticks(1+ticks, fontsize=fontsize)
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
    fig, axes = plt.subplots(2, 3, figsize=(figsize[0], 1.5*figsize[1]))
    for iB, (band, res) in enumerate(zip(bands, resnames)):
        for iT, test_name in enumerate(tests):
            axes[iB, iT] = half_matrix_plot(data[iB, iT].T, labels=results["short_labels"],
                                            label="Average COH in %s band" % band, ax=axes[iB, iT],
                                            vmin=vmins[iB], vmax=vmaxs[iB],
                                            colorbar=True if iT==2 else False,
                                            fontsize=fontsize)

    fig.tight_layout()
    return fig, axes

