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


def group_percent_barplot(data, errlows, errhighs,
                          index, legend,
                          data_label, index_title, legend_title,
                          **kwargs):  # ax in kwargs!!!

    # TEST:
    # data = np.random.normal(loc=1.0, scale=0.1, size=(3,3))
    # data_label = "COH"
    # errlows = 0.1*np.ones((3,3));
    # errhighs = 0.2*np.ones((3,3));
    # legend_title = "tests"
    # legend = ["cosim", "tvb-only", "cerebOFF"]
    # index_title = "regions"
    # index = ["m1s1", "m1c", "s1c"]
    # ax = plt.axes()
    # ax = group_percent_barplot(data, errlows, errhighs,
    #                            index, legend,  # cols, rows, respectively
    #                            data_label, index_title, legend_title, colormap="jet", ax=ax)

    data = np.array(data)

    nL = data.shape[0]  # number of legends, rows
    # nI = data.shape[1]  # number of indexes, cols

    df = pd.DataFrame({
        data_label: data.flatten(),
        legend_title: np.array(nL * ensure_list(legend)).flatten(),
        index_title: np.array([[ind] * nL for ind in ensure_list(index)]).flatten(),
        "errlows": errlows.flatten(),
        "errhighs": errhighs.flatten()
    })

    errLo = df.pivot(index=index_title, columns=legend_title, values="errlows")
    errHi = df.pivot(index=index_title, columns=legend_title, values="errhighs")
    err = []
    for col in errLo:  # Iterate over bar legend (represented as columns)
        err.append([errLo[col].values, errHi[col].values])
    df = df.pivot(index=index_title, columns=legend_title, values=data_label)

    return df.plot(kind='bar', yerr=err, ylabel=data_label, **kwargs)


def half_matrix_plot(data, labels=None, label=None, ax=None, **kwargs):

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
        ax.set_xticks(ticks, labels[:-1], rotation=90)
        ax.set_yticks(1+ticks, labels[1:])
    else:
        ax.set_xticks(ticks, rotation=90)
        ax.set_yticks(1+ticks)
    plt.colorbar(im, label=label)
    plt.box(False)
    return ax


def shorten_region_name(region_name, exclude=["of", "the", "to"]):
    return "".join([word[0] for word in region_name.split(" ") if word not in exclude])
