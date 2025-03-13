# -*- coding: utf-8 -*-

import os

import numpy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesX
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, is_float, ensure_list

from tvb_multiscale.core.utils.file_utils import dump_pickled_dict


def print_lbl(lbl, siz, prnt=""):
    prnt += lbl
    prnt += "." * (siz - len(lbl))
    return prnt


def print_row(vals, sizes, prnt=""):
    prnt += "\n"
    for iV, (val, siz) in enumerate(zip(vals, sizes)):
        if is_integer(val):
            prnt = print_lbl("%d" % val, siz, prnt)
        elif is_float(val):
            prnt = print_lbl("%g" % val, siz, prnt)  # %.3f
        else:
            prnt = print_lbl(str(val), siz, prnt)
    return prnt


def print_conn(d={}, prnt="", maxrow=200, printit=True):
    sizes = []
    values = []
    for col, val in d.items():
        sizes.append(col[1])
        values.append(np.array(val))
        prnt = print_lbl(col[0], sizes[-1], prnt)
    prnt += "\n" + "-" * np.sum(sizes)
    for iV, vals in enumerate(zip(*values)):
        if iV == maxrow:
            break
        prnt = print_row(vals, sizes, prnt)
    if printit:
        print(prnt)
    return prnt


def get_region_indice(reg, labels):
    if isinstance(reg, str):
        return np.where(labels==reg)[0].item()
    if is_integer(reg):
        return reg
    raise ValueError("reg should be either a region label or integer indice, but it is %s!" % str(reg))


def get_regions_indices(regs, labels):
    if regs is None:
        return slice(None)
    iR = []
    for reg in ensure_list(regs):
        iR.append(get_region_indice(reg, labels))
    return iR


def compute_nperseg(fs, Ndata):
    # # Window:
    # NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    # Trying to have a resolution of 1 Hz:
    # Nf = int(fs)
    # ...constraint by the length of the data:
    # nperseg = np.minimum(Ndata, NPERSEG[np.argmin(np.abs(NPERSEG - Nf))])
    return np.minimum(Ndata, int(fs))


def interpolate_freqs(data, f, fmin=0.0, fmax=100.0, ftarg=None):
    if ftarg is not None:
        # Compute spectrum interpolation...
        interp = interp1d(f, data, kind='linear', axis=1,
                          copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)
        # ...to the target frequencies:
        data = interp(ftarg)
        fout = ftarg
    else:
        finds = np.logical_and(f > fmin, f <= fmax)
        fout = f[finds]
        data = data[:, finds]
    return fout, data


def compute_selected_spectra_coherence(source_ts, inds, sample_period, transient=0, nperseg=None,
                                       fmin=0.0, fmax=100.0, ftarg=None):
    n_regions = len(inds)
    data = source_ts[transient:, 0, inds].squeeze().T
    fs = 1000/sample_period
    Ndata = data.shape[1]
    if nperseg is None:
        nperseg = compute_nperseg(fs, Ndata)
    f, Pxx_den = signal.welch(data, fs, nperseg=nperseg)
    fout, Pxx_den = interpolate_freqs(Pxx_den, f, fmin=fmin, fmax=fmax, ftarg=ftarg)
    Cxy = []
    n_regions2 = int(n_regions * (n_regions - 1)/2)
    pairs = []
    if n_regions2:
        nperseg = np.minimum(int(Ndata / 2), nperseg)
        for ii in range(n_regions):
            for jj in range(ii+1, n_regions):
                f, Cxyiijj = signal.coherence(data[ii], data[jj], fs, nperseg=nperseg)
                Cxy.append(Cxyiijj)
                pairs.append([ii, jj])
    fout, Cxy = interpolate_freqs(np.array(Cxy).squeeze(), f, fmin=fmin, fmax=fmax, ftarg=ftarg)
    return Pxx_den, Cxy, fout, np.array(pairs).squeeze()


def compute_plot_selected_spectra_coherence(source_ts, inds,
                                            transient=0.0, conn=None, nperseg=None, fmin=0.0, fmax=100.0,
                                            figsize=(15, 5), figures_path="", figname="", figformat="png", 
                                            show_flag=True, save_flag=True):
    n_regions = int(len(inds) / 2)
    transient = int(transient/source_ts.sample_period)
    data = source_ts[transient:, 0, inds].squeeze().T
    Ndata = data.shape[1]
    if conn is None:
        conn = source_ts.connectivity
    fs = 1000/source_ts.sample_period
    if nperseg is None:
        nperseg = compute_nperseg(fs, Ndata)
    f, Pxx_den = signal.welch(data, fs, nperseg=nperseg)
    fig, axes = plt.subplots(n_regions, 2, figsize=(figsize[0], figsize[1]*n_regions))
    if axes.ndim == 1:
        axes = np.array([axes])
    for ii in range(n_regions):
        iR = ii*2
        iL = ii*2 + 1
        axes[ii, 0].plot(f, Pxx_den[iR],
                         label="%.1fHz, %s" % (f[np.argmax(Pxx_den[iR])], conn.region_labels[inds[iR]]))
        axes[ii, 0].plot(f, Pxx_den[iL],
                         label="%.1fHz, %s" % (f[np.argmax(Pxx_den[iL])], conn.region_labels[inds[iL]]))
        axes[ii, 0].set_xlim([fmin, fmax])
        axes[ii, 0].set_xlabel('frequency [Hz]')
        axes[ii, 0].set_ylabel('PSD [V**2/Hz]')
        axes[ii, 0].grid(True, axis='x')
        axes[ii, 0].legend()
        axes[ii, 1].semilogy(f, Pxx_den[iR], label=conn.region_labels[inds[iR]])
        axes[ii, 1].semilogy(f, Pxx_den[iL], label=conn.region_labels[inds[iL]])
        axes[ii, 1].set_xlim([fmin, fmax])
        axes[ii, 1].set_xlabel('frequency [Hz]')
        axes[ii, 1].set_ylabel('PSD [log(V**2/Hz)]')
        axes[ii, 1].grid(True, axis='x')
        axes[ii, 1].legend()
    # plt.ylim([1e-7, 1e2])
    if save_flag and len(figures_path) + len(figname):
        plt.savefig(os.path.join(figures_path, figname + "_PSD.%s" % figformat))
    if show_flag:
        plt.show()
    else:
        plt.close(fig)

    CxyRs = []
    fR = []
    fL = []
    CxyLs = []
    n_regions2 = int(n_regions * (n_regions - 1)/2)
    if n_regions2:
        nperseg = np.minimum(int(Ndata / 2), nperseg)
        fig, axes = plt.subplots(n_regions2, 2, figsize=(figsize[0], figsize[1]*n_regions))
        if len(axes.shape) < 2:
            axes = axes[np.newaxis, :]
        ii = 0
        for i1 in range(0, n_regions-1):
            iR1 = 2*i1
            iL1 = 2*i1 + 1
            for i2 in range(i1+1, n_regions):
                iR2 = 2*i2
                iL2 = 2*i2 + 1
                fR, CxyR = signal.coherence(data[iR1], data[iR2], fs, nperseg=nperseg)
                CxyRs.append(CxyR)
                fL, CxyL = signal.coherence(data[iL1], data[iL2], fs, nperseg=nperseg)
                CxyLs.append(CxyL)
                axes[ii, 0].plot(fR, CxyR.T,
                                 label="%s - %s" % (conn.region_labels[inds[iR1]], conn.region_labels[inds[iR2]]))
                axes[ii, 0].plot(fL, CxyL.T,
                                 label="%s - %s" % (conn.region_labels[inds[iL1]], conn.region_labels[inds[iL2]]))
                axes[ii, 0].set_xlim([fmin, fmax])
                axes[ii, 0].set_ylim([fmin, 0.45])
                axes[ii, 0].set_xlabel('frequency [Hz]')
                axes[ii, 0].set_ylabel('Coherence')
                axes[ii, 0].legend()
                axes[ii, 1].semilogy(fR, CxyR.T,
                                     label="%s - %s" % (conn.region_labels[inds[iR1]], conn.region_labels[inds[iR2]]))
                axes[ii, 1].semilogy(fL, CxyL.T,
                                     label="%s - %s" % (conn.region_labels[inds[iL1]], conn.region_labels[inds[iL2]]))
                axes[ii, 1].set_xlim([fmin, fmax])
                axes[ii, 1].set_xlabel('frequency [Hz]')
                axes[ii, 1].set_ylabel('log10(Coherence)')
                axes[ii, 1].legend()
                ii += 1
        if save_flag and len(figures_path) + len(figname):
            plt.savefig(os.path.join(figures_path, figname + "_COH.%s" % figformat))
        if show_flag:
            plt.show()
        else:
            plt.close(fig)
    return Pxx_den, f, CxyRs, fR, CxyLs, fL


def only_plot_selected_spectra_coherence_and_diff(freq, avg_coherence, color, fmin=0.0, fmax=100.0,
                                                  figsize=(15, 5), figures_path="", figformat="png",
                                                  show_flag=True, save_flag=True):
    import numpy as np
    yranges = [[0,0.35], [-0.2, 0.2]]    # Ranges for coherence and diff plot respectively
    ylabel = ['Spectral coherence','Diff in spectral coherence']
    # avg_coherence is a dictionary with average coherence between L and R M1-S1 for each simulation test
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]*2))
    for test in avg_coherence.keys():
        # Plot coherence
        axes[0].plot(freq, avg_coherence[test], color=color[test])
    # Plot coherence diff cosim vs MF cereb-OFF
    #axes[1].plot(freq,np.subtract(avg_coherence['MF_cerebOFF'], avg_coherence['cosim']), color=color['cosim'])
    # Plot coherence diff MF cereb-ON vs MF cereb-OFF
    axes[1].plot(freq,np.subtract(avg_coherence['MF_cerebOFF'], avg_coherence['MF_cerebON']), color=color['MF_cerebON'])

    for ii in range(len(axes)):
        axes[ii].set_xlim([fmin, fmax])
        axes[ii].set_xlabel('frequency [Hz]')
        axes[ii].set_ylabel(ylabel[ii])
        axes[ii].vlines(25, yranges[ii][0], yranges[ii][1])
        axes[ii].vlines(45, yranges[ii][0], yranges[ii][1])
        axes[ii].grid(True, axis='x')
        
    axes[0].set_ylim(yranges[0])
    axes[0].set_title('M1-S1 coherence spectra during virtual whisking')
    axes[0].legend(['mean-field rising_net ON','cerebellar inactivation (OFF)','spiking rising_net ON (cosim)'])
    axes[1].set_ylim(yranges[1])
    axes[1].set_title('change in M1-S1 coherence after virtual cerebellar inactivation')
    axes[1].legend(['OFF-ON spiking','OFF-ON mean-field'])

    if show_flag:
        plt.show()
    else:
        plt.close(fig)
    
    if save_flag and len(figures_path):
        plt.savefig(os.path.join(figures_path, "COHselectDiff.%s" % figformat))

    return fig


def compute_plot_components(data, MODE=PCA, variable="BOLD", n_components=10, time=None, plotter=None):
    if MODE == PCA:
        mode = "PCA"
    else:
        mode = "ICA"
    ca = MODE(n_components=n_components)
    ca_ts = ca.fit_transform(data)
    if time is not None:
        ca_ts = TimeSeriesX(
            data=ca_ts[:, np.newaxis, :, np.newaxis], time=time,
            labels_ordering=["Time", "State Variable", mode, "Modes"],
            labels_dimensions={"State Variable": [variable],
                               mode: np.arange(ca_ts.shape[1])})
        ca_ts.configure()

        if plotter:
            ca_ts.plot_timeseries(plotter_config=plotter.config,
                                   hue="ICA" if ca_ts.shape[2] > plotter.config.MAX_REGIONS_IN_ROWS else None,
                                   per_variable=ca_ts.shape[1] > plotter.config.MAX_VARS_IN_COLS,
                                   figsize=plotter.config.DEFAULT_SIZE,
                                   figname="%s %s components Time Series" % (variable, mode))

            fig = plt.figure(figsize=(plotter.config.DEFAULT_SIZE[0], 5))
            plt.imshow(ca.components_)
            plt.xlabel("Region")
            plt.ylabel("%s component" % mode)
            plt.title("%s components" % mode)
            plt.colorbar()
            plt.tight_layout()
            if plotter.config.SAVE_FLAG:
                plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "%s.%s" % (mode, plotter.config.FIG_FORMAT)))
            if plotter.config.SHOW_FLAG:
                plt.show()
            else:
                plt.close(fig)
    return ca.components_, ca_ts, ca

# Example about how ICA works:
# from sklearn.decomposition import FastICA
# np.random.seed(0)
# n_samples = 2000
# time = np.linspace(0, 8, n_samples)
# s1 = np.sin(2 * time)
# s2 = np.sign(np.sin(3 * time))
# s3 = signal.sawtooth(2 * np.pi * time)
# S = np.c_[s1, s2, s3]
# S += 0.2 * np.random.normal(size=S.shape)
# S /= S.std(axis=0)
# A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
# X = np.dot(S, A.T)
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(X)
# fig = plt.figure()
# models = [X, S, S_]
# names = ['mixtures', 'real sources', 'predicted sources']
# colors = ['red', 'blue', 'orange']
# for i, (name, model) in enumerate(zip(names, models)):
#     plt.subplot(4, 1, i+1)
#     plt.title(name)
#     for sig, color in zip (model.T, colors):
#         plt.plot(sig, color=color)

# fig.tight_layout()
# plt.show()


def compute_plot_ica(data, variable="BOLD", n_components=10, time=None, plotter=None):
    return compute_plot_components(data, MODE=FastICA, variable=variable, n_components=n_components,
                                   time=time, plotter=plotter)


def compute_plot_pca(data, variable="BOLD", n_components=10, time=None, plotter=None):
    return compute_plot_components(data, MODE=PCA, variable=variable, n_components=n_components,
                                   time=time, plotter=plotter)


def compute_data_PSDs(data, dt, ftarg, transient=None, average_region_ps=False):
    # Time and frequency
    fs = 1000.0 / dt
    if transient is None:
        transient = 0
    else:
        transient = int(np.ceil(transient / dt))  # in data points
    # Remove possible transient and transpose time and signals:
    data = data[transient:].T
    Ndata = data.shape[1]

    # Window:
    nperseg = compute_nperseg(fs, Ndata)

    # Compute Power Spectrum
    f, Pxx_den = welch(data, fs, nperseg=nperseg)

    if average_region_ps:
        # Average power spectra across regions for the case of 1D computations
        Pxx_den = Pxx_den.mean(axis=0, keepdims=True)

    # Compute spectrum interpolation...
    interp = interp1d(f, Pxx_den, kind='linear', axis=1,
                      copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)
    # ...to the target frequencies:
    Pxx_den = interp(ftarg)

    # Normalize to get a density summing to 1.0:
    for ii in range(Pxx_den.shape[0]):
        Pxx_den[ii] = Pxx_den[ii] / np.sum(Pxx_den[ii])

    return Pxx_den


def raw_data_or_time_series(data):
    if isinstance(data, (tuple, list)):
        # For raw TVB monitor results
        ts = data[1]
        time = data[0]
    elif isinstance(data, np.ndarray):
        ts = data
        time = None
    else:
        # For TVB TimeSeries instances
        ts = data.data
        time = data.time
        sample_period = data.sample_period
        return ts, time, sample_period
    if time is not None:
        sample_period = np.mean(np.diff(time))
    else:
        sample_period = None
    return ts, time, sample_period


def compute_data_PSDs_from_raw(raw_results, ftarg, inds=None, transient=None, average_region_ps=False):
    if inds is None:
        inds = slice(None)
    data, time, sample_period = raw_data_or_time_series(raw_results)
    return compute_data_PSDs(data[:, 0, inds, 0].squeeze(),
                             sample_period, ftarg,
                             transient=transient, average_region_ps=average_region_ps)


# def _compute_tensorpac(xphases, xamplitudes, fs=10000.0, methods=(5, 0, 0),
#                        regions_label="", plot_flag=False, ax=None, **kwargs):
#     try:
#         from tensorpac import Pac
#     except:
#         import subprocess
#         print("Installing tensorpac...")
#         p = subprocess.Popen("pip install tensorpac", stdout=subprocess.PIPE, shell=True)
#         print(p.communicate())
#         from tensorpac import Pac
#
#     p = Pac(**kwargs)
#
#     # extract all of the phases and amplitudes
#     phases = p.filter(fs, xphases, ftype='phase', n_jobs=1)
#     amplitudes = p.filter(fs, xamplitudes, ftype='amplitude', n_jobs=1)
#
#     p.idpac = methods
#
#     # compute only the pac without filtering
#     xpac = p.fit(phases, amplitudes)
#
#     if plot_flag:
#         if ax is None:
#             plt.figure(figsize=(10, 5))
#         else:
#             plt.axes(ax)
#         # plot your Phase-Amplitude Coupling :
#         ax = p.comodulogram(xpac.mean(-1), title=regions_label, cmap='jet')
#         # Get the images on an axis
#         im = ax.images
#         # Assume colorbar was plotted last one plotted last
#         cb = im[-1].colorbar
#         cb.set_label("PAC (%s)" % str(methods), rotation=270)
#     return xpac, p
#
#
# def compute_tensorpac(results, pairs=None, methods=(5, 0, 0), transient=None,
#                       region_labels=[], plot_flag=False, figpath=None, **kwargs):
#     data, time, sample_period = raw_data_or_time_series(results)
#     if transient is None:
#         transient = 0
#     else:
#         transient = int(np.ceil(transient / sample_period))  # in data points
#     data = data[transient:]
#     fs = 1000 / sample_period
#     xpacs = []
#     if pairs is None:
#         pairs = np.tile(np.arange(0, data.shape[2]), (2, 1)).T
#
#     def fun(pair, region_labels):
#         if pair[0].item() == pair[1].item():
#             return "%s" % str(region_labels[0].item())
#         else:
#             return "%s - %s" % (str(region_labels[0].item()), str(region_labels[1].item()))
#
#     Nregs = len(region_labels)
#     if Nregs < len(pairs):
#         region_labels = np.array([fun(pair, pair) for pair in pairs])
#     else:
#         region_labels = np.array([fun(pair, region_labels[pair]) for pair in pairs])
#     if plot_flag:
#         Nregs2 = int(np.ceil(Nregs / 2))
#         axshape = (Nregs2, 2)
#         fig, axes = plt.subplots(nrows=Nregs2, ncols=2, figsize=(20, 10 * Nregs2),
#                                  sharex=True, sharey=True)
#     for iP, (pair, reg_lbl) in enumerate(zip(pairs, region_labels)):
#         xpacs.append(_compute_tensorpac(data[:, 0, pair[0]].squeeze().T,
#                                         data[:, 0, pair[1]].squeeze().T,
#                                         fs=fs, methods=methods, regions_label=reg_lbl,
#                                         plot_flag=plot_flag,
#                                         ax=axes[np.unravel_index(iP, axshape)] if plot_flag else None,
#                                         **kwargs)[0])
#     xpacs = np.array(xpacs).squeeze()
#     if plot_flag:
#         vmin = xpacs.min()
#         vmax = xpacs.max()
#         for ax in axes.flatten():
#             im = ax.images
#             im[-1].set_clim(vmin, vmax)
#             fig.canvas.flush_events()
#         fig.tight_layout()
#         if os.path.isdir(figpath):
#             plt.savefig(os.path.join(figpath, "TrasferMetricsPACs.png"))
#     return {"syncij": pairs, "pac": xpacs}
#
#
# def compute_task_transfer_metrics(raw_results, transient, region_labels, taskinds, theta, gamma, ftarg, Pxx_den=None,
#                                   methods=(5, 0, 0), plot_flag=True, figpath=None):
#     def compute_freq_P_ratio(P, band_freqs, ftarg):
#         finds = np.where(np.logical_and(ftarg >= band_freqs[0], ftarg <= band_freqs[-1]))[0]
#         return P[:, finds].sum(axis=1) / P.sum(axis=1)
#
#     if Pxx_den is None:
#         # Compute PSDs:
#         Pxx_den = compute_data_PSDs_from_raw(
#             raw_results, ftarg, inds=taskinds, transient=transient, average_region_ps=False)
#
#     Pth = compute_freq_P_ratio(Pxx_den, theta, ftarg)
#     Pgm = compute_freq_P_ratio(Pxx_den, gamma, ftarg)
#     Pgm_th_ratio = Pgm / Pth
#     PAC = compute_tensorpac(raw_results[:, 0, taskinds], transient=transient, methods=methods,
#                             region_labels=region_labels[taskinds], plot_flag=plot_flag, figpath=figpath,
#                             f_pha=theta, f_amp=gamma)['pac'].mean(axis=(1, 2))
#
#     metrics = np.array([Pth, Pgm, Pgm_th_ratio, PAC])
#
#     try:
#         from xarray import DataArray
#         metrics = DataArray(metrics,
#                             dims=["Measure", "Region"],
#                             coords={"Measure": ['Pth', 'Pgm', 'Pgm_th_ratio', 'PAC'],
#                                     "Region": region_labels[taskinds]},
#                             name="Task dynamics transfer metrics")
#         if plot_flag:
#             metrics.plot(y="Region", row="Measure", sharex=False, figsize=(5, 20))
#             if os.path.isdir(figpath):
#                 plt.savefig(os.path.join(figpath, "TrasferMetrics.png"))
#
#     except Exception as e:
#         import warnings
#         warning.warn(e)
#
#     return metrics


def intval(x):
    if x == 0.0:
        return 0
    elif np.abs(x) < 0.1:
        return int(np.abs(np.log10(x)))
    elif x < 1.0:
        return int(10*x)
    else:
        return int(x)


def dump_pickled_time_series(time_series, filepath):
    dump_pickled_dict({"data": time_series.data[:, :, :, 0],
                       "dimensions_labels": np.array(time_series.labels_ordering),
                       "Time": time_series.time, "time_unit": time_series.time_unit,
                       "sample_period": time_series.sample_period,
                       "State Variable": np.array(time_series.variables_labels),
                       "Region": np.array(time_series.space_labels)},
                      filepath)
    return filepath


def load_pickled_time_series(filepath, connectivity=None):
    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    from tvb.datatypes.connectivity import Connectivity

    tsdict = load_pickled_dict(filepath)

    data = tsdict.get("time_series", tsdict.get("data", None))
    if data is None:
        raise ValueError("Time Series data in %s is None!" % filepath)
    if isinstance(data, list):
        data = np.array(data)
    while data.ndim < 4:
        data = np.expand_dims(data, axis=-1)

    dimensions = list(tsdict.get("dims",  # xarray.Datarray
                                  tsdict.get("dimensions_labels",  # TVB TimeSeries
                                             tsdict.get("dimensions",
                                                        tsdict.get("labels_ordering",  # TVB TimeSeries
                                                                   [])))))
    DEFAULT_DIMENSIONS = ["Time", "State Variable", "Region", "Mode"]
    for ii in range(data.ndim):
        if len(dimensions) < ii+1:
            dimensions.append(DEFAULT_DIMENSIONS[ii])

    # Legacy:
    labels_dimensions = tsdict.get("coords", # xarray.Datarray
                                   tsdict.get("labels_dimensions", # TVB TimeSeries
                                              dict()))
    for label, key in zip(dimensions[:3],
                          ["time", "state_variables", "region_labels"]):  # TVB TimeSeries legacy
        if label not in labels_dimensions:
            val = tsdict.get(label, tsdict.get(key, None))
            if val is not None:
                labels_dimensions[label] = val
        if isinstance(labels_dimensions[label], dict):
            labels_dimensions[label] = np.array(labels_dimensions[label]["data"])

    time = tsdict.get("Time", tsdict.get("time", labels_dimensions.get("Time", None)))
    if time is not None:
        sample_period = tsdict.get("sample_period", np.mean(np.diff(time)))
    else:
        sample_period = None

    if isinstance(connectivity, Connectivity):
        from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray
        labels_dimensions["Region"] = connectivity.region_labels
        return TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
            data=data, time=time,
            connectivity=simulator.connectivity,
            labels_ordering=dimensions,
            labels_dimensions=labels_dimensions,
            sample_period=sample_period)
    else:
        from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesXarray
        return TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
                                data=data,  time=time,
                                labels_ordering=dimensions,
                                labels_dimensions=labels_dimensions,
                                sample_period=sample_period)


def joinstr(lst, connstr="_"):
    if len(lst):
        if len(lst[0]):
            outstr = lst[0]
            for lstr in lst[1:]:
                if len(lstr):
                    outstr += "%s%s" % (connstr, lstr)
        else:
            return joinstr(lst[1:], connstr)
    else:
        return ""
    return outstr
