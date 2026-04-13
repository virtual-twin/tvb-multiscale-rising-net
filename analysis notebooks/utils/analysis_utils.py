"""
Spike synchronization analysis utilities.

Functions for computing population rates, phase-locking values (PLV),
internal synchrony, cross-correlations, spike-triggered averages, and
cross-spectral density plots.
"""

import numpy as np
import matplotlib.colors as mcolors
from collections import Counter
from scipy.signal import butter, filtfilt, hilbert, correlate
from itertools import combinations, product
from astropy.stats import rayleightest


def darken_color(color, factor=0.6):
    """Return a darker version of *color* by scaling its RGB values.

    Parameters
    ----------
    color : any matplotlib-compatible color
    factor : float in [0, 1]  (0 = black, 1 = original)
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * factor)


# ============================================================
# Spike extraction & population rate
# ============================================================

def extract_spike_lists(spikes_data, cell_type, hemisphere='Right', t_min=0.0):
    """
    Extract per-neuron spike-time lists from a spikes dict.
    """
    spikes_series = spikes_data.get(cell_type, None)
    if spikes_series is None:
        raise KeyError(f"Cell type '{cell_type}' not found in spikes data. "
                       f"Available keys: {list(spikes_data.keys())}")

    if hemisphere is not None:
        indices = [idx for idx in spikes_series.index if hemisphere in str(idx)]
    else:
        indices = list(spikes_series.index)

    all_senders = []
    all_times = []
    for idx in indices:
        region_data = spikes_series[idx]
        senders = region_data.get('senders', [])
        times = region_data.get('times', [])
        all_senders.extend(senders)
        all_times.extend(times)

    all_senders = np.array(all_senders)
    all_times = np.array(all_times) / 1000.0  # ms -> seconds

    keep = all_times >= t_min
    all_senders = all_senders[keep]
    all_times = all_times[keep]

    spike_lists = []
    for uid in np.unique(all_senders):
        mask = all_senders == uid
        spike_lists.append(all_times[mask])
    return spike_lists


def population_rate(spike_lists, duration, dt):
    """Population firing rate from spike times.

    Computes the firing rate for each neuron individually
    (spike count per bin / dt) and then averages across neurons.
    """
    bins = np.arange(0, duration + dt, dt)
    n_bins = len(bins) - 1
    all_rates = np.zeros((len(spike_lists), n_bins))
    for i, spikes in enumerate(spike_lists):
        counts, _ = np.histogram(spikes, bins=bins)
        all_rates[i] = counts / dt
    mean_rate = np.mean(all_rates, axis=0)
    return mean_rate, bins[:-1]


def spike_train_binary(spikes, duration, dt):
    """Binary spike train from spike times."""
    bins = np.arange(0, duration + dt, dt)
    train, _ = np.histogram(spikes, bins=bins)
    return train


# ============================================================
# Synchrony & synchronization index
# ============================================================

def compute_internal_synchrony(spike_lists, duration, bin_size=0.003):
    """
    Mean pairwise Pearson correlation of binned spike trains.

    Parameters
    ----------
    spike_lists : list of arrays
        Per-neuron spike-time arrays (seconds).
    duration : float
        Total duration in seconds.
    bin_size : float
        Bin width for binarisation (default 3 ms).

    Returns
    -------
    float
        Mean pairwise correlation (NaN-safe).
    """
    binary_trains = [spike_train_binary(spk, duration, bin_size) for spk in spike_lists]
    pair_corrs = []
    for i, j in combinations(range(len(binary_trains)), 2):
        if np.std(binary_trains[i]) == 0 or np.std(binary_trains[j]) == 0:
            continue
        r = np.corrcoef(binary_trains[i], binary_trains[j])[0, 1]
        pair_corrs.append(r)
    return np.nanmean(pair_corrs)


def compute_cross_population_synchrony(spike_lists_a, spike_lists_b,
                                       duration, bin_size=0.003):
    """
    Mean pairwise Pearson correlation between two populations' binned spike trains.

    Parameters
    ----------
    spike_lists_a, spike_lists_b : list of arrays
        Per-neuron spike-time arrays (seconds) for populations A and B.
    duration : float
        Total duration in seconds.
    bin_size : float
        Bin width for binarisation (default 3 ms).

    Returns
    -------
    float
        Mean pairwise correlation across all (i, j) pairs with i in A, j in B.
    """
    trains_a = [spike_train_binary(spk, duration, bin_size) for spk in spike_lists_a]
    trains_b = [spike_train_binary(spk, duration, bin_size) for spk in spike_lists_b]
    pair_corrs = []
    for i, j in product(range(len(trains_a)), range(len(trains_b))):
        if np.std(trains_a[i]) == 0 or np.std(trains_b[j]) == 0:
            continue
        r = np.corrcoef(trains_a[i], trains_b[j])[0, 1]
        pair_corrs.append(r)
    return np.nanmean(pair_corrs)


def compute_synchronization_index(spike_times_list, t_start, t_stop, bin_size):
    """
    Computes the variance-based Synchronization Index (SI) for a population of neurons.

    Parameters
    ----------
    spike_times_list : list of arrays
        Per-neuron spike-time arrays (seconds).
    t_start : float
        Start time of the analysis window (seconds).
    t_stop : float
        End time of the analysis window (seconds).
    bin_size : float
        Size of the time bin (seconds).

    Returns
    -------
    float
        Synchronization Index (between 0 and 1).
    """
    num_neurons = len(spike_times_list)

    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    num_bins = len(bins) - 1

    binned_spikes = np.zeros((num_neurons, num_bins))
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes, bins=bins)
        binned_spikes[i, :] = counts

    pop_activity = np.mean(binned_spikes, axis=0)
    var_pop = np.var(pop_activity)
    var_indiv = np.var(binned_spikes, axis=1)
    mean_var_indiv = np.mean(var_indiv)

    if mean_var_indiv > 0:
        si = var_pop / mean_var_indiv
    else:
        si = 0.0

    return si


# ============================================================
# Phase-Locking Value (PLV) & Rayleigh test
# ============================================================

def compute_plv(input_spike_times, output_continuous_signal, sampling_rate, band=(25, 60)):
    """
    Phase-Locking Value between input spikes and gamma-filtered output signal.

    Steps:
      1. Bandpass-filter the output signal to the gamma band.
      2. Extract instantaneous phase via the Hilbert transform.
      3. Sample the phase at each spike time.
      4. Return the mean resultant vector length (PLV).

    Parameters
    ----------
    input_spike_times : array
        Spike times in seconds.
    output_continuous_signal : array
        Continuous signal (e.g., population rate or simulated LFP).
    sampling_rate : float
        Sampling frequency in Hz.
    band : tuple
        (low, high) frequency band for bandpass filtering.
    """
    nyquist = 0.5 * sampling_rate
    b, a = butter(N=4, Wn=[band[0] / nyquist, band[1] / nyquist], btype='bandpass')
    gamma_signal = filtfilt(b, a, output_continuous_signal)

    analytic_signal = hilbert(gamma_signal)
    instantaneous_phase = np.angle(analytic_signal)

    spike_indices = np.round(input_spike_times * sampling_rate).astype(int)
    spike_indices = spike_indices[spike_indices < len(instantaneous_phase)]
    spike_phases = instantaneous_phase[spike_indices]

    return np.abs(np.mean(np.exp(1j * spike_phases)))


def compute_rayleigh_pvalue(spike_phases):
    """
    Computes the Rayleigh test p-value for a set of circular phases.

    Parameters
    ----------
    spike_phases : array
        1-D array of phase angles (radians).
    """
    p_value = rayleightest(spike_phases)
    return p_value


def extract_spike_phases(spike_times, continuous_signal, sampling_rate, band=(25, 60)):
    """
    Bandpass-filter *continuous_signal* to *band*, extract instantaneous phase
    via Hilbert transform, and return the phase at each spike time.

    Returns
    -------
    spike_phases : 1-D array of phases in (-pi, pi]
    gamma_signal : 1-D array, the bandpass-filtered signal
    """
    nyquist = 0.5 * sampling_rate
    b, a = butter(N=4, Wn=[band[0] / nyquist, band[1] / nyquist], btype='bandpass')
    gamma_signal = filtfilt(b, a, continuous_signal)
    analytic = hilbert(gamma_signal)
    inst_phase = np.angle(analytic)

    spike_idx = np.round(spike_times * sampling_rate).astype(int)
    spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < len(inst_phase))]
    return inst_phase[spike_idx], gamma_signal


# ============================================================
# Amplitude Envelope Correlation (AEC) 
# ============================================================
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import pearsonr

def compute_aec_standard(x, y, fs, band=[25, 60]):
    """
    Computes the standard Amplitude Envelope Correlation (AEC) 
    without orthogonalization.
    
    Parameters:
    x, y : 1D numpy arrays (the raw signals)
    fs   : Sampling frequency in Hz
    band : List with [low_freq, high_freq] for the band-pass filter
    
    Returns:
    aec_value : Pearson correlation coefficient
    p_value   : Statistical significance of the correlation
    """
    
    # 1. Band-pass filter to isolate the frequency band
    nyq = 0.5 * fs
    b, a = butter(3, [band[0]/nyq, band[1]/nyq], btype='band')
    x_filt = filtfilt(b, a, x)
    y_filt = filtfilt(b, a, y)

    # 2. Extract analytic signals using the Hilbert transform
    x_ana = hilbert(x_filt)
    y_ana = hilbert(y_filt)

    # 3. Extract the amplitude envelopes (absolute magnitude)
    env_x = np.abs(x_ana)
    env_y = np.abs(y_ana)

    # 4. Compute Pearson correlation between the two envelopes
    aec_value, p_value = pearsonr(env_x, env_y)
    
    return aec_value, p_value



# ============================================================
# Cross-correlation
# ============================================================

def compute_spike_cross_correlation(input_spike_times, output_spike_times,
                                    duration, bin_size=0.0005):
    """
    Normalized cross-correlation between two spike trains.

    Parameters
    ----------
    input_spike_times : array
        Presynaptic spike times in seconds.
    output_spike_times : array
        Postsynaptic spike times in seconds.
    duration : float
        Total simulation time in seconds.
    bin_size : float
        Discretization bin size in seconds (default 0.5 ms).
    """
    time_bins = np.arange(0, duration + bin_size, bin_size)
    input_binned, _ = np.histogram(input_spike_times, bins=time_bins)
    output_binned, _ = np.histogram(output_spike_times, bins=time_bins)

    cross_corr = correlate(output_binned, input_binned, mode='full')

    max_lag_bins = len(input_binned) - 1
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size

    norm_factor = np.sqrt(np.sum(input_binned**2) * np.sum(output_binned**2))
    if norm_factor > 0:
        cross_corr = cross_corr / norm_factor

    return lags, cross_corr


# ============================================================
# Spike-Triggered Average (STA)
# ============================================================

def compute_sta(mf_gamma_wave, dcn_spike_times, fs, window_ms=50):
    """
    Compute the Spike-Triggered Average (no plotting).

    Parameters
    ----------
    mf_gamma_wave : 1-D array
        Continuous, bandpass-filtered MF gamma signal.
    dcn_spike_times : 1-D array
        DCN spike times in **seconds**.
    fs : float
        Sampling frequency (Hz).
    window_ms : float
        Half-window in ms before/after each spike.

    Returns
    -------
    sta : 1-D array – spike-triggered average
    time_axis : 1-D array – time axis in ms
    """
    half_window_idx = int((window_ms / 1000.0) * fs)
    spike_indices = np.round(dcn_spike_times * fs).astype(int)

    snippets = []
    for idx in spike_indices:
        if (idx - half_window_idx >= 0) and (idx + half_window_idx < len(mf_gamma_wave)):
            snippets.append(mf_gamma_wave[idx - half_window_idx : idx + half_window_idx])

    snippets = np.array(snippets)
    sta = np.mean(snippets, axis=0)
    time_axis = np.linspace(-window_ms, window_ms, len(sta))
    return sta, time_axis


# ============================================================
# Plotting helpers
# ============================================================

def plot_clean_overlay(time_array, mf_gamma_wave, dcn_spike_list,
                       window_start=200, window_end=300, num_neurons=20,
                       title='Spike-Phase Overlay (Micro-Zoom)'):
    """
    Plots a clean overlay of the MF gamma wave and a subset of DCN spikes.

    Parameters
    ----------
    time_array : array
        Time points (in ms).
    mf_gamma_wave : array
        Continuous filtered gamma signal.
    dcn_spike_list : list of arrays
        Per-neuron spike times (ms).
    window_start, window_end : float
        Micro-zoom window in ms.
    num_neurons : int
        How many neurons to show in the raster.
    title : str
        Plot title.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))

    idx_start = np.searchsorted(time_array, window_start)
    idx_end = np.searchsorted(time_array, window_end)

    ax1.plot(time_array[idx_start:idx_end], mf_gamma_wave[idx_start:idx_end],
             color='gray', linewidth=2, label='MF Gamma')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Gamma Amplitude', color='gray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='gray')

    subset_spikes = dcn_spike_list[:num_neurons]
    zoomed_spikes = []
    for spikes in subset_spikes:
        valid_spikes = spikes[(spikes >= window_start) & (spikes <= window_end)]
        zoomed_spikes.append(valid_spikes)

    ax2 = ax1.twinx()
    ax2.eventplot(zoomed_spikes, color='red', linelengths=0.6, linewidths=1.5, alpha=0.8)
    ax2.set_ylabel(f'{num_neurons} Example DCN Neurons', color='red', fontsize=12)
    ax2.set_yticks([])

    plt.xlim(window_start, window_end)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_cross_spectral_density_mean_std(csd_per_run, freqs, gamma_band=(25, 60),
                                          freq_max=100, condition_labels=None,
                                          condition_colors=None, output_path=None):
    """
    Plots mean +/- STD of CSD magnitude across simulation runs for each condition,
    together with a barplot of mean CSD in the gamma band.
    Uses log-log scale for the spectral plot.

    Parameters
    ----------
    csd_per_run : dict
        {condition: list of 1-D CSD magnitude arrays (one per run)}
    freqs : array
        Frequency vector (Hz).
    gamma_band : tuple
        (low, high) Hz for the gamma-band average.
    freq_max : float
        Upper frequency limit for the spectral plot.
    condition_labels : dict or None
        {condition: display label}.
    condition_colors : dict or None
        {condition: color}.
    output_path : str or None
        Base path (without extension) for saving the figure.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu as _mwu
    from NESTlesions.plot_utils import save_figure_multi_format, add_stat_annotation as _add_stat

    if condition_labels is None:
        condition_labels = {c: c for c in csd_per_run}
    if condition_colors is None:
        default_colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
        condition_colors = {c: default_colors[i % len(default_colors)]
                           for i, c in enumerate(csd_per_run)}

    gamma_mask = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])
    freq_mask = (freqs >= 1) & (freqs <= freq_max)

    gamma_means_per_cond = {}
    gamma_stds_per_cond = {}
    gamma_per_run = {}

    for cond, run_csds in csd_per_run.items():
        stack = np.array(run_csds)
        mean_csd = stack.mean(axis=0)
        std_csd = stack.std(axis=0)

        run_gamma_avgs = stack[:, gamma_mask].mean(axis=1)
        gamma_per_run[cond] = run_gamma_avgs
        gamma_means_per_cond[cond] = run_gamma_avgs.mean()
        gamma_stds_per_cond[cond] = run_gamma_avgs.std()

        csd_per_run[cond] = (mean_csd, std_csd)

    fig, (ax_csd, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={'width_ratios': [3, 1]})

    for cond, (mean_csd, std_csd) in csd_per_run.items():
        col = condition_colors[cond]
        lab = condition_labels[cond]
        ax_csd.plot(freqs[freq_mask], mean_csd[freq_mask],
                    label=lab, color=col, linewidth=2)
        ax_csd.fill_between(freqs[freq_mask],
                            (mean_csd - std_csd)[freq_mask],
                            (mean_csd + std_csd)[freq_mask],
                            color=col, alpha=0.2)

    ax_csd.axvline(x=gamma_band[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax_csd.axvline(x=gamma_band[1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax_csd.axvspan(gamma_band[0], gamma_band[1], alpha=0.1, color='gray',
                   label='Gamma Band')

    ax_csd.set_xscale('log')
    ax_csd.set_yscale('log')
    ax_csd.set_xlim(1, freq_max)
    ax_csd.set_xlabel('Frequency (Hz)')
    ax_csd.set_ylabel('CSD Magnitude |Pxy|')
    ax_csd.set_title('Cross-Spectral Density: MOS-CNe\n(mean \u00b1 STD across runs)')
    ax_csd.legend()
    ax_csd.grid(True, linestyle='--', alpha=0.6)

    conds_ordered = list(csd_per_run.keys())
    x_pos = np.arange(len(conds_ordered))
    bar_means = [gamma_means_per_cond[c] for c in conds_ordered]
    bar_stds = [gamma_stds_per_cond[c] for c in conds_ordered]
    bar_colors = [condition_colors[c] for c in conds_ordered]
    bar_labels = [condition_labels[c] for c in conds_ordered]

    ax_bar.bar(x_pos, bar_means, yerr=bar_stds, capsize=5, width=0.55,
               color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    rng = np.random.default_rng(42)
    for ci, cond in enumerate(conds_ordered):
        vals = gamma_per_run[cond]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        dot_color = darken_color(condition_colors[cond])
        ax_bar.scatter(np.full(len(vals), ci) + jitter, vals,
                       color=dot_color, s=12, zorder=5, alpha=0.8)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, rotation=20, ha='right')
    ax_bar.set_ylabel('Mean |Pxy| in gamma band')
    ax_bar.set_title(f'Gamma-band CSD\n({gamma_band[0]}\u2013{gamma_band[1]} Hz)')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)

    ctrl_cond = conds_ordered[0]
    ctrl_vals = gamma_per_run[ctrl_cond]
    max_bar_y = max(bar_means[i] + bar_stds[i] for i in range(len(conds_ordered)))
    spacing = 0.08 * max_bar_y
    for ti, cond in enumerate(conds_ordered[1:]):
        test_vals = gamma_per_run[cond]
        _, p_csd = _mwu(ctrl_vals, test_vals, alternative='two-sided')
        ci_test = conds_ordered.index(cond)
        _add_stat(ax_bar, 0, ci_test, max_bar_y, p_csd, h=spacing * (ti + 1))
    ax_bar.set_ylim(top=max_bar_y + spacing * (len(conds_ordered) + 1))

    plt.tight_layout()

    if output_path is not None:
        save_figure_multi_format(fig, output_path)
        print(f"Figure saved to: {output_path}.[png|eps|svg]")

    plt.show()


def get_significance_marker(p):
    """Return significance marker string for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ============================================================
# Transfer Entropy
# ============================================================

def _discretize_quantile(x, n_bins=5):
    """Discretize a continuous signal into n_bins levels via quantiles."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12  # include rightmost point
    return np.digitize(x, edges[1:])  # 0 … n_bins-1


def transfer_entropy(source, target, k=5, n_bins=5):
    """
    Plug-in (histogram) estimator of transfer entropy (bits):
        TE_{source → target} = H(Y_future | Y_past) − H(Y_future | Y_past, X_past)

    Parameters
    ----------
    source, target : 1-D array-like (same length, already discretized ints)
    k : int – history length (number of past steps)
    n_bins : int – alphabet size (max(source)+1 or max(target)+1)

    Returns
    -------
    te : float  – transfer entropy in bits
    """
    source = np.asarray(source, dtype=int)
    target = np.asarray(target, dtype=int)
    T = len(target)
    if T <= k:
        return 0.0

    base = n_bins
    y_past  = np.zeros(T - k, dtype=np.int64)
    x_past  = np.zeros(T - k, dtype=np.int64)
    for lag in range(k):
        y_past  += target[k - 1 - lag: T - 1 - lag].astype(np.int64) * (base ** lag)
        x_past  += source[k - 1 - lag: T - 1 - lag].astype(np.int64) * (base ** lag)
    y_future = target[k:]

    N = len(y_future)

    joint_yf_yp_xp = Counter(zip(y_future, y_past, x_past))
    joint_yp_xp    = Counter(zip(y_past, x_past))
    joint_yf_yp    = Counter(zip(y_future, y_past))
    count_yp       = Counter(y_past)

    te = 0.0
    for (yf, yp, xp), n_yf_yp_xp in joint_yf_yp_xp.items():
        n_yp_xp = joint_yp_xp[(yp, xp)]
        n_yf_yp = joint_yf_yp[(yf, yp)]
        n_yp    = count_yp[yp]
        te += (n_yf_yp_xp / N) * np.log2(
            (n_yf_yp_xp * n_yp) / (n_yp_xp * n_yf_yp + 1e-300)
        )
    return te


def transfer_entropy_with_shuffle(source_cont, target_cont,
                                  k=5, n_bins=5, n_surrogates=200,
                                  rng=None):
    """
    Compute TE on discretized signals and compare against a shuffle null.

    Parameters
    ----------
    source_cont, target_cont : 1-D float arrays (continuous population rates)
    k, n_bins : TE parameters
    n_surrogates : number of time-shuffled surrogates
    rng : numpy Generator

    Returns
    -------
    te_real : float
    p_value : float  (fraction of surrogates ≥ te_real)
    te_surrogates : 1-D array
    """
    if rng is None:
        rng = np.random.default_rng(0)

    src_d = _discretize_quantile(source_cont, n_bins)
    tgt_d = _discretize_quantile(target_cont, n_bins)

    te_real = transfer_entropy(src_d, tgt_d, k=k, n_bins=n_bins)

    te_surr = np.empty(n_surrogates)
    for s in range(n_surrogates):
        src_shuf = rng.permutation(src_d)
        te_surr[s] = transfer_entropy(src_shuf, tgt_d, k=k, n_bins=n_bins)

    p_value = np.mean(te_surr >= te_real)
    return te_real, p_value, te_surr
