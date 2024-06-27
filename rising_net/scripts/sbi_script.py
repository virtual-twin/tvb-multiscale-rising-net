# -*- coding: utf-8 -*-
import os
import warnings
import pickle

import numpy as np
from xarray import DataArray
import torch
from sbi.inference.base import infer, prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNPE
from sbi import utils as utils
from sbi import analysis as analysis

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from rising_net.scripts.base import *
from rising_net.scripts.tvb_script import run_workflow, load_connectome


def build_priors(config):
    if config.PRIORS_DIST.lower() == "normal":
        priors_normal = torch.distributions.Normal(loc=torch.as_tensor(config.prior_loc),
                                                   scale=torch.as_tensor(config.prior_sc))
        #     priors = torch.distributions.MultivariateNormal(loc=torch.as_tensor(config.prior_loc),
        #                                                     scale_tril=torch.diag(torch.as_tensor(config.prior_sc)))
        priors = torch.distributions.Independent(priors_normal, 1)
    else:
        priors = utils.torchutils.BoxUniform(low=torch.as_tensor(config.prior_min),
                                             high=torch.as_tensor(config.prior_max))
    return priors


def sample_priors_for_sbi(config=None):
    config = assert_config(config, return_plotter=False)
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)
    dummy_sim = lambda priors: priors
    priors = build_priors(config)
    simulator, priors = prepare_for_sbi(dummy_sim, priors)
    priors_samples, sim_res = simulate_for_sbi(dummy_sim, proposal=priors,
                                               num_simulations=config.N_SIMULATIONS,
                                               num_workers=config.SBI_NUM_WORKERS)
    return priors_samples, sim_res


def write_posterior(posterior, iG=None, iR=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_filepath(config, iG, iR, label)
    with open(filepath, "wb") as handle:
        pickle.dump(posterior, handle)


def compute_diagnostics(samples, config, priors=None, map=None, ground_truth=None):
    if priors is None:
        priors = build_priors(config)
    priors_std = priors.stddev.numpy()
    res = {}
    res["samples"] = samples.numpy()
    if map is not None:
        if not isinstance(map, np.ndarray):
            map = map.numpy()
        res['map'] = map
    res['mean'] = samples.mean(axis=0).numpy()
    res['std'] = samples.std(axis=0).numpy()
    if ground_truth is not None:
        res["diff"] = ground_truth - res['mean']
        res["accuracy"] = np.maximum(config.MIN_ACCURACY, 100*(1.0 - np.abs(res['diff']/ground_truth)))
        res["zscore"] = res["diff"] / res["std"]
        res["zscore_prior"] = res["diff"] / priors_std
    res["shrinkage"] = 1 - np.power(res['std'], 2) / np.power(priors_std, 2)
    return res


def safely_set_key_list_iR(d, iR, key, defval):
    # Make sure the key exists:
    if key not in d:
        d[key] = []
        if iR == -1:
            iR = 0
    else:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
        if iR == -1:
            iR = len(d[key])
    # Make sure that the size of the d[key] is adequate,
    # by filling in default values:
    while len(d[key]) < iR + 1:
        d[key].append(defval)
    if not isinstance(d[key][iR], list):
        d[key][iR] = [d[key][iR]]
    return d, iR


def safely_append_item_iR(d, iR, key, val):
    # Make sure the key exists and determine iR:
    d, iR = safely_set_key_list_iR(d, iR, key, [])
    # Append now the current value:
    d[key][iR].append(val)
    return d


def write_posterior_samples(results, config,
                            iG=None, iR=None, label="",
                            samples_fit=None, save_samples=True):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_samples_filepath(config, iG, iR, label)
    if samples_fit is None:
        if os.path.isfile(filepath):
            with open(filepath, "rb") as handle:
                samples_fit = pickle.load(handle)
        else:
            samples_fit = {}
    if iR is None:
        iR = -1
    # Get G for this run:
    if iG is not None:
        samples_fit["G"] = config.Gs[iG]
    for key, val in results.items():
        samples_fit = safely_append_item_iR(samples_fit, iR, key, val)
    if not save_samples:
        del samples_fit["samples"]
    with open(filepath, "wb") as handle:
        pickle.dump(samples_fit, handle)
    return samples_fit


def filepath_prefixes(filepath, iG=None, iR=None, label=""):
    if iG is not None:
        filepath += "_iG%02d" % iG
    if iR is not None:
        filepath += "_iR%02d" % iR
    if len(label):
        filepath += "_%s" % label
    return filepath


def construct_filepath(default_filepath, config, iG=None, iR=None, label="", filepath=None, extension=None):
    if filepath is None or extension is None:
        filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, default_filepath))
    filepath = filepath_prefixes(filepath, iG, iR, label)
    return "%s%s" % (filepath, extension)


def posterior_filepath(config, iG=None, iR=None, label="", filepath=None, extension=None):
    return construct_filepath(config.POSTERIOR_PATH, config, iG, iR, label, filepath, extension)


def posterior_samples_filepath(config, iG=None, iR=None, label="", filepath=None, extension=None):
    return construct_filepath(config.POSTERIOR_SAMPLES_PATH, config, iG, iR, label, filepath, extension)


def load_posterior(iG=None, iR=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_filepath(config, iG, iR, label)
    with open(filepath, "rb") as handle:
        posterior = pickle.load(handle)
    return posterior


def load_posterior_samples(iG=None, iR=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_samples_filepath(config, iG, iR, label)
    with open(filepath, "rb") as handle:
        samples_fit = pickle.load(handle)
    return samples_fit


def add_posterior_samples_iR(all_samples, samples_iR):
    for key, val in samples_iR.items():
        if key != "G":
            if key not in all_samples:
                all_samples[key] = []
            vals = []
            for vl in val[0]:
                if isinstance(vl, torch.Tensor):
                    vals.append(vl.numpy())
                elif isinstance(vl, DataArray):
                    vals.append(vl.values)
                else:
                    vals.append(vl)
            all_samples[key].append(vals)
    return all_samples


def load_posterior_samples_all_runs(iG, runs=None, label="", samples=None, config=None):
    config = assert_config(config, return_plotter=False)
    if samples is None:
        samples = OrderedDict()
    if runs is None:
        runs = list(range(config.N_FIT_RUNS))
    for iR in runs:
        try:
            samples_iR = load_posterior_samples(iG, iR, label, config)
            samples = add_posterior_samples_iR(samples, samples_iR)
        except Exception as e:
            warnings.warn("Failed to load posterior samples for iG=%d, G=%g, iR=%d!\n%s" % (iG, config.Gs[iG], iR, str(e)))
    return samples


def sbi_train(priors, priors_samples, sim_res, verbosity):
    # Initialize the inference algorithm class instance:
    inference = SNPE(prior=priors)
    # Append to the inference the priors samples and simulations results
    # and train the network:
    density_estimator = inference.append_simulations(priors_samples, sim_res).train()
    keep_building = -10
    posterior = None
    exception = "None"
    while keep_building < 0:
        try:
            # Build the posterior:
            if verbosity:
                print("\nBuilding the posterior...")
            posterior = inference.build_posterior(density_estimator)
            keep_building = 0
        except Exception as e:
            exception = e
            warnings.warn(str(e) + "\nTrying again for the %dth time!" % (10 + keep_building + 2))
            keep_building += 1
    if posterior is None:
        raise Exception(exception)
    return posterior


def sbi_estimate(posterior, target, n_samples_per_run, verbosity=1):
    posterior.set_default_x(target)
    if verbosity:
        print("\nSetting estimation target...")
        if verbosity > 1:
            print("\ntarget = %s" % str(target))
    if verbosity:
        print("\nSampling %d samples from the posterior..." % n_samples_per_run)
        tic = time.time()
    samples = posterior.sample((n_samples_per_run,), show_progress_bars=verbosity>0)
    if verbosity:
        print("\nDONE sampling in %g secs!" % (time.time() - tic))
        print("\nSampling to find MAP with %d initial samples and %d samples to optimize..." %
              (n_samples_per_run, int(0.1*n_samples_per_run)))
        tic = time.time()
    MAP = posterior.map(num_init_samples=n_samples_per_run,
                        num_to_optimize=int(0.1 * n_samples_per_run),
                        show_progress_bars=verbosity>0).numpy()
    if verbosity:
        print("\nDONE sampling for MAP in %g secs!" % (time.time() - tic))
        if verbosity > 1:
            print("\nMAP = %s" % str(MAP))
    return posterior, samples, MAP


def sbi_infer(priors, priors_samples, sim_res, n_samples_per_run, target, verbosity):
    # Train the neural network to approximate the posterior and return the posterior estimation:
    return sbi_estimate(sbi_train(priors, priors_samples, sim_res, verbosity),
                        target, n_samples_per_run, verbosity)
