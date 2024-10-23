# -*- coding: utf-8 -*-
import glob
import os
import random
import warnings
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy
import numpy as np
from pandas import Index
from xarray import DataArray, concat
import torch
from sbi.inference.base import infer, prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNPE
from sbi import utils as utils
from sbi import analysis as analysis

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from rising_net.scripts.base import *
from rising_net.scripts.filepaths import get_path, get_res_path, istr, construct_filepath, simres_filepath, \
    figs_filepath
from rising_net.scripts.plot_utils import sbi_pairplot, percent_plot
from rising_net.scripts.tvb_script import run_workflow, load_connectome
from rising_net.scripts.utils import dump_pickled_dict


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


def samples_filepath(config, default_filename="", folder="", iR=None, label="", filepath=None, extension=None):
    if len(default_filename) == 0:
        default_filename = config.SAMPLES_FILE
    return construct_filepath(get_res_path(config, folder),
                              default_filename,
                              iR=iR, label=label,
                              filepath=filepath, extension=extension)


def fitres_filepath(config, filename, iR=None, label="", filepath=None, extension=None):
    return simres_filepath(config, filename, folder=config.FIT_FOLDER,
                           iR=iR, label=label,
                           filepath=filepath, extension=extension)


def train_params_filepath(config, iR=None, label="", filepath=None, extension=None):
    return fitres_filepath(config, config.TRAIN_PARAMS_SAMPLES_FILE,
                           iR=iR, label=label,
                           filepath=filepath, extension=extension)


def fitfigs_filepath(config, filename, iR=None, label="", filepath=None, extension=None):
    return figs_filepath(config, filename, folder=config.FIT_FOLDER,
                         iR=iR, label=label,
                         filepath=filepath, extension=extension)


def posterior_filepath(config, iR=None, label="", filepath=None, extension=None):
    return fitres_filepath(config, config.POSTERIOR_FILE,
                           iR=iR, label=label,
                           filepath=filepath, extension=extension)


def posterior_samples_filepath(config, label="", filepath=None, extension=None):
    return samples_filepath(config, config.SAMPLES_FILE,
                            folder=config.FIT_FOLDER,
                            label=label, filepath=filepath, extension=extension)


def PPC_samples_filepath(config, label="", filepath=None, extension=None):
    return samples_filepath(config, config.SAMPLES_FILE,
                            folder=config.PPC_FOLDER,
                            label=label, filepath=filepath, extension=extension)


def params_pairplot(samples, points=None, metric=None, config=None, figname=None, figpath=None):
    config = assert_config(config, return_plotter=False)
    limits = []
    ticks = []
    labels = []
    if points is not None:
        for pmin, point, pmax in zip(config.prior_min, points, config.prior_max):
            limits.append([pmin, pmax])
            ticks.append(np.sort([pmin, point, pmax]).tolist())
        if metric is None:
            metric = config.OPT_RES_MODE
        for p, point in zip(config.PRIORS_PARAMS_NAMES, points):
            try:
                labels.append("%s %s = %g" % (p, metric, point))
            except Exception as e:
                print(p)
                print(metric)
                print(point)
                print(point.shape)
                raise e
    else:
        for pmin, pmax in zip(config.prior_min, config.prior_max):
            limits.append([pmin, pmax])
        ticks = deepcopy(limits)
        for p in config.PRIORS_PARAMS_NAMES:
            try:
                labels.append("%s" % p)
            except Exception as e:
                print(p)
                raise e
    if config.figures.SAVE_FLAG:
        if figpath is None:
            if figname is None:
                figname = "params_pairplot.png"
            figpath = os.path.join(config.figures.FOLDER_FIGURES, figname)
    return sbi_pairplot(samples, figpath=figpath,
                        save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG,
                        limits=limits, ticks=ticks, points=points, labels=labels)


def sample_train_params_for_sbi(config=None, label="", write_to_files=True, **kwargs):
    MODE = kwargs.pop("MODE",  "TRAIN_PARAMS")
    config = assert_config(config, return_plotter=False, MODE=MODE, **kwargs)
    dummy_sim = lambda priors: priors
    priors = build_priors(config)
    simulator, priors = prepare_for_sbi(dummy_sim, priors)
    samples, _ = simulate_for_sbi(dummy_sim, proposal=priors,
                                  num_simulations=config.N_SIMULATIONS,
                                  num_workers=config.SBI_NUM_WORKERS)
    samples_numpy = samples.numpy()
    print("samples.shape=%s" % str(samples.shape))
    stats = OrderedDict()
    for p in ["min", "max", "mean", "std"]:
        stats[p] = []
        stats[p] = getattr(samples_numpy, p)(axis=0)
        print("\nsamples.%s() =\n%s" % (p, str(stats[p])))

    params_pairplot(samples_numpy, points=stats["mean"], metric="mean", config=config,
                    figpath=fitfigs_filepath(config, "train_params_pairplot.png", label=label))
    if write_to_files:
        path = train_params_filepath(config, label=label)
        torch.save(samples, path)
        filepath, extension = path.split(".")
        dump_pickled_dict(stats, filepath + "_stats.pkl")
        dump_pickled_dict(config, filepath + "_config.pkl")
    return samples, stats


def load_train_params_samples(config, iR=None, label="", filepath=None, extension=None, **kwargs):
    config = assert_config(config, return_plotter=False, **kwargs)
    return torch.load(train_params_filepath(config, iR=iR, label=label, filepath=filepath, extension=extension))


def load_train_params_samples_selection(inds, config, iR=None, label="", filepath=None, extension=None, **kwargs):
    return load_train_params_samples(config,
                                     iR=iR, label=label, filepath=filepath, extension=extension,
                                     **kwargs)[inds]


def compute_diagnostics(samples, config, priors=None, map=None, ground_truth=None):
    if priors is None:
        priors = build_priors(config)
    priors_std = priors.stddev.numpy()
    res = {}
    if not isinstance(samples, np.ndarray):
        samples = samples.numpy()
    res["samples"] = samples
    if map is not None:
        if not isinstance(map, np.ndarray):
            map = map.numpy()
        res['map'] = map
    res['mean'] = samples.mean(axis=0).squeeze()
    res['std'] = samples.std(axis=0).squeeze()
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


def write_posterior(posterior, iR=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_filepath(config, iR, label)
    with open(filepath, "wb") as handle:
        pickle.dump(posterior, handle)


def write_posterior_samples(results_i, config,
                            iR=None, label="",
                            results=None, save_samples=True):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_samples_filepath(config, label=label)  # Write all runs to the same file
    if results is None:
        if os.path.isfile(filepath):
            with open(filepath, "rb") as handle:
                results = pickle.load(handle)
        else:
            results = {}
    if iR is None:
        iR = -1
    for key, val in results_i.items():
        results = safely_append_item_iR(results, iR, key, val)
    if not save_samples:
        del results["samples"]
    with open(filepath, "wb") as handle:
        pickle.dump(results, handle)
    return results


def load_posterior(iR=None, label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_filepath(config, iR, label)
    with open(filepath, "rb") as handle:
        posterior = pickle.load(handle)
    return posterior


def load_posterior_samples(label="", config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_samples_filepath(config, label)
    with open(filepath, "rb") as handle:
        results = pickle.load(handle)
    return results


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


def sbi_train(priors, train_params_samples, sim_res, verbosity):
    # Initialize the inference algorithm class instance:
    inference = SNPE(prior=priors)
    # Append to the inference the training parameter samples and simulations results
    # and train the network:
    density_estimator = inference.append_simulations(train_params_samples, sim_res).train()
    keep_building = -10
    posterior = None
    exception = "None"
    while keep_building < 0:
        try:
            # Build the posterior:
            if verbosity:
                print("\nBuilding the posterior...")
            posterior = inference.build_posterior(density_estimator,
                                                  # sample_with="mcmc"
                                                  )
            keep_building = 0
        except Exception as e:
            exception = e
            warnings.warn(str(e) + "\nTrying again for the %dth time!" % (10 + keep_building + 2))
            keep_building += 1
    if posterior is None:
        raise Exception(exception)
    return posterior


def plot_training_params_samples(params, label="", config=None):
    config = assert_config(config, return_plotter=False)
    fig, axes = params_pairplot(params, points=params.mean(axis=0), metric="mean", config=config,
                                figpath=fitfigs_filepath(config, "train_params_pairplot.png", label=label))
    return fig, axes


def train_posterior(train_params_samples, measures, priors=None,
                    label="", config=None, verbosity=None, plot_flag=True):
    config = assert_config(config, return_plotter=False)
    if verbosity is None:
        verbosity = config.VERBOSITY
    if plot_flag:
        fig, axes = \
            plot_training_params_samples(train_params_samples, label=label, config=config)
    if priors is None:
        priors = build_priors(config)
    posterior = sbi_train(priors,
                          torch.Tensor(train_params_samples),
                          torch.Tensor(measures),
                          verbosity)
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
    samples = posterior.sample((n_samples_per_run,),
                               show_progress_bars=verbosity>0).numpy()
    if verbosity:
        print("\nDONE sampling in %g secs!" % (time.time() - tic))
        print("\nSampling to find MAP with %d initial samples and %d samples to optimize..." %
              (n_samples_per_run, int(0.1*n_samples_per_run)))
        tic = time.time()
    MAP = posterior.map(num_init_samples=n_samples_per_run,
                        num_to_optimize=int(0.1 * n_samples_per_run),
                        show_progress_bars=verbosity>0).numpy().squeeze()
    if verbosity:
        print("\nDONE sampling for MAP in %g secs!" % (time.time() - tic))
        if verbosity > 1:
            print("\nMAP = %s" % str(MAP))
    return posterior, samples, MAP


def plot_samples_measures_and_targets(measures, target=None,
                                      label="", measure_labels=None, measures_plot_fun=None, config=None):
    if measures_plot_fun is not None:
        return measures_plot_fun(measures, target=target,  label=label, measure_labels=measure_labels, config=config)
    else:
        config = assert_config(config, return_plotter=False)
        metric = "target"
        if measures.shape[1] <= 10:
            points = target
            if points is None:
                metric = "mean"
                points = measures.mean(axis=0)
            fig, axes = sbi_pairplot(measures, points=points, metric=metric, labels=measure_labels,
                                     figpath=fitfigs_filepath(config, "measures_pairplot.png", label=label),
                                     save_flag=config.figures.SAVE_FLAG, show_flag=config.figures.SHOW_FLAG)
        else:
            x = np.arange(measures.shape[1])
            axes = percent_plot(x, measures,
                                percentile_min=10, percentile_max=90, n=5,
                                plot_mean=True, plot_median=False,
                                color='b', alpha=0.5, ax=None, mode="linear")
            if target is not None:
                axes.plot(x, target, color='r', linewidth=2)
            fig = plt.gcf()
            if config.figures.SAVE_FLAG:
                plt.savefig(fitfigs_filepath(config, "measure_vs_target_plot.png", label=label))
            if config.figures.SHOW_FLAG:
                plt.show()
            else:
                plt.close(fig)
        return fig, axes


def estimate_posterior_samples(target, posterior, n_samples_per_run=None, label="",
                               measures=None, measure_labels=None,
                               config=None, verbosity=None, plot_flag=True,
                               measures_plot_fun=None):
    config = assert_config(config, return_plotter=False)
    if verbosity is None:
        verbosity = config.VERBOSITY
    if n_samples_per_run is None:
        n_samples_per_run = config.N_POSTERIOR_SAMPLES_PER_RUN
    if plot_flag and measures is not None:
        plot_samples_measures_and_targets(measures, target=target, label=label,
                                          measure_labels=measure_labels, measures_plot_fun=measures_plot_fun,
                                          config=config)
    return sbi_estimate(posterior, target, n_samples_per_run, verbosity)


def sbi_infer(priors, train_params_samples, sim_res, n_samples_per_run, target, verbosity):
    # Train the neural network to approximate the posterior and return the posterior estimation:
    return sbi_estimate(sbi_train(priors, train_params_samples, sim_res, verbosity),
                        target, n_samples_per_run, verbosity)


def get_diagnostic(diagnostic, samples, config, params=None, iR=None):
    if params is None:
        params = config.PRIORS_PARAMS_NAMES
    if iR is None:
        iRs = slice(None)
    else:
        iRs = ensure_list(iR)
    res = np.vstack((samples[diagnostic]))[iRs]
    name = diagnostic
    coords = {"Parameter": params}
    if iR is None and res.shape[0] > 1:
        iR = np.arange(res.shape[0]).astype("i")
    if iR is not None:
        dims = ["Fitting repetition", "Parameter"]
        coords[dims[0]] = ensure_list(iR)
    else:
        dims = ["Parameter"]
        res = res.squeeze()
        name = "%s all samples fitting" % diagnostic
    return DataArray(res, name=name, dims=dims, coords=coords)


def plot_diagnostics(samples, config,
                     diagnostics=["map", "mean", "std", "diff", "accuracy", "zscore_prior", "zscore", "shrinkage"],
                     params=None, iR=None, label="", figsize=None, figname=None, figpath=None):
    res = []
    dnames = []
    for d in diagnostics:
        if d in samples:
            res.append(get_diagnostic(d, samples, config, params=params, iR=iR))
            dnames.append(d)
    res = concat(res, dim=Index(dnames, name="Inference diagnostics"))
    if figsize is None:
        figsize = config.figures.DEFAULT_SIZE
    if res.ndim > 2:
        res.plot.line(x=res.dims[1], row=res.dims[0], hue=res.dims[2],
                      marker="*", markersize=10, figsize=figsize)
    else:
        res.to_series().unstack().plot.bar(figsize=figsize)
    if config.figures.SAVE_FLAG:
        if figpath is None:
            if figname is None:
                figname = 'diagnostics' + "." + config.figures.FIG_FORMAT
        figpath = fitfigs_filepath(config, figname, iR=iR, label=label)
        plt.savefig(figpath)


def plot_infer(samples=None, results=None, points=None, metric=None, iR=None,
               label="", plot_diagnostics_flag=True, config=None):
    config = assert_config(config, return_plotter=False)
    if iR is None:
        iRinds = slice(None)
    else:
        iRinds = ensure_list(iR)
    if samples is None:
        if results is None:
            raise ValueError("Either samples or results dict must be given to plot_infer()!")
        samples = np.vstack(np.vstack(results["samples"])[iRinds]).squeeze()
    if metric is None:
        metric = config.OPT_RES_MODE
    try:
        if points is None:
            if results is not None:
                points = np.vstack(results[config.OPT_RES_MODE])[iRinds].mean(axis=0).squeeze()
    except Exception as e:
        warnings.warn("Failed to get metric %s!\n%s" % str(e))
        metric = None
        points = None
    if config.figures.SAVE_FLAG:
        figname = 'posterior_samples_pairplot' + "." + config.figures.FIG_FORMAT
        figpath = fitfigs_filepath(config, figname, iR=iR, label=label)
    else:
        figpath=None
    fig, axes = params_pairplot(samples, points=points, metric=metric, config=config, figpath=figpath)
    if plot_diagnostics_flag is None:
        plot_diagnostics_flag = len(config.FIT_DIAGNOSTICS)
    if plot_diagnostics_flag and results is not None:
        plot_diagnostics(results, config, diagnostics=config.FIT_DIAGNOSTICS, params=None, iR=iR,  label=label)
    return fig, axes


def infer_workflow(train_params_samples, sim_res, priors=None, target=None, ground_truth=None, config=None,
                   label="", n_samples_per_run=None, measure_labels=None,
                   results=None, iR=None, save_samples=True,
                   plot_flag=True, measures_plot_fun=None, plot_diagnostics_flag=True, verbosity=None):
    config = assert_config(config, return_plotter=False)
    if verbosity is None:
        verbosity = config.VERBOSITY
    if n_samples_per_run is None:
        n_samples_per_run = config.N_POSTERIOR_SAMPLES_PER_RUN
    labeliR = str(label)
    if iR is not None:
        labeliR = joinstr([labeliR, istr(iR)[1:]])
    if priors is None:
        priors = build_priors(config)
    posterior = train_posterior(train_params_samples, sim_res, priors=priors, label=labeliR,
                                config=config, verbosity=verbosity, plot_flag=plot_flag)
    write_posterior(posterior, iR=iR, label=label, config=config)
    posterior, samples, MAP = estimate_posterior_samples(target, posterior,
                                                         n_samples_per_run=n_samples_per_run, label=labeliR,
                                                         measures=sim_res, measure_labels=measure_labels,
                                                         config=config, verbosity=verbosity, plot_flag=plot_flag,
                                                         measures_plot_fun=measures_plot_fun)
    results_i = compute_diagnostics(samples, config, priors=priors, map=MAP, ground_truth=ground_truth)
    results_i["params"] = train_params_samples
    results_i["measures"] = sim_res
    results = write_posterior_samples(results_i, config,
                                      iR=iR, label=label,
                                      results=results, save_samples=save_samples)
    if plot_flag:
        plot_infer(samples, results=results, points=MAP, metric="MAP",
                   iR=iR, label=label, plot_diagnostics_flag=plot_diagnostics_flag, config=config)
    return posterior, results, MAP


def infer_nRuns(train_params_samples, sim_res, priors=None, target=None, groung_truth=None, config=None,
                label="", n_samples_per_run=None, measure_labels=None,
                save_samples=True, plot_flag=True, measures_plot_fun=None, verbosity=None):
    config = assert_config(config, return_plotter=False)
    if verbosity is None:
        verbosity = config.VERBOSITY
    if n_samples_per_run is None:
        n_samples_per_run = config.N_POSTERIOR_SAMPLES_PER_RUN
    if priors is None:
        priors = build_priors(config)
    results_i = None
    if config.N_FIT_RUNS > 1:
        n_samples = train_params_samples.shape[0]
        all_inds = list(range(n_samples))
        n_train_samples = int(np.ceil(1.0 * n_samples * config.SPLIT_RUN_SAMPLES))
        for iR in range(config.N_FIT_RUNS):
            if verbosity:
                print("\n\nFitting run %d / %d with %d training samples!..\n" %
                      (iR+1, config.N_FIT_RUNS, n_train_samples))
            ticR = time.time()
            # Choose a subsample of the whole set of samples:
            sampl_inds = random.sample(all_inds, n_train_samples)
            path = train_params_filepath(config, iR=iR, label=label)
            torch.save(train_params_samples[sampl_inds], path)
            filepath, extension = path.split(".")
            np.save(filepath + "_inds.npy", sampl_inds)
            # For every fitting run...
            results_i = infer_workflow(train_params_samples[sampl_inds], sim_res[sampl_inds], priors=priors,
                                       target=target, ground_truth=groung_truth, config=config,  label=label,
                                       n_samples_per_run=n_samples_per_run, measure_labels=measure_labels,
                                       results=results_i, iR=iR, save_samples=save_samples,
                                       plot_flag=plot_flag, measures_plot_fun=measures_plot_fun,
                                       plot_diagnostics_flag=False, verbosity=verbosity)[1]
            if verbosity:
                print("Done with run %d in %g sec!" % (iR, time.time() - ticR))
        # Plot with samples from all runs!:
        if verbosity:
            print("Plotting samples from all %d runs together..." % config.N_FIT_RUNS)
        plot_infer(samples=None, results=results_i, points=None, metric="MAP",
                   iR=None, label=joinstr([label, config.ALL_RUNS_LABEL]),
                   plot_diagnostics_flag=True, config=config)
    if verbosity:
        print("\n\nFitting with all samples!..\n")
    results = infer_workflow(train_params_samples, sim_res, priors=priors,
                             target=target, ground_truth=groung_truth, config=config,
                             label=joinstr([label, config.ALL_SAMPLES_LABEL]),
                             n_samples_per_run=n_samples_per_run, measure_labels=measure_labels,
                             results=None, iR=None, save_samples=save_samples,
                             plot_flag=plot_flag, measures_plot_fun=measures_plot_fun,
                             plot_diagnostics_flag=True, verbosity=verbosity)[1]
    return results, results_i


def plot_stats(measures, stat="PPC", target=None, params=None, label="",
               measure_labels=None, measures_plot_fun=None, config=None):
    config = assert_config(config, return_plotter=False, plot_flag=False, measures_plot_fun=None)
    fig1, axes1 = plot_samples_measures_and_targets(measures, target=target,
                                                    label="%s_%s" % (stat, label),
                                                    measure_labels=measure_labels, measures_plot_fun=measures_plot_fun,
                                                    config=config)
    if params is not None:
        fig2, axes2 = params_pairplot(params, points=params.mean(axis=0), metric="mean", config=config,
                                      figpath=fitfigs_filepath(config, "params_pairplot.png",
                                                               label="%s_%s" % (stat, label)))
        return fig1, axes1, fig2, axes2
    return fig1, axes1


def correlation_distance(data, target):
    return 1 - np.corrcoef(data, target)[:, -1][:-1]


def get_best_stat_sims_params_target(measures, target, params=None, label="",
                                     target_dist_fun=correlation_distance, Nbest=None):
    if measures.ndim < 2:
        measures = measures[np.newaxis]
    Nm = measures.shape[0]
    if params is not None:
        if params.ndim < 2:
            params = params[np.newaxis]
        Nparams = params.shape[0]
        if Nparams != Nm:
            raise ValueError("The number of parameters Nparams = %d "
                             "is not equal to the number of measures Nm = %d!" % (Nparams, Nm))

    if Nm == 1:
        if Nbest is not None and Nbest > Nm:
            warnings.warn("\nThe number of measures (Nm=%d) is smaller "
                          "than the required number of best measures (Nbest=%d)!\n"
                          "Setting Nbest=Nm=%d and using all measures!\n" % (Nm, Nbest, Nm))
        return measures, params, joinstr(["best", label])
    else:
        if Nbest is None:
            Nbest = np.maximum(int(np.ceil(Nm / 2)), 1)  # In this case we set safely Nbest as Nm/2
        elif Nbest > Nm:
            warnings.warn("\nThe number of measures (Nm=%d) is smaller "
                          "than the required number of best measures (Nbest=%d)!\n"
                          "Setting Nbest=Nm=%d and using all measures!\n" % (Nm, Nbest, Nm))
            return measures, params, joinstr(["all_%d_best" % Nm, label])
        if target.ndim < 2:
            target = target[np.newaxis]
        target_dist_best_sorted_inds = np.argsort(target_dist_fun(measures, target))[:Nbest]
        if params is not None:
            params = params[target_dist_best_sorted_inds]
        return measures[target_dist_best_sorted_inds], params, joinstr(["best_%d" % Nbest, label])


def plot_best_stat_sims_params_target(measures, target, stat="PPC", params=None, label="",
                                      target_dist_fun=correlation_distance, Nbest=None,
                                      measure_labels=None, measures_plot_fun=None, config=None):
    config = assert_config(config, return_plotter=False)
    measures, params, label = get_best_stat_sims_params_target(measures, target, params, label, target_dist_fun, Nbest)
    return plot_stats(measures, stat, target, params, label, measure_labels, measures_plot_fun, config)
