# -*- coding: utf-8 -*-
import numpy as np


def jobarr_id_to_task_ids(args):
    return np.unravel_index(args[0], args[1:], order='C')


def simulate_rest_ids_args(jobarr_id, Ngs=11, Nreps=3):
    output = print("--iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_train_ids_args(jobarr_id, Nps=1000, Ngs=11, Nreps=3):
    output = print("--iP %d --iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_ppc_ids_args(jobarr_id, Nps=100, Ngs=11, Nreps=3):
    # TODO: Correct this if iG = 0 comes back!
    iP, iG, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)])
    # iG = iG + 1
    iP = iP*10
    output = print("--iP %d --iG %d --iR %d" % (iP, iG, iR))
    return output


def simulate_rest_mapmean_ids_args(jobarr_id, Ngs=11, Nreps=3):
    iG, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nreps)])
    output = print("--iG %d --iR %d" % (iG, iR))
    return output


def simulate_task_train_ids_args(jobarr_id, Nps=1000, Ngs=11):
    iG, iP = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nps)])
    output = print("--iG %d --iP %d" % (iG, iP))
    return output


def simulate_task_ppc_ids_args(jobarr_id, Nps=100, Ngs=11):
    iG, iP = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nps)])
    output = print("--iG %d --iP %d" % (iG, iP))
    return output


def fit_rest(jobarr_id):
    from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR
    from rising_net.scripts.rest_run_fit_plot import get_config, infer_nRuns_for_iG

    # Fitting:
    config = get_config(FUNCMODE="FIT", BASENAME="FIT_REST")[0]

    return infer_nRuns_for_iG(int(jobarr_id),
                              priors=None, train_params_samples=None, sim_res=None, sim_res_path=None,
                              target=None, ground_truth=None,
                              config=config, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                              label="", n_samples_per_run=None,
                              save_samples=True, plot_flag=True, verbosity=2)


def fit_task(jobarr_id):
    from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR
    from rising_net.scripts.task_run_fit_plot import \
        get_config, get_sim_res_COHM1S1diffratio, target_COHM1S1diffratio_fun, infer_nRuns_for_task

    # Fitting:
    config = get_config(FUNCMODE="FIT", BASENAME="FIT_TASK")[0]

    if config.COHERENCE_FISHER_Z_TRANSFORM:
        measure_labels = ["M1S1R_ThCOHFisherZdiffratio", "M1S1L_ThCOHFisherZdiffratio",
                          "M1S1R_BtCOHFisherZdiffratio", "M1S1L_BtCOHFisherZdiffratio",
                          "M1S1R_GmCOHFisherZdiffratio", "M1S1L_GmCOHFisherZdiffratio"]
    else:
        measure_labels = ["M1S1R_ThCOHdiffratio", "M1S1L_ThCOHdiffratio",
                          "M1S1R_BtCOHdiffratio", "M1S1L_BtCOHdiffratio",
                          "M1S1R_GmCOHdiffratio", "M1S1L_GmCOHdiffratio"]

    return infer_nRuns_for_task(iG=int(jobarr_id),
                                priors=None, train_params_samples=None,
                                sim_res=None, sim_res_path=None, sim_res_fun=get_sim_res_COHM1S1diffratio,
                                target=None, target_fun=target_COHM1S1diffratio_fun, ground_truth=None,
                                config=config, folderstr=NSDSTR, resstr=RESSTR,
                                label="", n_samples_per_run=None, measure_labels=measure_labels,
                                save_samples=True, plot_flag=True, verbosity=2)


def multiply(x, y):
    output = print(x*y)
    return output


def condition(x, th=100.0):
    if x <= th:
        output = 1
    else:
        output = 0
    print(output)
    return output


if __name__ == '__main__':
    import sys

    if sys.argv[1] == "multiply":
        multiply(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "condition":
        x = float(sys.argv[2])
        if len(sys.argv) > 3:
            condition(x, float(sys.argv[3]))
        else:
            condition(x)
    elif sys.argv[1] == "fit_rest":
        fit_rest(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_ids_args":
        simulate_rest_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_train_ids_args":
        simulate_rest_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_ppc_ids_args":
        simulate_rest_ppc_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_mapmean_ids_args":
        simulate_rest_mapmean_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "fit_task":
        fit_task(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_train_ids_args":
        simulate_task_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_ppc_ids_args":
        simulate_task_ppc_ids_args(*sys.argv[2:])
