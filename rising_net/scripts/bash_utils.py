# -*- coding: utf-8 -*-
import numpy as np


def jobarr_id_to_task_ids(args):
    return np.unravel_index(args[0], args[1:], order='C')


def simulate_rest_train_ids_args(jobarr_id, Nps=1000, Ngs=11, Nreps=3):
    output = print("--iP %d --iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_ppc_ids_args(jobarr_id, Nps=100, Ngs=11, Nreps=3):
    output = print("--iP %d --iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_mapmean_ids_args(jobarr_id, Ngs=11, Nreps=3):
    output = print("--iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nreps)]))
    return output


def simulate_task_train_ids_args(jobarr_id, Nps=1000, Nreps=3):
    output = print("--iP %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Nreps)]))
    return output


def simulate_task_ppc_ids_args(jobarr_id, Nps=100, Nreps=3):
    output = print("--iP %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Nreps)]))
    return output


def simulate_task_mapmean_ids_args(jobarr_id, Nreps=3):
    output = print("--iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nreps)]))
    return output


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
    elif sys.argv[1] == "simulate_rest_train_ids_args":
        simulate_rest_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_ppc_ids_args":
        simulate_rest_ppc_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_train_ids_args":
        simulate_task_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_ppc_ids_args":
        simulate_task_ppc_ids_args(*sys.argv[2:])
