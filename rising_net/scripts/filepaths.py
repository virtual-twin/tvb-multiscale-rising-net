# -*- coding: utf-8 -*-

import os
import numpy as np

from tvb.contrib.scripts.utils.file_utils import safe_makedirs


def get_path(config, folder="", mode=""):
    if mode == "figs":
        base_path, mode = os.path.split(config.figures.FOLDER_FIGURES)
    elif mode == "res":
        base_path, mode = os.path.split(config.out.FOLDER_RES)
    else:
        base_path, _ = os.path.split(config.out.FOLDER_RES)
    return os.path.join(base_path, folder, mode)


def get_res_path(config, folder=""):
    return get_path(config, folder, mode="res")


def get_fig_path(config, folder=""):
    return get_path(config, folder, mode="figs")


def filepath_prefixes(filepath, iR=None, label=""):
    if len(label):
        filepath += "_%s" % label
    if iR is not None:
        filepath += istr(iR)
    return filepath


def istr(iR, nmin=1, Ns=100):
    n = np.maximum(nmin, int(np.ceil(np.log10(np.maximum(iR+1, Ns)))))
    format = "_%0" + "%d" % n + "d"
    return format % iR


def construct_filepath(default_filename, original_path, iR=None, label="", filepath=None, extension=None):
    if filepath is None or extension is None:
        filepath, extension = os.path.splitext(os.path.join(original_path, default_filename))
    filepath = filepath_prefixes(filepath, iR, label)
    filepath = "%s%s" % (filepath, extension)
    dirname = os.path.dirname(filepath)
    if not os.path.isdir(dirname):
        safe_makedirs(dirname)
    return filepath


def simres_filepath(config, default_filename="", folder="", iR=None, label="", filepath=None, extension=None):
    if len(default_filename) == 0:
        default_filename = config.SIM_RES_FILE
    return construct_filepath(default_filename,
                              get_res_path(config, folder),
                              iR=iR, label=label,
                              filepath=filepath, extension=extension)


def figs_filepath(config, default_filename, folder="", iR=None, label="", filepath=None, extension=None):
    return construct_filepath(default_filename,
                              get_fig_path(config, folder),
                              iR=iR, label=label,
                              filepath=filepath, extension=extension)