import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src: str, target: str):
    """Link file.

    Args:
        src (str): Source file path.
        target (str): Target file path.
    """
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system("ln -s {} {}".format(src, target))


def ensure_dir(path: str):
    """Ensure directory exists.

    Args:
        path (str): Directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed

    embed()
