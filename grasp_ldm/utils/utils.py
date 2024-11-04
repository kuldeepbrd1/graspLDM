import json
import multiprocessing

# from ptflops import get_model_complexity_info
from typing import Tuple

import torch
from scipy.spatial.transform import Rotation as R


def get_param_count(model: torch.nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:0.3f} M \n Total: {total/1e6:0.3f}")


def load_json(path: str) -> dict:
    """load json helper

    Args:
        path (str): json_path

    Returns:
        dict: data
    """
    with open(path, "r") as jf:
        data = json.load(jf)
    return data


def spawn_multiple_processes(n_proc, target_fn, process_args):
    assert (
        len(process_args) == n_proc
    ), f"Number of processes ({n_proc}) does not match the length of process_args ({len(process_args)})"

    read_processes = []

    for idx in range(n_proc):
        try:
            if isinstance(process_args[idx], list):
                p = multiprocessing.Process(target=target_fn, args=process_args[idx])
            elif isinstance(process_args[idx], dict):
                p = multiprocessing.Process(target=target_fn, kwargs=process_args[idx])
            else:
                raise TypeError

            p.start()
            read_processes.append(p)
        except:
            for p in read_processes:
                p.join()

    for p in read_processes:
        p.join()

    return


def split_list(lst, n):
    """Split a list into n sublists of approximately equal length

    Args:
        lst (list): list to split
        n (int): number of sublists

    Returns:
        list: list of sublists
    """
    # divisor, modulo for n splits of list length
    div, mod = divmod(len(lst), n)

    # Length of each sublist
    lengths = [div + 1 if i < mod else div for i in range(n)]

    # Split the original list into sublists
    # sum(lengths[:i]) is 0 for i=0, so the first sublist starts at 0
    sublists = [lst[sum(lengths[:i]) : sum(lengths[: i + 1])] for i in range(n)]

    # Remove empty sublists
    sublists = [sublist for sublist in sublists if sublist]

    return sublists
