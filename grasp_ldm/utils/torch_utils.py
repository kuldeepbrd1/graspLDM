import torch


def fix_state_dict_prefix(state_dict, prefix="model", ignore_all_others=False):
    """Fix state dict keys prefix

    Args:
        state_dict (dict): state dict
        prefix (str, optional): prefix to remove. Defaults to "model".

    Returns:
        dict: state dict with prefix removed
    """
    from collections import OrderedDict

    if isinstance(state_dict, dict):
        if ignore_all_others:
            return {
                k.partition(f"{prefix}.")[2]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            return {k.partition(f"{prefix}.")[2]: v for k, v in state_dict.items()}
    elif isinstance(state_dict, OrderedDict):
        if ignore_all_others:
            return OrderedDict(
                [
                    (k.partition(f"{prefix}.")[2], v)
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                ]
            )
        else:
            return OrderedDict(
                [(k.partition(f"{prefix}.")[2], v) for k, v in state_dict.items()]
            )


def minmax_normalize(
    t: torch.Tensor, dim: int, v_min: float = 0.0, v_max: float = 1.0, keepdim=True
) -> torch.Tensor:
    """min-max normalization in [0,1]

    Args:
        t (Tensor): tensor [B, D1, D2 ... Dn]
        dim (int): dimension to normalize on
        min (float, optional): min value. Defaults to 0.0.
        max (float, optional): max value. Defaults to 1.0.

    Returns:
        Tensor: [B, D1, D2 ... Dn]
    """
    t -= t.min(dim, keepdim=keepdim)[0]
    t /= t.max(dim, keepdim=keepdim)[0]

    t = t * (v_max - v_min) + v_min
    return t
