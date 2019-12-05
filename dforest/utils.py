from tqdm import trange, tnrange


__all__ = [
    'Range'
]


def Range(*args, notebook=False, **kwargs):
    if notebook:
        return tnrange(*args, **kwargs)
    else:
        return trange(*args, **kwargs)
