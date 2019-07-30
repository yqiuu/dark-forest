from tqdm import tqdm


__all__ = [
    'SnapRange'
]


def SnapRange(*args, progress_bar = True, **kwargs):
    if progress_bar:
        return tqdm(range(*args), **kwargs)
    else:
        return range(*args)
