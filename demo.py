import functools
import time
import numpy as np
import matrix_math


def timer(func):
    @functools.wraps(func)
    def timer_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        msg = f"{func.__name__} took {elapsed:.6f} seconds"
        print(msg)
        return result
    return timer_wrapper


def pure_python_comparison(matrix_1, matrix_2):
    score = 0
    dimensions = len(matrix_1.shape)

    def _compare_items(item_1, item_2):
        if dimensions == 2:
            return (item_1 - item_2) ** 2
        return sum((inner_item - item_2[idx]) ** 2 for idx, inner_item in enumerate(item_1))
    try:
        for row_idx, row in enumerate(matrix_1):
            for col_idx, item in enumerate(row):
                score += _compare_items(item, matrix_2[row_idx][col_idx])
    except IndexError:
        raise ValueError("Matrices must be of the same size and shape!")
    return score


def numpy_method(matrix_1, matrix_2):
    return ((np.array(matrix_1) - np.array(matrix_2)) ** 2).sum()


def cython_method(matrix_1, matrix_2):
    func = matrix_math.euclid_2d if len(matrix_1.shape) == 2 else matrix_math.euclid_3d
    return func(matrix_1, matrix_2)



@timer
def python_orderer(func, matrix_1, matrix_2):
    swapped = []
    total = len(matrix_1)
    for display_idx, original in enumerate(matrix_1):
        print(f"tile {display_idx: 5} of {total - 1}", end="\r")
        best_idx, best_value = 0, float('inf')
        for idx, test_item in enumerate(matrix_2):
            test_value = func(original, test_item)
            if test_value < best_value:
                best_idx, best_value = idx, test_value
        swapped.append(best_idx)
    return swapped


def cython_diff_lists(matrix_1, matrix_2):
    func = matrix_math.batch_euclid_2d if len(matrix_1.shape) == 3 else matrix_math.batch_euclid_3d
    buffer = np.zeros((matrix_1.shape[0], matrix_2.shape[0]), dtype=np.int32)
    return func(matrix_1, matrix_2, buffer)


@timer
def cython_orderer(matrix_1, matrix_2):
    values_to_compare = cython_diff_lists(matrix_1, matrix_2)
    result_buffer = np.zeros(matrix_1.shape[0], dtype=np.int32)
    return list(matrix_math.get_tile_order(values_to_compare, result_buffer))


def sort_tiles(func, image_splitter, tile_processor):
    image_data, tile_data = image_splitter.tile_data, tile_processor.tile_data
    return func(image_data, tile_data)


def make_python_sorter(func=None):
    orderer = functools.partial(python_orderer, func)
    return functools.partial(sort_tiles, orderer)


METHODS = {
    "pure_python": make_python_sorter(pure_python_comparison),
    "numpy": make_python_sorter(numpy_method),
    "hybrid": make_python_sorter(cython_method),
    "cython": functools.partial(sort_tiles, cython_orderer),
    "all": None
}


if __name__ == "__main__":
    import sys
    import subprocess
    import click
    from photomosaic.api import mosaicfy, is_animated


    def _get_default_open():
        platform = 'linux' if 'linux' in sys.platform.lower() else sys.platform.lower()
        return {
            'linux': 'eog',
            'darwin': 'open -a safari'
        }.get(platform, 'open')


    @click.command()
    @click.argument('filename', type=click.Path(exists=True))
    @click.option('--tile_size', default=8)
    @click.option('--scale', default=1)
    @click.option('--output_file')
    @click.option('--show/--no-show', default=False)
    @click.option('--open_with', default=_get_default_open())
    @click.option('--method', type=click.Choice(list(METHODS.keys())))
    @click.option('--tile_directory', type=click.Path(exists=True))
    @click.option('--image_type', type=click.Choice(['L', 'RGB']), default="L")
    def cli(filename, **kwargs):
        show = kwargs.pop('show', False)
        method_name = str(kwargs.pop("method", None)).lower()
        if method_name == "all":
            for func_name, func in METHODS.items():
                print(f"using {func_name}")
                mosaicfy(filename, method=func, **kwargs)
            return
        method = METHODS.get(method_name)
        kwargs["method"] = method
        result = mosaicfy(filename, **kwargs)
        if show:
            if is_animated(filename):
                cmd = f'{kwargs.get("default_open", _get_default_open())} {result.gif_path}'
                subprocess.run(cmd.split())
            else:
                result.image.show()
    cli()
