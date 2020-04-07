import re
import os
import h5py
import numpy as np
from glob import glob
import ruamel.yaml as yaml
from cytoolz import curry, compose
from functools import lru_cache, wraps
from cytoolz.curried import get_in, keyfilter, valmap
from cytoolz.itertoolz import peek, pluck, first, groupby
from cytoolz.dicttoolz import valfilter, merge_with, dissoc, assoc


# https://gist.github.com/jaytaylor/3660565
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile(r'([a-z0-9])([A-Z])')

def np_cache(function):
    @lru_cache(maxsize=None)
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def camel_to_snake(s):
    """ Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def check_video_parameters(index: dict) -> dict:
    ''' Iterates through each extraction parameter file to verify extraction parameters
    were the same. If they weren't this function raises a RuntimeError. Otherwise, it
    will return a dictionary containing the following parameters:
        crop_size, fps, max_height, min_height, resolution
    Args:
        index: a `sorted_index` dictionary of extraction parameters
    Returns:
        a dictionary with a subset of the used extraction parameters
    '''

    # define constants
    check_parameters = ['crop_size', 'fps', 'max_height', 'min_height']

    get_yaml = get_in(['path', 1])
    ymls = list(map(get_yaml, index['files'].values()))

    # load yaml config files when needed
    dicts = map(read_yaml, ymls)
    # get the parameters key within each dict
    params = pluck('parameters', dicts)

    first_entry, params = peek(params)
    if 'resolution' in first_entry:
        check_parameters += ['resolution']

    # filter for only keys in check_parameters
    params = map(keyfilter(lambda k: k in check_parameters), params)
    # turn lists (in the dict values) into tuples
    params = map(valmap(lambda x: tuple(x) if isinstance(x, list) else x), params)

    # get unique parameter values
    vid_parameters = merge_with(set, params)

    incorrect_parameters = valfilter(lambda x: len(x) > 1, vid_parameters)

    # if there are multiple values for a parameter, raise error
    if incorrect_parameters:
        raise RuntimeError('The following parameters are not equal ' +
                           f'across extractions: {incorrect_parameters.keys()}')

    # grab the first value in the set
    vid_parameters = valmap(first, vid_parameters)

    # update resolution
    if 'resolution' in vid_parameters:
        vid_parameters['resolution'] = tuple(x + 100 for x in vid_parameters['resolution'])
    else:
        vid_parameters['resolution'] = None

    return vid_parameters


def clean_dict(dct):

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    ans = {}
    if isinstance(file[path], h5py.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split('/')[-1]] = file[path][()]
    else:
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path: str = '/') -> dict:
    '''
    Args:
        h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
        path: path to the base dataset within the h5 file. Default: /
    Returns:
        a dict with h5 file contents with the same path structure
    '''
    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out


def get_timestamps_from_h5(h5file: str):
    with h5py.File(h5file, 'r') as f:
        # v0.1.3 new data format
        is_new = 'timestamps' in f
    if is_new:
        return h5_to_dict(h5file, 'timestamps')['timestamps']
    else:
        return h5_to_dict(h5file, 'metadata/timestamps')['timestamps']


def load_changepoints(cpfile):
    cps = h5_to_dict(cpfile, 'cps')
    cp_dist = map(compose(np.diff, np.squeeze), cps.values())
    # TODO: make sure that this is correct
    return np.concatenate(list(cp_dist))


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """
    ts = np.loadtxt(timestamp_file, delimiter=' ')
    if ts.ndim > 1:
        return ts[:, col]
    elif col > 0:
        raise Exception(f'Timestamp file {timestamp_file} does not have more than one column of data')
    else:
        return ts


def parse_index(index_file: str) -> tuple:
    ''' Load an index file, and use extraction UUIDs as entries in a sorted index.

    Returns:
        a tuple containing the loaded index file, and the index with extraction UUIDs as entries
    '''

    join = os.path.join
    index_dir = os.path.dirname(index_file)

    index = read_yaml(index_file)
    files = index['files']

    sorted_index = groupby('uuid', files)
    # grab first entry in list, which is a dict
    sorted_index = valmap(first, sorted_index)
    # remove redundant uuid entry
    sorted_index = valmap(lambda d: dissoc(d, 'uuid'), sorted_index)
    # tuple-ize the path entry, join with the index file dirname
    sorted_index = valmap(lambda d: assoc(d, 'path', tuple(join(index_dir, x) for x in d['path'])),
                          sorted_index)

    uuid_sorted = {
        'files': sorted_index,
        'pca_path': join(index_dir, index['pca_path'])
    }

    return index, uuid_sorted


def get_sorted_index(index_file: str) -> dict:
    ''' Just return the sorted index from an index_file path'''
    _, sorted_ind = parse_index(index_file)
    return sorted_ind


def h5_filepath_from_sorted(sorted_index_entry: dict) -> str:
    '''Gets the h5 extraction file path from a sorted index entry
    Returns:
        a str containing the extraction filepath
    '''
    return first(sorted_index_entry['path'])


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """

    def has_frames(h5f):
        '''Checks if the supplied h5 file has a frames key'''
        with h5py.File(h5f, 'r') as f:
            return 'frames' in f

    def h5_to_yaml(h5f):
        return yaml_string.format(os.path.splitext(h5f)[0])

    # make function to test if yaml file with same basename as h5 file exists
    yaml_exists = compose(os.path.exists, h5_to_yaml)

    # grab all files with ext = .h5
    files = glob(f'**/*{ext}', recursive=True)
    # keep h5s that have a yaml file associated with them
    to_keep = filter(yaml_exists, files)
    # keep h5s that have a frames key
    to_keep = filter(has_frames, to_keep)

    h5s = list(to_keep)
    yamls = list(map(h5_to_yaml, h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def read_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        loaded = yaml.safe_load(f)
    return loaded


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
# dang this is fast!
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


@curry
def star(f, args):
    '''Apply a function to a tuple of args, by expanding the tuple into
    each of the function's parameters. It is curried, which allows one to
    specify one argument at a time.
    Args:
        f: a function that takes multiple arguments
        args: a tuple to expand into ``f``
    Returns:
        the output of ``f``

    >>> instance_checker = star(isinstance)
    >>> instance_checker((1, int))
    True
    >>> star(max, (1, 2, 3))
    3'''
    return f(*args)