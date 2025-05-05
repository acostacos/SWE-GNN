import numpy as np

from . import file_utils

def get_cell_position(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = ['X', 'Y']
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_cell_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'Elevation1'
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_index(filepath: str) -> np.ndarray:
    columns = ['from_node', 'to_node']
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return data.astype(np.int64).transpose()

def get_edge_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'length'
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_slope(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'slope'
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)
