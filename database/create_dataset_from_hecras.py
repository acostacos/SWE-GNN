import numpy as np
import torch

from graph_creation import create_dataset_folders, save_database
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from transform_helper_files.hecras_data_retrieval import get_cell_area, get_water_level, get_facepoint_coordinates, get_velocity,\
    get_edge_direction_x, get_edge_direction_y, get_face_length, get_facecell_indexes, get_facepoint_indexes
from transform_helper_files.shp_data_retrieval import get_cell_position, get_cell_elevation, get_edge_index, get_edge_length, get_edge_slope

def get_cell_velocity(hec_ras_filepath: str, node_shp_filepath: str, perimeter_name: str = 'Perimeter 1') -> torch.Tensor:
    '''Adopted from https://doi.org/10.26188/24312658'''
    def dist_center2faces(center_xy,faces_xy):
        dist = np.sqrt(np.square(faces_xy[:,0]-center_xy[0]) + np.square(faces_xy[:,1]-center_xy[1]))
        return dist

    xy_coor = get_cell_position(node_shp_filepath)
    cell_area = get_cell_area(hec_ras_filepath, perimeter_name)
    facepoint_xy_coor = get_facepoint_coordinates(hec_ras_filepath, perimeter_name)
    edge_direction_x = get_edge_direction_x(hec_ras_filepath, perimeter_name)
    edge_direction_y = get_edge_direction_y(hec_ras_filepath, perimeter_name)
    face_length = get_face_length(hec_ras_filepath, perimeter_name)
    faces_cell_idx = get_facecell_indexes(hec_ras_filepath, perimeter_name)
    faces_facepoint_idx = get_facepoint_indexes(hec_ras_filepath, perimeter_name)
    face_vel = get_velocity(hec_ras_filepath, perimeter_name)

    # Find x-y components for each cell face in each cell
    n_timesteps = len(face_vel)
    n_cells = len(xy_coor)

    cell_velocity_x = np.zeros([n_timesteps, n_cells])
    cell_velocity_y = np.zeros([n_timesteps, n_cells])
    for cell_i in range(n_cells):
        # Find cell in FROM/TO table of faces
        find_faces_for_cell = np.column_stack(np.where(faces_cell_idx == cell_i))

        # Find cell velocity x-y components each cell
        # HEC-RAS method: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics/numerical-methods/cell-velocity
        # Vc = 1/A * SUM(dx * L * v_f)
        # where A is the cell area, 
        # dx is distance from cell center to facecenter, 
        # L is face length, v_f is facevelocity
        cell_xy = xy_coor[cell_i]
        facepoint1_xy = facepoint_xy_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,0]]
        facepoint2_xy = facepoint_xy_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,1]]
        face_center_xy = np.c_[np.mean([facepoint1_xy[:,0], facepoint2_xy[:,0]], axis=0),
                    np.mean([facepoint1_xy[:,1], facepoint2_xy[:,1]], axis=0)]

        dx_center2face = dist_center2faces(cell_xy,face_center_xy)

        if not cell_area[cell_i].any():
            continue
        else:
            cell_velocity_x[:,cell_i] = 1/cell_area[cell_i] * np.sum(dx_center2face * 
                                                face_length[find_faces_for_cell[:,0]] * 
                                                face_vel[:, find_faces_for_cell[:,0]] * edge_direction_x[find_faces_for_cell[:,0]], axis=1)

            cell_velocity_y[:,cell_i] = 1/cell_area[cell_i] * np.sum(dx_center2face *
                                                    face_length[find_faces_for_cell[:,0]] *
                                                    face_vel[:, find_faces_for_cell[:,0]] * edge_direction_y[find_faces_for_cell[:,0]], axis=1)

    return torch.FloatTensor(cell_velocity_x), torch.FloatTensor(cell_velocity_y)

def get_cell_slope(hec_ras_filepath: str, edge_shp_path: str, edge_index: torch.Tensor, n_cells: int, perimeter_name: str = 'Perimeter 1') -> tuple[torch.Tensor, torch.Tensor]:
    edge_slope = get_edge_slope(edge_shp_path)
    edge_direction_x = get_edge_direction_x(hec_ras_filepath, perimeter_name)
    edge_direction_y = get_edge_direction_y(hec_ras_filepath, perimeter_name)

    # Compute slope in x and y directions
    edge_slope_x = edge_slope * edge_direction_x
    edge_slope_y = edge_slope * edge_direction_y

    slope_x = np.zeros((n_cells))
    slope_y = np.zeros((n_cells))
    for cell_i in range(n_cells):
        cell_edge_idx = ((edge_index[0] == cell_i) | (edge_index[1] == cell_i)).nonzero()

        # Slope for each cell is the average of the slopes of the edges connected to it
        slope_x[cell_i] = np.average(edge_slope_x[cell_edge_idx])
        slope_y[cell_i] = np.average(edge_slope_y[cell_edge_idx])

    return torch.FloatTensor(slope_x), torch.FloatTensor(slope_y)

def get_edge_relative_distance(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    edge_relative_distance = pos[col] - pos[row]
    return edge_relative_distance

def get_dataset_features(hec_ras_file_path: str, node_shp_path: str, edge_shp_path: str) -> dict:
    # HEC-RAS data retrieval
    water_level = torch.FloatTensor(get_water_level(hec_ras_file_path))
    cell_velocity_x, cell_velocity_y = get_cell_velocity(hec_ras_file_path, node_shp_path)

    # Node Shapefile data retrieval
    dem = torch.FloatTensor(get_cell_elevation(node_shp_path))
    water_depth = water_level - dem
    pos = torch.FloatTensor(get_cell_position(node_shp_path))

    # Edge Shapefile data retrieval
    edge_index = torch.LongTensor(get_edge_index(edge_shp_path))
    edge_distance = torch.FloatTensor(get_edge_length(edge_shp_path))
    slope_x, slope_y = get_cell_slope(hec_ras_file_path, edge_shp_path, edge_index, len(dem))

    # Convert edges to undirected
    edge_index = to_undirected(edge_index)
    edge_distance = torch.cat([edge_distance, edge_distance], dim=0)
    edge_relative_distance = get_edge_relative_distance(pos, edge_index)

    # Get graph features
    num_nodes = len(dem)

    return {
        'num_nodes': num_nodes,
        'edge_index': edge_index,
        'water_depth': water_depth,
        'cell_velocity_x': cell_velocity_x,
        'cell_velocity_y': cell_velocity_y,
        'dem': dem,
        'pos': pos,
        'slope_x': slope_x,
        'slope_y': slope_y,
        'edge_distance': edge_distance,
        # 'edge_slope': edge_slope,
        'edge_relative_distance': edge_relative_distance,
    }

def convert_to_pyg(dataset_features: dict) -> Data:
    data = Data()

    data.edge_index = dataset_features['edge_index']
    data.edge_distance = dataset_features['edge_distance']
    # data.edge_slope = dataset_features['edge_slope']
    data.edge_relative_distance = dataset_features['edge_relative_distance']

    data.num_nodes = dataset_features['num_nodes']
    data.pos = dataset_features['pos']
    data.DEM = dataset_features['dem']
    data.slope_x = dataset_features['slope_x']
    data.slope_y = dataset_features['slope_y']
    data.WD = dataset_features['water_depth'].T
    data.VX = dataset_features['cell_velocity_x'].T
    data.VY = dataset_features['cell_velocity_y'].T

    return data

def main():
    datasets = {
        'event_key': {
            'hec_ras_file_path': "path/to/hec_ras/file",
            'node_shp_path': "path/to/node/shp/file",
            'edge_shp_path': "path/to/edge/shp/file",
        },
    }
    base_dataset_floder = "hecras_datasets"

    dataset_keys = list(datasets.keys())
    for key in dataset_keys:
        # Create dataset folder
        dataset_folder = f"{base_dataset_floder}/test_on_{key}"
        create_dataset_folders(dataset_folder=dataset_folder)
        print(f"Dataset folder created in: {dataset_folder}", flush=True)

    for key in dataset_keys:
        print(f"Processing event {key}", flush=True)

        paths = datasets[key]
        hec_ras_file_path = paths['hec_ras_file_path']
        node_shp_path = paths['node_shp_path']
        edge_shp_path = paths['edge_shp_path']

        dataset_features = get_dataset_features(hec_ras_file_path, node_shp_path, edge_shp_path)
        print(f"\tFinished obtaining features for event {key}", flush=True)
        pyg_dataset = [convert_to_pyg(dataset_features)]

        train_keys = [k for k in dataset_keys if k != key]
        for train_key in train_keys:
            print(f"\tSaving event {key} as train dataset for {train_key}", flush=True)
            train_folder = f"{base_dataset_floder}/test_on_{train_key}/train"
            save_database(pyg_dataset, name=key, out_path=train_folder)

        print(f"\tSaving event {key} as test dataset for {key}", flush=True)
        test_folder = f"{base_dataset_floder}/test_on_{key}/test"
        save_database(pyg_dataset, name=key, out_path=test_folder)

if __name__ == "__main__":
    main()