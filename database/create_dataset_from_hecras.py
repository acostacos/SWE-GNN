import h5py
import numpy as np
import torch

from transform_helper_files.hecras_data_retrieval import get_cell_area, get_water_level, get_facepoint_coordinates, get_velocity
from transform_helper_files.shp_data_retrieval import get_cell_position, get_cell_elevation, get_edge_index, get_edge_length, get_edge_slope

def get_cell_velocity(hec_ras_filepath: str, node_shp_filepath: str, perimeter_name: str = 'Perimeter 1') -> np.ndarray:
    def dist_center2faces(center_xy,faces_xy):
        dist = np.sqrt(np.square(faces_xy[:,0]-center_xy[0]) + np.square(faces_xy[:,1]-center_xy[1]))
        return dist

    XY_coor = get_cell_position(node_shp_filepath)
    Cell_Area = get_cell_area(hec_ras_filepath, perimeter_name)
    facepoint_XY_coor = get_facepoint_coordinates(hec_ras_filepath, perimeter_name)

    # TODO: Convert this to hecras data retrieval function
    hec = h5py.File(hec_ras_filepath, 'r')
    faces_cell_idx = np.array(hec['Geometry']['2D Flow Areas'][perimeter_name]['Faces Cell Indexes'])
    faces_norm_unit_vector = np.array(hec['Geometry']['2D Flow Areas'][perimeter_name]['Faces NormalUnitVector and Length'])
    faces_facepoint_idx = np.array(hec['Geometry']['2D Flow Areas'][perimeter_name]['Faces FacePoint Indexes'])

    # Face velocity
    Face_vel = get_velocity(hec_ras_filepath, perimeter_name)

    hec.close()  # Close

    # Find x-y components for each cell face in each cell
    n_timesteps = len(Face_vel)
    n_cells = len(XY_coor)

    V_c_x = np.zeros([n_timesteps,n_cells])
    V_c_y = np.zeros([n_timesteps,n_cells])
    for cell_i in range(n_cells):
        # Find cell in FROM/TO table of faces
        find_faces_for_cell = np.column_stack(np.where(faces_cell_idx == cell_i))

        # Find cell velocity x-y components each cell
        # HEC-RAS method: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics/numerical-methods/cell-velocity
        # Vc = 1/A * SUM(dx * L * v_f)
        # where A is the cell area, 
        # dx is distance from cell center to facecenter, 
        # L is face length, v_f is facevelocity
        cell_xy = XY_coor[cell_i]
        Facepoint1_xy = facepoint_XY_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,0]]
        Facepoint2_xy = facepoint_XY_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,1]]
        Face_center_xy = np.c_[np.mean([Facepoint1_xy[:,0], Facepoint2_xy[:,0]], axis=0),
                    np.mean([Facepoint1_xy[:,1], Facepoint2_xy[:,1]], axis=0)]

        dx_center2face = dist_center2faces(cell_xy,Face_center_xy)

        if not Cell_Area[cell_i].any():
            continue
        else:
            V_c_x[:,cell_i] = 1/Cell_Area[cell_i] * np.sum(dx_center2face * 
                                                faces_norm_unit_vector[find_faces_for_cell[:,0] ,2] * 
                                                Face_vel[:, find_faces_for_cell[:,0]] * faces_norm_unit_vector[find_faces_for_cell[:,0], 0], axis=1)

            V_c_y[:,cell_i] = 1/Cell_Area[cell_i] * np.sum(dx_center2face *
                                                    faces_norm_unit_vector[find_faces_for_cell[:,0] ,2] *
                                                    Face_vel[:, find_faces_for_cell[:,0]] * faces_norm_unit_vector[find_faces_for_cell[:,0], 1], axis=1)

    return V_c_x, V_c_y

def get_edge_relative_distance(node_shp_path: str, edge_index: torch.Tensor) -> torch.Tensor:
    pos = torch.FloatTensor(get_cell_position(node_shp_path))
    row, col = edge_index
    edge_relative_distance = pos[col] - pos[row]
    return edge_relative_distance

def main():
    hec_ras_file_path = "path/to/hec_ras/file"
    node_shp_path = "path/to/node/shapefile.shp"
    edge_shp_path = "path/to/edge/shapefile.shp"
    dataset_directory = "path/to/dataset/directory"

    # HEC-RAS data retrieval
    water_level = torch.FloatTensor(get_water_level(hec_ras_file_path))
    # velocity = get_cell_velocity(hec_ras_file_path, node_shp_path)

    # Node Shapefile data retrieval
    dem = torch.FloatTensor(get_cell_elevation(node_shp_path))

    # Edge Shapefile data retrieval
    edge_index = torch.FloatTensor(get_edge_index(edge_shp_path))
    edge_distance = torch.FloatTensor(get_edge_length(edge_shp_path))
    edge_slope = torch.FloatTensor(get_edge_slope(edge_shp_path))
    edge_relative_distance = get_edge_relative_distance(node_shp_path, edge_index)

    # Create raw dataset path if it doesn't exist
    # Create pickle dataset path if it doesn't exist


if __name__ == "__main__":
    main()