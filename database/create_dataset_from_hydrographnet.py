import os
import numpy as np
import torch

from tqdm import tqdm
from typing import Optional, Union
from scipy.spatial import KDTree
from graph_creation import create_dataset_folders, save_database
from torch_geometric.data import Data

config = {

}

def load_constant_data(folder: str, prefix: str,
                        norm_stats_static: Optional[dict] = None):
    epsilon = 1e-8
    stats = norm_stats_static if norm_stats_static is not None else {}

    def standardize(data: np.ndarray, key: str) -> np.ndarray:
        if key in stats:
            mean_val = np.array(stats[key]["mean"])
            std_val = np.array(stats[key]["std"])
        else:
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}
        return (data - mean_val) / (std_val + epsilon)

    # Load each file using the given prefix.
    xy_path = os.path.join(folder, f"{prefix}_XY.txt")
    ca_path = os.path.join(folder, f"{prefix}_CA.txt")
    ce_path = os.path.join(folder, f"{prefix}_CE.txt")
    cs_path = os.path.join(folder, f"{prefix}_CS.txt")
    aspect_path = os.path.join(folder, f"{prefix}_A.txt")
    curvature_path = os.path.join(folder, f"{prefix}_CU.txt")
    manning_path = os.path.join(folder, f"{prefix}_N.txt")
    flow_accum_path = os.path.join(folder, f"{prefix}_FA.txt")
    infiltration_path = os.path.join(folder, f"{prefix}_IP.txt")

    xy_coords = np.loadtxt(xy_path, delimiter='\t')
    xy_coords = standardize(xy_coords, "xy_coords")
    area_denorm = np.loadtxt(ca_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    area = standardize(area_denorm, "area")
    elevation = np.loadtxt(ce_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    elevation = standardize(elevation, "elevation")
    slope = np.loadtxt(cs_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    slope = standardize(slope, "slope")
    aspect = np.loadtxt(aspect_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    aspect = standardize(aspect, "aspect")
    curvature = np.loadtxt(curvature_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    curvature = standardize(curvature, "curvature")
    manning = np.loadtxt(manning_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    manning = standardize(manning, "manning")
    flow_accum = np.loadtxt(flow_accum_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    flow_accum = standardize(flow_accum, "flow_accum")
    infiltration = np.loadtxt(infiltration_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
    infiltration = standardize(infiltration, "infiltration")
    return (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
            manning, flow_accum, infiltration, stats)

def create_edge_features(xy_coords: np.ndarray, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row, col = edge_index
    relative_coords = xy_coords[row] - xy_coords[col]
    distance = np.linalg.norm(relative_coords, axis=1)
    epsilon = 1e-8
    # Normalize relative coordinates and distance.
    relative_coords = (relative_coords - np.mean(relative_coords, axis=0)) / (np.std(relative_coords, axis=0) + epsilon)
    distance = (distance - np.mean(distance)) / (np.std(distance) + epsilon)
    return distance, relative_coords

def load_dynamic_data(folder: str, hydrograph_id: str, prefix: str,
                        num_points: int, interval: int = 1, skip: int = 72):
    wd_path = os.path.join(folder, f"{prefix}_WD_{hydrograph_id}.txt")
    inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
    volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")
    vx_path = os.path.join(folder, f"{prefix}_VX_{hydrograph_id}.txt")
    vy_path = os.path.join(folder, f"{prefix}_VY_{hydrograph_id}.txt")
    precipitation_path = os.path.join(folder, f"{prefix}_Pr_{hydrograph_id}.txt")
    water_depth = np.loadtxt(wd_path, delimiter='\t')[skip::interval, :num_points]
    velocity_x = np.loadtxt(vx_path, delimiter='\t')[skip::interval, :num_points]
    velocity_y = np.loadtxt(vy_path, delimiter='\t')[skip::interval, :num_points]
    inflow_hydrograph = np.loadtxt(inflow_path, delimiter='\t')[skip::interval, 1]
    volume = np.loadtxt(volume_path, delimiter='\t')[skip::interval, :num_points]
    precipitation = np.loadtxt(precipitation_path, delimiter='\t')[skip::interval]
    # Limit data until 25 time steps after the peak inflow.
    peak_time_idx = np.argmax(inflow_hydrograph)
    water_depth = water_depth[:peak_time_idx + 25]
    velocity_x = velocity_x[:peak_time_idx + 25]
    velocity_y = velocity_y[:peak_time_idx + 25]
    volume = volume[:peak_time_idx + 25]
    precipitation = precipitation[:peak_time_idx + 25] * 2.7778e-7  # Unit conversion
    inflow_hydrograph = inflow_hydrograph[:peak_time_idx + 25]
    # Make sure water depth is non-negative.
    water_depth = np.clip(water_depth, a_min=0, a_max=None)
    return water_depth, inflow_hydrograph, volume, precipitation, velocity_x, velocity_y

def normalize(data: np.ndarray, mean: Union[float, list, np.ndarray],
                std: Union[float, list, np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
    mean = np.array(mean) if isinstance(mean, list) else mean
    std = np.array(std) if isinstance(std, list) else std
    return (data - mean) / (std + epsilon)

def denormalize(data: np.ndarray, mean: Union[float, list, np.ndarray],
                std: Union[float, list, np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
    mean = np.array(mean) if isinstance(mean, list) else mean
    std = np.array(std) if isinstance(std, list) else std
    return data * (std + epsilon) + mean

def process_dataset(data_dir: str,
                 prefix: str = 'M80',
                 k: int = 4,
                 hydrograph_ids_file: Optional[str] = None,
                 split: str = "train",
                 rollout_length: Optional[int] = None,
                 param_static_stats: Optional[dict] = None,
                 param_dynamic_stats: Optional[dict] = None):
    if split == "train":
        # For training, load constant data and compute static normalization stats.
        (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
            manning, flow_accum, infiltration, static_stats) = load_constant_data(
            data_dir, prefix, norm_stats_static=None)
    else:
        # For test or validation, load precomputed normalization stats.
        (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
            manning, flow_accum, infiltration, _) = load_constant_data(
            data_dir, prefix, norm_stats_static=param_static_stats)
        static_stats = param_static_stats

    # Build the graph connectivity using a k-d tree.
    num_nodes = xy_coords.shape[0]
    kdtree = KDTree(xy_coords)
    _, neighbors = kdtree.query(xy_coords, k=k + 1)
    edge_index = np.vstack([(i, nbr) for i, nbrs in enumerate(neighbors)
                                for nbr in nbrs if nbr != i]).T
    edge_distance, edge_relative_distance = create_edge_features(xy_coords, edge_index)

    # Read hydrograph IDs either from a file or from the directory.
    if hydrograph_ids_file is not None:
        file_path = os.path.join(data_dir, hydrograph_ids_file)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
            hydrograph_ids = [line.strip() for line in lines if line.strip()]
        else:
            raise FileNotFoundError(f"Hydrograph IDs file not found: {file_path}")
    else:
        all_files = os.listdir(data_dir)
        hydrograph_ids = []
        for f in all_files:
            if f.startswith(f"{prefix}_WD_") and f.endswith(".txt"):
                parts = f.split('_')
                if len(parts) >= 3:
                    hid = os.path.splitext(parts[2])[0]
                    hydrograph_ids.append(hid)

    # Process dynamic data (water depth, inflow, volume, precipitation) for each hydrograph.
    temp_dynamic_data = []
    water_depth_list = []
    velocity_x_list = []
    velocity_y_list = []
    for hid in tqdm(hydrograph_ids, desc="Processing Hydrographs"):
        water_depth, inflow_hydrograph, volume, precipitation, velocity_x, velocity_y = load_dynamic_data(
            data_dir, hid, prefix, num_points=num_nodes)

        if split != "train":
            water_depth = water_depth[:rollout_length]
            velocity_x = velocity_x[:rollout_length]
            velocity_y = velocity_y[:rollout_length]

        temp_dynamic_data.append({
            "water_depth": water_depth,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "hydro_id": hid,
        })
        water_depth_list.append(water_depth.flatten())
        velocity_x_list.append(velocity_x.flatten())
        velocity_y_list.append(velocity_y.flatten())

    # Compute dynamic normalization statistics for training or load precomputed stats.
    if split == "train":
        dynamic_stats = {}
        water_depth_all = np.concatenate(water_depth_list)
        dynamic_stats["water_depth"] = {"mean": float(np.mean(water_depth_all)),
                                                "std": float(np.std(water_depth_all))}
        velocity_x_all = np.concatenate(velocity_x_list)
        dynamic_stats["velocity_x"] = {"mean": float(np.mean(velocity_x_all)),
                                                "std": float(np.std(velocity_x_all))}
        velocity_y_all = np.concatenate(velocity_y_list)
        dynamic_stats["velocity_y"] = {"mean": float(np.mean(velocity_y_all)),
                                                "std": float(np.std(velocity_y_all))}
    else:
        dynamic_stats = param_dynamic_stats

    # Normalize the dynamic data.
    dynamic_data = []
    for dyn in temp_dynamic_data:
        dyn_std = {
            "water_depth": normalize(dyn["water_depth"],
                                            dynamic_stats["water_depth"]["mean"],
                                            dynamic_stats["water_depth"]["std"]),
            "velocity_x": normalize(dyn["velocity_x"],
                                            dynamic_stats["velocity_x"]["mean"],
                                            dynamic_stats["velocity_x"]["std"]),
            "velocity_y": normalize(dyn["velocity_y"],
                                            dynamic_stats["velocity_y"]["mean"],
                                            dynamic_stats["velocity_y"]["std"]),
            "hydro_id": dyn["hydro_id"],
        }
        dynamic_data.append(dyn_std)
    
    if split == "test":
        for h_idx, dyn in enumerate(dynamic_data):
            T = dyn["water_depth"].shape[0]
            if T < rollout_length:
                raise ValueError(
                    f"Hydrograph {dyn['hydro_id']} does not have enough time steps for the specified rollout_length."
                )

    return {
        'hydrograph_ids': hydrograph_ids,
        'num_nodes': num_nodes,
        "xy_coords": xy_coords,
        "area": area,
        "elevation": elevation,
        "edge_index": edge_index,
        "edge_distance": edge_distance,
        "edge_relative_distance": edge_relative_distance,
        'slope': slope,
        'dynamic_data': dynamic_data,
        'static_stats': static_stats,
        'dynamic_stats': dynamic_stats,
    }

def convert_to_pyg(dataset_features: dict) -> Data:
    data = Data()

    data.edge_index = torch.LongTensor(dataset_features['edge_index'])
    data.edge_distance = torch.FloatTensor(dataset_features['edge_distance'])
    data.edge_relative_distance = torch.FloatTensor(dataset_features['edge_relative_distance'])

    data.num_nodes = dataset_features['num_nodes']
    data.pos = torch.FloatTensor(dataset_features['pos'])
    data.DEM = torch.FloatTensor(dataset_features['dem'])
    data.slope_x = torch.FloatTensor(dataset_features['slope_x'])
    data.slope_y = torch.FloatTensor(dataset_features['slope_y'])
    data.WD = torch.FloatTensor(dataset_features['water_depth'].T)
    data.VX = torch.FloatTensor(dataset_features['cell_velocity_x'].T)
    data.VY = torch.FloatTensor(dataset_features['cell_velocity_y'].T)

    return data

def main():
    hydrographnet_data_folder = ""
    dataset_folder = "hydrographnet_datasets"
    prefix = "M80"
    k = 4
    num_val_timesteps = 30
    train_ids_file = '0_train.txt'
    test_ids_file = '0_test.txt'

    create_dataset_folders(dataset_folder=dataset_folder)
    print(f"Dataset folder created in: {dataset_folder}", flush=True)

    # ============= Create train dataset =============
    train_processed = process_dataset(
        data_dir=hydrographnet_data_folder,
        prefix=prefix,
        k=k,
        hydrograph_ids_file=train_ids_file,
        split="train",
    )

    constant_dataset_features = {
        'edge_index': train_processed['edge_index'],
        'edge_distance': train_processed['edge_distance'],
        'edge_relative_distance': train_processed['edge_relative_distance'],
        'num_nodes': train_processed['num_nodes'],
        'pos': train_processed['xy_coords'],
        'dem': train_processed['elevation'].squeeze(),
        # Temporary workaround: slope_x and slope_y are just the cell slope value.
        'slope_x': train_processed['slope'].squeeze(),
        'slope_y': train_processed['slope'].squeeze(),
    }

    grid_dataset = []
    for i, key in enumerate(train_processed['hydrograph_ids']):
        print(f"Processing training event {key}", flush=True)

        paths = train_processed['dynamic_data'][i]
        water_depth = paths['water_depth']
        velocity_x = paths['velocity_x']
        velocity_y = paths['velocity_y']

        dataset_features = {
            **constant_dataset_features,
            'water_depth': water_depth,
            'cell_velocity_x': velocity_x,
            'cell_velocity_y': velocity_y,
        }
        pyg_dataset = convert_to_pyg(dataset_features)
        grid_dataset.append(pyg_dataset)
    print('Saving training dataset', flush=True)
    save_database(grid_dataset, name='hydrographnet', out_path=f"{dataset_folder}/train")

    # ============= Create test dataset =============
    rollout_length = num_val_timesteps + 1
    test_processed = process_dataset(
        data_dir=hydrographnet_data_folder,
        prefix=prefix,
        k=k,
        hydrograph_ids_file=test_ids_file,
        split="test",
        rollout_length=rollout_length,
        param_static_stats=train_processed['static_stats'],
        param_dynamic_stats=train_processed['dynamic_stats'])

    constant_dataset_features = {
        'edge_index': test_processed['edge_index'],
        'edge_distance': test_processed['edge_distance'],
        'edge_relative_distance': test_processed['edge_relative_distance'],
        'num_nodes': test_processed['num_nodes'],
        'pos': test_processed['xy_coords'],
        'dem': test_processed['elevation'].squeeze(),
        # Temporary workaround: slope_x and slope_y are just the cell slope value.
        'slope_x': test_processed['slope'].squeeze(),
        'slope_y': test_processed['slope'].squeeze(),
    }

    grid_dataset = []
    for i, key in enumerate(test_processed['hydrograph_ids']):
        print(f"Processing testing event {key}", flush=True)

        paths = test_processed['dynamic_data'][i]
        water_depth = paths['water_depth']
        velocity_x = paths['velocity_x']
        velocity_y = paths['velocity_y']

        dataset_features = {
            **constant_dataset_features,
            'water_depth': water_depth,
            'cell_velocity_x': velocity_x,
            'cell_velocity_y': velocity_y,
        }
        pyg_dataset = convert_to_pyg(dataset_features)
        grid_dataset.append(pyg_dataset)
    print('Saving testing dataset', flush=True)
    save_database(grid_dataset, name='hydrographnet', out_path=f"{dataset_folder}/test")

if __name__ == "__main__":
    main()