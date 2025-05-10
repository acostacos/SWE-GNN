# Libraries
import torch
import numpy as np

from argparse import ArgumentParser
from utils.dataset import create_model_dataset, to_temporal_dataset
from utils.dataset import get_temporal_test_dataset_parameters
from utils.load import read_config
from utils.miscellaneous import get_model, SpatialAnalysis
from validation.validation_stats import ValidationStats
from utils.logging_utils import Logger

def main(config, model_path: str, output_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:\t', device)

    dataset_parameters = config['dataset_parameters']
    scalers = config['scalers']
    selected_node_features = config['selected_node_features']
    selected_edge_features = config['selected_edge_features']

    _, _, test_dataset, scalers = create_model_dataset(
        scalers=scalers, device='cpu', **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )

    temporal_dataset_parameters = config['temporal_dataset_parameters']

    # info for testing dataset
    previous_t = temporal_dataset_parameters['previous_t']
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters)

    temporal_test_dataset = to_temporal_dataset(test_dataset, previous_t=previous_t, rollout_steps=-1, 
                                **temporal_test_dataset_parameters)

    node_features, edge_features = temporal_test_dataset[0].x.size(-1), temporal_test_dataset[0].edge_attr.size(-1)
    num_nodes = temporal_test_dataset[0].x.size(0)

    model_parameters = config['models']
    model_type = model_parameters.pop('model_type')
    if model_type == 'GNN':
        model_parameters['edge_features'] = edge_features
    elif model_type == 'MLP':
        model_parameters.num_nodes = num_nodes

    model = get_model(model_type)(
        node_features=node_features,
        previous_t=previous_t,
        device=device,
        **model_parameters)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Rollout error and time
    type_loss = config['trainer_options']['type_loss']
    spatial_analyser = SpatialAnalysis(model, test_dataset, device=device, **temporal_test_dataset_parameters)

    # Prediction metrics
    validation_stats = ValidationStats(logger=Logger())
    DATASET_IDX = 0
    WATER_DEPTH_IDX = 0
    n_timesteps = spatial_analyser.predicted_rollout.shape[3]
    for i in range(n_timesteps):
        water_depth_pred = spatial_analyser.predicted_rollout[DATASET_IDX, :, WATER_DEPTH_IDX, i]
        water_depth_target = spatial_analyser.real_rollout[DATASET_IDX, :, WATER_DEPTH_IDX, i]
        validation_stats.update_stats_for_epoch(water_depth_pred.cpu(),
                                                water_depth_target.cpu(),
                                                water_threshold=0.05)

    validation_stats.print_stats_summary()

    # Prediction time
    average_pred_time = np.mean(spatial_analyser.prediction_times)
    print(f'Inference time for one timestep: {average_pred_time} seconds', flush=True)

    validation_stats.save_stats(output_path)

    # SWE-GNN metrics
    rollout_loss = spatial_analyser._get_rollout_loss(type_loss=type_loss)
    print('test roll loss WD:',rollout_loss.mean(0)[0].item(), flush=True)
    print('test roll loss V:',rollout_loss.mean(0)[1:].mean().item(), flush=True)

    # fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    # fig.savefig("results/temp_CSI.png")

    # best_id = rollout_loss.mean(1).argmin().item()
    # worst_id = rollout_loss.mean(1).argmax().item()
    
    # for id_dataset, name in zip([best_id, worst_id],['best', 'worst']):
    #     rollout_plotter = PlotRollout(model, test_dataset[id_dataset], scalers=scalers, 
    #         type_loss=type_loss, **temporal_test_dataset_parameters)
    #     fig = rollout_plotter.explore_rollout()
    #     fig.savefig("results/temp_summary.png")

if __name__ == '__main__':
    # Read configuration file with parameters
    parser = ArgumentParser(description='')
    # parser.add_argument("--model_path", type=str, required=True, help='Path to model to be validated')
    # parser.add_argument("--output_path", type=str, required=True, help='Path to numpy file to be saved')
    parser.add_argument("--model_path", type=str, help='Path to model to be validated')
    parser.add_argument("--output_path", type=str, help='Path to model to be validated')
    parser.add_argument("--config", type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    args.config = 'configs/initp01_config.yaml'
    args.model_path = 'wandb/offline-run-20250510_150831-9ykal8dx/files/9ykal8dx.h5'
    args.output_path = 'saved_metrics/initp01.npz'

    print('Reading config file: ', args.config, flush=True)
    cfg = read_config(args.config)

    main(cfg, args.model_path, args.output_path)
