import torch

from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

EPS = 1e-7 # Prevent division by zero

def RMSE(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    if mask is None:
        return torch.sqrt(mse_loss(pred, target))

    mse = mse_loss(pred, target, reduction='none')
    mse = (mse * mask).sum() / (mask.sum() + EPS)
    return torch.sqrt(mse)

def MAE(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    if mask is None:
        return l1_loss(pred, target, reduction='mean')

    mae = l1_loss(pred, target, reduction='none')
    mae = (mae * mask).sum() / (mask.sum() + EPS)
    return mae

def NSE(pred: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
    '''Nash Sutcliffe Efficiency'''
    if mask is None:
        model_sse = torch.sum((target - pred)**2)
        mean_model_sse = torch.sum((target - target.mean())**2)
        return 1 - (model_sse / mean_model_sse)

    target_mean = (target * mask).sum() / (mask.sum() + EPS)
    model_sse = torch.sum((target - pred)**2 * mask)
    mean_model_sse = torch.sum((target - target_mean)**2 * mask)
    return 1 - (model_sse / (mean_model_sse + EPS))

def CSI(binary_pred: Tensor, binary_target: Tensor):
    TP = (binary_pred & binary_target).sum() #true positive
    # TN = (~binary_pred & ~binary_target).sum() #true negative
    FP = (binary_pred & ~binary_target).sum() #false positive
    FN = (~binary_pred & binary_target).sum() #false negative

    return TP / (TP + FN + FP)
