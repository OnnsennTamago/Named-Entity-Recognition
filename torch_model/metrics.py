
import torch
import numpy as np
from sklearn.metrics import f1_score

def masking(lengths: torch.Tensor) -> torch.Tensor:
    return torch.arange(end=lengths.max(), device=lengths.device).expand(size=(lengths.shape[0], lengths.max())) < lengths.unsqueeze(1)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to np.ndarray.
    """
    return (tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy())

def calculate_metrics(metrics, loss, use_crf, loss_crf, y_true, y_pred, idx2label):
    '''
    y_true: ndarray
    y_pred: ndarray
    '''
    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
    if use_crf:
        metrics["loss_crf"].append(loss_crf)
    metrics["loss"].append(loss)

    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, labels=range(len(idx2label)), average=None)
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    for cls, f1 in enumerate(f1_per_class):
        metrics[f"f1 {idx2label[cls]}"].append(f1)
    metrics["f1-weighted"].append(f1_weighted)

    return metrics
