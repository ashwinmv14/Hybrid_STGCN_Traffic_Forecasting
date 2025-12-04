import torch


def masked_mse(preds, labels, null_val=0.0):
    """
    preds:  [B, H, N, 1]
    labels: [B, H, N, 1]
    """
    # convert scalar null_val to tensor
    if isinstance(null_val, float) or isinstance(null_val, int):
        null_val = torch.tensor([null_val], dtype=preds.dtype, device=preds.device)

    # mask invalid values
    mask = (labels != null_val).float()
    mask = mask / (mask.mean() + 1e-6)

    loss = (preds - labels) ** 2
    loss = loss * mask
    return loss.mean()


def masked_rmse(preds, labels, null_val=0.0):
    """Root Mean Squared Error with masking"""
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=0.0):
    """Mean Absolute Error with masking"""
    if isinstance(null_val, float) or isinstance(null_val, int):
        null_val = torch.tensor([null_val], dtype=preds.dtype, device=preds.device)

    mask = (labels != null_val).float()
    mask = mask / (mask.mean() + 1e-6)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    return loss.mean()


def masked_mape(preds, labels, null_val=0.0):
    """
    Mean Absolute Percentage Error with masking
    
    MAPE = mean(|y_pred - y_true| / |y_true|) * 100
    
    Note: Returns percentage (0-100 scale), not decimal
    """
    if isinstance(null_val, float) or isinstance(null_val, int):
        null_val = torch.tensor([null_val], dtype=preds.dtype, device=preds.device)

    mask = (labels != null_val).float()
    mask = mask / (mask.mean() + 1e-6)

    # Avoid division by zero - add small epsilon
    loss = torch.abs((preds - labels) / (torch.abs(labels) + 1e-6))
    loss = loss * mask
    
    # Return as percentage (multiply by 100)
    return loss.mean() * 100.0