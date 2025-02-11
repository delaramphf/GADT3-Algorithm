import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score

def compute_auc_ap(pred, labels):
    pred = np.array(torch.squeeze(pred).cpu().detach())
    pred = 1 - normalize_score(pred)
    auc = roc_auc_score(labels.cpu(), pred)
    ap = average_precision_score(labels.cpu(), pred, average='macro', pos_label=1)
    return auc, ap

def mmd_rbf_vectorized(target_features, source_features, kernel_mul=2.0):
    target_features = F.normalize(target_features, dim=1)
    source_features = F.normalize(source_features, dim=1)
    XX = torch.matmul(source_features, source_features.T)  
    YY = torch.matmul(target_features, target_features.T)
    XY = torch.matmul(source_features, target_features.T)
    
    XX_diag = torch.diagonal(XX).unsqueeze(0)
    YY_diag = torch.diagonal(YY).unsqueeze(0)
    
    X2 = XX_diag.T + XX_diag - 2*XX
    Y2 = YY_diag.T + YY_diag - 2*YY
    XY2 = XX_diag.T + YY_diag - 2*XY
    
    bandwidth = torch.median(torch.sqrt(torch.abs(X2) + 1e-8))
    bandwidth_sq = (bandwidth * kernel_mul).item() ** 2
    
    XX_kernel = torch.exp(-X2 / (2 * bandwidth_sq))
    YY_kernel = torch.exp(-Y2 / (2 * bandwidth_sq))
    XY_kernel = torch.exp(-XY2 / (2 * bandwidth_sq))
    
    mmd_per_target = (
        XX_kernel.mean(dim=0).mean() +
        torch.diagonal(YY_kernel) -
        2 * XY_kernel.mean(dim=0)
    )
    return mmd_per_target
