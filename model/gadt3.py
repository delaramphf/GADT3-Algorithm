# model/gadt3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from .sage import SAGE
from .mlp import MLPPredictor

@dataclass
class ModelConfig:
    """Configuration class for GADT3 model parameters"""
    ndim: int
    ndim_s: int
    ndim_t: int
    hid_dim: int = 128
    out_classes: int = 2
    dropout: float = 0.7
    use_attention: bool = True
    num_layers: int = 2
    activation: str = 'relu'
    residual: bool = True
    layer_norm: bool = True
    batch_norm: bool = False

class DomainProjector(nn.Module):
    """
    Single-layer neural network module for projecting features from source/target domains 
    into a common feature space.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection layer weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

class GADT3(nn.Module):
    """
    Graph Anomaly Detection with Test-time Training (GADT3)
    
    A novel framework for cross-domain graph anomaly detection that adapts to target
    domain data during test time using self-supervised learning objectives.
    
    Key Features:
    - Domain-specific feature projectors for handling heterogeneous feature spaces
    - Enhanced GraphSAGE backbone with attention mechanism
    - Test-time adaptation capability using homophily-based loss
    - Multi-scale feature fusion
    
    Args:
        config (ModelConfig): Model configuration parameters
        
    Attributes:
        transfer_mode (bool): If True, operates in target domain adaptation mode
        feature_fusion (bool): If True, uses multi-scale feature fusion
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transfer_mode = False
        self.feature_fusion = True
        
        # Simple domain projectors
        self.proj_s = DomainProjector(config.ndim_s, config.ndim)
        self.proj_t = DomainProjector(config.ndim_t, config.ndim)
        
        # Initialize activation function
        self.activation = getattr(F, config.activation)
        
        # Graph neural network backbone
        self.gnn = SAGE(
            ndim=config.ndim,
            activation=self.activation,
            dropout=config.dropout,
            hid_dim=config.hid_dim
        )
        
        # Prediction head
        pred_input_dim = config.hid_dim * 2 if self.feature_fusion else config.hid_dim
        self.pred = MLPPredictor(pred_input_dim, config.out_classes)
        
        # Additional components for enhanced feature processing
        if config.layer_norm:
            self.layer_norm = nn.LayerNorm(config.ndim)
        if config.batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.ndim)
            
        # Feature fusion components
        if self.feature_fusion:
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(config.hid_dim * 2, config.hid_dim),
                nn.ReLU(),
                nn.Linear(config.hid_dim, config.hid_dim)
            )
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def _transfer(self, transfer_mode: bool):
        """Switch between source and target domain modes"""
        self.transfer_mode = transfer_mode
        
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply normalization to features if configured"""
        if hasattr(self, 'layer_norm'):
            features = self.layer_norm(features)
        if hasattr(self, 'batch_norm'):
            features = self.batch_norm(features)
        return features
    
    def _fuse_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features using learned weights"""
        if not self.feature_fusion:
            return features[-1]
            
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted concatenation of features from different scales
        fused = torch.cat([
            w * f for w, f in zip(weights, features[-2:])
        ], dim=1)
        
        return self.fusion_mlp(fused)
    
    def get_domain_features(self, g) -> torch.Tensor:
        """Extract and project domain-specific features"""
        if self.transfer_mode:
            features = self.proj_t(g.ndata['h'])
        else:
            features = self.proj_s(g.ndata['h'])
        return self._normalize_features(features)
    
    def forward(self, g, extract_features: bool = False, 
               extract_attentions: bool = False) -> Union[
                   torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor],
                   Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]
               ]:
        """
        Forward pass of the GADT3 model.
        
        Args:
            g: Input graph
            extract_features: If True, return intermediate features
            extract_attentions: If True, return attention weights
            
        Returns:
            pred: Node predictions
            features: (Optional) Intermediate features
            attentions: (Optional) Attention weights
        """
        # Project domain features
        g.ndata['h_proj'] = self.get_domain_features(g)
        nfeats = g.ndata['h_proj']
        
        # Forward pass through GNN
        h, features, attentions = self.gnn(
            g, nfeats, 
            extract_features=True,  # Always extract for fusion
            extract_attentions=extract_attentions
        )
        
        # Feature fusion
        h_final = self._fuse_features(features)
        
        # Prediction
        pred = self.pred(g, h_final)
        
        # Return appropriate outputs based on flags
        if extract_features and extract_attentions:
            return pred, torch.cat(features, dim=1), attentions
        elif extract_features:
            return pred, torch.cat(features, dim=1)
        elif extract_attentions:
            return pred, attentions
        else:
            return pred

    def get_attention_stats(self) -> dict:
        """Get statistics about the attention weights"""
        stats = {}
        for i, weight in enumerate(self.gnn.attention_weights):
            stats[f'layer_{i}_mean'] = weight.mean().item()
            stats[f'layer_{i}_std'] = weight.std().item()
        return stats
    
    def get_feature_stats(self, features: torch.Tensor) -> dict:
        """Compute statistics of the learned features"""
        return {
            'mean': features.mean().item(),
            'std': features.std().item(),
            'norm_mean': features.norm(dim=1).mean().item()
        }
