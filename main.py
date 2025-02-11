import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from model.gadt3 import GADT3
from data.loader import load_mat, prepare_data, subsample_graph
from utils.loss import reg_edge, max_message, inference
from utils.metrics import compute_auc_ap
from utils.helper import EarlyStoppingCriterion
from config import Config

def train_source(model, graph, config):
    """Train the source domain model"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_source)
    criterion = nn.CrossEntropyLoss()
    best_auc = 0
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        pred, s_f = model(graph, extract_features=True)
        loss = criterion(pred[graph.ndata['train_mask']], 
                        graph.ndata['Label'][graph.ndata['train_mask']])
        
        # Add loss components
        loss_tam, _ = max_message(s_f, graph.adjacency_matrix().to_dense())
        loss_reg = reg_edge(s_f, graph.adjacency_matrix().to_dense())
        loss = loss + config.lambda_tam * (loss_tam + config.lambda_reg * loss_reg)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, s_f = model(graph, extract_features=True)
                pred = inference(s_f, graph.adjacency_matrix().to_dense())
                auc, ap = compute_auc_ap(pred[~graph.ndata['train_mask']], 
                                       graph.ndata['Label'][~graph.ndata['train_mask']])
                print(f"Epoch {epoch}: AUC = {auc*100:.2f}%, AP = {ap*100:.2f}%")
                
                if auc > best_auc:
                    best_auc = auc
            model.train()
    
    return model

def train_target(source_model, target_graph, config):
    """Adapt the model to target domain using test-time training"""
    # Create target model from source model
    target_model = deepcopy(source_model)
    target_model._transfer(True)
    
    # Freeze appropriate parts of the model
    target_model.proj_s.requires_grad_(False)
    target_model.proj_t.requires_grad_(True)
    target_model.gnn.layers[0].requires_grad_(False)
    target_model.gnn.layers[1].requires_grad_(False)
    target_model.pred.requires_grad_(False)
    
    optimizer = torch.optim.Adam(target_model.parameters(), lr=config.lr_target)
    early_stopping = EarlyStoppingCriterion(patience=config.patience)
    history = {'auc': [], 'ap': [], 'shift': []}
    
    for epoch in range(200):
        target_model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred, t_f = target_model(target_graph, extract_features=True)
        
        # Compute adaptation loss
        loss_tam, _ = max_message(t_f, target_graph.adjacency_matrix().to_dense())
        loss_reg = reg_edge(t_f, target_graph.adjacency_matrix().to_dense())
        loss = config.lambda_tam * (loss_tam + config.lambda_reg * loss_reg)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        target_model.eval()
        with torch.no_grad():
            _, t_f = target_model(target_graph, extract_features=True)
            pred = inference(t_f, target_graph.adjacency_matrix().to_dense())
            auc, ap = compute_auc_ap(pred, target_graph.ndata['Label'])
            
            # Update history
            history['auc'].append(auc)
            history['ap'].append(ap)
            
            print(f"Epoch {epoch}: AUC = {auc*100:.2f}%, AP = {ap*100:.2f}%")
            
            # Early stopping check
            if early_stopping(auc):
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    return target_model, history

def main():
    # Load configuration
    config = Config()
    
    # Load source domain data
    adj_source, features_source, labels_source = load_mat('Amazon')
    adj_source, features_source, labels_source = subsample_graph(
        adj_source, features_source, labels_source, config.source_percentage
    )
    source_graph = prepare_data(adj_source, features_source, labels_source, 
                              config.train_test_split, config.device)
    
    # Load target domain data
    adj_target, features_target, labels_target = load_mat('Facebook')
    target_graph = prepare_data(adj_target, features_target, labels_target,
                              config.train_test_split, config.device)
    
    # Create model
    model = GADT3(
        ndim=config.ndim_common,
        ndim_s=source_graph.ndata['h'].shape[1],
        ndim_t=target_graph.ndata['h'].shape[1],
        activation=F.relu,
        dropout=config.dropout,
        hid_dim=config.hid_dim
    ).to(config.device)
    
    # Train on source domain
    print("Training on source domain...")
    source_model = train_source(model, source_graph, config)
    
    # Adapt to target domain
    print("\nAdapting to target domain...")
    target_model, history = train_target(source_model, target_graph, config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
