import scipy.sparse as sp
import numpy as np
import torch
import dgl
from sklearn.preprocessing import StandardScaler
from .preprocess import preprocess_features

def load_mat(dataset_name):
    """
    Load the dataset from .mat file
    Returns adjacency matrix, features, labels
    """
    # This is a placeholder - you'll need to implement actual data loading
    # based on your .mat file structure
    pass

def adj_to_dgl_graph(adj):
    """Convert sparse adjacency matrix to DGL graph"""
    # Convert to sparse format if not already
    if not sp.isspmatrix(adj):
        adj = sp.csr_matrix(adj)
    
    # Create DGL graph
    graph = dgl.from_scipy(adj)
    return graph

def prepare_data(adj, features, ano_label, p=0.2, device='cuda:0'):
    """Prepare data for training"""
    # Process adjacency matrix
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj[np.newaxis])
    
    # Process features
    if isinstance(features, sp.spmatrix):
        features = features.todense()
    features = torch.FloatTensor(features[np.newaxis])
    
    # Create DGL graph
    dgl_graph = adj_to_dgl_graph(adj[0])
    
    # Add node features and labels
    dgl_graph.ndata['feature'] = torch.squeeze(features)
    dgl_graph.ndata['train_mask'] = torch.rand(len(features[0])) > p
    
    # Normalize features
    scaler = StandardScaler()
    train_features = dgl_graph.ndata['feature'][dgl_graph.ndata['train_mask']]
    scaler.fit(train_features)
    dgl_graph.ndata['h'] = torch.tensor(
        scaler.transform(dgl_graph.ndata['feature']), 
        dtype=torch.float32
    )
    
    # Add labels
    dgl_graph.ndata['Label'] = torch.from_numpy(ano_label)
    
    return dgl_graph.to(device)

def subsample_graph(adj, features, ano_label, sample_percentage):
    """Subsample the graph to reduce size"""
    num_nodes = adj.shape[0]
    sample_size = int(num_nodes * sample_percentage)
    sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
    
    # Subsample adjacency matrix, features and labels
    adj_sampled = adj[sample_indices][:, sample_indices]
    features_sampled = features[sample_indices]
    ano_label_sampled = ano_label[sample_indices]
    
    return adj_sampled, features_sampled, ano_label_sampled

