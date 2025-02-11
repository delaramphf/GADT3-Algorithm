import torch
import torch.nn.functional as F

def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = (1 - adj)
    sim_u_u = sim_u_u * adj_inverse
    
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    
    sim_u_u_no_diag = torch.sum(sim_u_u, 1) * r_inv
    loss_reg = torch.sum(sim_u_u_no_diag)
    return loss_reg

def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    return -torch.sum(message), message

def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    return message
