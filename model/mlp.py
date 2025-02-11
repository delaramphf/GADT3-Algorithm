import torch.nn as nn

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features, out_classes)

    def forward(self, graph, h):
        with graph.local_scope():
            return self.W(h)
