class Config:
    def __init__(self):
        # Model parameters
        self.ndim_common = 128  # shared domain dimension
        self.hid_dim = 32      # hidden dimension
        self.dropout = 0.7     # dropout rate
        
        # Training parameters
        self.source_percentage = 0.8  # percentage of source domain data to use
        self.lr_source = 0.001       # learning rate for source model
        self.lr_target = 0.005       # learning rate for target model
        self.patience = 5            # early stopping patience
        self.lambda_tam = 0.001      # weight for target adaptation loss
        self.lambda_reg = 0.1        # weight for regularization loss
        
        # Device configuration
        self.device = 'cuda:0'
        
        # Dataset parameters
        self.train_test_split = 0.2  # percentage of data for testing
