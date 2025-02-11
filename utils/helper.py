class EarlyStoppingCriterion:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        
    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def normalize_mmd(mmd_values):
    min_val = mmd_values.min()
    max_val = mmd_values.max()
    if max_val - min_val < 1e-8:
        return torch.ones_like(mmd_values)
    return (mmd_values - min_val) / (max_val - min_val)

def normalize_shift_gaussian(shift_history, current_shift):
    if len(shift_history) < 2:
        return current_shift
    history_tensor = torch.tensor(shift_history + [current_shift])
    mean = history_tensor.mean()
    std = history_tensor.std() + 1e-8
    normalized_shift = abs((current_shift - mean) / std)
    return normalized_shift
