import numpy as np
import random
# import torch  # Commented out as PyTorch is not used for actual computations

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed to use
    """
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)  # Commented out as PyTorch is not used
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False 