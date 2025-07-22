import random
import numpy as np
import torch
import os

def set_seed(seed: int = 0)-> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy
    random.seed(seed)
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              
    torch.cuda.manual_seed_all(seed)
    # cudnn          
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False    
    print(seed)

