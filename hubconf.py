import torch
import torch.nn as nn

def custom_model():
    """ # This docstring shows up in hub.help()
    Loads a model from a path
    and returns the model 
    """
    # load
    path = 'torch_hub\iris_classifier.pt'
    model = torch.load(path)
    return model


