import torch
from model import Model # imported model class

def custom_model():
    """ # This docstring shows up in hub.help()
    Loads a model from a path
    and returns the model 
    """
    # load
    path = 'iris_classifier.pt'
    model = torch.load(path)
    return model