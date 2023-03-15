Workflow

General
- set seed (important for reproducibility)
-- set universal seed per package
--- scikit-learn may have its own seed
--- torch may have its own seed

Data
- preprocessing
-- standardize features (mean center and std of train)
-- convert to tensor

- convert to TensorDataset

- load by batch using DataLoader

Model Setup
- model structure
- loss
- learning rate
- num epochs and log epochs

Model Training
- feedforward, loss, backprop
- optimizer
- epoch logging
