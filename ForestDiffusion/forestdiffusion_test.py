from ForestDiffusion import ForestDiffusionModel

# Iris: numpy dataset with 4 variables (all numerical) and 1 outcome (categorical)
from sklearn.datasets import load_iris
import numpy as np
my_data = load_iris()
X, y = my_data['data'], my_data['target']
Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

# Classification problem (outcome is categorical)
forest_model = ForestDiffusionModel(X, label_y=y, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='flow', n_jobs=-1)
Xy_fake = forest_model.generate(batch_size=X.shape[0]) # last variable will be the label_y
np.savetxt(f"ForestDiff_imputed_iris_misprop_categorical.csv", 
                   Xy_fake, delimiter=',')

# Regression problem (outcome is continuous)
Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
forest_model = ForestDiffusionModel(Xy, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[4], int_indexes=[], diffusion_type='flow', n_jobs=-1)
Xy_fake = forest_model.generate(batch_size=X.shape[0])
np.savetxt(f"ForestDiff_imputed_iris_misprop_cont.csv", 
                   Xy_fake, delimiter=',')