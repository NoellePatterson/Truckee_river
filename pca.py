# Perform principal components analysis to reduce dimension of ffm inputs to Bayesian model
# perform separately on upstream and downstream datasets

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pdb

# prep dataframes
ffc_blw_derby = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_pearson_vals.csv')
ffc_blw_derby = ffc_blw_derby.apply(pd.to_numeric, errors='coerce')
ffc_vista = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_pearson_vals.csv')
ffc_vista = ffc_vista.apply(pd.to_numeric, errors='coerce')

ffc_blw_derby.set_index('year')
ffc_vista.set_index('year')
# standardize data
ffc_blw_derby = StandardScaler().fit_transform(ffc_blw_derby)
ffc_vista = StandardScaler().fit_transform(ffc_vista)

# Fill NA's with an imputer that uses regressions to estimate missing values
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(ffc_blw_derby)
ffc_blw_derby_imp = imp.transform(ffc_blw_derby)
imp.fit(ffc_vista)
ffc_vista_imp = imp.transform(ffc_vista)

# run the PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(ffc_blw_derby_imp)
principal_derby = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
pca.explained_variance_ratio_
# pdb.set_trace()
principalComponents = pca.fit_transform(ffc_vista_imp)
principal_vista = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

principal_vista.to_csv('data_outputs/vista_pca.csv')
principal_derby.to_csv('data_outputs/derby_pca.csv')

pdb.set_trace()