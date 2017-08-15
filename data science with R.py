# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:47:07 2017

@author: winson
"""

import pandas as pd
from matplotlib import pyplot as plt
import sklearn.datasets

# Loading data

def get_iris_df():
  ds = sklearn.datasets.load_iris()
  df = pd.DataFrame(ds['data'],
    columns = ds['feature_names'])
  code_species_map = dict(zip(
    range(3), ds['target_names']))
  df['species'] = [code_species_map[c]
    for c in ds['target']]
  return df
df = get_iris_df()

means_by_species = df.groupby('species').mean()

df.plot(kind='hist', subplots=True, layout=(2,2))
plt.suptitle('Iris Histograms', fontsize=20)
plt.show()