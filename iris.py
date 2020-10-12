import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlp import Perceptron

dataset = pd.read_csv('iris.csv',header = None)
features = dataset[[0,1,2,3]]
labels = dataset[4]
features[0] = features[0]/features[0].max()
features[1] = features[1]/features[1].max()
features[2] = features[2]/features[2].max()
features[3] = features[3]/features[3].max()
features = np.array(features)
labels = np.array(labels)
mlp = Perceptron()
mlp.fit(features,labels,[2])
for i in range(300):
    mlp.train()
mlp.result()