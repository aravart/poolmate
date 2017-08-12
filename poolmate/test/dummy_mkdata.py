import string
import numpy as np
from sklearn.datasets import make_classification

n = 1000
x, y = make_classification(n,
                           n_features=2,
                           n_informative=1,
                           n_redundant=1,
                           n_clusters_per_class=1,
                           random_state=37)
z = np.hstack((np.reshape(y, (1000, 1)), x))

for r in z:
    print string.join(map(str, r.tolist()), sep=', ')
