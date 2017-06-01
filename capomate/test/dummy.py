import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier


def inline(inputfile, outputfile):
    # data = np.loadtxt(sys.stdin)
    data = np.loadtxt(inputfile, delimiter=',')
    if np.ndim(data) == 1:
        data = np.array([data])
    train_x = data[:, 1:]
    train_y = data[:, 0]

    candidate_size = 1000
    evaluation_size = 1000
    x, y = make_classification(n_samples=candidate_size + evaluation_size,
                               n_features=2,
                               n_informative=1,
                               n_redundant=1,
                               n_clusters_per_class=1,
                               random_state=37)
    eval_x = x[candidate_size:]
    eval_y = y[candidate_size:]

    learner = KNeighborsClassifier(n_neighbors=1)
    learner = learner.fit(train_x, train_y)
    pred_y = learner.predict(eval_x)
    with open(outputfile, 'w') as f:
        l = zero_one_loss(eval_y, pred_y)
        f.write(str(l))

if __name__ == "__main__":
    inline(sys.argv[1], sys.argv[2])
