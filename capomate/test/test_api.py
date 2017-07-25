import numpy as np
import StringIO

from capomate.teach import Runner, SVMLearner, build_options
from sklearn.datasets import make_classification


# TODO: Unneeded?
class ScikitTextLearner:
    def __init__(self, scikit_learner):
        self.scikit_learner = scikit_learner

    def fit(self, yx):
        pass

    def loss(self, model):
        pass

def make_example():
    x, y = make_classification(n_samples=100,
                               n_features=20,
                               n_informative=2,
                               n_redundant=2,
                               n_clusters_per_class=2,
                               flip_y=0.01)
    z = np.concatenate((np.reshape(y, (len(y), 1)), x), axis=1)
    return z


def test_numpy_python_api():
    z = make_example()
    runner = Runner()
    learner = SVMLearner(z)
    options = build_options(search_budget=10,
                            teaching_set_size=2)
    best_loss, best_set = runner.run_experiment(z, learner, options)

def test_text_python_api():
    z = make_example()
    runner = Runner()
    learner = SVMLearner(z)
    options = build_options(search_budget=10,
                            teaching_set_size=2)
    best_loss, best_set = runner.run_experiment(z, learner, options)

def test_log_stream():
    z = make_example()
    runner = Runner()
    learner = SVMLearner(z)
    log = StringIO.StringIO()
    options = build_options(search_budget=10,
                            teaching_set_size=2,
                            log=log)
    best_loss, best_set = runner.run_experiment(z, learner, options)
    print best_set, best_loss

# is this exactly like other api?
  # no this wrapper isn't taking indices
# can we output and plot performance?
# what about doing it for text?
# document
