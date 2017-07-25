import numpy as np
import pandas as pd
from tqdm import tqdm
from StringIO import StringIO

from ..teach import Runner, SVMLearner, build_options
from ..options import OptionsBuilder
from sklearn.datasets import make_classification

# Dimensions of variation:
# number of trials / search budget / teaching set size, other instance parameters / other learner parameters / proposals

# for proposals in [128,64,32,16,8,4,2,1]
# set it up so that you get 1 teaching set that tries everything and then go to uniform sampling

# Consider the following statistics:
# A plot of loss / timestep
# A plot of cumloss / timestep
# A plot of mean cumloss
# A plot of least loss
# And over several trials we an look at mean, min, and max

# What does this optimal proposals value vary as a function of?

options = OptionsBuilder()          \
          .log_most_proposals(7)    \
          .log_teaching_set_size(3) \
          .n_features(20)           \
          .n_informative(2)         \
          .n_redundant(2)           \
          .n_clusters_per_class(2)  \
          .trials(100)              \
          .flip_y(0.0)              \
          .derived('teaching_set_size',
                   lambda options: 2 ** options.log_teaching_set_size) \
          .derived('search_budget',
                   lambda options: 2 ** (options.log_most_proposals + options.log_teaching_set_size)) \
          .derived('n_samples',
                   lambda options: 2 ** options.log_most_proposals)


def make_classification_example(options):
    x, y = make_classification(n_samples=options.n_samples,
                               n_features=options.n_features,
                               n_informative=options.n_informative,
                               n_redundant=options.n_redundant,
                               n_clusters_per_class=options.n_clusters_per_class,
                               flip_y=options.flip_y)
    z = np.concatenate((np.reshape(y, (len(y), 1)), x), axis=1)
    return z


def run_example(z, exp_options):
    df = pd.DataFrame()
    for i in tqdm(range(exp_options.trials)):
        runner = Runner()
        learner = SVMLearner(z)
        options = build_options(proposals=exp_options.proposals,
                                search_budget=exp_options.search_budget,
                                teaching_set_size=exp_options.teaching_set_size,
                                log=StringIO())
        best_loss, best_set = runner.run_experiment(z, learner, options)
        res = pd.read_csv(StringIO(options.log.getvalue()), header=None)
        df = df.append(res[1])
    return df.transpose()


def main(options=options):
    opts = options.build()
    z = make_classification_example(opts)
    df = pd.DataFrame()
    for proposals in [2 ** x for x in range(opts.log_most_proposals+1)]:
        opts = options.proposals(proposals).build()
        df[proposals] = run_example(z, opts).cummin().mean(axis=1)
    return df


def plot_min_avg_loss(df):
    df.min(axis=0).plot.bar()


def plot_avg_loss(df):
    df.plot()

if __name__ == "__main__":
    main()
