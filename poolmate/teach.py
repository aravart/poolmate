import string
import random
import sys
import subprocess
import tempfile
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from search import Runner
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss


class AllOneLabel:
    def __init__(self, y):
        self.y = y

    def predict(self, xs):
        return [self.y] * len(xs)


class ScikitLearner(object):
    def __init__(self, learner_class, learner_params, z):
        self.learner_class = learner_class
        self.learner_params = learner_params
        self.z = z

    def loss(self, model):
        y_preds = self.current_learner.predict(self.z[:, 1:])
        return zero_one_loss(self.z[:, 0], y_preds, normalize=True)

    def fit(self, yx):
        yx = np.array(yx)
        labels = yx[:, 0]
        features = yx[:, 1:]
        if len(np.unique(labels)) == 1:
            self.current_learner = AllOneLabel(labels[0])
        else:
            self.current_learner = self.learner_class(**self.learner_params)
            self.current_learner.fit(features, labels)


class SVMLearner(ScikitLearner):
    def __init__(self, z):
        super(SVMLearner, self).__init__(SVC,
                                         {'C': 1.0, 'kernel': 'linear'},
                                         z)


class KNNLearner(ScikitLearner):
    def __init__(self, k, z):
        super(KNNLearner, self).__init__(KNeighborsClassifier,
                                         {'n_neighbors': k, 'weights': 'uniform'},
                                         z)

class ProcessLearner(object):
    def __init__(self, cmd):
        self.cmd = cmd
        dir = tempfile._get_default_tempdir()
        self.input_filename = os.path.join(dir,
                                           next(tempfile._get_candidate_names()))
        self.output_filename = os.path.join(dir,
                                            next(tempfile._get_candidate_names()))

    def loss(self, model):
        return self.loss_

    def fit(self, data):
        if os.path.exists(self.input_filename):
            raise Exception('Input file already exists')
        if os.path.exists(self.output_filename):
            raise Exception('Output file already exists')
        with open(self.input_filename, 'w') as f:
            for line in data:
                f.write(line)
        shell_cmd = self.cmd + ' ' + \
            self.input_filename + ' ' + \
            self.output_filename
        error_code = subprocess.call(shell_cmd.split(' '))
        if error_code != 0:
            print "Child process returned with error code %s" % error_code
            sys.exit(-1)
        if not os.path.exists(self.output_filename):
            raise Exception('Output file not written')
        with open(self.output_filename) as f:
            self.loss_ = float(string.strip(f.readline()))
        os.remove(self.input_filename)
        os.remove(self.output_filename)
        return None


def build_options(args=None,
                  candidate_pool_filename=None,
                  loss_executable=None,
                  output_filename=None,
                  teaching_set_size=None,
                  search_budget=None,
                  proposals=None,
                  seed=random.seed(),
                  algorithm='greedy-add',
                  initial_teaching_set=None,
                  log=None):

    parse_ints = lambda value: map(int, value.split(',')) if value else None
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-pool-filename",
                        help='Filename for candidate pool. The format of the file is that candidate items are represented one item per line.',
                        default=candidate_pool_filename)
    parser.add_argument("--loss-executable",
                        help="""Executable command which will return loss on teaching set. Executable must
                        take two command-line arguments, an `inputfilename`
                        containing the teaching set to train the learner on,
                        one item per line, and an `outputfilename` where the
                        loss should be written""",
                        default=loss_executable)
    parser.add_argument("--output-filename",
                        help="""Output filename where the
                        best found teaching set and loss are written at the
                        search procedure\'s termination""",
                        default=output_filename)
    parser.add_argument("--teaching-set-size",
                        type=int,
                        help='Size of teaching set to return.',
                        default=teaching_set_size)
    parser.add_argument("--search-budget",
                        type=int,
                        help='Budget of number of models to fit. This is the number of times `loss-executable` will be invoked.',
                        default=search_budget)
    parser.add_argument("--proposals",
                        type=int,
                        help='Number of proposals to consider at each search iteration. A tuning parameter for \'greedy-add\' and \'random-index-greedy-swap\' algorithms',
                        default=proposals)
    parser.add_argument("--seed",
                        type=int,
                        help='Set random seed to achieve reproducibility.',
                        default=seed)
    parser.add_argument('--algorithm',
                        help='Choice of search algorithm',
                        choices=['greedy-add', 'random-index-greedy-swap', 'uniform'],
                        default=algorithm)
    parser.add_argument("--initial-teaching-set",
                        help='A comma-separated list of zero-based indices to fix the initial teaching set. Used in \'random-index-greedy-swap\' and \'uniform\' algorithms (e.g., --initial-teaching-set 53,17 ).',
                        type=parse_ints,
                        default=initial_teaching_set)
    parser.add_argument('--log',
                        help="""Filename of log file, where interim
                        results are logged as comma-separated values (CSV). The
                        three colums of the output represent the iteration number,
                        the loss of the trained model for that iteration, and
                        the teaching set for that iterations.""",
                        default=log)

    options = parser.parse_args(args=args)
    options.no_progress = False
    return options




def main():
    options = build_options()

    with open(options.candidate_pool_filename) as f:
        candidate_pool = f.readlines()

    runner = Runner()
    learner = ProcessLearner(options.loss_executable)
    best_loss, best_set = runner.run_experiment(candidate_pool,
                                                learner,
                                                options)

    with open(options.output_filename, 'w') as f:
        f.write(str(best_loss) + '\n')
        for idx in best_set:
            f.write(candidate_pool[idx])

if __name__ == "__main__":
    main()
