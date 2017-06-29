import string
import random
import sys
import subprocess
import tempfile
import os
from argparse import ArgumentParser
from search import Search, Runner
from logger import Logger
# from capomate.test.dummy import inline


class ProcessSearch(Search):
    def __init__(self,
                 algorithm,
                 candidate_pool,
                 learner,
                 loss,
                 search_budget,
                 progress_bar,
                 attention_budget,
                 logger):
        super(ProcessSearch, self).__init__(algorithm,
                                            candidate_pool,
                                            learner,
                                            loss,
                                            search_budget,
                                            progress_bar,
                                            attention_budget,
                                            logger)

    def inds_to_dataset(self, inds):
        res = []
        for i in inds:
            res.append(self.candidate_pool[0][i])
        return res


class ProcessLearner(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.loss = None
        dir = tempfile._get_default_tempdir()
        self.input_filename = os.path.join(dir,
                                           next(tempfile._get_candidate_names()))
        self.output_filename = os.path.join(dir,
                                            next(tempfile._get_candidate_names()))

    def init_model(self):
        return None

    def fit(self, model, data):
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
        # inline(self.input_filename, self.output_filename)
        error_code = subprocess.call(shell_cmd.split(' '))
        if error_code != 0:
            print "Child process returned with error code %s" % error_code
            sys.exit(-1)
        if not os.path.exists(self.output_filename):
            raise Exception('Output file not written')
        with open(self.output_filename) as f:
            self.loss = float(string.strip(f.readline()))
        os.remove(self.input_filename)
        os.remove(self.output_filename)
        return None


class ProcessRunner(Runner):
    def __init__(self, cmd, logger=Logger()):
        super(ProcessRunner, self).__init__(logger)
        self.cmd = cmd

    def search_factory(self,
                       algorithm,
                       candidate_pool,
                       learner,
                       loss,
                       search_budget,
                       progress_bar,
                       attention_budget,
                       logger):
        return ProcessSearch(algorithm,
                             candidate_pool,
                             learner,
                             loss,
                             search_budget,
                             progress_bar,
                             attention_budget,
                             logger)

    def construct_learner(self, options):
        return ProcessLearner(self.cmd)

    def construct_loss(self, instance, options, learner):
        def f(model):
            return learner.loss
        return f, None


def main():
    parser = ArgumentParser()
    parser.add_argument("--candidate-pool-filename",
                        help='Filename for candidate pool, one item per line',
                        required=True)
    parser.add_argument("--loss-executable",
                        help='Executable command to return loss on teaching set',
                        required=True)
    parser.add_argument("--output-filename",
                        help='Filename to write best found teaching set and loss to',
                        required=True)
    parser.add_argument("--teaching-set-size",
                        type=int,
                        help='Size of teaching set to return',
                        required=True)
    parser.add_argument("--search-budget",
                        type=int,
                        help='Budget of models to fit',
                        required=True)
    parser.add_argument("--proposals",
                        type=int,
                        help='Number of proposals to consider at each search iteration')
    parser.add_argument("--seed",
                        type=int,
                        help='Random seed',
                        default=random.seed())
    parser.add_argument('--algorithm',
                        help='Teaching search algorithm',
                        choices=['greedy-add', 'random-index-greedy-swap'],
                        default='greedy-add')

    options = parser.parse_args()
    options.teaching_budget = options.teaching_set_size
    options.parallel = False
    options.searcher = options.algorithm
    options.num_trials = 1
    options.initial_training_set = None
    options.no_progress = False
    options.attention_budget = -1
    options.filename = None

    # Get candidate pool
    with open(options.candidate_pool_filename) as f:
        candidate_pool = f.readlines()
    instance = ((candidate_pool, candidate_pool),
                ([], []),
                ([], []))

    if options.proposals is None:
        options.proposals = len(candidate_pool) / options.teaching_budget

    log = Logger()
    log.store_instance = False
    runner = ProcessRunner(options.loss_executable)
    results = runner.run_experiment(instance, options)
    loss = results.results[0].best_evaluation_loss
    best_set_indices = results.results[0].best_set
    with open(options.output_filename, 'w') as f:
        f.write(str(loss) + '\n')
        for idx in best_set_indices:
            f.write(candidate_pool[idx])

if __name__ == "__main__":
    main()
