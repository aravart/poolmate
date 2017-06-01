from argparse import ArgumentParser
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import re
import sys
import scipy

from sklearn.metrics import zero_one_loss, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_mldata
from sklearn.metrics.pairwise import pairwise_distances
from learner import Sklearn_Learner

from logger import Logger


class Search(object):
    def __init__(self,
                 algorithm,
                 candidate_pool,
                 learner,
                 loss,
                 search_budget,
                 progress_bar,
                 attention_budget,
                 logger):
        self.algorithm = algorithm
        self.candidate_pool = candidate_pool
        self.learner = learner
        self.loss = loss
        self.search_budget = search_budget
        self.current_model = learner.init_model
        self.progress_bar = progress_bar
        self.attention_budget = attention_budget
        self.logger = logger

    def search(self):
        result = Result()
        self.algorithm.result = result
        m = self.learner.init_model
        rng = tqdm.trange(self.search_budget) \
            if self.progress_bar \
            else range(self.search_budget)
        for i in rng:
            if self.attention_budget != -1 and i > 0 and i % self.attention_budget == 0:
                self.algorithm.restart()
            s = self.algorithm.next_fit_request()
            m = self.learner.fit(m, self.inds_to_dataset(s))
            l = self.loss(m)
            result.fits.append((i, s, l))
            self.logger.model_fit(i=i,
                                  loss=l,
                                  training_set=s)
            self.algorithm.next_fit_result(m, l, s)
        self.algorithm.complete()
        result.best_model = self.algorithm.best_model
        result.best_evaluation_loss = self.algorithm.best_loss
        result.best_set = self.algorithm.best_set
        self.logger.best_set(best_loss=self.algorithm.best_loss,
                             best_set=self.algorithm.best_set)
        return result

    def inds_to_dataset(self, inds):
        return self.candidate_pool[0][inds], self.candidate_pool[1][inds]


class Algorithm(object):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_budget,
                 initial_training_set,
                 logger):
        self.random = random
        self.pool_size = pool_size
        self.teaching_budget = teaching_budget
        self.best_loss = np.inf
        self.best_model = None
        self.best_set = None
        self.logger = logger
        self.result = None
        self._initial_training_set = initial_training_set
        self.calls_to_initial_training_set = 0

    def validate(self, search_budget):
        pass

    def restart(self):
        pass

    def next_fit_request(self):
        pass

    def next_fit_result(self, model, loss, set):
        pass

    def complete(self):
        pass

    def accept_best(self, model, loss, set):
        if loss < self.best_loss:
            self.best_model = model
            self.best_loss = loss
            self.best_set = set
            i = len(self.result.fits)-1
            self.result.best_sets.append((i, self.best_set, self.best_loss))
            self.logger.best_sets(i=i,
                                  best_set=self.best_set,
                                  best_loss=self.best_loss)

    def initial_training_set(self):
        if self.calls_to_initial_training_set == 0 and self._initial_training_set:
            self.calls_to_initial_training_set += 1
            return self._initial_training_set
        else:
            self.calls_to_initial_training_set += 1
            return [self.random.randint(0, self.pool_size) for _ in range(self.teaching_budget)]


class UniformJumping(Algorithm):
    def next_fit_request(self):
        return self.initial_training_set()

    def next_fit_result(self, model, loss, set):
        self.logger.current_set(current_set=set, current_loss=loss)
        self.accept_best(model, loss, set)


class Greedy(Algorithm):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_budget,
                 initial_training_set,
                 logger,
                 proposals):
        super(Greedy, self).__init__(random,
                                     pool_size,
                                     teaching_budget,
                                     initial_training_set,
                                     logger)
        self.proposals = proposals or pool_size / teaching_budget
        self.current_set = []
        self.models_to_fetch = []
        self.models_fetched = []

    def validate(self, search_budget):
        if self.proposals * self.teaching_budget > search_budget:
            msg = 'Parameters will not produce a teaching set: ' + \
                  'proposals * teaching budget > search budget'
            raise Exception(msg)

    def next_fit_request(self):
        if self.models_to_fetch:
            return self.models_to_fetch.pop()
        else:
            self.models_fetched = []
            if len(self.current_set) == self.teaching_budget:
                self.current_set = []
            xs = self.random.choice(self.pool_size,
                                    size=self.proposals,
                                    replace=False).tolist()
            for x in xs:
                self.models_to_fetch.append(self.current_set + [x])
            return self.models_to_fetch.pop()

    def next_fit_result(self, model, loss, set):
        # notice that the [] is not considered: bug!
        self.models_fetched.append((model, loss, set))
        # option: stay at the current size if no better set is found
        if not self.models_to_fetch:
            best_loss = np.inf
            for m, l, s in self.models_fetched:
                if l < best_loss:
                    self.current_model = m
                    self.current_loss = l
                    self.current_set = s
                    best_loss = l
            self.logger.current_set(current_set=self.current_set,
                                    current_loss=self.current_loss)
            # to return any sized set remove following condition
            if len(self.current_set) == self.teaching_budget:
                self.accept_best(self.current_model,
                                 self.current_loss,
                                 self.current_set)

class Anneal(Algorithm):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_budget,
                 initial_training_set,
                 logger,
                 search_budget,
                 neighbors,
                 tmax,
                 tmin):
        super(Anneal, self).__init__(random,
                                     pool_size,
                                     teaching_budget,
                                     initial_training_set,
                                     logger)
        self.neighbors = neighbors
        self.search_budget = search_budget
        self.tmax = tmax
        self.tfactor = -math.log(tmax / tmin)
        self.current_set = None
        self.step = 0

        self.inferior_loss_take_step = 0
        self.inferior_loss_no_step = 0
        self.superior_loss = 0

    def restart(self):
        self.current_set = None
        self.current_model = None

    def next_fit_request(self):
        if self.current_set is None:
            return self.initial_training_set()
        else:
            candidate = list(self.current_set)
            for i in range(len(candidate)):
                candidates = self.neighbors(candidate[i])
                # include self among neighbors
                # TODO: there is a config option on this which is only respected by Neighbors algorithm
                candidates.append(candidate[i])
                candidate[i] = self.random.choice(candidates)
            return candidate

    def take_step(self, model, loss, set):
        self.current_model = model
        self.current_loss = loss
        self.current_set = set
        self.logger.current_set(current_set=self.current_set,
                                current_loss=self.current_loss)
        self.accept_best(model, loss, set)

    def complete(self):
        self.logger.anneal_summary(inferior_loss_no_step=self.inferior_loss_no_step,
                                   inferior_loss_take_step=self.inferior_loss_take_step,
                                   superior_loss=self.superior_loss)

    def next_fit_result(self, model, loss, set):
        if self.current_set is None:
            self.take_step(model, loss, set)
        else:
            self.step += 1
            t = self.tmax * math.exp(self.tfactor * self.step / self.search_budget)
            loss_difference = loss - self.current_loss
            if loss_difference <= 0.0:
                self.superior_loss += 1
                self.logger.anneal(decision='take_step_better_loss')
                self.take_step(model, loss, set)
            elif math.exp(-loss_difference / t) >= self.random.uniform():
                self.inferior_loss_take_step += 1
                self.logger.anneal(decision='take_step_inferior_loss',
                                   t=t,
                                   loss_difference=loss_difference,
                                   p=math.exp(-loss_difference / t))
                self.take_step(model, loss, set)
            else:
                self.inferior_loss_no_step += 1
                self.logger.anneal(decision='no_step_inferior_loss',
                                   t=t,
                                   loss_difference=loss_difference,
                                   p=math.exp(-loss_difference / t))


class AnnealEpsilon(Anneal):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_budget,
                 initial_training_set,
                 logger,
                 search_budget,
                 neighbors,
                 tmax,
                 tmin,
                 epsilon):
        super(AnnealEpsilon, self).__init__(random,
                                            pool_size,
                                            teaching_budget,
                                            initial_training_set,
                                            logger,
                                            search_budget - int(epsilon * search_budget),
                                            neighbors,
                                            tmax,
                                            tmin)
        self.i = 0
        self.explore_rounds = int(epsilon * search_budget)

    def restart(self):
        # This algorithm pointedly ignores the attention budget
        pass

    def next_fit_request(self):
        if self.i < self.explore_rounds:
            return self.initial_training_set()
        else:
            if self.i == self.explore_rounds:
                self.i += 1
                self.current_model = self.best_model
                self.current_loss = self.best_loss
                self.current_set = self.best_set
            return super(AnnealEpsilon, self).next_fit_request()

    def next_fit_result(self, model, loss, set):
        if self.i < self.explore_rounds:
            self.i += 1
            self.logger.current_set(current_set=set, current_loss=loss)
            self.accept_best(model, loss, set)
        else:
            super(AnnealEpsilon, self).next_fit_result(model, loss, set)


class Neighbors(Algorithm):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_budget,
                 schedule,
                 neighbors,
                 include_current,
                 initial_training_set,
                 logger):
        super(Neighbors, self).__init__(random,
                                        pool_size,
                                        teaching_budget,
                                        initial_training_set,
                                        logger)
        self.schedule = schedule
        self.current_set = None
        self.current_model = None
        self.current_index = 0
        self.schedule_index = 0
        self.models_to_fetch = []
        self.models_fetched = []
        self.neighbors = neighbors
        self.include_current = include_current

    def restart(self):
        self.current_set = None
        self.current_model = None
        self.current_index = 0
        self.schedule_index = 0
        self.models_to_fetch = []
        self.models_fetched = []

    def next_fit_request(self):
        if not self.current_set:
            self.current_set = self.initial_training_set()
            return self.current_set
        else:
            if self.models_to_fetch:
                return self.models_to_fetch.pop()
            else:
                self.models_fetched = []
                self.schedule_index += 1
                if self.schedule_index == self.schedule:
                    self.schedule_index = 0
                    self.current_index += 1
                    if self.current_index == self.teaching_budget:
                        self.current_index = 0
                for n in self.neighbors(self.current_set[self.current_index]):
                    ns = self.current_set[0:self.current_index] + \
                        [n] + \
                        self.current_set[self.current_index+1:]
                    self.models_to_fetch.append(ns)
                return self.models_to_fetch.pop()

    def next_fit_result(self, model, loss, set):
        if not self.current_model:
            self.current_model = model
            self.current_loss = loss
            self.logger.current_set(current_set=self.current_set,
                                    current_loss=self.current_loss)
            for n in self.neighbors(self.current_set[0]):
                ns = [n] + self.current_set[1:]
                self.models_to_fetch.append(ns)
        else:
            self.models_fetched.append((model, loss, set))
            if not self.models_to_fetch:
                if not self.include_current:
                    self.current_loss = np.inf
                    self.current_model = None
                for m, l, s in self.models_fetched:
                    if l < self.current_loss:
                        self.current_model = m
                        self.current_loss = l
                        self.current_set = s
                self.logger.current_set(current_set=self.current_set,
                                        current_loss=self.current_loss)
                self.accept_best(self.current_model,
                                 self.current_loss,
                                 self.current_set)


def loss_factory(learner, evaluation_set, loss_function, args):
    def f(model):
        preds = learner.predict(model, evaluation_set[0])
        return loss_function(evaluation_set[1], preds, **args)
    return f


class Result(object):
    def __init__(self):
        self.best_model = None
        self.best_set = None
        self.best_evaluation_loss = None
        self.current_sets = []
        self.best_sets = []
        self.test_loss = None
        self.fits = []


class ResultCollection(object):
    def __init__(self, results, instance, options):
        self.instance = instance
        self.options = options
        self.results = results
        if results:
            min_loss = min(lambda r: r.best_evaluation_loss, results)
            self.best = filter(lambda r: r.best_evaluation_loss != min_loss,
                               results)[0]
        self.neighbors = None

    def append(self, result):
        self.results.append(result)
        min_loss = min(lambda r: r.best_evaluation_loss, self.results)
        self.best = filter(lambda r: r.best_evaluation_loss != min_loss,
                           self.results)[0]

    def save(self, filename):
        output = open(filename, 'wb')
        pickle.dump(self, output)
        output.close()

    def summary(self):
        return pd.DataFrame(dict(test_loss=self.test_loss_summary(),
                                 evaluation_loss=self.best_evaluation_loss_summary()))

    def plot_current_loss(self):
        index = map(lambda x: x[0], self.results[0].fits)
        df = pd.DataFrame(index=index)
        for r in range(len(self.results)):
            index = map(lambda x: x[0], self.results[r].current_sets)
            vals = map(lambda x: x[2], self.results[r].current_sets)
            df[r] = pd.Series(vals, index=index)
        df = df.fillna(method='ffill')
        return df.plot(title="Loss of current set by model fit")

    def plot_best_loss(self):
        index = map(lambda x: x[0], self.results[0].fits)
        df = pd.DataFrame(index=index)
        for r in range(len(self.results)):
            index = map(lambda x: x[0], self.results[r].best_sets)
            vals = map(lambda x: x[2], self.results[r].best_sets)
            df[r] = pd.Series(vals, index=index)
        df = df.fillna(method='ffill')
        return df.plot(title="Loss of best set by model fit")

    def best_evaluation_loss_summary(self):
        losses = map(lambda r: r.best_evaluation_loss, self.results)
        return pd.Series(losses).describe()

    def test_loss_summary(self):
        losses = map(lambda r: r.test_loss, self.results)
        return pd.Series(losses).describe()

    @staticmethod
    def load(filename):
        import sys
        sys.path.append('./src')
        import search
        return pickle.load(open(filename, 'rb'))


def parse_dict(value):
    if value:
        kvs = value.split(',')
        return dict(map(lambda x: tuple(map(int, x.split('='))), kvs))


def delete_if_exists(options, key):
    if key in options:
        del options[key]


def delete_if_none(options, key):
    if key in options and not options[key]:
        del options[key]


def key_is_not(options, key, value):
    return key not in options or options[key] != value


def clean_options(options):
    options = dict(vars(options))

    delete_if_exists(options, 'parallel')
    delete_if_exists(options, 'couchdb')

    if 'loaded_from_mldata' not in options:
        delete_if_exists(options, 'mldata')
    else:
        delete_if_exists(options, 'loaded_from_mldata')
    if 'loaded_from_csv' not in options:
        delete_if_exists(options, 'csv')
        delete_if_exists(options, 'csv_label')
    else:
        delete_if_exists(options, 'loaded_from_csv')

    if key_is_not(options, 'learner', 'knn'):
        delete_if_exists(options, 'n_neighbors')
    if key_is_not(options, 'learner', 'svm'):
        delete_if_exists(options, 'c')

    if 'pca' not in options or options['pca'] is None:
        delete_if_exists(options, 'pca')

    if key_is_not(options, 'searcher', 'neighbor'):
        delete_if_exists(options, 'cycle')
    if 'searcher' not in options or (options['searcher'] != 'neighbor' and options['searcher'] != 'anneal'):
        delete_if_exists(options, 'neighbors')
        delete_if_exists(options, 'multiple_of_nearest_k')
        delete_if_exists(options, 'multiple_of_nearest_multiple')
        delete_if_exists(options, 'fixed_size')
        delete_if_exists(options, 'fixed_distance')
        delete_if_exists(options, 'neighbors_include_current')
    else:
        if key_is_not(options, 'neighbors', 'multiple-of-nearest'):
            delete_if_exists(options, 'multiple_of_nearest_k')
            delete_if_exists(options, 'multiple_of_nearest_multiple')
        if key_is_not(options, 'neighbors', 'fixed-size'):
            delete_if_exists(options, 'fixed_size')
        if key_is_not(options, 'neighbors', 'fixed-distance'):
            delete_if_exists(options, 'fixed_distance')
    delete_if_exists(options, 'initial_training_set')
    if key_is_not(options, 'searcher', 'anneal'):
        delete_if_exists(options, 'tmin')
        delete_if_exists(options, 'tmax')
        delete_if_exists(options, 'epsilon')
    if key_is_not(options, 'searcher', 'greedy'):
        delete_if_exists(options, 'proposals')

    delete_if_exists(options, 'rs')
    delete_if_exists(options, 'no_progress')
    delete_if_none(options, 'description')
    delete_if_none(options, 'filename')
    delete_if_none(options, 'label_map')
    delete_if_none(options, 'label_filter')

    # detect custom learner?

    # todo figure out exactly who is using the attention budget and
    # who is not and delete the option in the appropriate cases

    return options

def parse_options(args=None,
                  learner="knn",
                  n_neighbors=1,
                  c=1.0,
                  loss="zero_one_loss",
                  mldata=None,
                  csv=None,
                  csv_label=None,
                  pca=None,
                  label_filter=None,
                  label_map=None,
                  num_train=1000,
                  num_validate=1000,
                  num_test=1000,
                  teaching_budget=5,
                  search_budget=10,
                  attention_budget=-1,
                  num_trials=1,
                  seed=0,
                  searcher='neighbor',
                  cycle=1,
                  neighbors='multiple-of-nearest',
                  multiple_of_nearest_k=2,
                  multiple_of_nearest_multiple=3.0,
                  fixed_size=5,
                  fixed_distance=1.0,
                  initial_training_set=[],
                  tmin=0.005,
                  tmax=0.01,
                  epsilon=0.5,
                  proposals=5,
                  parallel=False,
                  couchdb='http://127.0.0.1:5984/',
                  description=None,
                  filename=None):

    parser = ArgumentParser()

    # learner options
    parser.add_argument("--learner", choices=["knn", "svm", "ols"], default=learner)

    # knn options
    parser.add_argument("--n-neighbors", type=int, default=n_neighbors)
    # linear svm options
    parser.add_argument("--c", type=float, default=c)

    # loss options
    parser.add_argument("--loss",
                        choices=["zero_one_loss",
                                 "mean_squared_error",
                                 "mean_absolute_error"],
                        default=loss)

    # data options
    parser.add_argument("--mldata", default=mldata)
    parser.add_argument("--csv", default=csv)
    parser.add_argument("--csv-label", default=csv_label)
    parser.add_argument("--pca", type=int, default=pca)
    parser.add_argument("-l", "--label-filter",
                        action="append",
                        type=int,
                        default=label_filter)
    parser.add_argument("--label-map",
                        type=parse_dict,
                        default=label_map)
    parser.add_argument("--num-train", type=int, default=num_train)
    parser.add_argument("--num-validate", type=int, default=num_validate)
    parser.add_argument("--num-test", type=int, default=num_test)

    # search options
    parser.add_argument("--teaching-budget", type=int, default=teaching_budget)
    parser.add_argument("--search-budget", type=int, default=search_budget)
    parser.add_argument("--attention-budget", type=int, default=attention_budget)
    parser.add_argument("--num-trials", type=int, default=num_trials)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--searcher",
                        choices=['uniform', 'neighbor', 'anneal', 'greedy'],
                        default=searcher)

    # searcher options
    parser.add_argument("--cycle", type=int, default=cycle)
    parser.add_argument("--neighbors",
                        choices=['multiple-of-nearest', 'fixed-size', 'fixed-distance'],
                        default=neighbors)
    parser.add_argument("--multiple-of-nearest_k",
                        type=int,
                        default=multiple_of_nearest_k)
    parser.add_argument("--multiple-of-nearest_multiple",
                        type=float,
                        default=multiple_of_nearest_multiple)
    parser.add_argument("--fixed-size",
                        type=int,
                        default=fixed_size)
    parser.add_argument("--fixed-distance",
                        type=float,
                        default=fixed_distance)
    parser.add_argument("--neighbors-include-current",
                        action="store_true",
                        default=False)
    parser.add_argument("--initial_training_set",
                        action="append",
                        type=int,
                        default=initial_training_set)
    parser.add_argument("--tmin", type=float, default=tmin)
    parser.add_argument("--tmax", type=float, default=tmax)
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--proposals", type=int, default=proposals)

    # persistence options
    parser.add_argument("--description", default=description)
    parser.add_argument("--filename", default=filename)

    # debugging options
    parser.add_argument("--parallel", action="store_true", default=parallel)
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--couchdb", default=couchdb)

    # todo add an option show_log and hide_store for logging events to
    # print out

    options = parser.parse_args(args=args)
    options.rs = np.random.RandomState(seed=options.seed)

    return options


def lazy_multiple_of_nearest_neighbors(i, neighbors, x, options):
    if not neighbors[i]:
        distance = np.transpose(pairwise_distances(x, x[i].reshape(1, -1)))[0]
        max_distance = sorted((set(distance)))[options.multiple_of_nearest_k] * options.multiple_of_nearest_multiple
        neighbors[i] = np.where(distance < max_distance + 1e-06)[0].tolist()
        neighbors[i].remove(i)
    return neighbors[i]


def lazy_fixed_size(i, neighbors, x, options):
    if not neighbors[i]:
        k = options.fixed_size
        distance = np.transpose(pairwise_distances(x, x[i].reshape(1, -1)))[0]
        max_distance = sorted(distance)[k+1]
        candidates = np.where(distance <= max_distance)[0].tolist()
        candidates.remove(i)
        neighbors[i] = options.rs.choice(candidates,
                                         size=k,
                                         replace=False).tolist()
    return neighbors[i]


def lazy_fixed_distance(i, neighbors, x, options):
    if not neighbors[i]:
        max_distance = options.fixed_distance
        distance = np.transpose(pairwise_distances(x, x[i].reshape(1, -1)))[0]
        candidates = np.where(distance <= max_distance)[0].tolist()
        candidates.remove(i)
        neighbors[i] = candidates
    return neighbors[i]


class Runner(object):
    def __init__(self, logger=Logger()):
        self.logger = logger
        self.algorithm = None
        self.loss_factory = loss_factory
        self.decorate_learner = None

    def search_factory(self,
                       algorithm,
                       candidate_pool,
                       learner,
                       loss,
                       search_budget,
                       progress_bar,
                       attention_budget,
                       logger):
        return Search(algorithm,
                      candidate_pool,
                      learner,
                      loss,
                      search_budget,
                      progress_bar,
                      attention_budget,
                      logger)

    def compute_neighbors(self, instance, options, results):
        x = instance[0][0]
        neighbors = [None] * len(instance[0][1])
        results.neighbors = neighbors
        if options.neighbors == 'multiple-of-nearest':
            return lambda n: lazy_multiple_of_nearest_neighbors(n, neighbors, x, options)
        elif options.neighbors == 'fixed-size':
            return lambda n: lazy_fixed_size(n, neighbors, x, options)
        elif options.neighbors == 'fixed-distance':
            return lambda n: lazy_fixed_distance(n, neighbors, x, options)

    def construct_learner(self, options):
        learner_class = {
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'ols': LinearRegression
        }[options.learner]
        init_params = {
            'svm': {'C': options.c, 'kernel': 'linear'},
            'knn': {'n_neighbors': options.n_neighbors, 'weights': 'uniform'},
            'ols': {'copy_X': True, 'fit_intercept': True}
        }[options.learner]
        learner_params = {}

        learner = Sklearn_Learner(init_params,
                                  learner_class,
                                  **learner_params)
        if self.decorate_learner:
            learner = self.decorate_learner(learner)
        return learner

    def construct_algorithm(self, options, neighbors, logger):
        algorithm = self.algorithm
        if algorithm is None:
            if options.searcher == 'uniform':
                algorithm = UniformJumping(options.rs,
                                           options.num_train,
                                           options.teaching_budget,
                                           options.initial_training_set,
                                           logger)
            elif options.searcher == 'neighbor':
                algorithm = Neighbors(options.rs,
                                      options.num_train,
                                      options.teaching_budget,
                                      options.cycle,
                                      neighbors,
                                      options.neighbors_include_current,
                                      options.initial_training_set,
                                      logger)
            elif options.searcher == 'anneal':
                if options.epsilon is None:
                    algorithm = Anneal(options.rs,
                                       options.num_train,
                                       options.teaching_budget,
                                       options.initial_training_set,
                                       logger,
                                       options.search_budget,
                                       neighbors,
                                       options.tmax,
                                       options.tmin)
                else:
                    algorithm = AnnealEpsilon(options.rs,
                                              options.num_train,
                                              options.teaching_budget,
                                              options.initial_training_set,
                                              logger,
                                              options.search_budget,
                                              neighbors,
                                              options.tmax,
                                              options.tmin,
                                              options.epsilon)
            elif options.searcher == 'greedy':
                algorithm = Greedy(options.rs,
                                   options.num_train,
                                   options.teaching_budget,
                                   options.initial_training_set,
                                   logger,
                                   options.proposals)
            else:
                msg = 'Algorithm %s not recognized' % (options.searcher)
                raise Exception(msg)
        return algorithm

    def construct_loss(self, instance, options, learner):
        loss_function = {"zero_one_loss": zero_one_loss,
                         "mean_squared_error": mean_squared_error,
                         "mean_absolute_error": mean_absolute_error}[options.loss]
        loss_function_args = {"zero_one_loss": {'normalize': True},
                              "mean_squared_error": {},
                              "mean_absolute_error": {}}[options.loss]
        evaluation_loss = self.loss_factory(learner,
                                            instance[1],
                                            loss_function,
                                            loss_function_args)
        test_loss = self.loss_factory(learner,
                                      instance[2],
                                      loss_function,
                                      loss_function_args)
        return evaluation_loss, test_loss

    def run_experiment(self, instance, options):
        # Once instance is passed in, assume these values are correct
        options.num_train = len(instance[0][1])
        options.num_validate = len(instance[1][1])
        options.num_test = len(instance[2][1])

        # Because we don't want order we call searchers to matter
        options.rs = np.random.RandomState(seed=options.seed)
        self.logger = self.logger.experiment(instance)
        self.logger.options(**clean_options(options))

        results = ResultCollection([], instance, clean_options(options))
        if options.searcher == 'neighbor' or options.searcher == 'anneal':
            neighbors = self.compute_neighbors(instance, options, results)
        else:
            neighbors = None
        for trial in range(options.num_trials):
            logger = self.logger.bind(trial=trial)
            algorithm = self.construct_algorithm(options, neighbors, logger)
            algorithm.validate(options.search_budget)
            learner = self.construct_learner(options)
            evaluation_loss, test_loss = self.construct_loss(instance,
                                                             options,
                                                             learner)
            searcher = self.search_factory(algorithm,
                                           instance[0],
                                           learner,
                                           evaluation_loss,
                                           options.search_budget,
                                           not options.no_progress,
                                           options.attention_budget,
                                           logger)
            result = searcher.search()
            if test_loss:
                result.test_loss = test_loss(result.best_model)
                logger.test_loss(test_loss=result.test_loss)
            results.append(result)
            logger.flush()
        if neighbors:
            self.logger.neighbors(neighbors=results.neighbors)
        # As long as nothing gets logged outside of for loop,
        # logger.bind() above is fine
        self.logger.finish()
        if options.filename:
            results.save(options.filename)
        return results


def load_data(options, X0=None, Y0=None):
    def replace_labels(labels):
        return np.array([options.label_map[y]
                         if y in options.label_map else y
                         for y in labels], dtype=float)

    if X0 is None or Y0 is None:
        X0, Y0 = load_all(options)
    if options.loss == "zero_one_loss":
        # Assume classification
        labels = options.label_filter
        if not labels:
            labels = list(set(Y0.tolist()))

        candidates = load_multilabel_items(X0,
                                           Y0,
                                           options,
                                           labels,
                                           options.num_train)
        evaluation = load_multilabel_items(X0,
                                           Y0,
                                           options,
                                           labels,
                                           options.num_validate)
        testing = load_multilabel_items(X0,
                                        Y0,
                                        options,
                                        labels,
                                        options.num_test)
        if options.label_map:
            candidates = (candidates[0], replace_labels(candidates[1]))
            evaluation = (evaluation[0], replace_labels(evaluation[1]))
            testing = (testing[0], replace_labels(testing[1]))
    else:
        # Assume regression
        n = options.num_train + options.num_validate + options.num_test
        inds = options.rs.choice(len(Y0), n, replace=False)
        i = options.num_train
        j = options.num_train + options.num_validate
        candidates = X0[inds[:i]], Y0[inds[:i]]
        evaluation = X0[inds[i:j]], Y0[inds[i:j]]
        testing = X0[inds[j:]], Y0[inds[j:]]

    instance = (candidates, evaluation, testing)
    if options.pca is not None:
        n_components = None if options.pca == 0 else options.pca
        # This is suspect: better to just use eval or fresh set
        train_and_eval = [instance[0][0], instance[1][0]]
        if scipy.sparse.issparse(instance[0][0]):
            pca = TruncatedSVD(n_components=n_components)
            train_and_eval = scipy.sparse.vstack(train_and_eval)
        else:
            pca = PCA(n_components=n_components,
                      copy=True)
            train_and_eval = np.vstack(train_and_eval)

        pca.fit(train_and_eval)
        instance = ((pca.transform(instance[0][0]),
                     instance[0][1]),
                    (pca.transform(instance[1][0]),
                     instance[1][1]),
                    (pca.transform(instance[2][0]),
                     instance[2][1]))
    elif scipy.sparse.issparse(instance[0][0]):
        print "Error: --pca must be set for sparse matrices"
        sys.exit(1)

    return instance


def load_multilabel_items(X0, Y0, options, labels, n, shuffle=True):
    # Because we are dividing up n items among len(labels)
    assert n % len(labels) == 0, "Number of items must be integer multiple of number of labels"
    Xs = []
    Ys = []
    for l in labels:
        X, Y = load_items(X0, Y0, options, l, n // len(labels))
        Xs.append(X)
        Ys.append(Y)
    Y = np.hstack(Ys)
    if scipy.sparse.issparse(Xs[0]):
        X = scipy.sparse.vstack(Xs)
    else:
        X = np.vstack(Xs)
    if shuffle:
        inds = range(X.shape[0])
        options.rs.shuffle(inds)
        inds = np.array(inds)
        return X[inds], Y[inds]
    else:
        return X, Y


def load_items(X0, Y0, options, label, n):
    inds = np.array([i for i, a in enumerate(Y0 == label) if a])
    inds = options.rs.choice(inds, n, replace=False)
    return X0[inds], Y0[inds]


def load_all(options):
    if options.mldata is None and (options.csv is None or options.csv_label is None):
        print "--mldata or (--csv and --csv-label) required"
        sys.exit(1)
    if options.mldata and options.csv:
        print "Both --mldata and --csv set. Please choose one."
        sys.exit(1)

    def slugify(value):
        value = re.sub('[^\w\s-]', '', value).strip().lower()
        value = re.sub('[-\s]+', '-', value)
        return value

    if options.mldata:
        options.loaded_from_mldata = True
        dataset = fetch_mldata(
            options.mldata,
            data_home="./samples/" + slugify(options.mldata))
        return (dataset.data, dataset.target)

    if options.csv:
        options.loaded_from_csv = True
        import urllib2
        url = re.compile(
            r'^(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        stream = urllib2.urlopen(options.csv) if url.match(options.csv) else open(options.csv, 'rb')
        df = pd.read_csv(stream, header=None)
        df.columns = df.columns.astype(str)
        x = df.drop(options.csv_label, 1)
        x = pd.get_dummies(x).values
        y = df[options.csv_label].values
        return (x, y)

if __name__ == "__main__":
    logger = Logger()
    logger.store_instance = False
    logger.show_log_options
    logger.show_log_best_set
    options = parse_options()
    instance = load_data(options)
    res = Runner(logger=logger).run_experiment(instance, options)
