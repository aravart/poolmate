from argparse import ArgumentParser
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import re
import sys
import scipy
import string
import StringIO

class Algorithm(object):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_set_size,
                 initial_training_set):
        self.random = random
        self.pool_size = pool_size
        self.teaching_set_size = teaching_set_size
        self.best_loss = np.inf
        self.best_model = None
        self.best_set = None
        self.result = None
        self._initial_training_set = initial_training_set
        self.calls_to_initial_training_set = 0

    def validate(self, search_budget):
        pass

    def next_fit_request(self):
        pass

    def next_fit_result(self, model, loss, set):
        pass

    def accept_best(self, model, loss, set):
        if loss < self.best_loss:
            self.best_model = model
            self.best_loss = loss
            self.best_set = set

    def initial_training_set(self):
        if self.calls_to_initial_training_set == 0 and self._initial_training_set:
            self.calls_to_initial_training_set += 1
            return self._initial_training_set
        else:
            self.calls_to_initial_training_set += 1
            return [self.random.randint(0, self.pool_size) for _ in range(self.teaching_set_size)]


class RandomIndexGreedySwap(Algorithm):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_set_size,
                 initial_training_set,
                 search_budget,
                 proposals):
        super(RandomIndexGreedySwap, self).__init__(random,
                                                    pool_size,
                                                    teaching_set_size,
                                                    initial_training_set)
        self.search_budget = search_budget
        self.proposals = proposals or pool_size
        self.current_set = None
        self.current_model = None
        self.current_loss = None
        self.models_to_fetch = []
        self.models_fetched = []
        self.step = 0

    def fill_models_to_fetch(self, base_set):
        idx = self.random.randint(0, self.teaching_set_size)
        if self.step + self.proposals > self.search_budget:
            rng = self.random.choice(self.pool_size,
                                     size=self.search_budget - self.step,
                                     replace=False).tolist()
        elif self.proposals < self.pool_size:
            rng = self.random.choice(self.pool_size,
                                     size=self.proposals,
                                     replace=False).tolist()
        else:
            rng = range(self.pool_size)
        for n in rng:
            ns = base_set[0:idx] + [n] + base_set[idx+1:]
            self.models_to_fetch.append(ns)

    def next_fit_request(self):
        if not self.current_set:
            self.fill_models_to_fetch(self.initial_training_set())
        elif not self.models_to_fetch:
            self.models_fetched = []
            self.fill_models_to_fetch(self.current_set)
        return self.models_to_fetch.pop()

    def next_fit_result(self, model, loss, set):
        self.step += 1
        if not self.current_loss:
            self.current_set = set
            self.current_model = model
            self.current_loss = loss
        self.models_fetched.append((model, loss, set))
        if not self.models_to_fetch:
            for m, l, s in self.models_fetched:
                if l < self.current_loss:
                    self.current_model = m
                    self.current_loss = l
                    self.current_set = s
            self.accept_best(self.current_model,
                             self.current_loss,
                             self.current_set)


class UniformSampling(Algorithm):
    def next_fit_request(self):
        return self.initial_training_set()

    def next_fit_result(self, model, loss, set):
        self.accept_best(model, loss, set)


class GreedyAdd(Algorithm):
    def __init__(self,
                 random,
                 pool_size,
                 teaching_set_size,
                 initial_training_set,
                 proposals,
                 search_budget):
        super(GreedyAdd, self).__init__(random,
                                        pool_size,
                                        teaching_set_size,
                                        initial_training_set)
        self.proposals = proposals or min(pool_size,
                                          search_budget / teaching_set_size)
        # TODO could also add a second flag for # of teaching sets you want to
        # produce which then sets proposals appropriately
        self.current_set = []
        self.models_to_fetch = []
        self.models_fetched = []

    def validate(self, search_budget):
        if self.proposals * self.teaching_set_size > search_budget:
            msg = 'Parameters will not produce a teaching set: ' + \
                  'proposals * teaching budget > search budget'
            raise Exception(msg)

    def next_fit_request(self):
        if self.models_to_fetch:
            return self.models_to_fetch.pop()
        else:
            self.models_fetched = []
            if len(self.current_set) == self.teaching_set_size:
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
            # to return any sized set remove following condition
            if len(self.current_set) == self.teaching_set_size:
                self.accept_best(self.current_model,
                                 self.current_loss,
                                 self.current_set)

class Result(object):
    def __init__(self):
        self.best_model = None
        self.best_set = None
        self.best_evaluation_loss = None
        self.current_sets = []
        self.best_sets = []
        self.test_loss = None
        self.fits = []


class Runner(object):
    def construct_algorithm(self, options):
        if options.algorithm == 'greedy-add':
            algorithm = GreedyAdd(options.rs,
                                  options.num_train,
                                  options.teaching_set_size,
                                  options.initial_training_set,
                                  options.proposals,
                                  options.search_budget)
        elif options.algorithm == 'random-index-greedy-swap':
            algorithm = RandomIndexGreedySwap(options.rs,
                                              options.num_train,
                                              options.teaching_set_size,
                                              options.initial_training_set,
                                              options.search_budget,
                                              options.proposals)
        elif options.algorithm == 'uniform':
            algorithm = UniformSampling(options.rs,
                                        options.num_train,
                                        options.teaching_set_size,
                                        options.initial_training_set)
        else:
            msg = 'Algorithm %s not recognized' % (options.algorithm)
            raise Exception(msg)
        return algorithm

    def run_experiment(self, instance, learner, options):
        options.num_train = len(instance)
        options.rs = np.random.RandomState(seed=options.seed)

        algorithm = self.construct_algorithm(options)
        algorithm.validate(options.search_budget)

        if options.log and type(options.log) == str:
            log = open(options.log, 'w')
        else:
            log = options.log
        rng = range(options.search_budget) if options.no_progress else tqdm.trange(options.search_budget)
        for i in rng:
            s = algorithm.next_fit_request()
            m = learner.fit([instance[x] for x in s])
            l = learner.loss(m)
            if log:
                log.write("%d, %f, %s\n" % (i,l,string.join(map(str,s),' ')))
            algorithm.next_fit_result(m, l, s)
        if log and not isinstance(log, StringIO.StringIO):
            log.close()

        return (algorithm.best_loss, algorithm.best_set)
