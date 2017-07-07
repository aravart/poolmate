import pickle
import datetime
import time
from uuid import uuid4


class Logger(object):
    def __init__(self,
                 bound_args={},
                 whitelist_log=[],
                 blacklist_store=[],
                 store_instance=True,
                 experiment=None,
                 show_all=False,
                 cache={},
                 f=None):
        self.bound_args = bound_args
        self.whitelist_log = whitelist_log
        self.blacklist_store = blacklist_store
        self.store_instance = store_instance
        self.exp = experiment
        self.show_all = show_all
        self.cache = cache
        self.f = f

    def log(self, event, **args):
        if event not in self.blacklist_store:
            if event not in self.cache:
                self.cache[event] = []
            args.update(self.bound_args)
            self.cache[event].append(args)
            if self.f:
                args['event'] = event
                self.f.write(str(args))
                self.f.write('\n')
                self.f.flush()
        if event in self.whitelist_log or self.show_all:
            print event, args

    def bind(self, **args):
        """ Returns copy of logger. Subclasses must override """
        args.update(self.bound_args)
        return Logger(args,
                      self.whitelist_log,
                      self.blacklist_store,
                      self.store_instance,
                      self.exp,
                      self.show_all,
                      self.cache,
                      self.f)

    def create_or_get_table(self, table):
        """ Returns reference to table """
        pass

    def insert_multiple(self, table, docs):
        pass

    def insert(self, table, doc):
        """ Returns identifier """
        pass

    def update(self, table, id, doc):
        pass

    def storage_flush(self):
        pass

    def experiment(self, instance=None, **options):
        id = uuid4().hex
        exp = dict(options)
        exp.update({'experiment': id})
        exp['elapsed'] = time.time()

        if instance and self.store_instance:
            r = {'instance': pickle.dumps(instance), 'experiment': id}
            self.insert(self.create_or_get_table('instances'), r)
        new_logger = self.bind(experiment=id)
        new_logger.exp = exp
        return new_logger

    def flush(self):
        for event in self.cache:
            self.insert_multiple(self.create_or_get_table(event),
                                 self.cache[event])
        self.cache.clear()
        self.storage_flush()

    def finish(self):
        if not self.exp:
            raise Exception("No experiment defined")
        t = self.create_or_get_table('experiments')
        self.exp['elapsed'] = time.time() - self.exp['elapsed']
        self.insert(t, self.exp)
        self.flush()

    def suppress(self, event, log=True, store=True):
        if log:
            self.do_not_log.append(event)
        if store:
            self.do_not_store.append(event)

    def __getattr__(self, name):
        if name[:11] == 'hide_store_':
            self.blacklist_store.append(name[11:])
            return
        if name[:9] == 'show_log_':
            self.whitelist_log.append(name[9:])
            return

        def method(**args):
            self.log(name, **args)
        return method
