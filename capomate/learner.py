import numpy as np


class Learner(object):
    """This base class defines a learner.
    Note that this is a bit funky, as you'll only have one copy of a learner.
    It defines globalesque functions to update models, and predict given a model.
    """
    def __init__(self, init_params, pos_class=1, neg_class=-1):
        self.pos_class = pos_class
        self.neg_class = neg_class
        # Record keeping:
        self.num_updates = 0
        self.num_preds = 0

    def update(self, model, data, new_point):
        """
        This function returns a new model, updated given the dataset so far, and the new point.
        """
        self.num_updates += 1

    def predict_point(self, model, query):
        """
        This function returns a prediction of query, given model.
        """
        self.num_preds += 1

    def predict(self, model, X):
        """
        This function returns predicts for a full matrix X, rather than a single query point.
        Note: for efficiency, this is overridden by sklearn based learners.
        """
        self.num_preds += X.shape[0]
        #return np.array([self.predict_point(model, x) for x in X])

    def should_consider(self, model, query):
        """
        Given a query and model, this function return True if the query point should be considered for exploring.
        For example, perceptrons do not consider points (for updating purposes) unless they predict them wrong.
        """
        return True  # By default, consider every point

    def fit(self, model, data):
        """
        Given a model and (full) dataset = (X, Y), this function returns a model which has been trained on data.
        Note that this is similar to update, but is useful for initializing models and makes more semantic sense
        when the learner is a batch learner.
        """
        self.num_updates += 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Type: " + str(type(self))


class Sklearn_Learner(Learner):
    """
    This learner is a wrapper around classifiers from scikit-learn.
    Given a class classify_class, the learner is:
    classify_class(**init_params)
    That is, an instance of the classifier provided, initialized with the specified initialization parameters.
    # TODO: provide example
    """
    def __init__(self, init_params, classify_class, pos_class=1, neg_class=-1):
        super(Sklearn_Learner, self).__init__(init_params, pos_class, neg_class)
        self.init_model = None
        self.init_params = init_params
        self.classify_class = classify_class

    def update(self, model, data, new_point):
        super(Sklearn_Learner, self).update(model, data, new_point)
        toR = self.classify_class(**self.init_params)

        X, Y = data
        if new_point is not None:
            X = np.vstack([X, new_point[0]])
            Y = np.concatenate([Y, [new_point[1]]])
        if len(Y) == 0:
            return None
        elif np.unique(Y).shape[0] <= 1:
            if Y[0] == self.pos_class:
                return "allpos"
            else:
                return "allneg"

        toR.fit(X, Y)
        return toR

    def predict_point(self, model, query):
        super(Sklearn_Learner, self).predict_point(model, query)
        if model is None:
            return 0
        elif model == "allpos":
            return self.pos_class
        elif model == "allneg":
            return self.neg_class
        else:
            query = query.reshape(1, -1)  # needed in sklearn 0.18 to avoid endless warnings
            return model.predict(query)

    def predict(self, model, X):
        super(Sklearn_Learner, self).predict(model, X)
        if model is None:
            return np.array([0 for _ in range(X.shape[0])])
        elif model == "allpos":
            return np.array([self.pos_class for _ in range(X.shape[0])])
        elif model == "allneg":
            return np.array([self.neg_class for _ in range(X.shape[0])])
        #self.num_preds += 1
        return model.predict(X)

    def fit(self, model, data):
        super(Sklearn_Learner, self).fit(model, data)
        X, Y = data
        toR = self.classify_class(**self.init_params)
        if len(Y) == 0:
            return None
        elif np.unique(Y).shape[0] <= 1:
            if Y[0] == self.pos_class:
                return "allpos"
            else:
                return "allneg"

        toR.fit(X, Y)
        return toR

class Sklearn_Trained_Learner(Learner):
    """
    This learner is a wrapper around classifiers from scikit-learn.
    However, it takes a fixed training set. Whenever asked to fit a set S, this learner fits it's fixed training set union S.
    Given a class classify_class, the learner is:
    classify_class(**init_params)
    That is, an instance of the classifier provided, initialized with the specified initialization parameters.
    # TODO: provide example
    """
    def __init__(self, init_params, classify_class, fixed_training_set,  pos_class=1, neg_class=-1):
        super(Sklearn_Trained_Learner, self).__init__(init_params, pos_class, neg_class)
        self.init_model = None
        self.init_params = init_params
        self.classify_class = classify_class
        self.fixed_training_set = fixed_training_set

    def update(self, model, data, new_point):
        raise NotImplemented 

    def predict_point(self, model, query):
        super(Sklearn_Trained_Learner, self).predict_point(model, query)
        if model is None:
            return 0
        elif model == "allpos":
            return self.pos_class
        elif model == "allneg":
            return self.neg_class
        else:
            query = query.reshape(1, -1)  # needed in sklearn 0.18 to avoid endless warnings
            return model.predict(query)

    def predict(self, model, X):
        super(Sklearn_Trained_Learner, self).predict(model, X)
        if model is None:
            return np.array([0 for _ in range(X.shape[0])])
        elif model == "allpos":
            return np.array([self.pos_class for _ in range(X.shape[0])])
        elif model == "allneg":
            return np.array([self.neg_class for _ in range(X.shape[0])])
        #self.num_preds += 1
        return model.predict(X)

    def fit(self, model, data):
        super(Sklearn_Trained_Learner, self).fit(model, data)
        X, Y = data
        X = np.vstack([self.fixed_training_set[0], X])
        Y = np.concatenate([self.fixed_training_set[1], Y])
        toR = self.classify_class(**self.init_params)
        if len(Y) == 0:
            return None
        elif np.unique(Y).shape[0] <= 1:
            if Y[0] == self.pos_class:
                return "allpos"
            else:
                return "allneg"
        toR.fit(X, Y)
        return toR

class Sklearn_Soft_Learner(Learner):
    """
    This learner is a wrapper around classifiers from scikit-learn.
    Unlike Sklearn_Learner, this will return "soft" preditions, using the sklearner's predict_proba instead predict.
    Note: only works for classifiers that support a predict_proba.
    Given a class classify_class, the learner is:
    classify_class(**init_params)
    That is, an instance of the classifier provided, initialized with the specified initialization parameters.
    # TODO: provide example
    """
    def __init__(self, init_params, classify_class, pos_class=1, neg_class=-1):
        super(Sklearn_Soft_Learner, self).__init__(init_params, pos_class, neg_class)
        self.init_model = None
        self.init_params = init_params
        self.classify_class = classify_class

    def update(self, model, data, new_point):
        super(Sklearn_Soft_Learner, self).update(model, data, new_point)
        toR = self.classify_class(**self.init_params)

        X, Y = data
        if new_point is not None:
            X = np.vstack([X, new_point[0]])
            Y = np.concatenate([Y, [new_point[1]]])
        if len(Y) == 0:
            return None
        elif np.unique(Y).shape[0] <= 1:
            if Y[0] == self.pos_class:
                return "allpos"
            else:
                return "allneg"

        toR.fit(X, Y)
        return toR

    def predict_point(self, model, query):
        super(Sklearn_Soft_Learner, self).predict_point(model, query)
        if model is None:
            return 0
        elif model == "allpos":
            return self.pos_class
        elif model == "allneg":
            return self.neg_class
        else:
            query = query.reshape(1, -1)  # needed in sklearn 0.18 to avoid endless warnings
            probs =  model.predict_proba(query)
            return probs[0]# * self.pos_class + probs[1] * self.neg_class #the probability that is is the first class
        
    def predict(self, model, X):
        super(Sklearn_Soft_Learner, self).predict(model, X)
        if model is None:
            return np.array([0 for _ in range(X.shape[0])])
        elif model == "allpos":
            return np.array([self.pos_class for _ in range(X.shape[0])])
        elif model == "allneg":
            return np.array([self.neg_class for _ in range(X.shape[0])])
        #self.num_preds += 1
        probs = model.predict_proba(X)

        return probs[:, 0]# * self.pos_class + probs[:, 1] * self.neg_class
    
    def fit(self, model, data):
        super(Sklearn_Soft_Learner, self).fit(model, data)
        X, Y = data
        toR = self.classify_class(**self.init_params)
        if len(Y) == 0:
            return None
        elif np.unique(Y).shape[0] <= 1:
            if Y[0] == self.pos_class:
                return "allpos"
            else:
                return "allneg"

        toR.fit(X, Y)
        return toR
