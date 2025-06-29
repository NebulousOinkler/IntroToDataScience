import numpy as np


def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string

    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1 * m :]
        v[hash(featurestring) % B] = 1
    return v


def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, "r") as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split("\n")
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i, :] = hashfeatures(babynames[i], B, FIX)
    return X


def genTrainFeatures(dimension=128):
    """
    Input:
        dimension: desired dimension of the features
    Output:
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """

    # Load in the data
    Xgirls = name2features("girls.train", B=dimension)
    Xboys = name2features("boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])

    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])

    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])

    return X[ii, :], Y[ii]


def naivebayesPY(X, Y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1, 1]])
    unique_labels = sorted(np.unique(Y), reverse=True)
    return [np.mean(Y == label) for label in unique_labels]


def naivebayesPXY(X, Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]

    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)

    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2, d)), np.zeros((2, d))])
    Y = np.concatenate([Y, [-1, 1, -1, 1]])
    unique_labels = sorted(np.unique(Y), reverse=True)
    data_by_label = [X[np.where(Y == c), :] for c in unique_labels]
    return [np.mean(data, axis=1).reshape((d,)) for data in data_by_label]


def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test

    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)

    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    log_pos = np.log(posprob)
    log_pos_comp = np.log(1 - posprob)
    log_neg = np.log(negprob)
    log_neg_comp = np.log(1 - negprob)

    for i in range(n):
        if Y_test[i] == 1:
            loglikelihood[i] = np.sum(
                X_test[i] * log_pos + (1 - X_test[i]) * log_pos_comp
            )
        if Y_test[i] == -1:
            loglikelihood[i] = np.sum(
                X_test[i] * log_neg + (1 - X_test[i]) * log_neg_comp
            )

    return loglikelihood


def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test

    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)

    Output:
        prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    pred = (
        loglikelihood(posprob, negprob, X_test, np.ones(n))
        - loglikelihood(posprob, negprob, X_test, -np.ones(n))
        + np.log(pos)
        - np.log(neg)
    )
    return np.array([1 if p > 0 else -1 for p in pred])
