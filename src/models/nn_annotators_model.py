import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import KERNEL_PARAMS
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.exceptions import NotFittedError

from src.models.parzen_window_classifier import PWC
from src.utils.mathematical_functions import rand_arg_max, compute_vote_vectors


class NNAnnotatorsModel(BaseEstimator):
    """NNAnnotatorModel

    The Nearest-Neighbor Annotators Model [1] estimates the annotation performances, i.e. label accuracies, of multiple
    annotators per sample. Given several samples and corresponding label vectors of these annotators, the majority vote
    per sample-label-vector-pair is computed. To estimate an annotator's label accuracies for a given sample,
    its k nearest samples labeled by the annotator are used in combination with the majority vote labels.

    Parameters
    ----------
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs: dict,
        Any further parameters are passed directly to the metric/kernel function.

    Attributes
    ----------
    metric_: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors_: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs_: dict,
        Any further parameters are passed directly to the kernel function.
    pwc_list_: array-like, shape (n_annotators)
        For each annotator one fitted Parzen Window Classifier [2] used to estimate the annotation performance.

    References
    ----------
    [1] Huang, S. J., Chen, J. L., Mu, X., & Zhou, Z. H. (2017). Cost-effective Active Learning from Diverse Labelers.
        Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), 1879–1885.
        Melbourne, Australia.
    [2] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, n_classes, n_neighbors=5, metric='rbf', random_state=None, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 1:
            raise ValueError("'n_classes' must be an integer greater than one")

        self.n_neighbors_ = int(n_neighbors)
        if self.n_neighbors_ < 0:
            raise ValueError("'n_neighbors' must be a positive integer")

        self.metric_ = metric
        if self.metric_ not in KERNEL_PARAMS.keys() and self.metric_ != PWC.PRECOMPUTED:
            raise ValueError("'metric' must be in {}".format(KERNEL_PARAMS.keys()))

        self.random_state_ = check_random_state(random_state)

        self.kwargs_ = kwargs
        self.pwc_list_ = None

    def fit(self, X, y, c=None):
        """
        Given the labels of multiple annotators, this method fits annotator models to estimate annotation performances,
        i.e. label accuracies, of these multiple annotators.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples, n_annotators)
            Labels provided by multiple annotators. An entry y[i, j] indicates that the annotator with index j has not
            provided a label for the sample with index i.
        c: array-like, shape (n_samples, n_annotators)
            Weights for the individual labels.
            Default is c[i, j]=1 as weight for the label entry y[i, j].
        """
        # check input parameters
        X = check_array(X)
        y = check_array(y, force_all_finite=False)
        check_consistent_length(X, y)

        # determine number of annotators
        n_annotators = np.size(y, axis=1)

        # flag for labeled entries
        is_labeled = ~np.isnan(y)

        # compute (confidence weighted majority) vote
        V = compute_vote_vectors(y=y, c=c, n_unique_votes=self.n_classes_)
        y_mv = rand_arg_max(arr=V, axis=1, random_state=self.random_state_)

        # fit PWC per annotator
        self.pwc_list_ = []
        for a_idx in range(n_annotators):
            is_correct = np.array(np.equal(y_mv[is_labeled[:, a_idx]], y[is_labeled[:, a_idx], a_idx]), dtype=int)
            pwc = PWC(n_classes=2, metric=self.metric_, n_neighbors=self.n_neighbors_, random_state=self.random_state_,
                      **self.kwargs_)
            self.pwc_list_.append(pwc.fit(X[is_labeled[:, a_idx]], is_correct))

        return self

    def predict(self, X):
        """
        This method estimates the annotation performances, i.e. label accuracies, of the multiple annotators for each
        given sample in X.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.

        Returns
        -------
        Y: matrix-like, shape (n_samples, n_annotators)
            Estimate label accuracy for each sample-annotator-pair.
        """
        if self.pwc_list_ is None:
            raise NotFittedError("This NNAnnotatorsModel instance is not fitted yet. Call 'fit' with appropriate "
                                 "arguments before using this estimator.")
        n_annotators = len(self.pwc_list_)
        Y = np.column_stack([self.pwc_list_[a_idx].predict_proba(X, normalize=False)[:, 1]
                             for a_idx in range(n_annotators)])
        return Y
