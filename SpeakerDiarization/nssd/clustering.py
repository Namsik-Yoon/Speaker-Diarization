"""Clustering modules."""
from functools import partial

from scipy.cluster.hierarchy import linkage, fcluster

def _setup_clusterer(config):
    """Factory method to setup clusterer.

    Args:
        config: dictionary, hparams for initializing clusterer.

    Returns:
        Function that requests clustering.
    """
    def _ahc(embeddings, config):
        # initialize hparams related with ahc.
        num_cluster = config['num_cluster']
        threshold = config['threshold']
        ahc_metric = config['ahc_metric']
        ahc_method = config['ahc_method']
        ahc_criterion = config['ahc_criterion']

        if ahc_metric == "euclidean": # default method = 'single'
            _linkage = linkage(embeddings, metric=ahc_metric, method=ahc_method)
        else: # default method = 'complete'
            _linkage = linkage(embeddings, metric=ahc_metric, method=ahc_method)

        if num_cluster != "None": # default criterion = 'maxclust'
            cluster_labels = fcluster(_linkage, float(num_cluster),
                                      criterion=ahc_criterion)
        else: # default criterion = 'distance'
            cluster_labels = fcluster(_linkage, threshold,
                                      criterion=ahc_criterion)
        return cluster_labels

    method = config['method']

    if method == "ahc":
        return partial(_ahc, config=config)

    raise ValueError("Unsupported clustering method : %s" % method)

class Clusterer():
    """Wrapper class for clustering module."""
    def __init__(self, config):
        self.config = config
        self.clusterer = _setup_clusterer(config)

    def predict(self, embeddings):
        """Predict speaker label.

        Args:
            embeddings: list of embeddings will be clustered.

        Returns:
            labels: ndarray of shape (n_samples,).
                    Index of cluster each sample belong to.
        """
        return self.clusterer(embeddings)

    def update_num_cluster(self, n_clusters):
        """Update num_cluster.

        If given n_clusters is different from one that clusterer has,
        update clusterer.

        Args:
            n_clusters: integer, the number of clusters.
        """
        if self.config['num_cluster'] != n_clusters:
            self.config['num_cluster'] = n_clusters
            self.clusterer = _setup_clusterer(self.config)
