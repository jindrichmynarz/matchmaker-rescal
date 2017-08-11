"""
Metrics for evaluation of the predictions of RESCAL-based matchmakers.
"""
import numpy as np

def rank_predictions(top_predictions, fold_indices):
    """
    Rank top predictions given ground truth.

    :param top_predictions: nxk matrix of top-k predictions for given n.
    :param fold_indices: Tuple matching contracts indices to winner indices by order.
    :returns: List of arrays with the ranks of the winnings bidders.
    """
    return [np.argwhere(predictions == winner)[:1]
            for predictions, winner in zip(top_predictions, fold_indices[1])]

def hit_rate(ranks):
    """
    Compute the ratio of cases when the winner is found in top k predictions.
    
    :param ranks: List of arrays with the ranks of the winning bidders.
    :returns: Hit rate from the interval [0, 1].
    """
    return np.concatenate(ranks).size / len(ranks)

def mean_reciprocal_rank(ranks):
    """
    Compute mean reciprocal rank of the winners in top k predictions.
    When the bidder is not found, its reciprocal rank is 0.

    :param ranks: List of arrays with the ranks of the winning bidders.
    :returns: Mean reciprocal rank from the interval [0, 1].
    """
    return np.mean([1/rank[0][0] if rank else 0 for rank in ranks])

def catalog_coverage(top_predictions, bidder_indices):
    """
    Compute the share of distinct predicted bidders out of all bidders.

    :param top_predictions: nxk matrix of top-k predictions for given n.
    :param bidder_indices: Array of indices of all bidders.
    :returns: Catalog coverage from the interval [0, 1].
    """
    return np.intersect1d(np.unique(top_predictions),
                          bidder_indices,
                          assume_unique = True).size / len(bidder_indices)

def prediction_coverage(top_predictions, bidder_indices):
    """
    Compute the share of contracts for which a bidder is recommended.

    :param top_predictions: nxk matrix of top-k predictions for given n.
    :param bidder_indices: Array of indices of all bidders.
    :returns: Prediction coverage from the interval [0, 1].
    """
    return np.sum(np.any(np.isin(top_predictions, bidder_indices), axis = 1)) / top_predictions.shape[0]

def long_tail_bidders(ground_truth, bidder_indices, long_tail = 0.8):
    """
    Get indices of long-tail bidders from `bidder_indices`.
    Given contract awards in `ground_truth` it sorts bidders by their counts of awarded contracts
    and returns the least successful bidders awarded with `long_tail` of contracts.

    :param ground_truth: Square matrix of contract awards.
    :param bidder_indices: Array of indices of bidders.
    :param long_tail: Ratio of contracts awards that forms the long tail.
    :returns: Array of indices of bidders forming the long tail.
    """
    awards = ground_truth.T[bidder_indices].getnnz(axis = 1)
    short_head = np.sum(awards) * (1 - long_tail)
    sort_by_awards = awards.argsort()[::-1] # Sort award counts in the descending order
    long_tail_indices = np.where(np.cumsum(awards[sort_by_awards]) > short_head)
    return bidder_indices[sort_by_awards][long_tail_indices]

def long_tail_percentage(top_predictions, long_tail_bidders):
    """
    Compute the share of predicted bidders that are from the long tail.

    :param top_predictions: nxk matrix of top-k predictions for given n.
    :param long_tail_bidders: Array of indices of bidders forming the long tail.
    :returns: Long tail percentage from the interval [0, 1].
    """
    return np.sum(np.isin(top_predictions, long_tail_bidders)) / np.count_nonzero(top_predictions)
