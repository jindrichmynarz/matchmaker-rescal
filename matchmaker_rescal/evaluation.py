"""
Runner for n-fold cross validation of the RESCAL-based matchmakers.
"""
import edn_format
import logging
import metrics
import numpy as np
import uuid
from functools import partial
from rescal import rescal_als
from scipy.sparse import coo_matrix

_log = logging.getLogger("matchmaker-rescal.evaluation")
_log.setLevel(logging.INFO)

def fold_to_indices(fold):
    """
    Helper function that transposes 2 × n matrix fold into two arrays in the (n, n) tuple,
    which is used for indexing.

    :param fold: 2 × n matrix with the indices of ground truth in the given fold.
    :returns: Tuple of indices.
    """
    transposed = fold.T
    return (transposed[0], transposed[1])

def sorted_indices(matrix):
    """
    Get non-zero indices of components in a matrix, sorted in ascending order.

    :param matrix: matrix to sort indices in
    :returns: Tuple of non-zero indices sorted by value in ascending order.
    """
    nzidx = matrix.nonzero()
    ranking = np.argsort(matrix[nzidx])
    x, y = np.array(nzidx)[:, ranking]
    return np.stack([x[0], y[0]], axis = 1)

def split_folds(organized_ground_truth, number_of_folds):
    """
    :param organized_ground_truth: n × n matrix with the tested relations
                                   organized according to the evaluation type.
    :param number_of_folds: Positive integer.
    :returns: Iterable with pairs of arrays of indices for each fold.
    """
    return map(fold_to_indices, np.array_split(organized_ground_truth, number_of_folds))

def fold_tensor(ground_truth, slices, withheld_indices):
    """
    Make a tensor for a fold by removing the tested relations from its ground truth.

    :param ground_truth: n × n matrix with the tested relations.
    :param slices: List of n × n matrices to include as slices in the tensor.
    :param withheld_indices: Pair of arrays of indices to withhold from the ground truth.
    :returns: List of n × n matrices, where the ground truth without the tested relations is the first.
    """
    fold_truth = ground_truth.copy()
    # Remove the test indices from the fold's ground truth
    fold_truth[withheld_indices] = 0
    # Ground truth has the index 0
    return [fold_truth] + slices

def n_folds_splits(ground_truth, slices, number_of_folds):
    """
    Generate train/test splits for the provided folds.

    :param ground_truth: n × n matrix with the tested relations.
    :param slices: List of n × n matrices to include as slices in the tensor.
    :param number_of_folds: Positive integer.
    :returns: Iterator returning pairs of tested indices and training tensors.
    """
    organized_ground_truth = np.random.permutation(np.stack(ground_truth.nonzero(), axis = 1))
    folds = split_folds(organized_ground_truth, number_of_folds)
    return ((fold_indices, fold_tensor(ground_truth, slices, fold_indices))
            for fold_indices in folds)

def time_series_splits(ground_truth, slices, number_of_folds):
    """
    Generate train/test splits for the provided folds ordered as time series.
    
    :param ground_truth: n × n matrix with the tested relations.
    :param slices: List of n × n matrices to include as slices in the tensor.
    :param number_of_folds: Positive integer.
    :returns: Iterator returning pairs of tested indices and training tensors.
    """
    organized_ground_truth = sorted_indices(ground_truth)
    folds = list(split_folds(organized_ground_truth, number_of_folds))
    return ((testing_fold, fold_tensor(ground_truth,
                                       slices,
                                       tuple(np.concatenate(folds[index+1:], axis = 1))))
            for index, testing_fold in enumerate(folds[1:]))

def run_rescal(tensor, config):
    """
    Run the RESCAL-ALS algorithm on given tensor.

    :param tensor: List of n × n adjacency matrices.
    :param config: Dict of parameters for RESCAL, includes "rank", "init", "conv", and "regularization".
                   "regularization" includes "lambdaA", "lambdaR", "lambdaV".
                   See DEFAULT_CONFIG in cli.py for default configuration values.
    :returns: (A, R) tuple where A is an n × r matrix with interactions of n entities with r latent components.
              R is a list of r × r matrices with interactions of latent components in the individual matrix
              slices. R[0] are the interactions for the ground truth.
    """
    rank = config["rank"]
    regularization = config["regularization"]
    A, R, _, _, _ = rescal_als(tensor, rank,
            init = config["init"],
            conv = config["conv"],
            lambda_A = regularization["lambdaA"],
            lambda_R = regularization["lambdaR"],
            lambda_V = regularization["lambdaV"],
            compute_fit = False)
    return (A, R)

def predict_bidders_for_contract(A, A_T, R_ground_truth, bidder_mask, top_k, contract):
    """
    Predict top-k recommendations of bidders for a contract given a RESCAL decomposition.

    :param contract: Index of a contract.
    :param A: n × r matrix with interactions of n entities with r latent components.
    :param A_T: Transposed matrix A.
    :param R_ground_truth: r × r matrix with interactions of latent components for the ground_truth.
    :param bidder_mask: Boolean array masking non-bidder indices.
    :param top_k: How many predictions should be produced.
    :returns: Array of top-k predictions.
    """
    predictions = A[contract].dot(R_ground_truth).dot(A_T)
    predictions[~bidder_mask] = -float("inf") # Set predictions for links to non-bidders to -infinity.
    # .copy() breaks the link to predictions, allowing it to be GC-ed.
    return predictions.argsort()[-top_k:][::-1].copy()

def predict_bidders(A, R_ground_truth, fold_indices, top_k = 10):
    """
    Predict top-k recommendations of bidders given a RESCAL decomposition.

    :param A: n × r matrix with interactions of n entities with r latent components.
    :param R_ground_truth: r × r matrix with interactions of latent components for the ground_truth.
    :param fold_indices: Tuple matching contracts indices to winner indices by order.
    :param top_k: Number of top predictions to consider.
    :returns: n × top_k matrix where each row contains indices of the top predicted bidder.
    """
    contract_indices, bidder_indices = fold_indices
    bidder_mask = np.zeros(A.shape[0], dtype = bool)
    bidder_mask[bidder_indices] = True
    # Reconstruct predictions slice for the contract indices
    predict_fn = partial(predict_bidders_for_contract, A, A.T, R_ground_truth, bidder_mask, top_k)
    return np.vstack(map(predict_fn, contract_indices))

def run_fold(index, args, _, split):
    evaluation = args.config["evaluation"]
    evaluation_type = evaluation["type"]
    if evaluation_type == "time-series":
        number_of_folds = evaluation["folds"] - 1
    else:
        number_of_folds = evaluation["folds"]
    top_k = evaluation["top-k"]
    config = args.config["matchmaker"]

    _log.info("Running the fold {}/{}...".format(index, number_of_folds))

    fold_indices, tensor = split
    # Run RESCAL factorization
    A, R = run_rescal(tensor, config)
    # Reconstruct top predictions from the factor matrices
    top_predictions = predict_bidders(A, R[0], fold_indices, top_k = top_k)
    ranks = metrics.rank_predictions(top_predictions, fold_indices)
    return (top_predictions, ranks)

def run_random_fold(index, args, bidder_indices, split):
    evaluation = args.config["evaluation"]
    number_of_folds = evaluation["folds"]
    top_k = evaluation["top-k"]
    _log.info("Running the fold {}/{} with random matches...".format(index, number_of_folds))
    
    fold_indices, _ = split

    top_predictions = np.vstack((np.random.permutation(bidder_indices)[:top_k].copy()
                                 for contract in fold_indices[0]))
    ranks = metrics.rank_predictions(top_predictions, fold_indices)
    return (top_predictions, ranks)

def run(args):
    config = args.config
    evaluation_type = config["evaluation"]["type"]

    _log.info("Running {} evaluation...".format(evaluation_type))
    m, n = args.ground_truth.shape
    slice_count = len(args.slices) + 1 if args.slices else 1 
    _log.info("Tensor: {} × {} × {}".format(m, n, slice_count))

    number_of_folds = config["evaluation"]["folds"]
    bidder_indices = np.unique(args.ground_truth.nonzero()[1])
    long_tail_bidders = metrics.long_tail_bidders(args.ground_truth, bidder_indices)
    
    # Split the tensor into training/testing data
    split_fn = {
        "n-folds": n_folds_splits,
        "time-series": time_series_splits
    }[evaluation_type]
    splits = split_fn(args.ground_truth, args.slices, number_of_folds)

    # Compute the predictions for each fold
    run_fn = {
        "random": run_random_fold,
        "rescal": run_fold
    }[config["matchmaker"]["type"]]
    predictions = [run_fn(index + 1, args, bidder_indices, split)
                   for index, split in enumerate(splits)]

    # Merge predictions and ranks from each fold
    top_predictions = np.concatenate([fold_predictions[0] for fold_predictions in predictions])
    ranks = sum([fold_predictions[1] for fold_predictions in predictions], [])

    # Compute evaluation metrics
    results = {
        "hit_rate": metrics.hit_rate(ranks),
        "mean_reciprocal_rank": metrics.mean_reciprocal_rank(ranks),
        "catalog_coverage": metrics.catalog_coverage(top_predictions, bidder_indices),
        "prediction_coverage": metrics.prediction_coverage(top_predictions, bidder_indices),
        "long_tail_percentage": metrics.long_tail_percentage(top_predictions, long_tail_bidders)
    }

    # Save the evaluation results
    file_name = "results_" + str(uuid.uuid4()) + ".edn"
    with open(file_name, "w") as f:
       f.write(edn_format.dumps({
                   "config": config,
                   "results": results
               }))

    _log.info("Evaluation results written to {}.".format(file_name))
