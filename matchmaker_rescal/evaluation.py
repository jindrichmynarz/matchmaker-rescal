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

def fold_tensor(ground_truth, slices, fold_indices):
    """
    Make a tensor for a fold by removing the tested relations from its ground truth.

    :param ground_truth: n × n matrix with the tested relations.
    :param slices: List of n × n matrices to include as slices in the tensor.
    :returns: List of nxn matrices, where the ground truth without the tested relations is the first.
    """
    fold_truth = ground_truth.copy()
    # Remove the test indices from the fold's ground truth
    fold_truth[fold_indices] = 0
    # Ground truth has the index 0
    return [fold_truth] + slices

def run_rescal(tensor, config):
    """
    Run the RESCAL-ALS algorithm on given tensor.

    :param tensor: List of nxn adjacency matrices.
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
    :param bidder_mask: A boolean array masking non-bidder indices.
    :param top_k: How many predictions should be produced.
    :returns: Array of top-k predictions.
    """
    predictions = A[contract].dot(R_ground_truth).dot(A_T)
    predictions[~bidder_mask] = -float("inf") # Set predictions for links to non-bidders to -infinity.
    # .copy() breaks the link to predictions, allowing it to be GC-ed.
    return predictions.argsort()[-top_k:][::-1].copy()

def predict_bidders(A, R_ground_truth, fold_indices, bidder_indices, top_k = 10):
    """
    Predict top-k recommendations of bidders given a RESCAL decomposition.

    :param A: n × r matrix with interactions of n entities with r latent components.
    :param R_ground_truth: r × r matrix with interactions of latent components for the ground_truth.
    :param fold_indices: Tuple matching contracts indices to winner indices by order.
    :param bidder_indices: Indices of all bidders.
    :param top_k: Number of top predictions to consider.
    :returns: n × top_k matrix where each row contains indices of the top predicted bidder.
    """
    contract_indices, bidder_indices = fold_indices
    bidder_mask = np.zeros(A.shape[0], dtype = bool)
    bidder_mask[bidder_indices] = True
    # Reconstruct predictions slice for the contract indices
    predict_fn = partial(predict_bidders_for_contract, A, A.T, R_ground_truth, bidder_mask, top_k)
    return np.vstack(map(predict_fn, contract_indices))

def run_fold(index, args, fold_indices):
    _log.info("Running the fold {}/{}...".format(index, args.config["evaluation"]["folds"]))

    config = args.config["matchmaker"]
    # Create a tensor for the fold
    tensor = fold_tensor(args.ground_truth, args.slices, fold_indices)
    # Run RESCAL factorization
    A, R = run_rescal(tensor, config)
    # Reconstruct top predictions from the factor matrices
    top_predictions = predict_bidders(A, R[0], fold_indices, args.headers["bidders"],
                                      top_k = args.config["evaluation"]["top-k"])
    ranks = metrics.rank_predictions(top_predictions, fold_indices)
    return (top_predictions, ranks)

def run_random_fold(index, args, fold_indices):
    _log.info("Running the fold {}/{} with random matches...".format(index, args.config["evaluation"]["folds"]))

    bidder_indices = args.headers["bidders"]
    top_k = args.config["evaluation"]["top-k"]
    top_predictions = np.vstack((np.random.permutation(bidder_indices)[:top_k].copy()
                                 for contract in fold_indices[0]))
    ranks = metrics.rank_predictions(top_predictions, fold_indices)
    return (top_predictions, ranks)

def fold_to_indices(fold):
    """
    Helper function that transposes 2 × n matrix fold into two arrays in the (n, n) tuple,
    which is used for indexing.

    :param fold: 2 × n matrix with the indices of ground truth in the given fold.
    """
    transposed = fold.T
    return (transposed[0], transposed[1])

def run(args):
    _log.info("Running evaluation...")
    _log.info("Tensor: {} × {} × {}".format(args.ground_truth.shape + (len(args.slices) + 1,)))

    config = args.config
    number_of_folds = config["evaluation"]["folds"]
    bidder_indices = args.headers["bidders"]
    long_tail_bidders = metrics.long_tail_bidders(args.ground_truth, bidder_indices)
    run_fn = {
        "random": run_random_fold,
        "rescal": run_fold
    }[config["matchmaker"]["type"]]

    shuffled_ground_truth = np.random.permutation(np.stack(args.ground_truth.nonzero(), axis = 1))
    folds = map(fold_to_indices, np.array_split(shuffled_ground_truth, number_of_folds))
    # Compute the predictions for each fold
    predictions = [run_fn(index + 1, args, fold_indices) for index, fold_indices in enumerate(folds)]

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
