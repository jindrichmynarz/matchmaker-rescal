"""
Runner for n-fold cross validation of the RESCAL-based matchmakers.
"""
import edn_format
import logging
import metrics
import numpy as np
import uuid
from rescal import rescal_als
from scipy.sparse import coo_matrix

_log = logging.getLogger("matchmaker-rescal.evaluation")
_log.setLevel(logging.INFO)

def matrix_density(matrix):
    return matrix.nnz / np.prod(matrix.shape)

def matrix_sparsity(matrix):
    return 1 - matrix_density(matrix)

def random_matrix(n, density = 0.01):
    """
    Create a random square matrix of 0s and 1s given size `n` and density.

    :param n: Rank of the square matrix to create.
    :param density: Expected density of the created matrix.
    :returns: nxn CSR matrix with random values from {0, 1}.
    """
    elements = int(n ** 2 * density)
    idxs = np.random.randint(n, size = (2, elements))
    data = np.random.randint(2, size = elements, dtype = np.int32)
    return coo_matrix((data, idxs), (n, n)).tocsr()

def random_truth(ground_truth, contract_indices):
    """
    Create random matrix for ground truth based on the actual ground truth.

    :param ground_truth: nxn matrix
    :param contract_indices: Array of indices of contracts of length m.
    :returns: mxn matrix of random predictions.
    """
    return random_matrix(ground_truth.shape[0], matrix_density(ground_truth))[contract_indices].toarray()

def fold_tensor(ground_truth, slices, fold_indices):
    """
    Make a tensor for a fold by removing the tested relations from its ground truth.

    :param ground_truth: nxn matrix with the tested relations.
    :param slices: List of nxn matrices to include as slices in the tensor.
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
    :returns: (A, R) tuple where A is an nxr matrix with interactions of n entities with r latent components.
              R is a list of rxr matrices with interactions of latent components in the individual matrix
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

def predict_bidders(A, R_ground_truth, fold_indices, bidder_indices, top_k = 10):
    """
    Predict top-k recommendations of bidders given a RESCAL decomposition.

    :param A: nxr matrix with interactions of n entities with r latent components.
    :param R_ground_truth: rxr matrix with interactions of latent components for the ground_truth.
    :param fold_indices: Tuple matching contracts indices to winner indices by order.
    :param bidder_indices: Indices of all bidders.
    :param top_k: Number of top predictions to consider.
    :returns: n x top_k matrix where each row contains indices of the top predicted bidder 
    """
    contract_indices, bidder_indices = fold_indices
    # Reconstruct predictions slice for the contract indices
    predictions = A[contract_indices].dot(R_ground_truth.dot(A.T))
    bidder_mask = np.zeros(predictions.shape, dtype = bool)
    bidder_mask[:,bidder_indices] = True
    predictions[~bidder_mask] = float("inf") # Set predictions for links to non-bidders to -infinity.
    return np.fliplr(predictions.argsort()[:,-top_k:])

def run_fold(index, args, fold_indices):
    _log.info("Running the fold %d/%d..." % (index, args.config["evaluation"]["folds"]))

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

def fold_to_indices(fold):
    """
    Helper function that transposes 2xn matrix fold into two arrays in the (n, n) tuple,
    which is used for indexing.

    :param fold: 2xn matrix with the indices of ground truth in the given fold.
    """
    transposed = fold.T
    return (transposed[0], transposed[1])

def run(args):
    _log.info("Running evaluation...")
    _log.info("Tensor: %d x %d x %d" % (args.ground_truth.shape + (len(args.slices) + 1,)))

    config = args.config
    number_of_folds = config["evaluation"]["folds"]
    bidder_indices = args.headers["bidders"]
    long_tail_bidders = metrics.long_tail_bidders(args.ground_truth, bidder_indices)

    shuffled_ground_truth = np.random.permutation(np.stack(args.ground_truth.nonzero(), axis = 1))
    folds = map(fold_to_indices, np.array_split(shuffled_ground_truth, number_of_folds))
    # Compute the predictions for each fold
    predictions = [run_fold(index + 1, args, fold_indices) for index, fold_indices in enumerate(folds)]
    
    # Merge predictions and ranks from each fold
    top_predictions = np.concatenate([fold_predictions[0] for fold_predictions in predictions])
    ranks = np.concatenate([fold_predictions[1] for fold_predictions in predictions])

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

    _log.info("Evaluation results written to %s." % file_name)
