import numpy as np


def compute_precision_recall_and_matching_score(gt: np.array, pred: np.array,
                                                prob: np.array, thresh: float):
    """Given the groundtruth, prediction, probability arrays and the probability threshold that is
    used to count a prediction as a true match, returns the precision, recall and matching score of
    the prediction. Assumes that keypoints without matches have gt = -1."""
    # Mask out all predictions with prob < thresh
    mask = np.where(prob < thresh)
    masked_pred = np.array(pred)
    masked_pred[mask] = -1
    #    print(f'masked_pred: {masked_pred}')

    total_positive = np.count_nonzero(gt != -1)
    tp = np.count_nonzero(np.logical_and(gt != -1, masked_pred == gt))
    fp = np.count_nonzero(np.logical_and(masked_pred != -1, masked_pred != gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_positive
    matching_score = tp / len(
        gt)  # num_correct / num_kps according to SuperGlue's implementation

    print(f'total positive: {total_positive}, tp: {tp}, fp: {fp}')
    print(f'precision: {precision}, recall: {recall}, matching_score: {matching_score}')
    return {
        'precision': precision,
        'recall': recall,
        'matching_score': matching_score
    }
