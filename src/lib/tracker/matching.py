import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def dious(atlbrs, btlbrs):
    """
    Compute cost based on DIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype dious np.ndarray
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    # Convert to numpy arrays if they aren't already
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float)

    # Calculate IoU
    _ious = bbox_ious(atlbrs, btlbrs)

    # Calculate centers of boxes
    a_centers = (atlbrs[:, :2] + atlbrs[:, 2:]) / 2
    b_centers = (btlbrs[:, :2] + btlbrs[:, 2:]) / 2

    # Calculate center distance for each pair
    center_dist = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            center_dist[i, j] = np.sum((a_centers[i] - b_centers[j]) ** 2)

    # Calculate enclosing box diagonal distance
    enclosing_dist = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            enclosing_x1 = min(atlbrs[i, 0], btlbrs[j, 0])
            enclosing_y1 = min(atlbrs[i, 1], btlbrs[j, 1])
            enclosing_x2 = max(atlbrs[i, 2], btlbrs[j, 2])
            enclosing_y2 = max(atlbrs[i, 3], btlbrs[j, 3])
            enclosing_dist[i, j] = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

    # Calculate DIoU
    dious = _ious - center_dist / enclosing_dist

    return dious


def giou(atlbrs, btlbrs):
    """
    Compute cost based on GIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype gious np.ndarray
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    # Convert to numpy arrays if they aren't already
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float)

    # Calculate IoU
    _ious = bbox_ious(atlbrs, btlbrs)

    # Initialize GIoU matrix
    gious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            # Calculate area of boxes
            area_a = (atlbrs[i, 2] - atlbrs[i, 0]) * (atlbrs[i, 3] - atlbrs[i, 1])
            area_b = (btlbrs[j, 2] - btlbrs[j, 0]) * (btlbrs[j, 3] - btlbrs[j, 1])

            # Calculate area of enclosing box
            enclosing_x1 = min(atlbrs[i, 0], btlbrs[j, 0])
            enclosing_y1 = min(atlbrs[i, 1], btlbrs[j, 1])
            enclosing_x2 = max(atlbrs[i, 2], btlbrs[j, 2])
            enclosing_y2 = max(atlbrs[i, 3], btlbrs[j, 3])
            enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)

            # Calculate union area
            union_area = area_a + area_b - _ious[i, j] * area_a  # IoU definition: intersection / union

            # Calculate GIoU
            gious[i, j] = _ious[i, j] - (enclosing_area - union_area) / enclosing_area

    return gious


def ciou(atlbrs, btlbrs):
    """
    Compute cost based on CIoU (Complete IoU)
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype cious np.ndarray
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    # Convert to numpy arrays if they aren't already
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float)

    # Calculate IoU
    _ious = bbox_ious(atlbrs, btlbrs)

    # Initialize CIoU matrix
    cious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            # Calculate centers of boxes
            a_center = (atlbrs[i, :2] + atlbrs[i, 2:]) / 2
            b_center = (btlbrs[j, :2] + btlbrs[j, 2:]) / 2

            # Calculate center distance
            center_dist = np.sum((a_center - b_center) ** 2)

            # Calculate enclosing box diagonal distance
            enclosing_x1 = min(atlbrs[i, 0], btlbrs[j, 0])
            enclosing_y1 = min(atlbrs[i, 1], btlbrs[j, 1])
            enclosing_x2 = max(atlbrs[i, 2], btlbrs[j, 2])
            enclosing_y2 = max(atlbrs[i, 3], btlbrs[j, 3])
            enclosing_dist = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

            # Calculate aspect ratio consistency term
            a_width = atlbrs[i, 2] - atlbrs[i, 0]
            a_height = atlbrs[i, 3] - atlbrs[i, 1]
            b_width = btlbrs[j, 2] - btlbrs[j, 0]
            b_height = btlbrs[j, 3] - btlbrs[j, 1]

            # Avoid division by zero
            if a_width == 0 or a_height == 0 or b_width == 0 or b_height == 0:
                v = 0
            else:
                a_arctan = np.arctan(a_width / a_height)
                b_arctan = np.arctan(b_width / b_height)
                v = (4 / (np.pi ** 2)) * ((a_arctan - b_arctan) ** 2)

            # Calculate alpha parameter
            alpha = v / (1 - _ious[i, j] + v)

            # Calculate CIoU
            cious[i, j] = _ious[i, j] - center_dist / enclosing_dist - alpha * v

    return cious


def diou_distance(atracks, btracks):
    """
    Compute cost based on DIoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _dious = dious(atlbrs, btlbrs)
    # Convert to cost matrix (higher DIoU means lower cost)
    cost_matrix = 1 - _dious

    return cost_matrix


def giou_distance(atracks, btracks):
    """
    Compute cost based on GIoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _gious = giou(atlbrs, btlbrs)
    # Convert to cost matrix (higher GIoU means lower cost)
    cost_matrix = 1 - _gious

    return cost_matrix


def ciou_distance(atracks, btracks):
    """
    Compute cost based on CIoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _cious = ciou(atlbrs, btlbrs)
    # Convert to cost matrix (higher CIoU means lower cost)
    cost_matrix = 1 - _cious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_cosineiou(tracks, detections, lambda_):
    # 計算 IoU 距離矩陣
    iou_dist = iou_distance(tracks, detections)

    # 計算特徵距離矩陣
    cost_matrix = embedding_distance(tracks, detections)

    # 結合成本矩陣
    combined_cost = lambda_ * cost_matrix + (1 - lambda_) * iou_dist

    return combined_cost


def fuse_cosinediou(tracks, detections, lambda_):
    """
    Fusion of cosine distance and DIoU distance with weighted lambda
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type lambda_: float, weight for cosine distance
    :rtype combined_cost np.ndarray
    """
    # 計算 DIoU 距離矩陣
    diou_dist = diou_distance(tracks, detections)

    # 計算特徵距離矩陣
    cosine_dist = embedding_distance(tracks, detections)

    # 結合成本矩陣
    combined_cost = lambda_ * cosine_dist + (1 - lambda_) * diou_dist

    return combined_cost


def fuse_cosinegiou(tracks, detections, lambda_):
    """
    Fusion of cosine distance and GIoU distance with weighted lambda
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type lambda_: float, weight for cosine distance
    :rtype combined_cost np.ndarray
    """
    # 計算 GIoU 距離矩陣
    giou_dist = giou_distance(tracks, detections)

    # 計算特徵距離矩陣
    cosine_dist = embedding_distance(tracks, detections)

    # 結合成本矩陣
    combined_cost = lambda_ * cosine_dist + (1 - lambda_) * giou_dist

    return combined_cost


def fuse_cosineciou(tracks, detections, lambda_):
    """
    Fusion of cosine distance and CIoU distance with weighted lambda
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type lambda_: float, weight for cosine distance
    :rtype combined_cost np.ndarray
    """
    # 計算 CIoU 距離矩陣
    ciou_dist = ciou_distance(tracks, detections)

    # 計算特徵距離矩陣
    cosine_dist = embedding_distance(tracks, detections)

    # 結合成本矩陣
    combined_cost = lambda_ * cosine_dist + (1 - lambda_) * ciou_dist

    return combined_cost

