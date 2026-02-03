import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

def focal_eiou(bboxes1, bboxes2, alpha=0.25, gamma=2.0):
    """
    Calculate Focal-EIoU between two sets of bounding boxes.
    :param bboxes1: (N, 4) ndarray of float
    :param bboxes2: (M, 4) ndarray of float
    :param alpha: float, balancing parameter
    :param gamma: float, focusing parameter
    :return: (N, M) ndarray of Focal-EIoU
    """
    # Calculate IoU
    ious = bbox_ious(bboxes1, bboxes2)

    # Calculate the center points of each bounding box
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2

    # Calculate the distance between the center points
    center_distance = (center_x1[:, np.newaxis] - center_x2) ** 2 + (center_y1[:, np.newaxis] - center_y2) ** 2

    # Calculate the diagonal length of the smallest enclosing box
    enclose_x1 = np.minimum(bboxes1[:, 0][:, np.newaxis], bboxes2[:, 0])
    enclose_y1 = np.minimum(bboxes1[:, 1][:, np.newaxis], bboxes2[:, 1])
    enclose_x2 = np.maximum(bboxes1[:, 2][:, np.newaxis], bboxes2[:, 2])
    enclose_y2 = np.maximum(bboxes1[:, 3][:, np.newaxis], bboxes2[:, 3])
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Calculate aspect ratio
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    v = (4 / (np.pi ** 2)) * (np.arctan(w1 / h1)[:, np.newaxis] - np.arctan(w2 / h2)) ** 2
    alpha = v / (1 - ious + v)

    # Calculate EIoU
    eiou = ious - (center_distance / enclose_diagonal + alpha * v)

    # Apply Focal Loss
    focal_eiou = alpha * (1 - eiou) ** gamma * eiou
    return focal_eiou

def ciou(bboxes1, bboxes2):
    """
    Calculate CIoU between two sets of bounding boxes.
    :param bboxes1: (N, 4) ndarray of float
    :param bboxes2: (M, 4) ndarray of float
    :return: (N, M) ndarray of CIoU
    """
    # Calculate IoU
    ious = bbox_ious(bboxes1, bboxes2)

    # Calculate the center points of each bounding box
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2

    # Calculate the distance between the center points
    center_distance = (center_x1[:, np.newaxis] - center_x2) ** 2 + (center_y1[:, np.newaxis] - center_y2) ** 2

    # Calculate the diagonal length of the smallest enclosing box
    enclose_x1 = np.minimum(bboxes1[:, 0][:, np.newaxis], bboxes2[:, 0])
    enclose_y1 = np.minimum(bboxes1[:, 1][:, np.newaxis], bboxes2[:, 1])
    enclose_x2 = np.maximum(bboxes1[:, 2][:, np.newaxis], bboxes2[:, 2])
    enclose_y2 = np.maximum(bboxes1[:, 3][:, np.newaxis], bboxes2[:, 3])
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Calculate aspect ratio
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    v = (4 / (np.pi ** 2)) * (np.arctan(w1 / h1)[:, np.newaxis] - np.arctan(w2 / h2)) ** 2
    alpha = v / (1 - ious + v)

    # Calculate CIoU
    ciou = ious - (center_distance / enclose_diagonal + alpha * v)
    return ciou


def diou(bboxes1, bboxes2):
    """
    Calculate DIoU between two sets of bounding boxes.
    :param bboxes1: (N, 4) ndarray of float
    :param bboxes2: (M, 4) ndarray of float
    :return: (N, M) ndarray of DIoU
    """
    # Calculate IoU
    ious = bbox_ious(bboxes1, bboxes2)

    # Calculate the center points of each bounding box
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2

    # Calculate the distance between the center points
    center_distance = (center_x1[:, np.newaxis] - center_x2) ** 2 + (center_y1[:, np.newaxis] - center_y2) ** 2

    # Calculate the diagonal length of the smallest enclosing box
    enclose_x1 = np.minimum(bboxes1[:, 0][:, np.newaxis], bboxes2[:, 0])
    enclose_y1 = np.minimum(bboxes1[:, 1][:, np.newaxis], bboxes2[:, 1])
    enclose_x2 = np.maximum(bboxes1[:, 2][:, np.newaxis], bboxes2[:, 2])
    enclose_y2 = np.maximum(bboxes1[:, 3][:, np.newaxis], bboxes2[:, 3])
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Calculate DIoU
    diou = ious - center_distance / enclose_diagonal
    return diou