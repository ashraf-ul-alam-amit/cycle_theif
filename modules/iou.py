def iou(bbox1, bbox2):
    # calculate the intersection area
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # calculate the union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    # calculate the IOU
    iou = intersection_area / union_area

    return iou