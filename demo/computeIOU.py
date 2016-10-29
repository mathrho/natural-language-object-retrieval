import numpy as np


'''
compute interval over union for boxes
'''

def computeIOU(gt,dt):
    '''
    compute interval overlapval
    '''
    if not gt.ndim > 1:
        gt = np.reshape(gt, (1, gt.shape[0]))
    if not dt.ndim > 1:
        dt = np.reshape(dt, (1, dt.shape[0]))

    nr_gt = gt.shape[0]
    nr_dt = dt.shape[0]
    ov = np.zeros((nr_gt,nr_dt))

    for i in range(nr_gt):
        for j in range(nr_dt):
            ov[i,j] = computeBoxOverlap(gt[i,:].astype(float),dt[j,:].astype(float))

    return ov


def computeBoxOverlap(gt,dt):
    '''
    '''
    if gt.size > 0 and dt.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(gt[0], dt[0])
        iymin = np.maximum(gt[1], dt[1])
        ixmax = np.minimum(gt[2], dt[2])
        iymax = np.minimum(gt[3], dt[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((dt[2] - dt[0] + 1.) * (dt[3] - dt[1] + 1.) +
               (gt[2] - gt[0] + 1.) *
               (gt[3] - gt[1] + 1.) - inters)

        overlap = inters / uni
        return overlap

