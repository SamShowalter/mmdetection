import copy
import cv2
import torch.nn.functional as F
import os.path as osp
import glob
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmcv.parallel import collate, scatter
from mmcv.ops.nms import batched_nms
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def my_multiclass_nms(multi_bboxes,
                      multi_scores,
                      score_thr,
                      nms_cfg,
                      max_num=-1, ):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
    Returns:
        tuple: (bboxes, labels, probs ), tensors of shape (k, 5),
            (k), and (k, 80). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # print(multi_bboxes.shape)
    # num_classes = multi_scores.size(1) 
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    # Background probability ignored, but why? let's include
    # scores = multi_scores[:, :-1]
    # print(bboxes.shape)
    # final_scores = copy.deepcopy(scores)
    scores = multi_scores[:, :-1]
    # print(scores.shape)
    

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    # choose max score for each ROI
    _, indices = torch.max(scores, dim=1)

    device = scores.device
    num_ROIs = indices.size(0)
    new_bboxes = torch.zeros(num_ROIs, 4).to(device)
    new_scores = torch.zeros(num_ROIs, ).to(device)
    new_labels = torch.zeros(num_ROIs, ).to(device)
    for i in range(num_ROIs):
        idx = indices[i]
        new_bboxes[i, :] = bboxes[i, idx, :]
        new_scores[i] = scores[i, idx]
        new_labels[i] = labels[i, idx]

    # remove low scoring boxes
    valid_mask = new_scores > score_thr

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    new_bboxes, new_scores, new_labels = new_bboxes[inds], new_scores[inds], new_labels[inds]

    probs = scores[inds]  # same as applying softmax operation on logits

    dets, keep = batched_nms(new_bboxes, new_scores, new_labels, nms_cfg)
    probs = probs[keep]
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
        probs = probs[:max_num]

    return dets, new_labels[keep], probs


def inference_detector_with_probs(model, img, score_thresh = None):
    """Inference image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img (str): image file.
    Returns:
    """

    cfg = model.cfg
    
    cfg.model.test_cfg.rcnn.max_per_img = 50
    device = next(model.parameters()).device  # model device
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    # add information into dict
    data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    datas.append(test_pipeline(data))

    data = collate(datas, samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    # scatter to specified GPU
    data = scatter(data, [device])[0]

    img_metas = data["img_metas"][0]
    # forward the model
    with torch.no_grad():
        x = model.extract_feat(data['img'][0])
        # print(img_metas)
        # print(x[4].shape)
        # print(len(x))
        proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
        # print(proposal_list[0].shape)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        rois = torch.stack(proposal_list, dim=0)

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = model.roi_head._bbox_forward(x, rois)
        cls_logits = bbox_results['cls_score']
        # print(cls_logits.shape)
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
		# logits
        cls_logits = cls_logits.reshape(batch_size, num_proposals_per_img, -1)
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, -1)

        scores = F.softmax(cls_logits, dim=-1)
        bboxes = model.roi_head.bbox_head.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shapes)
        # B, 1, bboxes.size(-1)
        scale_factor = bboxes.new_tensor(np.array(scale_factors)).unsqueeze(1).repeat(
            1, 1,
            bboxes.size(-1) // 4)
        bboxes /= scale_factor

        det_bboxes = []
        det_labels = []
        det_probs = []
        for (bbox, score) in zip(bboxes, scores):
            if score_thresh is not None:
                cfg.model.test_cfg.rcnn.score_thr = score_thresh
            det_bbox, det_label, det_prob = my_multiclass_nms(bbox, score,
                                                              cfg.model.test_cfg.rcnn.score_thr,
                                                              cfg.model.test_cfg.rcnn.nms,
                                                              cfg.model.test_cfg.rcnn.max_per_img)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_probs.append(det_prob)

    return det_bboxes, det_labels, det_probs