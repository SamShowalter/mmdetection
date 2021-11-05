import os
import datetime as dt
import sys

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from inference_det_with_probs import *
import mmcv
import pickle as pkl
from tqdm import tqdm


if __name__ in '__main__':
    with open("../../Scene_Graph_Novelty/data/coco/clean/ms_coco_test2017_filename_ordering.pkl", 'rb') as file:
        paths = pkl.load(file)

    print("Loaded COCO training data with {} samples.".format(len(paths)))

    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../models/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:4')
    print('Model loaded. Beginning run')

    bboxes = []
    labels = []
    probs = []
    i = 0
    for p in paths:
        img_path = "../../Scene_Graph_Novelty/data/coco/images/test2017/{}"\
        .format(p)
        det_bboxes, det_labels, det_probs = inference_detector_with_probs(model, img_path,score_thresh = 0.00)
        det_bboxes = det_bboxes[0].cpu().numpy()
        det_labels = det_labels[0].cpu().numpy()
        det_probs = det_probs[0].cpu().numpy()
        det_probs_background = 1 - det_probs.sum(axis = 1)
        # print(det_probs.shape)
        # print(np.expand_dims(det_probs_background, axis = 0).shape)
        det_probs = np.append(det_probs,
                              np.expand_dims(det_probs_background, axis = 0).T,
                             axis = 1)

        i += 1
        if i%1000 == 0:
            print("{}:    {}% complete".format(dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),round(i/len(paths),3)))
        # print(det_probs.sum(axis = 1))
        # print(det_bboxes.shape)
        assert det_bboxes.shape == (50,5), "ERROR"
        assert det_labels.shape[0] == 50, "ERROR"
        assert det_probs.shape == (50,81), "ERROR"
        bboxes.append(det_bboxes)
        labels.append(det_labels)
        probs.append(det_probs)
        sys.stdout.flush()

    bboxes = np.stack(bboxes)
    labels = np.stack(labels)
    probs = np.stack(probs)

    torch.save(bboxes, '../../Scene_Graph_Novelty/data/coco/extracted/ms_coco_test2017_bboxes_top50.pt')
    torch.save(labels, '../../Scene_Graph_Novelty/data/coco/extracted/ms_coco_test2017_labels_top50.pt')
    torch.save(probs, '../../Scene_Graph_Novelty/data/coco/extracted/ms_coco_test2017_probs_top50.pt')
