import argparse
import mmdet.apis
import mmdet.utils
from mmdet.structures.bbox import cat_boxes, get_box_tensor, get_box_wh, scale_boxes
from mmcv.transforms import Compose
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmengine.config import Config
from mmengine.runner import Runner
import glob
import cv2 as cv
import os
import torch
import tqdm
import numpy as np
import json
import copy

import sys



# modified inference_detector to return output tensor of network
def inference_detector(model, imgs, test_pipeline = None):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = mmdet.utils.get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            data = model.data_preprocessor(data_, False)
            if isinstance(data, dict):
                results = model(**data, mode='tensor')
            elif isinstance(data, (list, tuple)):
                results = model(*data, mode='tensor')
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                f'list, tuple or dict, but got {type(data)}')

            # from bbox_head.predict function
            batch_img_metas = [
                data_samples.metainfo for data_samples in data['data_samples']
            ]

            predictions = predict_by_feat(model, *results, batch_img_metas=batch_img_metas)

            # extract GT bboxes
            outputs = mmdet.models.utils.unpack_gt_instances(data['data_samples'])
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs
            print(batch_gt_instances)

    return results



def predict_by_feat(model, targets, cls_scores, bbox_preds, score_factors = None,
                    batch_img_metas = None, cfg = None,
                    rescale = False, with_nms = True):
    """Transform a batch of output features extracted from the head into
    bbox results.

    Note: When score_factors is not None, the cls_scores are
    usually multiplied by it then obtain the real score used in NMS,
    such as CenterNess in FCOS, IoU branch in ATSS.

    Args:
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Defaults to None.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        list[:obj:`InstanceData`]: Object detection results of each image
        after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    assert len(cls_scores) == len(bbox_preds)

    if score_factors is None:
        # e.g. Retina, FreeAnchor, Foveabox, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
        with_score_factors = True
        assert len(cls_scores) == len(score_factors)

    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = model.bbox_head.prior_generator.grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device)

    # get targets
    (batch_gt_instances, batch_gt_instances_ignore,
     batch_img_metas) = targets

    result_list = []

    for img_id in range(len(batch_img_metas)):
        img_meta = batch_img_metas[img_id]
        cls_score_list = mmdet.models.utils.select_single_mlvl(
            cls_scores, img_id, detach=True)
        bbox_pred_list = mmdet.models.utils.select_single_mlvl(
            bbox_preds, img_id, detach=True)
        if with_score_factors:
            score_factor_list = mmdet.models.utils.select_single_mlvl(
                score_factors, img_id, detach=True)
        else:
            score_factor_list = [None for _ in range(num_levels)]

        results = _predict_by_feat_single(
            model,
            gt_instances=batch_gt_instances[img_id],
            gt_instances_ignore=batch_gt_instances_ignore[img_id],            
            cls_score_list=cls_score_list,
            bbox_pred_list=bbox_pred_list,
            score_factor_list=score_factor_list,
            mlvl_priors=mlvl_priors,
            img_meta=img_meta,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms)
        result_list.append(results)
    return result_list



def _predict_by_feat_single(model, gt_instances, gt_instances_ignore, cls_score_list, bbox_pred_list, score_factor_list,
                            mlvl_priors, img_meta, cfg, rescale = False,
                            with_nms = True):
    """Transform a single image's features extracted from the head into
    bbox results.

    Args:
        cls_score_list (list[Tensor]): Box scores from all scale
            levels of a single image, each item has shape
            (num_priors * num_classes, H, W).
        bbox_pred_list (list[Tensor]): Box energies / deltas from
            all scale levels of a single image, each item has shape
            (num_priors * 4, H, W).
        score_factor_list (list[Tensor]): Score factor from all scale
            levels of a single image, each item has shape
            (num_priors * 1, H, W).
        mlvl_priors (list[Tensor]): Each element in the list is
            the priors of a single level in feature pyramid. In all
            anchor-based methods, it has shape (num_priors, 4). In
            all anchor-free methods, it has shape (num_priors, 2)
            when `with_stride=True`, otherwise it still has shape
            (num_priors, 4).
        img_meta (dict): Image meta info.
        cfg (mmengine.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    if score_factor_list[0] is None:
        # e.g. Retina, FreeAnchor, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, etc.
        with_score_factors = True

    cfg = model.bbox_head.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)
    img_shape = img_meta['img_shape']

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    if with_score_factors:
        mlvl_score_factors = []
    else:
        mlvl_score_factors = None
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                          score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        dim = model.bbox_head.bbox_coder.encode_size
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
        if with_score_factors:
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2,
                                      0).reshape(-1, model.bbox_head.cls_out_channels)
        if model.bbox_head.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)

        if with_score_factors:
            mlvl_score_factors.append(score_factor)

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = cat_boxes(mlvl_valid_priors)
    bboxes = model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
    if rescale:
        assert img_meta.get('scale_factor') is not None
        scale_factor = [1 / s for s in img_meta['scale_factor']]
        bboxes = scale_boxes(bboxes, scale_factor)

    num_level_priors = [len(s) for s in mlvl_scores]
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_score_factors = torch.cat(mlvl_score_factors)
    
    # assign a ground truth label to each predicted bbox
    assign_result = model.bbox_head.assigner.assign(InstanceData(priors=bboxes), num_level_priors,
                                                    gt_instances, gt_instances_ignore)
    gt_labels = assign_result.labels

    # filter prediction by score, and topk
    max_scores, labels = torch.max(mlvl_scores, 1)
    score_thr = cfg.get('score_thr', 0)
    valid_mask = max_scores > score_thr
    valid_idxs = torch.nonzero(valid_mask) 
    
    nms_pre = cfg.get('nms_pre', -1)
    num_topk = min(nms_pre, valid_idxs.size(0))
    sorted_scores, idxs = max_scores[valid_mask].sort(descending=True)
    topk_idxs = valid_idxs[idxs[:num_topk]].squeeze()

    mlvl_scores = mlvl_scores[topk_idxs]
    mlvl_labels = labels[topk_idxs]
    gt_labels = gt_labels[topk_idxs]
    mlvl_score_factors = mlvl_score_factors[topk_idxs]
    bboxes = bboxes[topk_idxs]
        
    results = InstanceData()
    results.bboxes = bboxes
    results.scores = mlvl_scores
    results.labels = mlvl_labels
    results.gt_labels = gt_labels

    # apply scale factors, e.g. centerness for anchor-free detectors
    if with_score_factors:
        results.scores = (results.scores.T * mlvl_score_factors).T

    # filter small size bboxes
    if cfg.get('min_bbox_size', -1) >= 0:
        w, h = get_box_wh(results.bboxes)
        valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
        if not valid_mask.all():
            results = results[valid_mask]

    # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    if with_nms and results.bboxes.numel() > 0:
        bboxes = get_box_tensor(results.bboxes)
        det_bboxes, keep_idxs = batched_nms(bboxes, results.scores.max(dim=1)[0],
                                            results.labels, cfg.nms)
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        #results.scores = det_bboxes[:, -1]
        results = results[:cfg.max_per_img]

    
    return results


    
def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    config = Config.fromfile(args.config)

    config.work_dir = 'outputs/conformal_prediction/'
    config.load_from = args.checkpoint
    
    runner = Runner.from_cfg(config)

    # weird results when taking model from runner instead of from inference detector.
    # only diff is init_cfg option in backbone with path to pretrained model
    #    detector = runner.model
    #    detector.cfg = config

    calib_loader = runner.val_dataloader

    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=device,
    )

    for img in calib_loader:
        img = detector.data_preprocessor(img, False)
        with torch.no_grad():
            results = detector(**img, mode='tensor')

        # from bbox_head.predict function
        batch_img_metas = [
            data_samples.metainfo for data_samples in img['data_samples']
        ]

        # extract GT bboxes
        targets = mmdet.models.utils.unpack_gt_instances(img['data_samples'])

        # get scores for bboxes filtered by score and nms
        predictions = predict_by_feat(detector, targets, *results, batch_img_metas=batch_img_metas, rescale=True)
        print(predictions[0].scores.size())  
        print(predictions[0].scores.max(dim=1))  
        print(predictions[0].gt_labels)
        print(1-predictions[0].scores.sum(dim=1))
        print('')
        

#    res = inference_detector(detector, args.im_dir)
#    # for dyhead: res[0] --> classification scores, res[1] --> location regression, res[2] --> centerness


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
    parser.add_argument("--im-dir", help="Directory containing the images")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    args = parser.parse_args()

    main(args)
