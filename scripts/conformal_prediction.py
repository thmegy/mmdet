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
from uncertainties import unumpy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


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
    mlvl_logits = []
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
        mlvl_logits.append(cls_score)

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
    mlvl_logits = torch.cat(mlvl_logits)
    
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

    if topk_idxs.size() == torch.Size([]): # case only 1 idx left, tensor is just a number
        topk_idxs = topk_idxs.unsqueeze(0)

    mlvl_scores = mlvl_scores[topk_idxs]
    mlvl_logits = mlvl_logits[topk_idxs]
    mlvl_labels = labels[topk_idxs]
    gt_labels = gt_labels[topk_idxs]
    mlvl_score_factors = mlvl_score_factors[topk_idxs]
    bboxes = bboxes[topk_idxs]
        
    results = InstanceData()
    results.bboxes = bboxes
    results.scores = mlvl_scores
    results.logits = mlvl_logits
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

    # add class corresponding to absence of object
    results = add_empty_class(results)

    return results



def predict(detector, img_batch):
    '''
    Predict the bboxes and scores for a batch of images.
    Returns bboxes, scores, predicted labels and groud-truth labels.
    '''
    img_batch = detector.data_preprocessor(img_batch, False)
    with torch.no_grad():
        results = detector(**img_batch, mode='tensor')

    # from bbox_head.predict function
    batch_img_metas = [
        data_samples.metainfo for data_samples in img_batch['data_samples']
    ]

    # extract GT bboxes
    targets = mmdet.models.utils.unpack_gt_instances(img_batch['data_samples'])

    # get bboxes and corresponding scores and ground truth, filtered by score and nms
    predictions = predict_by_feat(detector, targets, *results, batch_img_metas=batch_img_metas, rescale=True)

    return predictions



def platt_scaling(predictions, max_iters=10, lr=0.01, epsilon=0.01):
    '''
    Original function from https://github.com/aangelopoulos/conformal_classification
    '''
    nll_criterion = torch.nn.CrossEntropyLoss().cuda()

    n_classes = predictions[0].scores.shape[1]
    Temp = torch.nn.Parameter(torch.ones(n_classes).cuda()*1.3) # use 1.3 as initial value

    optimizer = torch.optim.SGD([Temp], lr=lr)
    for iter in range(max_iters):
        Temp_old = Temp
        for pred in predictions:
            x, target = pred.logits, pred.gt_labels # convert scores to logits
            optimizer.zero_grad()
            x.requires_grad = True
            out = x/Temp
            loss = nll_criterion(out, target.long())
            loss.backward()
            optimizer.step()
        if torch.abs(Temp_old - Temp).max() < epsilon:
            break
    
    return Temp 



def apply_platt_scaling(predictions, Temp_hat):
    for ip, pred in enumerate(predictions):
        scores = torch.sigmoid( pred.logits / Temp_hat )
#        scores = torch.softmax( pred.logits / Temp_hat, dim=1 )
        predictions[ip].scores = scores
    return predictions



def add_empty_class(pred):
    '''
    Add a class for "absence of object", whose score is computed as 1 - sum(other classes).
    '''
    scores_empty = 1-pred.scores.sum(dim=1)
    scores_empty[scores_empty < 0] = 0.01
    scores = torch.column_stack( (pred.scores, scores_empty) )
    pred.scores = scores

    logits = torch.column_stack( (pred.logits, torch.special.logit(scores_empty)) )
    pred.logits = logits

    # initial gt label for "empty" class is -1, change to N(classes)
    pred.gt_labels[pred.gt_labels==-1] = scores.size()[1]-1

    return pred



def compute_conformity_scores(scores):
    '''
    Get conformity score of every class for each bbox.
    '''
    sorted_scores, sorted_idxs = scores.sort(descending=True)
    conformity_score = torch.cumsum(sorted_scores, dim=1)

    return conformity_score, sorted_idxs



def plot_size_vs_difficulty(ranking, size, target, classes, outpath):
    '''
    Compute and plot size vs ranking of true class for each class + mean over all classes
    '''
    def get_size_vs_difficulty(ranking, size, outpath, plot_name_suffix='overall'):
        size_matrix = np.histogram2d(size, ranking, bins=[np.arange(14)+0.5, np.arange(14)-0.5])[0]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlabel('true-class ranking')
        ax.set_ylabel('prediction-set size')

        ticks = np.linspace(1, 13, 13).astype(int)
        c = ax.pcolor(size_matrix / size_matrix.sum(axis=0), cmap='Greens')
        for irow in range(size_matrix.shape[0]):
            for icol in range(size_matrix.shape[1]):
                ax.text(icol+0.5, irow+0.5, f'{int(size_matrix[irow][icol])}',
                           ha="center", va="center", color="black")

        mean = ticks @ size_matrix / size_matrix.sum(axis=0)
        ax.plot(ticks-0.5, mean, linestyle="None", marker='o', color='darkblue', label='mean', alpha=0.7)
        ax.plot(ticks-0.5, [ np.median(size[ranking == idiff]) for idiff in range(13) ], linestyle="None", marker='o', color='red', label='median', alpha=0.7)

        ax.set_xticks(ticks-0.5)
        ax.set_xticklabels(ticks-1, fontsize=10)
        ax.set_yticks(ticks-0.5)
        ax.set_yticklabels(ticks, fontsize=10)

        ax.legend()
        cbar = fig.colorbar(c)
        cbar.set_label('p.d.f')
        fig.set_tight_layout(True)
        fig.savefig(f'{outpath}/size_vs_ranking_{plot_name_suffix}.png')

        return mean

    mean_array_overall = get_size_vs_difficulty(ranking, size, outpath)

    mean_array_list = []
    for icls, cls in enumerate(classes):
        mask = (target == icls)
        mean_array = get_size_vs_difficulty(ranking[mask], size[mask], outpath, plot_name_suffix=cls)
        mean_array_list.append(mean_array)

    # plot average of classes of size vs ranking of true class
    fig_svr, ax_svr = plt.subplots()
    ax_svr.set_ylabel('prediction-set size')
    ax_svr.set_xlabel('true-class ranking')

    ax_svr.plot( range(len(classes)), mean_array_overall, linestyle="None", marker='p', color='black', label=f'overall', alpha=0.7)
    ax_svr.plot( range(len(classes)), np.stack(mean_array_list).mean(axis=0), linestyle="None", marker='p', color='darkblue', label=f'average w/ "empty"', alpha=0.7)
    ax_svr.plot( range(len(classes)), np.stack(mean_array_list)[:-1].mean(axis=0), linestyle="None", marker='p', color='red', label=f'average w/o "empty"', alpha=0.7)

    ax_svr.legend()
    fig_svr.set_tight_layout(True)
    fig_svr.savefig(f'{outpath}/average_size_vs_ranking.png')



def plot_coverage_per_class(covered, target, classes, alpha, outpath):
    fig, ax = plt.subplots()
    ax.set_ylabel('coverage')

    coverage_list = []
    coverage_list.append(covered.sum() / len(covered)) # overall coverage
    
    for icls, cls in enumerate(classes):
        mask = (target == icls)
        coverage = covered[mask].sum() / len(covered[mask])
        coverage_list.append(coverage)

    plt.bar(range(len(coverage_list)), coverage_list)
    ax.plot([-1, len(coverage_list)], [1-alpha, 1-alpha], color='red', linestyle='--', linewidth=1)

    ax.set_xticks( range(len(coverage_list)) )
    ax.set_xticklabels(['overall']+[c[:12] for c in classes], fontsize=10, rotation=45, ha='right')

    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/coverage_per_class.png')



def plot_coverage_vs_size(size, covered, target, classes, alpha, outpath):
    def get_coverage_vs_size(size, covered):
        size_list = []
        coverage_list = []
        n_sample = [] # number of samples per category
        for isize in range(size.max()):
            mask = (size == isize+1)
            coverage = covered[mask].sum() / len(covered[mask])
            n_sample.append(mask.sum())

            size_list.append(isize+1)
            coverage_list.append(coverage)

        return np.array(size_list), np.array(coverage_list), np.array(n_sample)
    
    # compute and plot coverage vs size
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('coverage')
    ax.set_xlabel('prediction-set size')

    # overall coverage
    size_list, coverage_list, n_sample = get_coverage_vs_size(size, covered)
    ax.scatter(size_list-0.45, coverage_list, 500*n_sample/n_sample.sum(), color='black', marker='o', label=f'Overall ({n_sample.sum()})')
    ax.plot([0.4, max(size_list)+0.5], [1-alpha, 1-alpha], color='red', linestyle='--', linewidth=1)

    # coverage per class
    markerstyle = ["p", "p", "p", "p", "p", "p", "p", "s", "p", "s", "s", "s", "*"]
    for icls, cls in enumerate(classes):
        mask = (target == icls)

        size_list, coverage_list, n_sample = get_coverage_vs_size(size[mask], covered[mask])
        ax.scatter(size_list-0.45+(icls+1)*0.064, coverage_list, 500*n_sample/n_sample.sum(), linestyle="None", marker=markerstyle[icls], label=f'{cls} ({n_sample.sum()})')
        ax.plot([icls+1.5, icls+1.5], [-0.03, 1.03], color='black', linestyle='--', linewidth=1)

    ax.legend(ncol=3, fontsize='small', framealpha=1)
    ax.set_xlim(0.5, max(size_list)+0.5)
    ax.set_ylim(-0.03, 1.03)
    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/coverage_vs_size.png')
        
            
    
def main(args):
    config = Config.fromfile(args.config)
    
    if not args.post_process:
        alpha_list = [0.05, 0.1, 0.15, 0.2]
        if args.alpha not in alpha_list:
            alpha_list.append(args.alpha)
            alpha_list.sort()
        alpha_list = np.array(alpha_list)
        select_id = np.where(alpha_list==args.alpha)[0].item()
            
        config.work_dir = 'outputs/conformal_prediction/'
        config.load_from = args.checkpoint

        runner = Runner.from_cfg(config)
        calib_loader = runner.val_dataloader

        # weird results when taking model from runner instead of from inference detector.
        # only diff is init_cfg option in backbone with path to pretrained model
        #    detector = runner.model
        #    detector.cfg = config

        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        detector = mmdet.apis.init_detector(
            args.config,
            args.checkpoint,
            device=device,
        )

        # calibration: compute conformity score for true class oach image of the calibration set
        if not args.skip_calibration:
            predictions = []
            for img_batch in tqdm.tqdm(calib_loader):
                predictions += predict(detector, img_batch)

            # Platt scaling
            Temp_hat = platt_scaling(predictions)
            print(f'Optimal Temperature scaling: {Temp_hat.tolist()}')

            predictions = apply_platt_scaling(predictions, Temp_hat)
            
            # CP calibration
            conformity_score_list = []
            gt_label_list = []
            for pred in predictions:
                conformity_scores, sorted_idxs = compute_conformity_scores(pred.scores)
                box_id, true_class_ranking = torch.where( sorted_idxs==pred.gt_labels.unsqueeze(1) )  # get conformity score of true class for each bbox
                conformity_score_list.append( conformity_scores[box_id, true_class_ranking].cpu().detach().numpy() )
                gt_label_list.append(pred.gt_labels.cpu().detach().numpy())
                    
            true_class_conformity_score = np.concatenate(conformity_score_list)

            # test several significance levels, apply later only the inputed one
            tau_hat = np.quantile(true_class_conformity_score, 1-alpha_list, method='higher')
            print(f'conformity-score threshold for alpha={args.alpha}: {tau_hat[select_id]:.3f}')

            true_class = np.concatenate(gt_label_list)
            tau_hat_dict = { 'classes':list(config.classes)+['empty'], 'temperature_scaling':Temp_hat.cpu().detach().tolist(), 'significance':alpha_list.tolist(), 'overall':tau_hat.tolist()}
            tau_hat_cls_list = []
            for icls, cls in enumerate(list(config.classes)+['empty']):
                tau_hat_cls = np.quantile(true_class_conformity_score[true_class==icls], 1-alpha_list, method='higher')
                print(f'{cls} {tau_hat_cls[select_id]:.3f}')
                tau_hat_cls_list.append(tau_hat_cls.tolist())
            tau_hat_dict['classes_thr'] = tau_hat_cls_list
                
            with open(f'{args.outpath}/conformity_thresholds_cracks.json', 'w') as f:
                json.dump(tau_hat_dict, f, indent = 6)

        else:
            with open(f'{args.outpath}/conformity_thresholds_cracks.json', 'r') as f:
                tau_hat_dict = json.load(f)
            Temp_hat = torch.tensor(tau_hat_dict['temperature_scaling'], device=device)
                
        if args.per_class_thr:
            tau_hat = torch.tensor(tau_hat_dict['classes_thr'], device=device).T[select_id]
        else:
            tau_hat = torch.tensor(tau_hat_dict['overall'][select_id], device=device)



                
        # test performance of the model with CP
        test_loader = runner.test_dataloader

        size_list = []
        target_list = []
        ranking_list = [] # ranking of true class based on predicted scores
        argmax_list = []
        covered_list = []
        for img_batch in tqdm.tqdm(test_loader):
            predictions = predict(detector, img_batch)
            predictions = apply_platt_scaling(predictions, Temp_hat)
            for pred in predictions:
                conformity_scores, sorted_idxs = compute_conformity_scores(pred.scores)
                # construct prediction set of every bbox
                for cs, idxs, gt_label in zip(conformity_scores, sorted_idxs, pred.gt_labels): #loop over bboxes
                    if args.per_class_thr:
                        tau_hat = tau_hat[idxs]
                    prediction_set = idxs[cs<=tau_hat].cpu().detach().numpy()

                    # do not allow empty sets
                    if len(prediction_set) == 0:
                        prediction_set = [idxs[0].cpu().detach().item()]
                        
                    size_list.append(len(prediction_set))
                    target_list.append(gt_label.item())
                    ranking_list.append(torch.where(idxs==gt_label)[0].item())
                    covered_list.append(gt_label.item() in prediction_set)

        results = {
            'size' : size_list,
            'target' : target_list,
            'ranking' : ranking_list,
            'covered' : covered_list
            }
        with open(f'{args.outpath}/results_cp_cracks.json', 'w') as f:
            json.dump(results, f, indent = 6)

            
            
    # post-processing
    with open(f'{args.outpath}/results_cp_cracks.json', 'r') as f:
        results = json.load(f)

    size, target, ranking, covered = np.array(results['size']), np.array(results['target']), np.array(results['ranking']), np.array(results['covered'])
    classes = list(config.classes)+['empty']

    plot_coverage_per_class(covered, target, classes, args.alpha, args.outpath)
    
    # compute and plot size vs ranking of true class for each class
    plot_size_vs_difficulty(ranking, size, target, classes, args.outpath)

    # compute and plot coverage vs size
    plot_coverage_vs_size(size, covered, target, classes, args.alpha, args.outpath)
    
#    res = inference_detector(detector, args.im_dir)
#    # for dyhead: res[0] --> classification scores, res[1] --> location regression, res[2] --> centerness


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmdetection config")
    parser.add_argument("--checkpoint", required=True, help="mmdetection checkpoint")
    parser.add_argument("--im-dir", help="Directory containing the images")
    parser.add_argument("--outpath", required=True, help="Path to output directory")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    
    parser.add_argument("--alpha", type=float, default=0.1, help="significance level")
    parser.add_argument("--per-class-thr", action='store_true', help="Determine and apply per-class conformity-score thresholds.")
    
    parser.add_argument("--skip-calibration", action='store_true', help="Skip calibration and run directly validation.")
    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    
    main(args)
