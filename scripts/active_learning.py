import argparse
import mmdet.apis
import os
import mmcv
import json
from mmdet.utils import select_images
import numpy as np
import torch



def select_random(n_pool, n_sel):
    '''
    Select randomly images from pool.
    '''
    return torch.randint(n_pool, (n_sel,))



def main(args):    
    config = mmcv.Config.fromfile(args.config)
    test_cfg = config.model.bbox_head.test_cfg
    
    train_set_orig = config.data.train.ann_file
    img_path = config.data.train.img_prefix
    train_set_name = train_set_orig.replace('.json', '_AL_init.json')
    pool_set_name = train_set_orig.replace('.json', '_AL_pool_init.json')

    # split training data into start training set and data pool for active learning
    if args.do_split or not os.path.isfile(train_set_name):
        os.system( f'python scripts/split_train_val.py --original {train_set_orig} --train {train_set_name} --val {pool_set_name} --ratio {args.ratio}' )

    # 1st training
    config.data.train.ann_file = train_set_name
    config_AL = args.config.replace('.py', '_AL.py')
    workdir = f'{args.work_dir}/init_training/'
    mmcv.Config.dump(config, config_AL)
    if args.do_init_train or not os.path.isfile(f'{workdir}/latest.pth'):
        os.system( f'python mmdetection/tools/train.py {config_AL} --work-dir {workdir}/' )

    # create list of pool images
    with open(pool_set_name, 'rt') as f:
        pool = json.load(f)
        pool_img = [os.path.join(img_path, im['file_name']) for im in pool['images']]

    with open(train_set_name, 'rt') as f:
        train = json.load(f)
    

    # loop over active learning iterations
    for ir in range(args.n_round):
        if test_cfg.active_learning.selection_method == 'random':
            selection = select_random(len(pool_img), test_cfg.active_learning.n_sel)
        else:
            # inference over pool images
            detector = mmdet.apis.init_detector(
                config_AL,
                f'{workdir}/latest.pth',
                device='cuda:0',
            )
            img_n_batch = 100
            pool_img_batch = np.array_split( np.array(pool_img), img_n_batch )
            uncertainty = []
            for img_batch in pool_img_batch:
                uncertainty.append( mmdet.apis.inference_detector(detector, img_batch.tolist(), active_learning=True) )
            uncertainty = torch.concat(uncertainty)

            # select images to be added to the training set
            selection = select_images(test_cfg.active_learning.selection_method, uncertainty, test_cfg.active_learning.n_sel, **test_cfg.active_learning.selection_kwargs)

        # update training set and pool according to selected images
        sel_id = []
        for idx in selection.sort(descending=True)[0]:
            train['images'].append( pool['images'][idx] )
            sel_id.append( pool['images'][idx]['id'] )

            pool['images'].remove( pool['images'][idx] )
            pool_img.remove( pool_img[idx] )

        # Pick annotations corresponding to selected images
        for annot in pool["annotations"]:
            if annot["image_id"] in sel_id:
                train["annotations"].append(annot)
                pool["annotations"].remove(annot)

        # save new train and pool sets
        with open(pool_set_name.replace('init.json', f'{ir}.json'), "wt") as f_out:
            json.dump(pool, f_out)
        with open(train_set_name.replace('init.json', f'{ir}.json'), "wt") as f_out:
            json.dump(train, f_out)

        # training with updated set
        config.data.train.ann_file = train_set_name.replace('init.json', f'{ir}.json') # use updated training set
        config.load_from = f'{workdir}/latest.pth' # checkpoint from previous iteration
        config.runner['max_epochs'] = args.n_epoch
        mmcv.Config.dump(config, config_AL)

        if test_cfg.active_learning.selection_method == 'random':
            workdir = f'{args.work_dir}/{test_cfg.active_learning.selection_method}/round_{ir}/'
        else:
            workdir = f'{args.work_dir}/{test_cfg.active_learning.score_method}/{test_cfg.active_learning.aggregation_method}/{test_cfg.active_learning.selection_method}/round_{ir}/'

        os.system( f'python mmdetection/tools/train.py {config_AL} --work-dir {workdir}' )

        # test
        os.system( f'python mmdetection/tools/test.py {config_AL} {workdir}/latest.pth --work-dir {workdir} --eval bbox' )


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='mmdetection config')
    parser.add_argument('--work-dir', required=True, help='output directory')
    parser.add_argument('--n-round', default=10, type=int, help='Number of iterations for active learning')
    parser.add_argument('--n-epoch', default=5, type=int, help='Number of epochs to update training at each iteration')
    parser.add_argument('--do_split', action='store_true', help='Split original training set into starting training set and pool set')
    parser.add_argument('--ratio', default='0.9', help='The pool will be ratio * N(images) in original training set')
    parser.add_argument('--do_init_train', action='store_true', help='Perform initial training')

    args = parser.parse_args()

    main(args)
