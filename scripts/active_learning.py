import argparse
import mmdet.apis
import os
import mmcv
import json
from mmdet.utils import select_images
import numpy as np
import torch
import tqdm
os.environ['MKL_THREADING_LAYER'] = 'GNU'



def select_random(n_pool, n_sel):
    '''
    Select randomly images from pool.
    '''
    return torch.randint(n_pool, (n_sel,))



def update_train_pool(train, pool, selection, pool_img=None):
    # update training set and pool according to selected images
    sel_id = []
    for idx in selection.sort(descending=True)[0]:
        train['images'].append( pool['images'][idx] )
        sel_id.append( pool['images'][idx]['id'] )
                
        pool['images'].remove( pool['images'][idx] )
        if pool_img is not None:
            pool_img.remove( pool_img[idx] )

    # Pick annotations corresponding to selected images
    for annot in pool["annotations"]:
        if annot["image_id"] in sel_id:
            train["annotations"].append(annot)
            pool["annotations"].remove(annot)
        
    if pool_img is not None:
        return train, pool, pool_img
    else:
        return train, pool



def main(args):    
    config_AL = args.config.replace('.py', '_AL.py')

    if args.auto_resume:
        config = mmcv.Config.fromfile(config_AL)
        train_set_name = config.data.train.ann_file
        pool_set_name = train_set_name.replace('AL', 'AL_pool')
    else:
        config = mmcv.Config.fromfile(args.config)
    
        train_set_orig = config.data.train.ann_file
        train_set_name = train_set_orig.replace('.json', '_AL_init.json')
        pool_set_name = train_set_orig.replace('.json', '_AL_pool_init.json')

        # split training data into start training set and data pool for active learning
        if args.do_split or not os.path.isfile(train_set_name):
            with open(train_set_orig, 'rt') as f:
                pool = json.load(f)
            train = {'images':[], 'type':pool['type'], 'categories':pool['categories'], 'annotations':[]}
                
            selection = select_random(len(pool['images']), args.n_init)
            train, pool = update_train_pool(train, pool, selection)

            with open(pool_set_name, "wt") as f_out:
                json.dump(pool, f_out)
            with open(train_set_name, "wt") as f_out:
                json.dump(train, f_out)

        # 1st training
        config.data.train.ann_file = train_set_name
        workdir = f'{args.work_dir}/init_training/'
        mmcv.Config.dump(config, config_AL)
        if args.do_init_train or not os.path.isfile(f'{workdir}/latest.pth'):
            if args.n_gpu == 1:
                os.system( f'CUDA_VISIBLE_DEVICES={args.gpu_id} python mmdetection/tools/train.py {config_AL} --work-dir {workdir}/ --auto-scale-lr' )
            else:
                os.system( f'./mmdetection/tools/dist_train.sh {config_AL} {args.n_gpu} --work-dir {workdir}/' )
                
            # test
            os.system( f'CUDA_VISIBLE_DEVICES={args.gpu_id} python mmdetection/tools/test.py {config_AL} {workdir}/latest.pth --work-dir {workdir} --eval bbox' )

    # create list of pool images
    img_path = config.data.train.img_prefix
    with open(pool_set_name, 'rt') as f:
        pool = json.load(f)
        pool_img = [os.path.join(img_path, im['file_name']) for im in pool['images']]

    with open(train_set_name, 'rt') as f:
        train = json.load(f)

    try:
        test_cfg = config.model.bbox_head.test_cfg
    except:
        test_cfg = config.model.test_cfg

    if test_cfg.active_learning.selection_method == 'random':
        method = 'random'
    else:
        method = f'{test_cfg.active_learning.score_method}_{test_cfg.active_learning.aggregation_method}_{test_cfg.active_learning.selection_method}'

    if test_cfg.active_learning.selection_method == 'CoreSet': # needed to use correct gpu in feature vector calculation
        torch.cuda.set_device(args.gpu_id)

        
    # loop over active learning iterations
    if args.auto_resume:
        round_range = range(args.resume_round, args.n_round)
    else:
        round_range = range(args.n_round)
        
    for ir in round_range:
        print(f'\nRound {ir}\n')

        if not args.auto_resume or ir > args.resume_round:
            if test_cfg.active_learning.selection_method == 'random':
                selection = select_random(len(pool_img), test_cfg.active_learning.n_sel)
            else:
                # inference over pool images
                detector = mmdet.apis.init_detector(
                    config_AL,
                    f'{workdir}/latest.pth',
                    device=f'cuda:{args.gpu_id}',
                )
                
                # split images into batches to run the inference
                img_batch_size = args.batch_size
                img_n_batch = len(pool_img) // img_batch_size
                img_batches = np.array_split( np.array(pool_img), img_n_batch )

                uncertainty = []
                representation = []
                
                print('\nRunning inference on pool set')
                for img_batch in tqdm.tqdm(img_batches):
                    if test_cfg.active_learning.selection_method == 'CoreSet':
                        unc, rep = mmdet.apis.inference_detector(detector, img_batch.tolist(), active_learning=True, repr_selection=True)
                        uncertainty.append(unc)
                        representation.append(rep)
                    else:
                        uncertainty.append( mmdet.apis.inference_detector(detector, img_batch.tolist(), active_learning=True) )

                uncertainty = torch.concat(uncertainty)
                if test_cfg.active_learning.selection_method == 'CoreSet':
                    representation = torch.concat(representation)
                torch.cuda.empty_cache()
                del detector
                # select images to be added to the training set
                selection = select_images(test_cfg.active_learning.selection_method, uncertainty, test_cfg.active_learning.n_sel, **test_cfg.active_learning.selection_kwargs, embedding=representation)
                # free gpu memory
                del uncertainty
                
            # update training set and pool according to selected images
            train, pool, pool_img = update_train_pool(train, pool, selection, pool_img=pool_img)

            # free gpu memory
            del selection
            torch.cuda.empty_cache()

            
            # save new train and pool sets
            if args.auto_resume:
                with open(pool_set_name.replace(f'{args.resume_round}.json', f'{ir}.json'), "wt") as f_out:
                    json.dump(pool, f_out)
                with open(train_set_name.replace(f'{args.resume_round}.json', f'{ir}.json'), "wt") as f_out:
                    json.dump(train, f_out)
                config.data.train.ann_file = train_set_name.replace(f'{args.resume_round}.json', f'{ir}.json') # use updated training set
            else:
                with open(pool_set_name.replace('init.json', f'{method}_{ir}.json'), "wt") as f_out:
                    json.dump(pool, f_out)
                with open(train_set_name.replace('init.json', f'{method}_{ir}.json'), "wt") as f_out:
                    json.dump(train, f_out)
                config.data.train.ann_file = train_set_name.replace('init.json', f'{method}_{ir}.json') # use updated training set

            # training with updated set
            if args.incremental_learning:
                config.load_from = f'{workdir}/latest.pth' # checkpoint from previous iteration
                config.runner = dict(type='IterBasedRunner', max_iters=args.n_iter)
                config.evaluation['interval'] = args.n_iter + 1 # do not make intermediary evaluation
                config.checkpoint_config['interval'] = args.n_iter+1
            mmcv.Config.dump(config, config_AL)

        workdir = f'{args.work_dir}/{method.replace("_", "/")}/{test_cfg.active_learning.n_sel}/round_{ir}/'         
        options = f' --work-dir {workdir}/'
        if args.auto_resume and ir == args.resume_round:
            options += f' --auto-resume'
        if args.incremental_learning:
            options += ' --incremental-learning'

        if args.n_gpu > 1:
            os.system( f'./mmdetection/tools/dist_train.sh {config_AL} {args.n_gpu} {options}' )
        else:
            os.system( f'CUDA_VISIBLE_DEVICES={args.gpu_id} python mmdetection/tools/train.py {config_AL} {options} --auto-scale-lr' )

        # test
        os.system( f'CUDA_VISIBLE_DEVICES={args.gpu_id} python mmdetection/tools/test.py {config_AL} {workdir}/latest.pth --work-dir {workdir} --eval bbox' )


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='mmdetection config')
    parser.add_argument('--work-dir', required=True, help='output directory')
    parser.add_argument('--n-round', default=10, type=int, help='Number of iterations for active learning')
    parser.add_argument('--batch-size', default=50, type=int, help='Number of images in inference batch')
    parser.add_argument('--n-gpu', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ')

    parser.add_argument('--do-split', action='store_true', help='Split original training set into starting training set and pool set')
    parser.add_argument('--n-init', default=1000, type=int, help='Number of initially labelled images')
    parser.add_argument('--do-init-train', action='store_true', help='Perform initial training')

    parser.add_argument('--incremental-learning', action='store_true', help='Do not train from scratch at each round, start from latest checkpoint of previous round')
    parser.add_argument('--n-iter', default=100, type=int, help='Number of training iteration at each round')

    parser.add_argument('--auto-resume', action='store_true', help='Resume training from latest checkpoint of round given in --resume-round')
    parser.add_argument('--resume-round', default=0, type=int, help='Round to resume from if --auto-resume is used')

    args = parser.parse_args()

    main(args)
