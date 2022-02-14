import argparse
import mmdet.apis
import os
import mmcv
import json



def main(args):    
    config = mmcv.Config.fromfile(args.config)
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
    os.system( f'python mmdetection/tools/train.py {config_AL} --work-dir {workdir}/' )

    # create list of pool images
    with open(pool_set_name, 'rt') as f:
        pool = json.load(f)
        pool_img = [os.path.join(img_path, im['file_name']) for im in pool['images']]

    with open(train_set_name, 'rt') as f:
        train = json.load(f)
    

    # loop over active learning iterations
    for ir in range(args.n_round):
        # inference over pool images
        detector = mmdet.apis.init_detector(
            config_AL,
            f'{workdir}/latest.pth',
            device='cuda:0',
        )
        selected = mmdet.apis.inference_detector(detector, pool_img, active_learning=True)

        # update training set and pool according to selected images
        sel_img = [pool_img[idx] for idx in selected] 
        sel_id = [pool_id[idx] for idx in selected]
        for idx in selected:
            train['images'].append( pool['image'][idx] )
            sel_id.append( pool['image'][idx]['id'] )
            pool_img.remove( pool_img[idx] )
            pool['image'].remove( pool['image'][idx] )
            
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
        config.data.train.ann_file = train_set_name.replace('init.json', f'{ir}.json')
        mmcv.Config.dump(config, config_AL)
        test_cfg = config.model.bbox_head.test_cfg
        workdir = f'{args.work_dir}/{test.cfg.score_method}/{test.cfg.aggregation_method}/{test.cfg.selection_method}/round_{ir}/'
        # TO DO: update load_from and n_epoch
        os.system( f'python mmdetection/tools/train.py {config_AL} --work-dir {workdir}/' )


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='mmdetection config')
    parser.add_argument('--work-dir', required=True, help='output directory')
    parser.add_argument('--n_round', default=10, type=int, help='Number of iterations for active learning')
    parser.add_argument('--do_split', action='store_true', help='Split original training set into starting training set and pool set')
    parser.add_argument('--ratio', default='0.9', help='The pool will be ratio * N(images) in original training set.')

    args = parser.parse_args()

    main(args)
