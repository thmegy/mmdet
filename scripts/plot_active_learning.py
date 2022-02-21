import argparse
import matplotlib.pyplot as plt
import json
from glob import glob
from datetime import datetime



def get_latest(path_list):
    '''
    Get path containing the latest timestamp
    '''
    latest = datetime(1,1,1)
    latest_path = ''
    for path in path_list:
        p = path.split('_')
        date = p[-2]
        time = p[-1].replace('.json', '')
        stamp = datetime( int(date[:4]), int(date[4:6]), int(date[6:8]), int(time[:2]), int(time[2:4]), int(time[4:6]) )
        if stamp > latest:
            latest = stamp
            latest_path = path

    return latest_path



def main(args):

    for indir in args.input_dir:
        # get method used for active learning
        if 'random' in indir:
            al_method = 'random'
        else:
            indir_split = indir.split('/')
            al_method = f'{indir_split[-3]} / {indir_split[-2]} / {indir_split[-1]}'

        # get results of each active learning round
        results = []
        for ir in range(args.n_round):
            eval_list = glob(f'{indir}/round_{ir}/eval*')
            with open(get_latest(eval_list), 'r') as f_in: # read results from latest evaluation available in directory
                results.append( json.load(f_in)['metric'][args.metric_name] )

        print(results)
            


                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-round', required=True, type=int, help='Number of iterations for active learning')
    parser.add_argument('--input-dir', nargs='*', required=True, help='path to active learning results for a given set of method, e.g. output/<dataset_nn>/Entropy/sum/batch/')
    parser.add_argument('--metric-name', default='bbox_mAP_50', help='Name of evaluation metric')

    args = parser.parse_args()

    main(args)
