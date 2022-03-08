import argparse
import matplotlib.pyplot as plt
import json
from glob import glob
from datetime import datetime
import numpy as np
import os



color = { # uncertainty method
    '_' : 'black',
    'Entropy' : 'blue',
    'MarginSampling' : 'red',
    'VarRatio' : 'green'
}

line_style = { # aggregation method
    '_' : 'solid',
    'maximum' : 'solid',
    'average' : 'dotted',
    'sum' : 'dashed',
    }

marker_style = { # selection method
    'random' : '.',
    'maximum' : 's',
    'batch' : 'o',
    'CoreSet' : '^'
    }



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

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('mAP (IoU=0.5)')
    ax1.set_xlabel('# Added images')
    
    for indir in args.input_dir:
        # get method used for active learning
        if indir[-1] == '/':
            indir = indir[:-1]
        indir_split = indir.split('/')

        n_sel = int(indir_split[-1])
        unc_method = indir_split[-4]
        agg_method = indir_split[-3]
        sel_method = indir_split[-2]

        # get results of each active learning round
        results = []
        for ir in range(args.n_round):
            eval_list = glob(f'{indir}/round_{ir}/eval*')
            with open(get_latest(eval_list), 'r') as f_in: # read results from latest evaluation available in directory
                results.append( json.load(f_in)['metric'][args.metric_name] )

        print(results)
        n_sel_array = np.linspace(n_sel, n_sel*args.n_round, args.n_round)

        if 'random' in indir:
            label = sel_method
            unc_method = '_'
            agg_method = '_'
        else:
            label = f'{unc_method} / {agg_method} / {sel_method}'

        ax1.plot(n_sel_array, results, color=color[unc_method], linestyle=line_style[agg_method], marker=marker_style[sel_method], label=label)
        

    ax1.legend()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig(f'plots/active_learning_{args.n_round}_rounds_{n_sel}_sel.pdf')
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-round', required=True, type=int, help='Number of iterations for active learning')
    parser.add_argument('--input-dir', nargs='*', required=True, help='path to active learning results for a given set of method, e.g. output/<dataset_nn>/Entropy/sum/batch/<n_sel>')
    parser.add_argument('--metric-name', default='bbox_mAP_50', help='Name of evaluation metric')

    args = parser.parse_args()

    main(args)
