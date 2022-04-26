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
    'VarRatio' : 'orange',
    'LossPrediction' : 'green'
}

line_style = { # aggregation method
    '_' : 'solid',
    'none' : 'solid',
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

metric_label = {
    'bbox_mAP' : 'mAP',
    'bbox_mAP_50' : 'mAP (IoU=0.5)',
    'bbox_mAP_75' : 'mAP (IoU=0.75)'
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
    ax1.set_ylabel(metric_label[args.metric_name])
    ax1.set_xlabel('# Added images')
    
    for indir in args.input_dir:
        # get method used for active learning
        if indir[-1] == '/':
            indir = indir[:-1]
        indir_split = indir.split('/')

        n_sel = int(indir_split[-1])
        sel_method = indir_split[-2]

        if 'random' in indir:
            unc_method = '_'
            agg_method = '_'
            init_train_path = '/'.join(indir_split[:-2])
            label = sel_method
        else:
            unc_method = indir_split[-4]
            agg_method = indir_split[-3]
            init_train_path = '/'.join(indir_split[:-4])
            label = f'{unc_method} / {agg_method} / {sel_method}'
            

        results = []
        results_std = []

        #get results for initial training
        eval_list_init = glob(f'{init_train_path}/init_training*/eval*')
        res_init = []
        for ev_init in eval_list_init:
            with open(ev_init, 'r') as f_in: # read results from latest evaluation available in directory
                res_init.append( json.load(f_in)['metric'][args.metric_name] )
        res_init = np.array(res_init)
        results.append(res_init.mean())
        results_std.append(res_init.std())

        # get results of each active learning round
        for ir in range(args.n_round):
            eval_list = glob(f'{indir}/round_{ir}/eval*')
            if args.latest:
                with open(get_latest(eval_list), 'r') as f_in: # read results from latest evaluation available in directory
                    results.append( json.load(f_in)['metric'][args.metric_name] )
            else:
                res = []
                for ev in eval_list:
                    with open(ev, 'r') as f_in: # read results from latest evaluation available in directory
                        res.append( json.load(f_in)['metric'][args.metric_name] )
                res = np.array(res)
                results.append(res.mean())
                results_std.append(res.std())
                    
        print(results)
        n_sel_array = np.linspace(0, n_sel*args.n_round, args.n_round+1)

        results = np.array(results)
        results_std = np.array(results_std)
            
        ax1.plot(n_sel_array, results, color=color[unc_method], linestyle=line_style[agg_method], marker=marker_style[sel_method], label=label)
        if args.plot_std:
            ax1.plot(n_sel_array, results+results_std, color=color[unc_method], linestyle=line_style[agg_method], marker=marker_style[sel_method], linewidth=0.5, markersize=2, label='$\pm$std')
            ax1.plot(n_sel_array, results-results_std, color=color[unc_method], linestyle=line_style[agg_method], marker=marker_style[sel_method], linewidth=0.5, markersize=2)
        

    ax1.legend()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig(f'plots/active_learning_{args.n_round}_rounds_{n_sel}_sel_{args.metric_name}.pdf')
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-round', required=True, type=int, help='Number of iterations for active learning')
    parser.add_argument('--input-dir', nargs='*', required=True, help='path to active learning results for a given set of method, e.g. output/<dataset_nn>/Entropy/sum/batch/<n_sel>')
    parser.add_argument('--metric-name', default='bbox_mAP_50', help='Name of evaluation metric')
    parser.add_argument('--latest', action='store_true', help='Use latest evaluation instead of mean of all evaluations')
    parser.add_argument('--plot-std', action='store_true', help='Plot standard deviation')

    args = parser.parse_args()

    main(args)
