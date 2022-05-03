import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    '''
    Get (height+width)/2 for all bboxes and draw distributions to get an idea of object sizes in the dataset.
    Useful to decide of the crop/resize to apply in the data pipeline.
    '''
    with open(args.infile, 'r') as f:
        in_dict = json.load(f)

    size_list = []
    for ann in in_dict['annotations']:
        size_list.append( (ann['bbox'][2]+ann['bbox'][3])/2 )

    size_list = np.array(size_list)
    print(size_list.mean(), size_list.std())

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('# bboxes')
    ax1.set_xlabel('(height+width)/2')
    ax1.hist(size_list, 25, label=f'mean={size_list.mean():.1f}\nstd={size_list.std():.1f}')
    ax1.legend()

    name = args.infile.split('/')[-1].replace('.json', '')
    fig.savefig(f'plots/preprocessing/{name}_bbox_size.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='coco annotation file (.json)')
    args = parser.parse_args()

    os.makedirs('plots/preprocessing/', exist_ok=True)
    main(args)
    
