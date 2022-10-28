import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def main(args):
    '''
    Get (height+width)/2 for all bboxes and draw distributions to get an idea of object sizes in the dataset.
    Also get number of small, medium and large objects, according to bboxes area:
    - small: area < 32^2
    - medium: 32^2 < area < 96^2
    - large: area > 96^2
    Useful to decide of the crop/resize to apply in the data pipeline.
    '''
    with open(args.infile, 'r') as f:
        in_dict = json.load(f)

    n_class = len(in_dict['categories'])
        
    size_list = [[] for _ in range(n_class)]
    
    n_small = [0 for _ in range(n_class)]
    n_medium = [0 for _ in range(n_class)]
    n_large = [0 for _ in range(n_class)]
    for ann in in_dict['annotations']:
        cat_id = ann['category_id']
        size_list[cat_id].append( (ann['bbox'][2]+ann['bbox'][3])/2 )
        area = ann['bbox'][2]*ann['bbox'][3]
        if area < 32*32:
            n_small[cat_id] += 1
        elif area > 96*96:
            n_large[cat_id] += 1
        else:
            n_medium[cat_id] += 1

    for cat in in_dict['categories']:
        print('')
        print(cat['name'])
        print(f'small objects: {n_small[cat["id"]]}')
        print(f'medium objects: {n_medium[cat["id"]]}')
        print(f'large objects: {n_large[cat["id"]]}')
        print('')

        size_list_cat = np.array(size_list[cat["id"]])
        label = f'mean={size_list_cat.mean():.1f}\nstd={size_list_cat.std():.1f}'
        print('(height+width)/2')
        print(label)
        print('')

        fig, ax1 = plt.subplots()
        ax1.set_ylabel('# bboxes')
        ax1.set_xlabel('(height+width)/2')
        ax1.hist(size_list_cat, 25, label=label)
        ax1.legend()

        name = args.infile.split('/')[-1].replace('.json', '')
        fig.savefig(f'plots/preprocessing/{name}_{cat["name"]}_bbox_size.pdf')
        print(f'Saved plots/preprocessing/{name}_{cat["name"]}_bbox_size.pdf\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, help='coco annotation file (.json)')
    args = parser.parse_args()

    os.makedirs('plots/preprocessing/', exist_ok=True)
    main(args)
    
