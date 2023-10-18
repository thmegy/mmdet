import json
import argparse
import os
import glob
import tqdm

def main(args):
    '''
    Check if there are duplicated images in dataset, then move them to subfolder
    '''
    images = glob.glob(f'{args.indir}/*.{args.format}')

    duplicate_list = []
    for im in tqdm.tqdm(images):
        # find images with same names
        split_name = im.split('/')
        file_name = '_'.join(split_name[-1].split('_')[1:])
        path = '/'.join(split_name[:-1])
        dup = glob.glob(f'{path}/*{file_name}')
        if len(dup)>1:
            duplicate_list.append(dup)
            for d in dup:
                images.remove(d)

    # check if images are really duplicates by comparing them
    for i, duplicate in enumerate(tqdm.tqdm(duplicate_list)):
        imhash = os.popen(f'identify -quiet -format "%#" "{duplicate[0]}"').read()
        for dup in duplicate[1:]:
            imhash2 = os.popen(f'identify -quiet -format "%#" "{dup}"').read()
            if imhash != imhash2:
                print(dup)
                duplicate_list[i].remove(dup)

#    with open('duplicate.json', 'w') as f:
#        json.dump(duplicate_list, f)

    # check which duplicate has the most annotations
    remove_duplicate_list = []
    for i, duplicate in enumerate(tqdm.tqdm(duplicate_list)):
        max_len = 0
        for dup in duplicate:
            with open(dup.replace(args.format, 'txt'), 'r') as f:
                length = len(f.readlines())
                if length > max_len:
                    max_len = length
                    keep_duplicate = dup
        if max_len > 0:
            duplicate.remove(keep_duplicate)
            remove_duplicate_list += duplicate
                
    #with open('duplicate_keep.json', 'w') as f:
        #json.dump(keep_duplicate_list, f)

    # move duplicates to other repository
    for duplicate in remove_duplicate_list:
        os.system(f'mv {duplicate} {args.outdir}')
        os.system(f'mv {duplicate.replace(args.format, "txt")} {args.outdir}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, help='path to images')
    parser.add_argument('--outdir', required=True, help='move duplicates to this path')
    parser.add_argument('--format', default='jpg', help='image format')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    main(args)
    
