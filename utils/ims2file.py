import pickle
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
import lmdb
from torchvision import transforms


MAX_SIZE = 1e12


def load_and_resize(root, path, imscale):

    transf_list = []
    transf_list.append(transforms.Resize(imscale))
    transf_list.append(transforms.CenterCrop(imscale))
    transform = transforms.Compose(transf_list)

    img = Image.open(os.path.join(root, path[0], path[1], path[2], path[3], path)).convert('RGB')
    img = transform(img)

    return img


def main(args):

    parts = {}
    datasets = {}
    
    for split in ['test']:
        datasets[split] = pickle.load(open(os.path.join(args.save_dir, args.suff + 'recipe1m_' + split + '.pkl'), 'rb'))

        parts[split] = lmdb.open(os.path.join(args.save_dir, 'lmdb_'+split), map_size=int(MAX_SIZE))

        # with parts[split].begin() as txn:
        #     present_entries = [key for key, _ in txn.cursor()]
        # j = 0
        
        for entry in tqdm(datasets[split]):
            impaths = entry['images'][0:5]

            for p in impaths:
                im_dir = os.path.join(args.root,split,p[0],p[1],p[2],p[3],p)
                if os.path.exists(im_dir):
                    im = load_and_resize(os.path.join(args.root, split), p, args.imscale)
                    im = np.array(im).astype(np.uint8)
                    with parts[split].begin(write=True) as txn:
                        txn.put(p.encode(), im)

def test(args):

    imname2pos = pickle.load(open(os.path.join(args.save_dir, 'imname2pos.pkl'), 'rb'))
    paths = imname2pos['val']

    for k, v in paths.items():
        path = k
        break
    image_file = lmdb.open(os.path.join(args.save_dir, 'lmdb_' + 'val'), max_readers=1, readonly=True,
                           lock=False, readahead=False, meminit=False)
    with image_file.begin(write=False) as txn:
        image = txn.get(path.encode())
        image = np.fromstring(image, dtype=np.uint8)
        image = np.reshape(image, (args.imscale, args.imscale, 3))
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    print (np.shape(image))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/r8v10/git/InvCo/dataset/Recipe1M',
                        help='path to the recipe1m dataset')
    parser.add_argument('--save_dir', type=str, default='/home/r8v10/git/InvCo/dataset',
                        help='path where the lmdbs will be saved')
    parser.add_argument('--imscale', type=int, default=256,
                        help='size of images (will be rescaled and center cropped)')
    parser.add_argument('--maxnumims', type=int, default=5,
                        help='maximum number of images to allow for each sample')
    parser.add_argument('--suff', type=str, default='',
                        help='id of the vocabulary to use')
    parser.add_argument('--test_only', dest='test_only', action='store_true')
    parser.set_defaults(test_only=False)
    args = parser.parse_args()

    if not args.test_only:
        main(args)
    #test(args)
