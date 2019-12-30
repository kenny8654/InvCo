import pickle as pk
import os

dataset = {}
dir_file = '/home/r8v10/git/InvCo/dataset/'

for split in ['train','val']:

    dataset[split] = pk.load(open(os.path.join(dir_file,'recipe1m_'+ split +'.pkl'), 'rb')) 

    ids = []
    for i, entry in tqdm(enumerate(dataset[split])):
        if len(entry['images']) == 0:
            continue
        for path in entry['images'][0:5]:
            image_dir = os.path.join(dir_file,'Recipe1M','train',path[0], path[1], path[2], path[3], path)
            if os.path.exists(image_dir):
                ids.append(i)
                break

    for i in ids:
        remove_item = dataset[split][i]
        dataset[split].remove(remove_item)

    with open(os.path.join(dir_file + 'filter_recipe1m_' + split + '.pkl'), 'wb') as f:
        pk.dump(dataset[split], f)