import pickle
import os
from tqdm import tqdm

dir_file = '/home/r8v10/git/InvCo/dataset'
print('Loading Val data')
with open(os.path.join(dir_file,'recipe1m_val.pkl'), 'rb') as f:
    data = pickle.load(f)

print('Remove data with images!')
dataset = []

for i, item in tqdm(enumerate(data)):
    exist_data = False
    images_list = []
    for img_id in item['images']:
        img_dir = os.path.join(dir_file,'Recipe1M','val',img_id[0],img_id[1],img_id[2],img_id[3],img_id)

        if os.path.exists(img_dir):
            images_list.append(img_id)
            exist_data = True

    if exist_data:
        ingrs_list = item['ingredients']
        unit_list = item['ingrs_unit']
        title = item['title']
        newentry = {'id': item['id'], 'ingredients': ingrs_list,
                    'ingrs_unit': unit_list,
                    'images': images_list, 'title': title}
        dataset.append(newentry)

print('Finished')

with open('/home/r8v10/git/InvCo/dataset/1M_val.pkl', 'wb') as f:
    pickle.dump(dataset,f)

print('Saved Train data')
