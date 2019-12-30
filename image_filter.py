import pickle
import os
from tqdm import tqdm

dir_file = '/home/r8v10/git/InvCo/dataset'

print('Loading Test data')
with open(os.path.join(dir_file,'recipe1m_test.pkl'), 'rb') as f:
    dataset = pickle.load(f)

print('Detect imgs in Test_data with not found')
index = []
new_data = []
for i, item in tqdm(enumerate(dataset)):
    gotit = False
    if len(item['images']) == 0:
        continue
    for img_id in item['images'][0:5]:
        img_dir = os.path.join(dir_file,'Recipe1M','test',img_id[0],img_id[1],img_id[2],img_id[3],img_id)

        if os.path.exists(img_dir):
            gotit = True
            break

    if gotit:
        new_data.append(item)

with open('/home/r8v10/git/InvCo/dataset/filter_test.pkl', 'wb') as f:
    pickle.dump(new_data,f)
print('Saved Test data')

