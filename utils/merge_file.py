import json
import os
import numpy as np
import pandas as pd

# Local import
from invco import DATASET_DIR

def openfile(dir_file, name):
    input_data = os.path.join(dir_file, name)
    

    with open(input_data, 'r') as reader:
        json_data = json.loads(reader.read())

    return json_data

#Get all item form layer2
def get_info(file):
    id_list = [] 
    numImage = []
    imageID = []
    url = []

    for index in range(len(file)):
        id_list.append(file[index]['id'])
        numImage.append(len(file[index]['images']))
        Img_tmp = ''
        url_tmp = ''
        for ImgID_index in range(len(file[index]['images'])):
            Img_tmp += file[index]['images'][ImgID_index]['id']
            url_tmp += file[index]['images'][ImgID_index]['url']
            if ImgID_index < len(file[index]['images'])-1:
                Img_tmp += '|'
                url_tmp += '|'
        imageID.append(Img_tmp)
        url.append(url_tmp)

    return id_list, numImage, imageID, url

#Get item from layer1
def get_Item(file, item):
    item_list = []
    for index in range(len(file)):
        item_list.append(file[index][item])

    return item_list

#Merge two file and save as CSV
def Mergedata(First_data, Second_data, dir_file, output):
    
    res_csv = pd.merge(First_data, Second_data, on='recipe_ID')
    
    dir_file = os.path.join(dir_file, output)
    res_csv.to_csv(dir_file,index=False)

def main():
    dir_file = F'{DATASET_DIR}/recipe1M_layers'
    
    layer1 = openfile(dir_file, 'layer1.json')
    layer2 = openfile(dir_file, 'layer2.json')
    
    recipe_ID, numofImage, img_ID, url = get_info(layer2)
    
    receipe = get_Item(layer1, 'id')
    title = get_Item(layer1, 'title')
    partition = get_Item(layer1, 'partition')

    layer1_tuples = list(zip(recipe_ID, numofImage, img_ID, url))
    layer2_tuples = list(zip(receipe,title,partition))
    csv_layer1 = pd.DataFrame(layer1_tuples, columns =['recipe_ID','numofImage','img_ID','url'])
    csv_layer2 = pd.DataFrame(layer2_tuples, columns =['recipe_ID','partition','title'])

    Mergedata(csv_layer1, csv_layer2, dir_file, output= 'recipeid.csv')

if __name__ =='__main__':
    main()