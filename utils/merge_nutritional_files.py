import pickle

i1 = pickle.load(open('../dataset/nutritional_training_1.pickle', 'rb'))
i2 = pickle.load(open('../dataset/nutritional_training_2.pickle', 'rb'))
i3 = pickle.load(open('../dataset/nutritional_training_3.pickle', 'rb'))
i4 = pickle.load(open('../dataset/nutritional_training_4.pickle', 'rb'))

print(len(i1),len(i2),len(i3),len(i4))

merged_list = []

merged_list.extend(i1)
merged_list.extend(i2)
merged_list.extend(i3)
merged_list.extend(i4)

print(len(merged_list))

file = open('../dataset/nutritional_training_all.pickle', 'wb')
pickle.dump(merged_list, file)
file.close()
