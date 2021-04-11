import numpy as np
import pickle as pk
import json, os, csv
from PIL import Image, ImageDraw

def index2class(input):
    out = {}
    for key, value in input.items():
        out[value] = key
    return out

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

ds = 'data/'
obj2idx = json.load(open(os.path.join(ds, 'object1_object2.json'), 'r'))
rel2idx = json.load(open(os.path.join(ds, 'relationship.json'), 'r'))
img2target = json.load(open(os.path.join(ds, 'training_annotation.json'), 'r'))

idx2rel = index2class(rel2idx)
idx2obj = index2class(obj2idx)

with open('features/train.pk', 'rb') as f:
    train_dict = pk.load(f)
with open('features/test.pk', 'rb') as f:
    test_dict = pk.load(f)

train_k = sorted(train_dict.keys())
test_k = sorted(test_dict.keys())

train_feature = np.array([list(sigmoid(np.array(train_dict[i]))) for i in train_k])
test_feature = np.array([list(sigmoid(np.array(test_dict[i]))) for i in test_k])

### remove noise
train_feature[train_feature<0.2] = 0
test_feature[test_feature<0.2] = 0

#### Eu distance
# x1 = np.sum(np.power(test_feature,2),axis=-1, keepdims=True)
# x2 = np.sum(np.power(train_feature,2),axis=-1, keepdims=True)
# x3 = -2 * np.matmul(test_feature, train_feature.T)
# sim = np.power(x3 + x1 + x2.T, 0.5)

### cos distance
x1 = test_feature #/ np.linalg.norm(test_feature, ord=2, axis=-1, keepdims=True)
x2 = train_feature #/ np.linalg.norm(train_feature, ord=2, axis=-1, keepdims=True)
sim = np.matmul(x1, x2.T)
top_k_idx=np.argsort(sim, axis=-1)[:,::-1]

img_list = sorted(list(os.listdir(os.path.join(ds, 'test'))))
id = 0
with open('features/submit.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'label'])
    for i in range(len(img_list)):
        video_id = img_list[i]
        ### get output
        topk = top_k_idx[test_k.index(video_id)]
        result = np.array([img2target[train_k[j]] for j in topk])
        obj1, rel, obj2 = list(result[:, 0]), list(result[:, 1]), list(result[:, 2])
        obj_list1 = sorted(set(obj1), key=obj1.index)[:5]
        rel_list = sorted(set(rel), key=rel.index)[:5]
        obj_list2 = sorted(set(obj2), key=obj2.index)[:5]

        writer.writerows([[id, ' '.join([str(o1) for o1 in obj_list1])],
                          [id + 1, ' '.join([str(o2) for o2 in rel_list])],
                          [id + 2, ' '.join([str(o3) for o3 in obj_list2])]])
        id += 3
        ### plot image

        temp_path = os.path.join(ds, 'test', video_id)
        img_ids = sorted(list(os.listdir(temp_path)))[14]
        img = Image.open(os.path.join(temp_path, img_ids)).convert('RGB')
        img = img.resize((224, 224))
        draw = ImageDraw.Draw(img)
        for j in range(5):
            text = '({}-{}-{}) {} {} {}'.format(obj_list1[j], rel_list[j], obj_list2[j],
                                                idx2obj[obj_list1[j]], idx2rel[rel_list[j]], idx2obj[obj_list2[j]])
            draw.text((10, 10 + j * 15), text, fill="#ff0000", direction=None)

        img.save(os.path.join('features', video_id + '.jpg'))

a=1