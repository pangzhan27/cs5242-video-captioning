import json, os, pickle
import numpy as np
import csv

#1. Data-driven probability
ds = 'data/'
obj2idx = json.load(open(os.path.join(ds, 'object1_object2.json'), 'r'))
rel2idx = json.load(open(os.path.join(ds, 'relationship.json'), 'r'))
img2target = json.load(open(os.path.join(ds, 'training_annotation.json'), 'r'))

head_tail = {}
head_rel_tail ={}
head,rel = {},{}
for key,value in img2target.items():
    h,r,t = value
    ht = '_'.join([str(i) for i in [h,t]])
    hrt = '_'.join([str(i) for i in [h,r,t]])
    if ht not in head_tail:  head_tail[ht] = 0
    if hrt not in head_rel_tail:  head_rel_tail[hrt] = 0
    if str(h) not in head: head[str(h)] = 0
    if str(r) not in rel: rel[str(r)] = 0
    head_tail[ht] =1
    head[str(h)] =1
    head_rel_tail[hrt] = 1
    rel[str(r)] = 1

#2. NN
with open('result/rel_results.pickle', 'rb') as f:
    nn_result = pickle.load(f)
image_id = list(nn_result.keys())
image_id.remove('idx2rel')
image_id.remove('idx2obj')
image_id.remove('rel2idx')
image_id.remove('obj2idx')
image_id.sort()

def get_pairs(l):
    out = []
    for i in range(len(l)):
        for j in range(len(l)):
            out.append((l[i],l[j]))
    return out

id =0
with open('result/submit.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'label'])
    for key in image_id:
        rel, rel_p = nn_result[key]['rel'][0], nn_result[key]['rel'][1]
        obj, obj_p = nn_result[key]['obj'][0], nn_result[key]['obj'][1]

        all_pairs = get_pairs(obj)
        ## relevent objects
        p_objs = {}
        for ht_pairs in all_pairs:
            obj1, obj2 = ht_pairs
            co_occr = 0.001
            if str(obj1)+ '_' + str(obj2) in head_tail:
                co_occr = 1
            bias = int(obj1 != obj2) * 1.2
            p_obj1 = obj_p[np.where(obj == obj1)[0][0]]
            p_obj2 = obj_p[np.where(obj == obj2)[0][0]]
            p_objs[(obj1, obj2)] = co_occr * (p_obj1+p_obj2 +bias)

        p_objs_sort = sorted(p_objs.items(), key=lambda d: d[1], reverse=True)
        # relevent relation
        for i in range(3):
            obj1, obj2 = p_objs_sort[i][0]
            for j in range(len(rel)):
                r = rel[j]
                if str(obj1) + '_' + str(r) + '_' + str(obj2) in head_rel_tail:
                    rel_p[j] += 1.5

        tempt = sorted(range(len(rel_p)), key=lambda k: rel_p[k], reverse = True)
        rel_list = rel[tempt]
        #rel_list = rel
        ### only use obj_paris:
        list1 = [op[0][0] for op in p_objs_sort]
        list2 = [op[0][1] for op in p_objs_sort]
        obj_list1 = sorted(set(list1), key=list1.index)
        obj_list2 = sorted(set(list2), key=list2.index)
        ### use obj_rel_triplets
        # list1 = [op[0][0] for op in p_hrt_sort]
        # list2 = [op[0][1] for op in p_hrt_sort]
        # list3 = [op[0][2] for op in p_hrt_sort]
        # obj_list1 = sorted(set(list1), key=list1.index)
        # rel_list = sorted(set(list2), key=list2.index)
        # obj_list2 = sorted(set(list3), key=list3.index)

        writer.writerows([[id, ' '.join([str(o1) for o1 in obj_list1[:5]])],
                          [id+1, ' '.join([str(o2) for o2 in rel_list[:5]])],
                          [id+2, ' '.join([str(o3) for o3 in obj_list2[:5]])]])
        id += 3





a=1