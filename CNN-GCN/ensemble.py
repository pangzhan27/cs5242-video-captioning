import json, os, pickle
import numpy as np
import csv
from copy import deepcopy
from PIL import Image, ImageDraw


def get_pairs(l):
    out = []
    for i in range(len(l)):
        for j in range(len(l)):
            out.append((l[i], l[j]))
    return out

def get_prob(ranking, prob, pred):
    prob, pred = list(prob), list(pred)
    out_prob = [ prob[pred.index(i)]/3 for i in ranking]
    for i in range(len(out_prob) -1):
        if out_prob[i+1] > out_prob[i]:
            out_prob[i + 1] = 0.9 * out_prob[i]
    return out_prob

def ensemble(pred, prob):
    set_pred = list(set(pred))
    out_dic = []
    for sp in set_pred:
        sp_prob = 0
        k = 0
        for i in range(len(pred)):
            if pred[i] == sp:
                sp_prob += prob[i]
                k+=1
        out_dic.append([sp, sp_prob/k])
    out_sort = sorted(out_dic, key=lambda d: d[1], reverse=True)
    return out_sort

def index2class(input):
    out = {}
    for key, value in input.items():
        out[value] = key
    return out

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


epochs = [ '35', '45'] #'30',
collect = {}
v_id = None
for e in epochs:
    collect[e] ={}
    with open('result/' + e + '/rel_results.pickle', 'rb') as f:
        nn_result = pickle.load(f)
    image_id = list(nn_result.keys())
    image_id.remove('idx2rel')
    image_id.remove('idx2obj')
    image_id.remove('rel2idx')
    image_id.remove('obj2idx')
    image_id.sort()
    v_id = image_id

    for key in image_id:
        collect[e][key] = {}
        rel, rel_p = nn_result[key]['rel'][0], nn_result[key]['rel'][1]
        obj, obj_p = nn_result[key]['obj'][0], nn_result[key]['obj'][1]
        obj_org, rel_org = deepcopy(obj_p), deepcopy(rel_p)

        all_pairs = get_pairs(obj)
        ## relevent objects
        p_objs = {}
        for ht_pairs in all_pairs:
            obj1, obj2 = ht_pairs
            co_occr = 0.001
            if str(obj1)+ '_' + str(obj2) in head_tail:
                co_occr = 1
            bias = 0
            if (obj1!=obj2) and (obj1 in obj[:2]) and (obj2 in obj[:2]):
                bias = 1.5
            p_obj1 = obj_p[np.where(obj == obj1)[0][0]]
            p_obj2 = obj_p[np.where(obj == obj2)[0][0]]
            p_objs[(obj1, obj2)] = co_occr * (p_obj1+p_obj2 +bias)

        if key == '000010':
            a=1
        p_objs_sort = sorted(p_objs.items(), key=lambda d: d[1], reverse=True)
        # relevent relation
        for i in range(3):
            obj1, obj2 = p_objs_sort[i][0]
            for j in range(len(rel)):
                r = rel[j]
                if str(obj1) + '_' + str(r) + '_' + str(obj2) in head_rel_tail:
                    rel_p[j] += 0.6

        tempt = sorted(range(len(rel_p)), key=lambda k: rel_p[k], reverse = True)
        rel_list = rel[tempt]
        list1 = [op[0][0] for op in p_objs_sort]
        list2 = [op[0][1] for op in p_objs_sort]
        obj_list1 = sorted(set(list1), key=list1.index)
        obj_list2 = sorted(set(list2), key=list2.index)

        obj_prob1 = get_prob(obj_list1, obj_org, obj)
        obj_prob2 = get_prob(obj_list2, obj_org, obj)
        rel_prob = get_prob(rel_list, rel_org, rel)
        collect[e][key]['rel'] = (rel_list, rel_prob)
        collect[e][key]['obj1'] = (obj_list1, obj_prob1)
        collect[e][key]['obj2'] = (obj_list2, obj_prob2)


ds = 'data/'
obj2idx = json.load(open(os.path.join(ds, 'object1_object2.json'), 'r'))
rel2idx = json.load(open(os.path.join(ds, 'relationship.json'), 'r'))
img2target = json.load(open(os.path.join(ds, 'training_annotation.json'), 'r'))

idx2rel = index2class(rel2idx)
idx2obj = index2class(obj2idx)

id = 0
ensem_out = {}
with open('result/submit.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'label'])
    for key in v_id:
        rel, rel_prob, obj1, obj1_prob, obj2, obj2_prob = [],[],[],[],[],[]
        for e in epochs:
            rel.extend(collect[e][key]['rel'][0])
            rel_prob.extend(collect[e][key]['rel'][1])
            obj1.extend(collect[e][key]['obj1'][0])
            obj1_prob.extend(collect[e][key]['obj1'][1])
            obj2.extend(collect[e][key]['obj2'][0])
            obj2_prob.extend(collect[e][key]['obj2'][1])
        obj1_f = ensemble(obj1, obj1_prob)
        obj2_f = ensemble(obj2, obj2_prob)
        rel_f = ensemble(rel, rel_prob)
        ensem_out[key] = (obj1_f, rel_f, obj2_f)

        if key == '000029':
            a=1

        writer.writerows([[id, ' '.join([str(int(o1)) for o1 in np.array(obj1_f)[:5,0]])],
                          [id + 1, ' '.join([str(int(o2)) for o2 in np.array(rel_f)[:5,0]])],
                          [id + 2, ' '.join([str(int(o3)) for o3 in np.array(obj2_f)[:5,0]])]])
        id += 3

        temp_path = os.path.join(ds, 'test', key)
        img_ids = sorted(list(os.listdir(temp_path)))[14]
        img = Image.open(os.path.join(temp_path, img_ids)).convert('RGB')
        img = img.resize((224, 224))
        draw = ImageDraw.Draw(img)
        for j in range(5):
            text = '({}-{}-{}) {} {} {}'.format(obj1_f[j][0], rel_f[j][0], obj2_f[j][0],
                                                idx2obj[obj1_f[j][0]], idx2rel[rel_f[j][0]], idx2obj[obj2_f[j][0]])
            draw.text((10, 10 + j * 15), text, fill="#ff0000", direction=None)

        img.save(os.path.join('result', key + '.jpg'))

with open('result/ensemble.pk', 'wb') as f:
    pickle.dump(ensem_out, f)
a =1

