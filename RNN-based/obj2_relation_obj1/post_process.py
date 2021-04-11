import json, os, pickle
import numpy as np
import csv


with open('result/rel_results.pickle', 'rb') as f:
    nn_result = pickle.load(f)
image_id = list(nn_result.keys())
image_id.remove('idx2rel')
image_id.remove('idx2obj')
image_id.remove('rel2idx')
image_id.remove('obj2idx')
image_id.sort()


id =0
with open('result/submit.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'label'])
    for key in image_id:
        rel, rel_p = nn_result[key]['rel'][0], nn_result[key]['rel'][1]
        obj1, obj1_p = nn_result[key]['obj1'][0], nn_result[key]['obj1'][1]
        obj2, obj2_p = nn_result[key]['obj2'][0], nn_result[key]['obj2'][1]
        writer.writerows([[id, ' '.join([str(o1) for o1 in obj1[:5]])],
                          [id+1, ' '.join([str(o2) for o2 in rel[:5]])],
                          [id+2, ' '.join([str(o3) for o3 in obj2[:5]])]])
        id += 3





a=1