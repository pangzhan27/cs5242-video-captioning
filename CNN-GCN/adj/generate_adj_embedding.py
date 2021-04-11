import pickle as pk
import json, os
import numpy as np

## 1. generate embeddings for objects
# dimension =300
# words = []
# idx = 0
# word2idx = {}
# vectors = []
# with open('glove.6B.{}d.txt'.format(dimension), 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
#
# ds = '../data/'
# obj2idx = json.load(open(os.path.join(ds, 'object1_object2.json'), 'r'))
# objs = list(obj2idx.keys())
#
# embedding= {}
# for o in objs:
#     if o not in word2idx:
#         print(o, 'not found')
#         o_list = o.split('_')
#         feature = 0
#         for o_l in o_list:
#             feature += vectors[word2idx[o_l]]
#         feature /= len(o_list)
#     else:
#         feature = vectors[word2idx[o]]
#     embedding[obj2idx[o]] =  feature
#
# sorted_embedding = []
# for i in range(len(embedding)):
#     sorted_embedding.append(embedding[i])
#
# with open("embedding.pickle", "wb") as fp:
#     pk.dump(np.array(sorted_embedding), fp)
# with open("embedding.pickle", "rb") as fp:
#     a = pk.load(fp)
# a =1
#2. generate adjacency matrix


ds = '../data/'
thre = 0.2
obj2idx = json.load(open(os.path.join(ds, 'object1_object2.json'), 'r'))
img2target = json.load(open(os.path.join(ds, 'training_annotation.json'), 'r'))

adj = np.zeros((len(obj2idx),len(obj2idx)))
for key,value in img2target.items():
    h,r,t = value
    if h!=t:
        adj[h, t] += 1
        adj[t, h] += 1

num = np.sum(adj,axis =0)
num = num[:, np.newaxis]
template = adj >2
adj = adj / (num+ 1e-8)
adj[adj < thre] = 0
adj[adj >= thre] = 1
adj *= template
adj = adj * 0.25 / (adj.sum(0, keepdims=True) + 1e-6)
adj = adj + np.identity(len(obj2idx), np.int)

D = np.power(np.sum(adj, 1), -1)
D = np.diag(D)
norm_adj = np.matmul(adj, D)
with open("adj.pickle", "wb") as fp:
    pk.dump(norm_adj, fp)

