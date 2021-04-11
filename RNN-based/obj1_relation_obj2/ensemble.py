import json, os, pickle
import numpy as np
import csv

def read_result(file):
    with open(file, 'rb') as f:
        nn_result = pickle.load(f)
    image_id = list(nn_result.keys())
    image_id.remove('idx2rel')
    image_id.remove('idx2obj')
    image_id.remove('rel2idx')
    image_id.remove('obj2idx')
    image_id.sort()
    return image_id, nn_result

def read_ensemble(file):
    with open(file, 'rb') as f:
        nn_result = pickle.load(f)
    return nn_result

def gen_ensemble():
    image_id, oro = read_result('result/15/0/rel_results.pickle')
    _, oro_r = read_result('../lstm_oro_reverse/result/15/0/rel_results.pickle')
    _, oor = read_result('../lstm_oor/result/25/0/rel_results.pickle')
    id = 0
    result_save = {}
    with open('ensemble/submit.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'label'])
        for key in image_id:
            oro_score, oro_path, oro_prob = np.exp(np.array(oro[key]['score'])), oro[key]['paths'], oro[key]['prob']
            oro_r_score, oro_r_path, oro_r_prob = np.exp(np.array(oro_r[key]['score'])), oro_r[key]['paths'], \
                                                  oro_r[key]['prob']
            oor_score, oor_path, oor_prob = np.exp(np.array(oor[key]['score'])), oor[key]['paths'], oor[key]['prob']
            oro_str = ['_'.join([str(j) for j in i]) for i in oro_path]
            oro_str_r = ['_'.join([str(j) for j in i]) for i in oro_r_path]
            oor_str = ['_'.join([str(j) for j in i]) for i in oor_path]

            final_output = []
            all_names = set(oro_str) | set(oro_str_r) | set(oor_str)
            oro_set = (oro_str, oro_score, oro_path, oro_prob)
            oro_set_r = (oro_str_r, oro_r_score, oro_r_path, oro_r_prob)
            oor_set = (oor_str, oor_score, oor_path, oor_prob)
            for name in all_names:
                final_scores, final_paths, final_path_ps = [], [], []
                for rlt in [oro_set, oro_set_r, oor_set]:
                    if name in rlt[0]:
                        index = rlt[0].index(name)
                        final_scores.append(rlt[1][index])
                        final_paths.append(rlt[2][index])
                        final_path_ps.append(rlt[3][index])
                final_score = np.sum(np.array(final_scores))
                final_path = final_paths[0]
                final_path_p = np.mean(np.array(final_path_ps), axis=0)
                final_output.append([final_score, final_path, list(final_path_p)])
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)
            assert len(final_output) == len(all_names)
            result_save[key] = final_output

            scores, paths, paths_p = [], [], []
            for i in range(len(final_output)):
                scores.append(final_output[i][0])
                paths.append(final_output[i][1])
                paths_p.append(final_output[i][2])
            paths, paths_p = np.array(paths), np.array(paths_p)
            predicted_obj1 = sorted(set(list(paths[:, 0])), key=list(paths[:, 0]).index)
            predicted_rel = sorted(set(list(paths[:, 1])), key=list(paths[:, 1]).index)
            predicted_obj2 = sorted(set(list(paths[:, 2])), key=list(paths[:, 2]).index)

            writer.writerows([[id, ' '.join([str(o1) for o1 in predicted_obj1[:5]])],
                              [id + 1, ' '.join([str(o2) for o2 in predicted_rel[:5]])],
                              [id + 2, ' '.join([str(o3) for o3 in predicted_obj2[:5]])]])
            id += 3

    with open("ensemble/rel_results.pickle", "wb") as fp:
        pickle.dump(result_save, fp)

def gen_batch_ensemble():
    result0 = read_ensemble('ensemble/0/rel_results.pickle')
    result1 = read_ensemble('ensemble/1/rel_results.pickle')
    image_ids = sorted(result0.keys())
    id = 0
    result_save = {}
    with open('ensemble/submit.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'label'])
        for key in image_ids:
            path_0 = ['_'.join([str(j) for j in i[1]]) for i in result0[key]]
            path_1 = ['_'.join([str(j) for j in i[1]]) for i in result1[key]]
            all_names = set(path_0) | set(path_1)
            final_output = []
            set0 = (path_0, result0[key])
            set1 = (path_1, result1[key])
            for name in all_names:
                final_scores, final_paths, final_path_ps = [], [], []
                for rlt in [set0, set1]:
                    if name in rlt[0]:
                        index = rlt[0].index(name)
                        final_scores.append(rlt[1][index][0])
                        final_paths.append(rlt[1][index][1])
                        final_path_ps.append(rlt[1][index][2])
                final_score = np.sum(np.array(final_scores))
                final_path = final_paths[0]
                final_path_p = np.mean(np.array(final_path_ps), axis=0)
                final_output.append([final_score, final_path, list(final_path_p)])
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)
            result_save[key] = final_output

            scores, paths, paths_p = [], [], []
            for i in range(len(final_output)):
                scores.append(final_output[i][0])
                paths.append(final_output[i][1])
                paths_p.append(final_output[i][2])
            paths, paths_p = np.array(paths), np.array(paths_p)
            predicted_obj1 = sorted(set(list(paths[:, 0])), key=list(paths[:, 0]).index)
            predicted_rel = sorted(set(list(paths[:, 1])), key=list(paths[:, 1]).index)
            predicted_obj2 = sorted(set(list(paths[:, 2])), key=list(paths[:, 2]).index)

            writer.writerows([[id, ' '.join([str(o1) for o1 in predicted_obj1[:5]])],
                              [id + 1, ' '.join([str(o2) for o2 in predicted_rel[:5]])],
                              [id + 2, ' '.join([str(o3) for o3 in predicted_obj2[:5]])]])
            id += 3
    with open("ensemble/rel_results.pickle", "wb") as fp:
        pickle.dump(result_save, fp)


def gen_color_ensemble():
    result0 = read_ensemble('ensemble/0/rel_results.pickle')
    result1 = read_ensemble('ensemble/1/rel_results.pickle')
    result1_ = read_ensemble('ensemble/-1/rel_results.pickle')
    image_ids = sorted(result0.keys())
    id = 0
    result_save = {}
    with open('ensemble/submit.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'label'])
        for key in image_ids:
            path_0 = ['_'.join([str(j) for j in i[1]]) for i in result0[key]]
            path_1 = ['_'.join([str(j) for j in i[1]]) for i in result1[key]]
            path_1_ = ['_'.join([str(j) for j in i[1]]) for i in result1_[key]]
            all_names = set(path_0) | set(path_1) | set(path_1_)
            final_output = []
            set0 = (path_0, result0[key])
            set1 = (path_1, result1[key])
            set1_ = (path_1_, result1_[key])
            for name in all_names:
                final_scores, final_paths, final_path_ps = [], [], []
                for rlt in [set0, set1, set1_]:
                    if name in rlt[0]:
                        index = rlt[0].index(name)
                        final_scores.append(rlt[1][index][0])
                        final_paths.append(rlt[1][index][1])
                        final_path_ps.append(rlt[1][index][2])
                final_score = np.sum(np.array(final_scores))
                final_path = final_paths[0]
                final_path_p = np.mean(np.array(final_path_ps), axis=0)
                final_output.append([final_score, final_path, list(final_path_p)])
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)
            result_save[key] = final_output

            scores, paths, paths_p = [], [], []
            for i in range(len(final_output)):
                scores.append(final_output[i][0])
                paths.append(final_output[i][1])
                paths_p.append(final_output[i][2])
            paths, paths_p = np.array(paths), np.array(paths_p)
            predicted_obj1 = sorted(set(list(paths[:, 0])), key=list(paths[:, 0]).index)
            predicted_rel = sorted(set(list(paths[:, 1])), key=list(paths[:, 1]).index)
            predicted_obj2 = sorted(set(list(paths[:, 2])), key=list(paths[:, 2]).index)

            writer.writerows([[id, ' '.join([str(o1) for o1 in predicted_obj1[:5]])],
                              [id + 1, ' '.join([str(o2) for o2 in predicted_rel[:5]])],
                              [id + 2, ' '.join([str(o3) for o3 in predicted_obj2[:5]])]])
            id += 3
    with open("ensemble/rel_results.pickle", "wb") as fp:
        pickle.dump(result_save, fp)




if __name__ == '__main__':
    gen_ensemble()
    #gen_color_ensemble() useless