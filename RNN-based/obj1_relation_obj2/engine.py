import torch
import numpy as np
import torch.nn.functional as F
import os, time, pickle
from PIL import Image, ImageDraw, ImageFont


class Trainer(object):
    def __init__(self, optimizer, criterion, args, log):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.criterion_obj = criterion[0]
        self.criterion_rel = criterion[1]
        self.args = args
        self.log = log

    def train(self, model, train_loader):
        model.to(self.device)
        for epoch in range(self.args.epochs):
            model.train()
            self.adjust_learning_rate(self.optimizer, epoch)
            epoch_loss, obj1_loss, obj2_loss, rel_loss = 0, 0, 0, 0
            correct, total, c_obj1, c_obj2, c_rel = 0, 0, 0, 0, 0
            end = time.time()
            for num_batches, (input, target) in enumerate(train_loader):
                print(c_rel,total)
                images, id = input
                obj_target1, rel_target, obj_target2 = target
                images = images.to(self.device)
                obj_target1, rel_target, obj_target2 = obj_target1.to(self.device), rel_target.to(self.device), obj_target2.to(self.device)
                self.optimizer.zero_grad()
                images = images.view(images.shape[0]*images.shape[1], images.shape[2], images.shape[3],images.shape[4])
                (obj1_pred, rel_pred, obj2_pred), _ = model(images, (obj_target1, rel_target, obj_target2), True)

                obj1_l = self.criterion_obj(obj1_pred, obj_target1)
                rel_l = self.criterion_rel(rel_pred , rel_target)
                obj2_l = self.criterion_obj(obj2_pred, obj_target2)

                loss = obj1_l + 1.5 * rel_l + obj2_l
                epoch_loss += loss.item()
                obj1_loss += obj1_l.item()
                rel_loss += rel_l.item()
                obj2_loss += obj2_l.item()

                loss.backward()
                self.optimizer.step()
                _, predicted_rel = torch.max(rel_pred.data, 1)
                correct_rel = (predicted_rel.view(-1) == rel_target.view(-1)).int()

                _, predicted_obj1 = torch.max(obj1_pred.data, 1)
                _, predicted_obj2 = torch.max(obj2_pred.data, 1)
                correct_obj1 = (predicted_obj1  == obj_target1).int()
                correct_obj2 = (predicted_obj2 == obj_target2).int()
                c_obj1 += correct_obj1.sum().item()
                c_obj2 += correct_obj2.sum().item()
                c_rel += correct_rel.sum().item()
                correct += (correct_obj1 * correct_rel * correct_obj2).sum().item()
                total += len(correct_rel)

            self.log.logger.info("[epoch %d]: epoch loss = %.3f, obj1 loss = %.3f, rel loss = %.3f, obj2 loss = %.3f, "
                                 "acc = %.3f, acc_obj1 = %.3f, acc_rel = %.3f, acc_obj2 = %.3f, time = %.1f" %
                  (epoch + 1, epoch_loss / (num_batches+1), obj1_loss / (num_batches+1), rel_loss / (num_batches+1), obj2_loss / (num_batches+1),
                   float(correct) / total, float(c_obj1) / total, float(c_rel) / total, float(c_obj2) / total, time.time() - end))
            if epoch >= 5:
                save_dir = 'model'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
            if epoch >= 5 and epoch %3 ==0 :
                self.valid(model,train_loader,epoch)

    def valid(self,model,data_loader,epoch):
        model.eval()
        model.to(self.device)
        correct, total, c_obj1, c_obj2, c_rel = 0, 0, 0, 0, 0
        with torch.no_grad():
            for num_batches, (input, target) in enumerate(data_loader):
                #print('VALID', c_rel, total)
                images, id = input
                obj_target1, rel_target, obj_target2 = target
                images = images.to(self.device)
                obj_target1, rel_target, obj_target2 = obj_target1.to(self.device), rel_target.to(
                    self.device), obj_target2.to(self.device)
                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])
                (obj1_pred, rel_pred, obj2_pred), _ = model(images, (obj_target1, rel_target, obj_target2), True)

                _, predicted_rel = torch.max(rel_pred.data, 1)
                correct_rel = (predicted_rel.view(-1) == rel_target.view(-1)).int()

                _, predicted_obj1 = torch.max(obj1_pred.data, 1)
                _, predicted_obj2 = torch.max(obj2_pred.data, 1)
                correct_obj1 = (predicted_obj1 == obj_target1).int()
                correct_obj2 = (predicted_obj2 == obj_target2).int()
                c_obj1 += correct_obj1.sum().item()
                c_obj2 += correct_obj2.sum().item()
                c_rel += correct_rel.sum().item()
                correct += (correct_obj1 * correct_rel * correct_obj2).sum().item()
                total += len(correct_rel)

        self.log.logger.info(
            "[Validating epoch %d]: acc = %.3f, obj1 acc = %.3f, rel acc = %.3f, obj1 acc = %.3f" %
            (epoch + 1, float(correct) / total, float(c_obj1) / total, float(c_rel) / total, float(c_obj2) / total))

    def test(self, model, data_loader, epoch, ref, istrain=False, isbeam = False):
        model.load_state_dict(torch.load("model/epoch-" + str(epoch) + ".model", map_location=self.device))
        model.eval()
        model.to(self.device)

        def index2class(input):
            out = {}
            for key, value in input.items():
                out[value] = key
            return out

        idx2rel = index2class(ref.rel2idx)
        idx2obj = index2class(ref.obj2idx)
        save_dir = 'result'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results_save = {'idx2rel': idx2rel, 'rel2idx': ref.rel2idx, 'idx2obj': idx2obj, 'obj2idx': ref.obj2idx}
        correct, total, c_obj1, c_obj2, c_rel = 0, 0, 0, 0, 0
        with torch.no_grad():
            for num_batches, (input, target) in enumerate(data_loader):
                images, id = input
                images = images.to(self.device)
                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])
                if isbeam:
                    beam_result, _ = model(images, None, False, True)
                    scores, paths, paths_p = [], [], []
                    for i in range(len(beam_result)):
                        scores.append(beam_result[i][0])
                        paths.append(beam_result[i][1])
                        paths_p.append(beam_result[i][2])
                    paths, paths_p = np.array(paths), np.array(paths_p)
                    predicted_obj1 = sorted(set(list(paths[:,0])), key = list(paths[:,0]).index)
                    predicted_rel = sorted(set(list(paths[:, 1])), key=list(paths[:, 1]).index)
                    predicted_obj2 = sorted(set(list(paths[:, 2])), key=list(paths[:, 2]).index)

                    results_save[id[0]] = {'score': scores, 'paths':paths, 'prob':paths_p}
                    ### plot image
                    split_image = images.split(30, dim=0)
                    for i, s_image in enumerate(split_image):

                        org_images = np.array(
                            (s_image[15].permute(1, 2, 0).cpu() * torch.tensor(ref.normalization_std).view(1, 1, 3)
                             + torch.tensor(ref.normalization_mean).view(1, 1, 3)).detach())

                        img = Image.fromarray(np.uint8((org_images * 255).astype('int'))).convert('RGB')
                        draw = ImageDraw.Draw(img)
                        for j in range(5):
                            text = '{} {}'.format(predicted_obj1[j], idx2obj[predicted_obj1[j]])
                            draw.text((10, 10 + j * 15), text, fill="#0000ff", direction=None)
                            text1 = '{} {}'.format(predicted_rel[j], idx2rel[predicted_rel[j]])
                            draw.text((10, 80 + j * 15), text1, fill="#ff0000", direction=None)
                            text2 = '{} {}'.format(predicted_obj2[j], idx2obj[predicted_obj2[j]])
                            draw.text((100, 10 + j * 15), text2, fill="#0000ff", direction=None)

                        img.save(os.path.join('result', id[i] + '.jpg'))

                else:
                    (obj1_prob, rel_prob, obj2_prob), pred = model(images, None, False, False)

                    if istrain:
                        obj_target1, rel_target, obj_target2 = target
                        obj_target1, rel_target, obj_target2 = obj_target1.to(self.device), rel_target.to(
                            self.device), obj_target2.to(self.device)

                        _, predicted_rel = torch.max(rel_prob.data, 1)
                        correct_rel = (predicted_rel.view(-1) == rel_target.view(-1)).int()

                        _, predicted_obj1 = torch.max(obj1_prob.data, 1)
                        _, predicted_obj2 = torch.max(obj2_prob.data, 1)
                        correct_obj1 = (predicted_obj1 == obj_target1).int()
                        correct_obj2 = (predicted_obj2 == obj_target2).int()
                        c_obj1 += correct_obj1.sum().item()
                        c_obj2 += correct_obj2.sum().item()
                        c_rel += correct_rel.sum().item()
                        correct += (correct_obj1 * correct_rel * correct_obj2).sum().item()
                        total += len(correct_rel)

                    ### relations
                    rel_prob = torch.sum(torch.exp(rel_prob), dim=0, keepdim=True)
                    prob_rel, predicted_rel = torch.topk(rel_prob.data, k=10, dim=1)
                    prob_rel = np.array(prob_rel.cpu().detach())
                    predicted_rel = np.array(predicted_rel.cpu().detach())
                    ### objects
                    obj1_prob = torch.sum(torch.exp(obj1_prob), dim=0, keepdim=True)
                    obj2_prob = torch.sum(torch.exp(obj2_prob), dim=0, keepdim=True)
                    prob_obj1, predicted_obj1 = torch.topk(obj1_prob.data, k=10, dim=1)
                    prob_obj1 = np.array(prob_obj1.cpu().detach())
                    predicted_obj1 = np.array(predicted_obj1.cpu().detach())
                    prob_obj2, predicted_obj2 = torch.topk(obj2_prob.data, k=10, dim=1)
                    prob_obj2 = np.array(prob_obj2.cpu().detach())
                    predicted_obj2 = np.array(predicted_obj2.cpu().detach())

                    ### plot image
                    split_image = images.split(30, dim=0)
                    for i, s_image in enumerate(split_image):
                        results_save[id[i]] = {'rel': (predicted_rel[i], prob_rel[i]),
                                               'obj1': (predicted_obj1[i], prob_obj1[i])
                            , 'obj2': (predicted_obj2[i], prob_obj2[i])}
                        org_images = np.array(
                            (s_image[15].permute(1, 2, 0).cpu() * torch.tensor(ref.normalization_std).view(1, 1, 3)
                             + torch.tensor(ref.normalization_mean).view(1, 1, 3)).detach())

                        img = Image.fromarray(np.uint8((org_images * 255).astype('int'))).convert('RGB')
                        draw = ImageDraw.Draw(img)
                        for j in range(5):
                            text = '{} {} {:.2f}'.format(predicted_obj1[i][j], idx2obj[predicted_obj1[i][j]],
                                                         prob_obj1[i][j])
                            draw.text((10, 10 + j * 15), text, fill="#0000ff", direction=None)
                            text1 = '{} {} {:.2f}'.format(predicted_rel[i][j], idx2rel[predicted_rel[i][j]],
                                                          prob_rel[i][j])
                            draw.text((10, 80 + j * 15), text1, fill="#ff0000", direction=None)
                            text2 = '{} {} {:.2f}'.format(predicted_obj2[i][j], idx2obj[predicted_obj2[i][j]],
                                                          prob_obj2[i][j])
                            draw.text((100, 10 + j * 15), text2, fill="#0000ff", direction=None)

                        img.save(os.path.join('result', id[i] + '.jpg'))

        if istrain:
            print(("[Validating epoch %d]: acc = %.3f, obj1 acc = %.3f, rel acc = %.3f, obj2 acc = %.3f" %
                   (epoch, float(correct) / total, float(c_obj1) / total, float(c_rel) / total, float(c_obj2) / total)))
        with open("result/rel_results.pickle", "wb") as fp:
            pickle.dump(results_save, fp)

    def adjust_learning_rate(self, optimizer, epoch, thre=[52]):
        decay = 0.1
        lr_list = []
        if epoch in thre:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay
                lr_list.append(param_group['lr'])
            lr = np.unique(lr_list)
            string = ' '.join([str(i) for i in lr])
            self.log.logger.info('Changing lr:'+string)

