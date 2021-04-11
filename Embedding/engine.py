import torch
import numpy as np
import torch.nn.functional as F
import os, time, pickle, json
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
            epoch_loss, obj_loss = 0, 0
            total, c_obj = 0, 0
            end = time.time()
            for num_batches, (input, target) in enumerate(train_loader):
                images, id, inp = input
                obj_target, rel_target = target
                images, obj_target, rel_target, inp= images.to(self.device), obj_target.to(self.device), rel_target.to(self.device), inp.to(self.device)
                self.optimizer.zero_grad()
                images = images.view(images.shape[0]*images.shape[1], images.shape[2], images.shape[3],images.shape[4])
                # aa = torch.tensor(np.array([[[[[1,1],[1,1]]],[[[1,1],[1,1]]]]]))
                # bb = torch.tensor(np.array([[[[[2, 2], [2, 2]]], [[[2, 2], [2, 2]]]]]))
                # images = torch.cat([aa,bb],dim=0)
                # dd = images.view(images.shape[0]*images.shape[1], images.shape[2], images.shape[3],images.shape[4])
                # k = dd.split(2,dim=0)
                obj_pred = model(images,inp[0].float(), obj_target)

                loss = self.criterion_obj(obj_pred, obj_target)

                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                predicted_obj = (F.sigmoid(obj_pred) + 0.5).int()
                correct_obj = torch.sum((predicted_obj - obj_target.int()), dim=1)
                c_obj += ((correct_obj == 0).int()).sum().item()

                total += len(correct_obj)

            self.log.logger.info("[epoch %d]: epoch loss = %.3f, acc = %.3f,time = %.1f" %
                  (epoch + 1, epoch_loss / (num_batches+1), float(c_obj) / total,  time.time() - end))
            if epoch >= 15:
                save_dir = 'model'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
            if epoch >= 5 and epoch %5 ==0 :
                self.valid(model,train_loader,epoch)

    def valid(self,model,data_loader,epoch):
        model.eval()
        model.to(self.device)
        total, c_obj = 0, 0
        with torch.no_grad():
            for num_batches, (input, target) in enumerate(data_loader):
                #print('VALID', c_rel, total)
                images, id, inp = input
                obj_target, rel_target = target
                images, obj_target, rel_target, inp = images.to(self.device), obj_target.to(self.device), rel_target.to(
                    self.device), inp.to(self.device)

                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])
                obj_pred= model(images,inp[0].float(),None,False)

                predicted_obj = (F.sigmoid(obj_pred) + 0.5).int()
                correct_obj = torch.sum((predicted_obj - obj_target.int()), dim=1)
                c_obj += ((correct_obj == 0).int()).sum().item()
                total += len(correct_obj)

        self.log.logger.info(
            "[Validating epoch %d]: acc = %.3f" %(epoch + 1,  float(c_obj) / total))

    def test(self, model, data_loader, epoch, ref, istrain = False):
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
        c_obj,total = 0, 0
        with torch.no_grad():
            for num_batches, (input, target) in enumerate(data_loader):
                images, id, inp = input
                images, inp = images.to(self.device), inp.to(self.device)
                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])

                obj_pred = model(images,inp[0].float(),None, False)

                if istrain:
                    obj_target, rel_target = target
                    obj_target, rel_target =  obj_target.to(self.device), rel_target.to(self.device)

                    predicted_obj = (F.sigmoid(obj_pred) + 0.5).int()
                    correct_obj = torch.sum((predicted_obj - obj_target.int()), dim=1)
                    c_obj += ((correct_obj == 0).int()).sum().item()
                    total += len(correct_obj)

                ### objects
                prob_obj, predicted_obj = torch.topk(F.sigmoid(obj_pred).data + 0.5, k=5, dim=1)
                prob_obj = np.array(prob_obj.cpu().detach())
                predicted_obj = np.array(predicted_obj.cpu().detach())

                ### plot image
                split_image = images.split(30,dim=0)
                for i, s_image in enumerate(split_image):
                    results_save[id[i]] = {'obj': (predicted_obj[i],prob_obj[i])}
                    org_images = np.array(
                        (s_image[15].permute(1, 2, 0).cpu() * torch.tensor(ref.normalization_std).view(1, 1,3)
                         + torch.tensor(ref.normalization_mean).view(1, 1, 3)).detach())

                    img = Image.fromarray(np.uint8((org_images * 255).astype('int'))).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    for j in range(5):
                        text = '{} {} {:.2f}'.format(predicted_obj[i][j], idx2obj[predicted_obj[i][j]], prob_obj[i][j])
                        draw.text((10, 10 + j * 15), text, fill="#0000ff", direction=None)

                    img.save(os.path.join('result', id[i] + '.jpg'))
        if istrain:
            print("[Validating epoch %d]: acc = %.3f" %(epoch, float(c_obj) / total))
        with open("result/obj_results.pickle", "wb") as fp:
            pickle.dump(results_save, fp)

    def gen_feature(self, model, data_loader, epoch, phase = 'train'):
        model.load_state_dict(torch.load("model/epoch-" + str(epoch) + ".model", map_location=self.device))
        model.eval()
        model.to(self.device)
        save_dir = 'features/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with torch.no_grad():
            output = {}
            for num_batches, (input, target) in enumerate(data_loader):
                images, id, inp = input
                print(id)
                images, inp = images.to(self.device), inp.to(self.device)
                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])

                obj_pred = model(images, inp[0].float(), None, False)
                output[id[0]] = list(F.sigmoid(obj_pred).cpu().detach().numpy().reshape(-1))

            with open(save_dir + phase + '.pk', 'wb') as f:
                pickle.dump(output, f)



    def adjust_learning_rate(self, optimizer, epoch, thre=[18]):
        decay = 0.1
        lr_list = []
        if epoch in thre:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay
                lr_list.append(param_group['lr'])
            lr = np.unique(lr_list)
            string = ' '.join([str(i) for i in lr])
            self.log.logger.info('Changing lr:'+string)

