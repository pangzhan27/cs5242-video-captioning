import torch.nn as nn
import torchvision.models as models
import copy
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Resnet_TCN(nn.Module):
    def __init__(self, num_classes, num_layers, num_f_maps, dim, adj, embedding =300):
        super(Resnet_TCN, self).__init__()
        self.num_obj = num_classes[0]
        self.num_rel = num_classes[1]
        ###### 1.resnet101 ######
        model = models.resnet101(pretrained=True)
        for name, param in model.named_parameters():
            param.requires_grad = False
            if ('layer3' in name) or ('layer4' in name) :
                param.requires_grad = True

        self.features_3 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.features_4 = model.layer4
        self.conv34 = nn.Conv2d(1024, 2048, 3, padding = 1)
        # for name, para in self.features.named_parameters():
        #     print(name, para.requires_grad)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = torch.from_numpy(adj).float().to(device)
        self.pooling = nn.MaxPool2d(14, 14)
        #self.pooling = nn.MaxPool2d(14, 14)
        self.gc1 = GraphConvolution(embedding , 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        ###### 2.Multi-stage TCN ######
        #self.temp_pool = nn.MaxPool1d(3, stride=3)
        self.pool_obj = nn.MaxPool1d(10, stride=1)
        self.dropout = nn.Dropout()

        self.stage = SingleStageModel(num_layers, num_f_maps, dim, num_classes)

    def forward(self, feature, inp, targets, epoch=None, istrain = True):
        feature3 = self.features_3(feature)
        feature4 = self.features_4(feature3)
        feature4 = F.interpolate(feature4, scale_factor=2, mode='bilinear')
        feature34 = self.conv34(feature3)
        feature = feature4 + feature34
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        split_feature = feature.split(10,dim=0)
        new_feature = torch.cat([i.transpose(0,1).unsqueeze(0) for i in split_feature])

        # object
        objs = self.pool_obj(new_feature)
        objs = objs.view(objs.shape[0],-1)
        #objs = self.dropout(objs)
        # gcn
        x = self.gc1(inp, self.adj)
        x = self.relu(x)
        x = self.gc2(x, self.adj)
        x_t = x.transpose(0, 1)
        objs = torch.matmul(objs, x_t)

        #temp_feature = self.temp_pool(new_feature)
        rels = self.stage(new_feature)
        return objs, rels

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features_3.parameters(), 'lr': lr * lrp},
                {'params': self.features_4.parameters(), 'lr': lr * lrp},
                {'params': self.conv34.parameters(), 'lr': lr},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                {'params': self.stage.parameters(), 'lr': lr},
                ]


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.layers1 = nn.Conv1d(dim, num_f_maps, 3, dilation = 1)
        self.layers2 = nn.Conv1d(num_f_maps, num_f_maps, 3, dilation =2)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.conv_rel = nn.Linear(num_f_maps, num_classes[1])
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.layers1(x))
        out = F.relu(self.layers2(out))
        out = self.pool1(out)
        out = out.view(out.shape[0], -1)
        out = self.dropout(out)
        rel_out = self.conv_rel(out)

        return rel_out

