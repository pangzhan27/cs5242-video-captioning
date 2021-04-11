import argparse
from Dataset import VID2016
import torch.utils.data
from models import *
from torch import optim
from engine import *
from logger import Logger
import pickle as pk

parser = argparse.ArgumentParser(description='CNN-GNN Training')
parser.add_argument('--data', default='data/', type=str, help='path to dataset')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--w-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--action', default='feature', help='train or test')

### hyper-parameters for TCN
num_layers = 5
num_f_maps = 128
features_dim = 2048

### number of classes
num_obj = 35
num_rel = 82

#### test model epoch
test_epoch = 30

log_dir = 'logs'
logfile = log_dir +'/'+ time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
log = Logger(log_dir, logfile, fmt="[%(asctime)s - %(levelname)s]: %(message)s")
log.logger.info('########################################################################')

def main_vrd():
    args = parser.parse_args()
    log.logger.info(args)
    with open("adj/adj.pickle", "rb") as fp:
        adj = pk.load(fp)
    model = Resnet_TCN((num_obj, num_rel), num_layers, num_f_maps, features_dim, adj)
    if args.action == 'train':
        train_dataset = VID2016(args.data, phase='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        criterion_obj = nn.MultiLabelSoftMarginLoss()
        criterion_rel = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.get_config_optim(args.lr, 0.1), lr=args.lr, weight_decay=args.w_decay)
        vrd = Trainer(optimizer, (criterion_obj, criterion_rel), args, log)
        vrd.train(model, train_loader)
    elif args.action == 'test':
        test_dataset = VID2016(args.data, phase='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        vrd = Trainer(None, (None, None), args, log)
        vrd.test(model, test_loader, test_epoch, test_dataset)
    else:
        train_dataset = VID2016(args.data, phase='train')
        test_dataset = VID2016(args.data, phase='test')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        vrd = Trainer(None, (None, None), args, log)
        vrd.gen_feature(model, test_loader, test_epoch, phase = 'test')
        vrd.gen_feature(model, train_loader, test_epoch, phase='train')

if __name__ == '__main__':
    main_vrd()




