import argparse
from Dataset import VID2016
import torch.utils.data
from models import *
from torch import optim
from engine import *
from logger import Logger
import json

parser = argparse.ArgumentParser(description='CNN-GNN Training')
parser.add_argument('--data', default='data/', type=str, help='path to dataset')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=8, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--w-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--action', default='test', help='train or test')

### hyper-parameters for TCN
features_dim = 2048

### number of classes
num_obj = 35
num_rel = 82

#### test model epoch
test_epoch = 15

log_dir = 'logs'
logfile = log_dir +'/'+ time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
log = Logger(log_dir, logfile, fmt="[%(asctime)s - %(levelname)s]: %(message)s")
log.logger.info('########################################################################')


def main_vrd():
    args = parser.parse_args()
    log.logger.info(args)
    model = Resnet_LSTM((num_obj, num_rel), features_dim)
    if args.action == 'train':
        train_dataset = VID2016(args.data, phase='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        criterion_obj =  FocalLoss(num_obj, alpha=None, gamma=2, size_average=True)
        criterion_rel =  FocalLoss(num_rel, alpha=None, gamma=2, size_average=True)
        optimizer = optim.Adam(model.get_config_optim(args.lr, 0.1), lr=args.lr, weight_decay=args.w_decay)
        vrd = Trainer(optimizer, (criterion_obj, criterion_rel), args, log)
        vrd.train(model, train_loader)
    elif args.action == 'test':
        test_dataset = VID2016(args.data, phase='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        vrd = Trainer(None, (None, None), args, log)
        vrd.test(model, test_loader, test_epoch, test_dataset, False, True)




if __name__ == '__main__':
    main_vrd()




