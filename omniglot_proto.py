import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data.omniglotloader import Omniglot_bg_loader
from models.vgg import VGG
from modules.prototypical_loss import prototypical_loss
from utils.util import  AverageMeter, Identity
import os

def train(model, train_loader, args):
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        for batch_idx, (x, _, label, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            _, feat = model(x)
            loss, acc = prototypical_loss(feat, label, n_support=5) 
            loss.backward()
            optimizer.step()
            acc_record.update(acc.item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))

        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
        torch.save(model.state_dict(), args.model_dir)
        test(model, eva_loader, args)
    print("model saved to {}.".format(args.model_dir))

def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    for batch_idx, (x, _, label, _) in enumerate(test_loader):
        x = x.to(device)
        _, feat = model(x)
        loss, acc = prototypical_loss(feat, label, n_support=5) 
        acc_record.update(acc.item(), x.size(0))
    print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--model_name', type=str, default='vgg6_omniglot_proto')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    runner_name = os.path.basename(__file__).split(".")[0] 
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'

    bg_loader, eva_loader = Omniglot_bg_loader(batch_size=args.batch_size, num_workers=2, root=args.dataset_root)
    model = VGG(n_layer='4+2', in_channels=1).to(device)
    model.last = Identity()
    train(model, bg_loader, args)
    test(model, eva_loader, args)
