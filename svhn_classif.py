import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from utils.util import AverageMeter, accuracy
from data.svhnloader import SVHNLoader 
from models.resnet_3x3 import ResNet, BasicBlock 
import os

def train(model, train_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion=nn.CrossEntropyLoss().cuda(device)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, _) in enumerate(train_loader):
            x, target = x.to(device), label.to(device)
            optimizer.zero_grad()
            output= model(x)
            loss = criterion(output, target) 
            acc = accuracy(output, target)
            loss.backward()
            optimizer.step()
            acc_record.update(acc[0].item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
        test(model, eva_loader, args)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    for batch_idx, (x, label, _) in enumerate(test_loader):
        x, target = x.to(device), label.to(device)
        output= model(x)
        acc = accuracy(output, target)
        acc_record.update(acc[0].item(), x.size(0))
    print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=180, type=int)
    parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet18_svhn_classif_5')
    parser.add_argument('--dataset_root', type=str, default='data/datasets/SVHN')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    runner_name = os.path.basename(__file__).split(".")[0] 
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'

    train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = True, aug='once', shuffle=True)
    eva_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', labeled = True, aug=None, shuffle=False)
    model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(device)
    train(model, train_loader, args)
    test(model, eva_loader, args)
