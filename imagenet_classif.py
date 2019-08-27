import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter, accuracy
from data.imagenetloader import ImageNetLoader800from882 
from models.resnet import resnet18
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

def train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss().cuda(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(args.epochs):
        scheduler.step()
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        for batch_idx, (x, label, _) in enumerate(tqdm(train_loader)):
            x, label = x.to(device), label.to(device)
            output = model(x)
            loss = criterion(output, label) 
            acc = accuracy(output, label)
            acc_record.update(acc[0].item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
        test(model, eva_loader, args)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output = model(x)
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), x.size(0))
    print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_ids', default=[0, 1, 2, 3], type=int, nargs='+',
                            help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--step_size', default=30, type=int)
    parser.add_argument('--num_classes', default=800, type=int)
    parser.add_argument('--model_name', type=str, default='resnet18_imagenet800')
    parser.add_argument('--dataset_root', type=str, default='data/datasets/ImageNet/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    runner_name = os.path.basename(__file__).split(".")[0] 
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'

    loader_train = ImageNetLoader800from882(batch_size=256, num_workers=2, split='train', path=args.dataset_root)
    loader_val = ImageNetLoader800from882(batch_size=256, num_workers=2, split='val', path=args.dataset_root)
    model = resnet18(num_classes=800) 
    model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)
    train(model, loader_train, loader_val, args)
    test(model, loader_val, args)
