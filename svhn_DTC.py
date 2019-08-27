import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool
from utils import ramps 
from models.resnet_3x3 import ResNet, BasicBlock 
from modules.module import feat2prob, target_distribution 
from data.svhnloader import SVHNLoader 
from tqdm import tqdm
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

def init_prob_kmeans(model, eval_loader, args):
    torch.manual_seed(1)
    model = model.to(device)
    # cluster parameter initiate
    model.eval()
    targets = np.zeros(len(eval_loader.dataset)) 
    feats = np.zeros((len(eval_loader.dataset), 512))
    for _, (x, label, idx) in enumerate(eval_loader):
        x = x.to(device)
        feat = model(x)
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.data.cpu().numpy()
        targets[idx] = label.data.cpu().numpy()
    # evaluate clustering performance
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(feats) 
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs 

def warmup_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.warmup_epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x  = x.to(device)
            feat = model(x)
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
    args.p_targets = target_distribution(probs) 

def Baseline_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = model(x)
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def PI_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            feat = model(x)
            feat_bar = model(x_bar)
            prob = feat2prob(feat, model.center)
            prob_bar = feat2prob(feat_bar, model.center)
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)
            loss = sharp_loss + w * consistency_loss 
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        acc, _, _, probs = test(model, eva_loader, args)

        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def TE_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_clusters).float().to(device)        # intermediate values
    z_ema = torch.zeros(ntrain, args.n_clusters).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_clusters).float().to(device)  # current outputs
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = model(x)
            prob = feat2prob(feat, model.center)
            z_epoch[idx, :] = prob
            prob_bar = Variable(z_ema[idx, :], requires_grad=False)
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)
            loss = sharp_loss + w * consistency_loss 
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = alpha * Z + (1. - alpha) * z_epoch
        z_ema = Z * (1. / (1. - alpha ** (epoch + 1)))
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        acc, _, _, probs = test(model, eva_loader, args)

        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def TEP_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_clusters).float().to(device)        # intermediate values
    z_bars = torch.zeros(ntrain, args.n_clusters).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_clusters).float().to(device)  # current outputs
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = model(x)
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        z_epoch = probs.float().to(device)
        Z = alpha * Z + (1. - alpha) * z_epoch
        z_bars = Z * (1. / (1. - alpha ** (epoch + 1)))
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(z_bars).float().to(device) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs= np.zeros((len(test_loader.dataset), args.n_clusters))
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)
    return acc, nmi, ari, probs 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warmup_lr', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_txt', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/pretrained/resnet18_svhn_classif_5.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/SVHN/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--save_txt_name', type=str, default='result.txt')
    parser.add_argument('--DTC', type=str, default='PI')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0] 
    model_dir= args.exp_root + '{}/{}'.format(runner_name, args.DTC)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'

    args.save_txt_path = args.exp_root + '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

    model = ResNet(BasicBlock, [2,2,2,2], 5).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    model.linear= Identity()
    init_feat_extractor = model
    model = ResNet(BasicBlock, [2,2,2,2], args.n_clusters).to(device)
    model.load_state_dict(init_feat_extractor.state_dict(), strict=False)
    model.center= Parameter(torch.Tensor(args.n_clusters, args.n_clusters))

    train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = False, aug='twice', shuffle=True)
    eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = False, aug=None, shuffle=False)
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, eval_loader, args)
    args.p_targets = target_distribution(init_probs) 
    model.center.data = torch.tensor(init_centers).float().to(device)
    warmup_train(model, train_loader, eval_loader, args)
    if args.DTC == 'Baseline':
        Baseline_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'PI':
        PI_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'TE':
        TE_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'TEP':
        TEP_train(model, train_loader, eval_loader, args)
    acc, nmi, ari, _ = test(model, eval_loader, args)
    print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
    print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))

    if args.save_txt:
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))
