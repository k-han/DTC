import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.decomposition import PCA
from data.omniglotloader import omniglot_alphabet_func, omniglot_evaluation_alphabets_mapping 
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool
from utils import ramps 
from models.vgg import VGG
from modules.module import feat2prob, target_distribution 
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

def init_prob_kmeans(model, eval_loader, args):
    torch.manual_seed(1)
    model = model.to(device)
    # cluster parameter initiate
    model.eval()
    targets = np.zeros(len(eval_loader.dataset)) 
    feats = np.zeros((len(eval_loader.dataset), 1024))
    for _, (x, _, label, idx) in enumerate(eval_loader):
        x = x.to(device)
        _, feat = model(x)
        feat = feat.view(x.size(0), -1)
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
    return kmeans.cluster_centers_, probs 

def warmup_train(model, alphabetStr, train_loader, eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.warmup_lr)
    for epoch in range(args.warmup_epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, (x, g_x, _, idx) in enumerate(train_loader):
            _,  feat = model(x.to(device))
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), x.size(0))
        print('Warmup Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        test(model, eval_loader, args)

def Baseline_train(model, alphabetStr, train_loader, eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, (x, g_x, _, idx) in enumerate(train_loader):
            _, feat = model(x.to(device))
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eval_loader, args)

        if epoch % args.update_interval==0:
            args.p_targets= target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def PI_train(model, alphabetStr, train_loader, eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, (x, g_x, _, idx) in enumerate(train_loader):
            _,  feat = model(x.to(device))
            _,  feat_g = model(g_x.to(device))
            prob = feat2prob(feat, model.center)
            prob_g = feat2prob(feat_g, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            mse_loss = F.mse_loss(prob, prob_g)
            loss=loss + w*mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eval_loader, args)

        if epoch % args.update_interval==0:
            args.p_targets= target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def TE_train(model, alphabetStr, train_loader, eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
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
        for batch_idx, (x, _, _, idx) in enumerate(train_loader):
            _, feat = model(x.to(device))
            prob = feat2prob(feat, model.center)
            z_epoch[idx, :] = prob
            prob_bar = Variable(z_ema[idx, :], requires_grad=False)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            mse_loss = F.mse_loss(prob, prob_bar)
            loss=loss+w*mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), x.size(0))

        Z = alpha * Z + (1. - alpha) * z_epoch
        z_ema = Z * (1. / (1. - alpha ** (epoch + 1)))
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eval_loader, args)

        if epoch % args.update_interval==0:
            args.p_targets = target_distribution(probs) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def TEP_train(model, alphabetStr, train_loader, eval_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_clusters).float().to(device)        # intermediate values
    z_ema = torch.zeros(ntrain, args.n_clusters).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_clusters).float().to(device)  # current outputs

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, (x, g_x, _, idx) in enumerate(train_loader):
            _, feat = model(x.to(device))
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), x.size(0))

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eval_loader, args)
        z_epoch = probs.float().to(device)
        Z = alpha * Z + (1. - alpha) * z_epoch
        z_bars = Z * (1. / (1. - alpha ** (epoch + 1)))

        if epoch % args.update_interval==0:
            args.p_targets = target_distribution(z_bars).float().to(device) 
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def test(model, eval_loader, args):
    model.eval()
    targets = np.zeros(len(eval_loader.dataset)) 
    y_pred = np.zeros(len(eval_loader.dataset)) 
    probs= np.zeros((len(eval_loader.dataset), args.n_clusters))
    for _, (x, _, label, idx) in enumerate(eval_loader):
        x = x.to(device)
        _, feat = model(x)
        prob = feat2prob(feat, model.center)
        #  prob = F.softmax(logit, dim=1)
        idx = idx.data.cpu().numpy()
        y_pred[idx] = prob.data.cpu().detach().numpy().argmax(1)
        targets[idx] = label.data.cpu().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    # evaluate clustering performance
    y_pred = y_pred.astype(np.int64)
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)
    return acc, nmi, ari, probs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup_lr', type=float, default=0.001)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_txt', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', default=100.0, type=float)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/pretrained/vgg6_omniglot_proto.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--subfolder_name', type=str, default='run')
    parser.add_argument('--save_txt_name', type=str, default='result.txt')
    parser.add_argument('--DTC', type=str, default='PI')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    model = VGG(n_layer='4+2', in_channels=1).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    model.last = Identity()
    init_feat_extractor = model 
    acc = {}
    nmi = {}
    ari = {}
    for _, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
        runner_name = os.path.basename(__file__).split(".")[0] 
        model_dir= args.exp_root + '{}/{}/{}'.format(runner_name, args.DTC, args.subfolder_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        args.model_dir = model_dir+'/'+'vgg6_{}.pth'.format(alphabetStr)
        args.save_txt_path= args.exp_root + '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)
        train_Dloader, eval_Dloader = omniglot_alphabet_func(alphabet=alphabetStr, background=False, root=args.dataset_root)(batch_size=args.batch_size, num_workers=args.num_workers)
        args.n_clusters = train_Dloader.num_classes 
        model = VGG(n_layer='4+2', out_dim=args.n_clusters, in_channels=1).to(device)
        model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        model.center= Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
        init_centers, init_probs = init_prob_kmeans(init_feat_extractor, eval_Dloader, args)
        args.p_targets = target_distribution(init_probs) 
        model.center.data = torch.tensor(init_centers).float().to(device)
        warmup_train(model, alphabetStr, train_Dloader, eval_Dloader, args)
        if args.DTC == 'Baseline':
            Baseline_train(model, alphabetStr, train_Dloader, eval_Dloader, args)
        elif args.DTC == 'PI':
            PI_train(model, alphabetStr, train_Dloader, eval_Dloader, args)
        elif args.DTC == 'TE':
            TE_train(model, alphabetStr, train_Dloader, eval_Dloader, args)
        elif args.DTC == 'TEP':
            TEP_train(model, alphabetStr, train_Dloader, eval_Dloader, args)
        acc[alphabetStr], nmi[alphabetStr], ari[alphabetStr], _ = test(model, eval_Dloader, args)
    print('ACC for all alphabets:',acc)
    print('NMI for all alphabets:',nmi)
    print('ARI for all alphabets:',ari)
    avg_acc, avg_nmi, avg_ari = sum(acc.values())/float(len(acc)), sum(nmi.values())/float(len(nmi)), sum(ari.values())/float(len(ari))
    print('avg ACC {:.4f}, NMI {:.4f} ARI {:.4f}'.format(avg_acc, avg_nmi, avg_ari))

    if args.save_txt: 
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(avg_acc, avg_nmi, avg_ari))
