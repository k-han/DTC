import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from utils.faster_mix_k_means_pytorch import K_Means
from utils.util import cluster_acc, Identity, seed_torch
from data.cifarloader import CIFAR100Loader
from models.vgg import VGG
from tqdm import tqdm
from collections import Counter
import random
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def estimate_k(model, unlabeled_loader, labeled_loader, args):
    u_num = len(unlabeled_loader.dataset)
    u_targets = np.zeros(u_num) 
    u_feats = np.zeros((u_num, 512))
    print('extracting features for unlabeld data')
    for _, (x, label, idx) in enumerate(tqdm(unlabeled_loader)):
        x = x.to(device)
        _, feat = model(x)
        feat = feat.view(x.size(0), -1)
        idx = idx.data.cpu().numpy()
        u_feats[idx, :] = feat.data.cpu().numpy()
        u_targets[idx] = label.data.cpu().numpy()
    cand_k = np.arange(args.max_cand_k)
    #get acc for labeled data with short listed k
    l_num = len(labeled_loader.dataset)
    l_targets = np.zeros(l_num) 
    l_feats = np.zeros((l_num, 512))
    print('extracting features for labeld data')
    for _, (x, label, idx) in enumerate(tqdm(labeled_loader)):
        x = x.to(device)
        _, feat = model(x)
        feat = feat.view(x.size(0), -1)
        idx = idx.data.cpu().numpy()
        l_feats[idx, :] = feat.data.cpu().numpy() 
        l_targets[idx] = label.data.cpu().numpy()

    l_classes = set(l_targets) 
    num_lt_cls = int(round(len(l_classes)*args.split_ratio))
    lt_classes = set(random.sample(l_classes, num_lt_cls)) #random sample 5 classes from all labeled classes
    lv_classes = l_classes - lt_classes

    lt_feats = np.empty((0, l_feats.shape[1]))
    lt_targets = np.empty(0)
    for c in lt_classes:
        lt_feats = np.vstack((lt_feats, l_feats[l_targets==c]))
        lt_targets = np.append(lt_targets, l_targets[l_targets==c])

    lv_feats = np.empty((0, l_feats.shape[1]))
    lv_targets = np.empty(0)
    for c in lv_classes:
        lv_feats = np.vstack((lv_feats, l_feats[l_targets==c]))
        lv_targets = np.append(lv_targets, l_targets[l_targets==c])

    cvi_list = np.zeros(len(cand_k))
    acc_list = np.zeros(len(cand_k))
    cat_pred_list = np.zeros([len(cand_k),u_num+l_num])
    print('estimating K ...')
    for i in range(len(cand_k)):
        cvi_list[i],  cat_pred_i = labeled_val_fun(np.concatenate((lv_feats, u_feats)), lt_feats, lt_targets, cand_k[i]+args.num_val_cls)
        cat_pred_list[i, :] = cat_pred_i
        acc_list[i] = cluster_acc(lv_targets, cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
        best_k = get_best_k(cvi_list[:i+1], acc_list[:i+1], cat_pred_list[:i+1], l_num) 
        print('current best K {}'.format(best_k))
    kmeans = KMeans(n_clusters=best_k)
    u_pred = kmeans.fit_predict(u_feats).astype(np.int32) 
    acc, nmi, ari = cluster_acc(u_targets, u_pred), nmi_score(u_targets, u_pred), ari_score(u_targets, u_pred)
    print('Final K {}, acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(best_k, acc, nmi, ari))
    return best_k

def labeled_val_fun(u_feats, l_feats, l_targets, k):
    if device=='cuda':
        torch.cuda.empty_cache()
    l_num=len(l_targets)
    kmeans = K_Means(k, pairwise_batch_size=256)
    kmeans.fit_mix(torch.from_numpy(u_feats).to(device), torch.from_numpy(l_feats).to(device), torch.from_numpy(l_targets).to(device))
    cat_pred = kmeans.labels_.cpu().numpy() 
    u_pred = cat_pred[l_num:]
    silh_score = silhouette_score(u_feats, u_pred)
    return silh_score, cat_pred 


def get_best_k(cvi_list, acc_list, cat_pred_list, l_num):
    idx_cvi = np.max(np.argwhere(cvi_list==np.max(cvi_list)))
    idx_acc = np.max(np.argwhere(acc_list==np.max(acc_list)))
    idx_best = int(math.ceil((idx_cvi+idx_acc)*1.0/2))
    cat_pred = cat_pred_list[idx_best, :]
    cnt_cat = Counter(cat_pred.tolist())
    cnt_l = Counter(cat_pred[:l_num].tolist())
    cnt_ul = Counter(cat_pred[l_num:].tolist())
    bin_cat = [x[1] for x in sorted(cnt_cat.items())]
    bin_l = [x[1] for x in sorted(cnt_l.items())]
    bin_ul = [x[1] for x in sorted(cnt_ul.items())]
    best_k = np.sum(np.array(bin_ul)/np.max(bin_ul).astype(float)>args.min_max_ratio)
    return best_k

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--num_val_cls', default=10, type=int)
    parser.add_argument('--max_cand_k', default=100, type=int)
    parser.add_argument('--split_ratio', type=float, default=0.6)
    parser.add_argument('--min_max_ratio', type=float, default=0.01)
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/pretrained/vgg6_cifar100_classif_80.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed) 
    model = VGG(n_layer='5+1', out_dim=80).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    model.last = Identity()
    
    val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = True, aug=None, shuffle=True, mode='probe')
    eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = False, aug=None, shuffle=False)
    args.n_clusters = estimate_k(model, eval_loader, val_loader, args)

