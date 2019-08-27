import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from utils.faster_mix_k_means_pytorch import K_Means
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool
from data.omniglotloader import omniglot_alphabet_func, omniglot_evaluation_alphabets_mapping, omniglot_background_val_alphabets  
from models.vgg import VGG
from tqdm import tqdm
from collections import Counter
import random
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def estimate_k(model, unlabeled_loader, labeled_loaders, args):
    u_num = len(unlabeled_loader.dataset)
    u_targets = np.zeros(u_num) 
    u_feats = np.zeros((u_num, 1024))
    print('extracting features for unlabeld data')
    for _, (x, _, label, idx) in enumerate(unlabeled_loader):
        x = x.to(device)
        _, feat = model(x)
        feat = feat.view(x.size(0), -1)
        idx = idx.data.cpu().numpy()
        u_feats[idx, :] = feat.data.cpu().numpy()
        u_targets[idx] = label.data.cpu().numpy()
    cand_k = np.arange(args.max_cand_k)
    #get acc for labeled data with short listed k
    best_ks = np.zeros(len(omniglot_background_val_alphabets))
    print('extracting features for labeld data')
    for alphabetStr in omniglot_background_val_alphabets: 
        labeled_loader = labeled_loaders[alphabetStr]
        args.num_val_cls = labeled_loader.num_classes

        l_num = len(labeled_loader.dataset)
        l_targets = np.zeros(l_num) 
        l_feats = np.zeros((l_num, 1024))
        for _, (x, _, label, idx) in enumerate(labeled_loader):
            x = x.to(device)
            _, feat = model(x)
            feat = feat.view(x.size(0), -1)
            idx = idx.data.cpu().numpy()
            l_feats[idx, :] = feat.data.cpu().numpy()
            l_targets[idx] = label.data.cpu().numpy()

        l_classes = set(l_targets) 
        num_lt_cls = int(round(len(l_classes)*args.split_ratio))
        lt_classes = set(random.sample(l_classes, num_lt_cls)) 
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
        expectation = u_num*1.0 / (cand_k[idx_best]+args.num_val_cls)
        best_k = np.sum(np.array(bin_ul)/np.max(bin_ul).astype(float)>args.min_max_ratio)
        print('current best K {}'.format(best_k))
        i_alpha = omniglot_background_val_alphabets.index(alphabetStr)
        best_ks[i_alpha] = best_k
    best_k = np.ceil(np.mean(best_ks)).astype(np.int32)
    kmeans = KMeans(n_clusters=best_k)
    u_pred = kmeans.fit_predict(u_feats).astype(np.int32) 
    acc, nmi, ari = cluster_acc(u_targets, u_pred), nmi_score(u_targets, u_pred), ari_score(u_targets, u_pred)
    print('Final K {}, acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(best_k, acc, nmi, ari))
    return best_k

def labeled_val_fun(u_feats, l_feats, l_targets, k):
    if device=='cuda':
        torch.cuda.empty_cache()
    l_num=len(l_targets)
    kmeans = K_Means(k, pairwise_batch_size = 200)
    kmeans.fit_mix(torch.from_numpy(u_feats).to(device), torch.from_numpy(l_feats).to(device), torch.from_numpy(l_targets).to(device))
    cat_pred = kmeans.labels_.cpu().numpy() 
    u_pred = cat_pred[l_num:]
    silh_score = silhouette_score(u_feats, u_pred)
    del kmeans
    return silh_score, cat_pred 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_val_cls', default=0, type=int)
    parser.add_argument('--max_cand_k', default=100, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--split_ratio', type=float, default=0.7)
    parser.add_argument('--min_max_ratio', type=float, default=0.01)
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/pretrained/vgg6_omniglot_proto.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets')
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_torch(args.seed)
    model = VGG(n_layer='4+2', in_channels=1).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    model.last = Identity()

    labeled_loaders = {}
    for alphabetStr in omniglot_background_val_alphabets: 
        _, labeled_loaders[alphabetStr] = omniglot_alphabet_func(alphabet=alphabetStr, background=True, root=args.dataset_root)(batch_size=args.batch_size, num_workers=args.num_workers)

    acc = {}
    nmi = {}
    ari = {}
    gtK = {}
    predK = {}
    for _, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
        _, eval_Dloader = omniglot_alphabet_func(alphabet=alphabetStr, background=False, root=args.dataset_root)(batch_size=args.batch_size, num_workers=args.num_workers)
        gtK[alphabetStr] = eval_Dloader.num_classes
        predK[alphabetStr] = estimate_k(model, eval_Dloader, labeled_loaders, args)
    print('GT K:', gtK)
    print('Pred K:', predK)
    print('Average K error: {:.4f}'.format(np.mean(abs(np.array(gtK.values())-np.array(predK.values())))))
