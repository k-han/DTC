# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        #  _assert_no_grad(target)
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    
    #  print('support_idxs', support_idxs)

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    #  print('dist', F.log_softmax(-dists, dim=1).shape)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    #  print('log_p_y', log_p_y.shape)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #  print(target_inds.shape)
    #  print(log_p_y)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val

def prototypical_loss_pair(input, target, n_support, normalize=False):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    if normalize:
        radius = 30.0
        input = F.normalize(input)*radius
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    
    #  print('support_idxs', support_idxs)

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    if normalize:
        prototypes = F.normalize(prototypes)
        prototypes = prototypes*radius
    #  print('prototypes', prototypes.shape, prototypes.norm(dim=1))
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    #  print('dists', dists.shape)

    #  print('dist', F.log_softmax(-dists, dim=1).shape)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    #  print('log_p_y', log_p_y.shape)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #  print(target_inds.shape)
    #  print(log_p_y)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    #  print('target, y_hat', target_inds.squeeze(2).shape, y_hat.shape)

    return loss_val,  acc_val

def prototypical_loss_pair_cyc(input, target, n_support, normalize=False):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    if normalize:
        radius = 30.0
        input = F.normalize(input)*radius
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    
    #  print('support_idxs', support_idxs)

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    if normalize:
        prototypes = F.normalize(prototypes)
        prototypes = prototypes*radius
    #  print('prototypes', prototypes.shape, prototypes.norm(dim=1))
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    #  print('dists', dists.shape)

    #  print('dist', F.log_softmax(-dists, dim=1).shape)
    log_p_y_AB = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    log_p_y_BA = F.log_softmax(-dists.transpose(0, 1), dim=1).view(n_classes, n_query, -1)
    #  print('log_p_y', log_p_y.shape)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #  print(target_inds.shape)
    #  print(log_p_y)
    loss_val_AB = -log_p_y_AB.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat_AB = log_p_y_AB.max(2)
    acc_val_AB = y_hat_AB.eq(target_inds.squeeze(2)).float().mean()

    loss_val_BA = -log_p_y_BA.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat_BA = log_p_y_BA.max(2)
    acc_val_BA = y_hat_BA.eq(target_inds.squeeze(2)).float().mean()

    loss_val = (loss_val_AB+loss_val_BA)/2.0
    acc_val = (acc_val_AB + acc_val_BA)/2.0
    return loss_val,  acc_val

def prototypical_center_loss(input, target, n_support=None):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero().squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu, sorted=True)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() 
    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.cat(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    class_to_idx = {classes.numpy()[i]: i for i in range(len(classes))}
    query_samples = input.to('cpu')[query_idxs]
    target_cpu = target_cpu[query_idxs] #sort targets according to query_idxs
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1)
    target_indx = [class_to_idx[i] for i in target_cpu.numpy()]
    target_indx = torch.Tensor(target_indx).long()
    loss_val = -log_p_y.gather(1, target_indx.unsqueeze(1)).view(-1).mean()
    _, y_hat = log_p_y.max(1)
    acc_val = y_hat.eq(target_indx).float().mean()

    return loss_val,  acc_val

def prototypical_mpair_loss(input, target, n_support=None):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    #  print('target', target_cpu)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    dists = euclidean_dist(input_cpu, input_cpu)
    n_per_cls = len(target)/n_classes
    dists_np = dists.detach().numpy()
    #  print('dist', dists)
    block_mask = np.kron(np.eye(n_classes), np.ones([n_per_cls, n_per_cls])) 
    dists_np = dists_np - 2*dists_np*block_mask
    #  print('dist np', dists_np)
    ind_tuple = np.zeros([len(target), n_classes])
    for i in range(n_classes):
        ind_tuple[:, i] = np.argmin(dists_np[:, i*n_per_cls:(i+1)*n_per_cls], axis=1) + i*n_per_cls

    ind_tuple = torch.from_numpy(ind_tuple).long()
    #  print(ind_tuple)
    dists = dists.gather(1, ind_tuple)
    #  print('dist after', dists) 
    #  class_to_idx = {classes.numpy()[i]: i for i in range(len(classes))}
    log_p_y = F.log_softmax(-dists, dim=1)
    #  target_indx = [class_to_idx[i] for i in target_cpu.numpy()]
    #  target_indx = torch.Tensor(target_indx).long()

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1)
    target_inds = target_inds.expand(n_classes, n_per_cls).long().contiguous().view(-1, 1)
    #  print('target_inds', target_inds)

    loss_val = -log_p_y.gather(1, target_inds).view(-1).mean()
    _, y_hat = log_p_y.max(1)
    #  print('y_hat, target_inds', y_hat, target_inds.squeeze())
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

def prototypical_mpair_l2_loss(input, target, radius=None):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    #  print('target', target_cpu)
    if radius is not None:
        input = F.normalize(input)*radius

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    dists = euclidean_dist(input_cpu, input_cpu)
    n_per_cls = len(target)/n_classes
    dists_np = dists.detach().numpy()
    #  print('dist', dists)
    block_mask = np.kron(np.eye(n_classes), np.ones([n_per_cls, n_per_cls])) 
    dists_np = dists_np - 2*dists_np*block_mask
    #  print('dist np', dists_np)
    ind_tuple = np.zeros([len(target), n_classes])
    for i in range(n_classes):
        ind_tuple[:, i] = np.argmin(dists_np[:, i*n_per_cls:(i+1)*n_per_cls], axis=1) + i*n_per_cls

    ind_tuple = torch.from_numpy(ind_tuple).long()
    #  print(ind_tuple)
    dists = dists.gather(1, ind_tuple)
    #  print('dist after', dists) 
    #  class_to_idx = {classes.numpy()[i]: i for i in range(len(classes))}
    log_p_y = F.log_softmax(-dists, dim=1)
    #  target_indx = [class_to_idx[i] for i in target_cpu.numpy()]
    #  target_indx = torch.Tensor(target_indx).long()

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1)
    target_inds = target_inds.expand(n_classes, n_per_cls).long().contiguous().view(-1, 1)
    #  print('target_inds', target_inds)

    loss_val = -log_p_y.gather(1, target_inds).view(-1).mean()
    _, y_hat = log_p_y.max(1)
    #  print('y_hat, target_inds', y_hat, target_inds.squeeze())
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val



if __name__ == "__main__":
    import numpy as np
    x = torch.randn(10, 256)
    y = torch.Tensor([10, 10, 2, 2, 3, 3, 4, 4, 6, 6])
    l, a = prototypical_mpair_loss(x, y)
    #  print(l, a)
    #  classes = torch.unique(y) 
    #  print('classes', classes)
    #  class_to_idx = {classes[i]: i for i in range(len(classes))}
    #  print('class_to_idx', class_to_idx)

    n_classes = 5 
    n_query = 2 
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #  print(target_inds.contiguous().view(-1))
    #  print(target_inds.contiguous().view(-1, 1))
