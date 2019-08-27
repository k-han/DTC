r'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''
import torch

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)
    
    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2 
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
            for group_index_c, group_c in enumerate(groups):
                    R, C = X[group_r], X[group_c]
                    if device!=-1:
                            R = R.cuda(device)
                            C = C.cuda(device)
                    group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict

