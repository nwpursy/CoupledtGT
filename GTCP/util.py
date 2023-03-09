import torch
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi']=300

def softplus(x):
    return torch.log(1+torch.exp(x))

def calculate_XY_vectors(points_set):
    '''
    :param points_set: shape=(stations_size,2)
    :return: coupled_locations , shape=(stations_size,stations_size,2)
    '''
    points_size = points_set.size(0)
    coupled_locations = []
    for n in range(points_size):
        pn = points_set[n]
        pn_end = points_set - pn
        coupled_locations.append(pn_end)
    coupled_locations = torch.stack(coupled_locations)
    return coupled_locations

def mask_diag_element(batch_matrix):
    batch_size=batch_matrix.size(0)
    ones=torch.ones_like(batch_matrix,device=batch_matrix.device)
    batched_eye=torch.eye(batch_matrix.size(1),device=batch_matrix.device).unsqueeze(0).repeat(batch_size,1,1)
    mask=ones-batched_eye
    return batch_matrix*mask
def mask_diag_element_by_value(batch_matrix,value):
    batch_size = batch_matrix.size(0)
    full=torch.full_like(batch_matrix,value,device=batch_matrix.device)
    batched_eye = torch.eye(batch_matrix.size(1), device=batch_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
    b=batched_eye.bool()
    return torch.where(b,full,batch_matrix)

    # m=torch.diagonal(batch_matrix,dim1=-2,dim2=-1)
    # d=torch.diag_embed(m)
    # zero_masked_batch_matrix=batch_matrix-d
    # batched_eye = torch.eye(batch_matrix.size(1), device=batch_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
    # mask=value*batched_eye
    # return zero_masked_batch_matrix+mask
# 输出模型训练参数量
def model_size(model, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:.4f} KB'.format(model._get_name(), para * type_size / 1000 ))

def plotLossCurve(train_losses, valid_losses,model_name, start_index=0):
    index = np.arange(start=start_index, stop=len(train_losses))
    plt.figure()
    plt.title('Ubuntu 1 '+model_name)
    plt.tick_params(labelsize=14)
    plt.plot(index, train_losses[index], color='r', label='train_losses')
    plt.plot(index, valid_losses[index], color='k', label='valid_losses')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20, }
    plt.xlabel('Epoch', font)
    plt.ylabel('Loss', font)
    plt.grid(linestyle='-.')
    # pyplot.title('RMSE:' + rmse + '  MAE:' + mae + '  R2_SCORE:' + r2)
    plt.legend()
    # pyplot.savefig('d:/Result.png')
    plt.show()

def plotLineChart(inp,truth,pred,title):
    plt.figure()
    plt.title('Ubuntu 1 '+title)
    in_len = len(inp)
    out_len = len(pred)
    ground_truth = np.concatenate([inp, truth])
    plt.plot(ground_truth, color='b', label='real')
    out_indices = np.arange(start=in_len, stop=in_len + out_len)
    plt.plot(out_indices,pred, color='r', label='predict')
    plt.show()

def plotLineCharts(n_row,n_col,input_group,truth_group,pred_group,title,names):
    fig, ax = plt.subplots(n_row, n_col)
    plt.suptitle('Ubuntu 1 '+title)
    # plt.grid(linestyle='-.')
    # plt.legend()
    k = 0
    for i in range(n_row):
        for j in range(n_col):
            inp=input_group[k]
            truth=truth_group[k]
            pred=pred_group[k]
            in_len=len(inp)
            out_len=len(pred)
            ground_truth=np.concatenate([inp,truth])
            ax[i][j].plot(ground_truth, color='b', label='real')
            out_indices = np.arange(start=in_len, stop=in_len+out_len)
            ax[i][j].plot(out_indices,pred, color='r', label='predict')
            ax[i][j].set_title(names[k])
            k += 1
    plt.show()

def plotOneHeatmap(data,title=''):
    im=plt.imshow(data,cmap=plt.cm.coolwarm)
    plt.colorbar(orientation='vertical')
    plt.title('Ubuntu 1 '+title)
    plt.show()
def plotManyHeatmap(n_row,n_col,data_group,title,names):
    fig, ax = plt.subplots(n_row, n_col, sharex=True, sharey=True, constrained_layout=False, figsize=(5, 7))
    fig.suptitle('Ubuntu 1 '+title)
    # plt.grid(linestyle='-.')
    # plt.legend()
    k = 0
    for i in range(n_row):
        for j in range(n_col):
            data=data_group[k]
            im=ax[i][j].imshow(data, cmap=plt.cm.coolwarm)
            # ax[i][j].colorbar(orientation='horizontal')
            ax[i][j].set_title(names[k])
            # plt.legend()
            k += 1
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.05,left=0.1,right=0.84,top=0.9,wspace=0.1)
    cax = plt.axes([0.85, 0.05, 0.02, 0.85])
    plt.colorbar(im,cax=cax)
    # cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])  # 设置colarbar位置
    # cbar = fig.colorbar(im, cax=cb_ax)  # 共享colorbar
    # plt.subplots_adjust()
    # plt.tight_layout()
    plt.show()

def min_max_scale(data,data_lowerbound,data_upperbound,feature_range=(0,1)):
    feature_min=feature_range[0]
    feature_max=feature_range[1]
    data_std=(data-data_lowerbound)/(data_upperbound-data_lowerbound)
    data_scaled=data_std*(feature_max-feature_min)+feature_min
    return data_scaled
def min_max_inv_scale(data_scaled,data_lowerbound,data_upperbound,feature_range=(0,1)):
    feature_min = feature_range[0]
    feature_max = feature_range[1]
    data_std=(data_scaled-feature_min)/(feature_max-feature_min)
    data_origin=data_std*(data_upperbound-data_lowerbound)+data_lowerbound
    return data_origin

if __name__=='__main__':
    # print(torch.eye(3).repeat(2,1,1))
    # import torch.nn as nn

    a=torch.randn(12,25,25)
    # names=np.arange(12)
    # plotOneHeatmap(a[0],'test')

    names = []
    for i in range(len(a)):
        names.append('step {}'.format(i + 1))
    plotManyHeatmap(4,3,a,'test',names)
    # inp=torch.randn(6,72)
    # truth=torch.randn(6,12)
    # pred=torch.randn(6,12)
    # title='ggg'
    # names=[1,2,3,4,5,6]
    # plotLineCharts(3,2,inp,truth,pred,title,names)
    x=torch.randn(2,3,3)
    # print(x)
    import torch.nn as nn
    s=nn.Softmax(dim=1)
    # print(s(x))
    m=mask_diag_element_by_value(x,-np.inf)
    print(m)
    print(s(m))

    # from torch import nn
    # a=torch.rand(2,4,5)
    # print(a)
    # bn=nn.BatchNorm1d(1)
    # print(bn(a))