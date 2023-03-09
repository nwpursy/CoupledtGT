import torch
import torch.nn as nn
import numpy as np

from GTCP_u2 import Const,util

class GTCPredictor(nn.Module):
    def __init__(self,points_set,coupled_locations):
        super(GTCPredictor,self).__init__()
        self.node_size=points_set.size(0)
        self.gd2dpg=GaussianDiff2dPG(len(Const.non_wind_cols),points_set)
        self.bgdm=BatchedGaussianDiffModel2d(coupled_locations)
        self.decoder=DenseDecoder(self.node_size,Const.framework_hidden_size)
        # self.data_lowerbound=data_lowerbound
        # self.data_upperbound=data_upperbound
        # self.out_act=nn.Softplus(beta=2)

    def forward(self,hidden_states,factors,wind_vectors):
        '''
        :param hidden_states: shape=(batch,node_size,hidden_size)
        :param factors: shape=(batch,3)
        :param wind_vectors: shape=(batch,2)
        :param point_set: shape=(node_size,2)
        :return: scaled_y_hat: shape=(batch,node_size)
        :return: coupling_matrix: shape=(batch,node_size,node_size)
        '''

        batch_size=wind_vectors.size(0)
        if Const.intra_coupling_weight>=0 and Const.intra_coupling_weight<1:
            sigma,q=self.gd2dpg(factors)
            # util.plotOneHeatmap(sigma[0:32].cpu(),title='sigma')
            # util.plotOneHeatmap(q[0:32].cpu(),title='q')
            self.bgdm.set_sigma(sigma)
            if Const.ones_q == True:
                self.bgdm.set_q(torch.ones_like(q, device=q.device))
            else:
                self.bgdm.set_q(q)
            coupling_matrix, pdf_value=self.bgdm.forward(wind_vectors)
            batched_eye=torch.eye(self.bgdm.node_size,device=Const.device).unsqueeze(0).repeat(batch_size,1,1)
            coef=Const.intra_coupling_weight*batched_eye+(1-Const.intra_coupling_weight)*coupling_matrix
            rep=torch.bmm(coef,hidden_states)
        else:
            rep=hidden_states
            coupling_matrix=torch.randn(batch_size,self.bgdm.node_size,self.bgdm.node_size,device=Const.device)
            sigma=torch.randn(batch_size,2,device=Const.device)
            q=torch.randn(batch_size,self.bgdm.node_size,device=Const.device)
        scaled_y_hat=self.decoder(rep)
        return scaled_y_hat,coupling_matrix,sigma,q

class DenseDecoder(nn.Module):
    def __init__(self,node_size,input_size):
        super(DenseDecoder, self).__init__()
        self.innerDense=nn.Linear(input_size,2*input_size)
        self.outerDense=nn.Linear(2*input_size,1)
        self.bn1=nn.BatchNorm1d(node_size)
        self.inner_act=nn.LeakyReLU()
        self.out_act=nn.Softplus(beta=2)
    def forward(self,x):
        middle=self.innerDense(x)
        middle = self.bn1(middle)
        middle=self.inner_act(middle)
        added_rep=self.outerDense(middle).squeeze(-1)
        scaled_y_hat = self.out_act(added_rep)
        return scaled_y_hat

class GaussianDiff2dPG(nn.Module):
    '''
    :param factors_batch shape=(batch,3) \n
    :param point_set shape=(node,2) \n
    :return sigma shape=(batch,2) \n
    :return q shape=(batch,node)
    '''
    def __init__(self,factor_in_size,points_set):
        super(GaussianDiff2dPG, self).__init__()
        self.points_set=points_set
        node_size=points_set.size(0)
        self.factor_encoder=nn.Linear(factor_in_size,Const.gaussian_PG_hidden_size)
        self.position2d_encoder=nn.Linear(2,Const.gaussian_PG_hidden_size)
        self.lpg=nn.Linear(Const.gaussian_PG_hidden_size*2,1)
        self.gpg=nn.Linear(Const.gaussian_PG_hidden_size,2)
        self.act=nn.LeakyReLU()
        self.scaler=nn.Softplus()
        # self.bn_factor_in=nn.BatchNorm1d(factor_in_size)
        self.bn_points_in=nn.BatchNorm1d(2)
        self.bn_gpg_prepare=nn.BatchNorm1d(Const.gaussian_PG_hidden_size)
        self.bn_lpg_prepare=nn.BatchNorm1d(node_size)
    def forward(self,factors_batch):
        scaled_f1=util.min_max_scale(factors_batch[:,0],-100,400)
        scaled_f2=util.min_max_scale(factors_batch[:,1],-200,250)
        scaled_f3=util.min_max_scale(factors_batch[:,2],10000,10400)
        scaled_factors=torch.stack([scaled_f1,scaled_f2,scaled_f3],dim=-1)
        factors_encoding=self.act(self.factor_encoder(scaled_factors))
        sigma=self.scaler(self.gpg(self.bn_gpg_prepare(factors_encoding)))
        position2d_encoding=self.act(self.position2d_encoder(self.bn_points_in(self.points_set)))
        batch_size=factors_encoding.size(0)
        node_size=position2d_encoding.size(0)
        fm=factors_encoding.unsqueeze(1).repeat(1,node_size,1)
        pm=position2d_encoding.unsqueeze(0).repeat(batch_size,1,1)
        local_encoding=torch.cat([fm,pm],dim=-1)
        q=self.lpg(self.bn_lpg_prepare(local_encoding)).squeeze(-1)
        # print(sigma.size())
        # print(q.size())
        return sigma,q

class BatchedGaussianDiffModel2d:
    '''
        :arg coupled_locations: shape=(node,node,2)
        :arg sigma: shape=(batch,2)
        :arg q: shape=(batch,node)
    '''
    def __init__(self,coupled_locations):
        super(BatchedGaussianDiffModel2d,self).__init__()
        self.coupled_locations=coupled_locations
        self.node_size=coupled_locations.size(0)
        # self.softmax=nn.Softmax(dim=-1)
    def set_sigma(self,sigma):
        self.sigma=sigma
    def set_q(self,q):
        self.q=q
    def forward(self,batch_wind_vecters):
        '''
        :param batch_wind_vecters: shape=(batch,2)
        :returns: coupling_matrix,pdf_value: shape=(batch,node,node)
        '''
        batch_size=batch_wind_vecters.size(0)
        batched_inputs=self.get_batched_inputs(batch_wind_vecters)
        pdf_value=self.get_pdf_value(batched_inputs)
        broadcast_q=self.q.unsqueeze(1).repeat(1,self.node_size,1)
        coupling_matrix=broadcast_q*pdf_value
        # coupling_matrix=util.mask_diag_element(coupling_matrix)
        return coupling_matrix,pdf_value
    def get_batched_inputs(self,batch_wind_vecters):
        batch_size, wind_dim = batch_wind_vecters.size()
        batch_coupled_loc = self.coupled_locations.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        broadcast_wind = batch_wind_vecters.view(batch_size, 1, 1, wind_dim).repeat(1, self.node_size, self.node_size,1)
        batched_inputs = batch_coupled_loc - broadcast_wind  # (batch,node,node,2)
        return batched_inputs
    def get_pdf_value(self,batched_inputs):
        batch_size=batched_inputs.size(0)
        batched_inputs_x = batched_inputs[:, :, :, 0]
        batched_inputs_y = batched_inputs[:, :, :, 1]
        # analysis(batched_inputs_x,'x')
        # analysis(batched_inputs_y,'y')
        broadcast_sigma_x = self.sigma[:, 0].view(batch_size, 1, 1).repeat(1, self.node_size, self.node_size)
        broadcast_sigma_y = self.sigma[:, 1].view(batch_size, 1, 1).repeat(1, self.node_size, self.node_size)
        core = (batched_inputs_x ** 2) / (broadcast_sigma_x ** 2) + (batched_inputs_y ** 2) / (broadcast_sigma_y ** 2)
        log_pdf_value=-torch.log(2 * np.pi * broadcast_sigma_x * broadcast_sigma_y)-core/2
        masked_log_pdf=util.mask_diag_element_by_value(log_pdf_value,-np.inf)
        pdf_value=torch.exp(masked_log_pdf)
        # pdf_value=self.softmax(masked_log_pdf)
        return pdf_value

def analysis(batched_inputs,title):
    print('min', torch.min(batched_inputs))
    print('max', torch.max(batched_inputs))
    print('mean', torch.mean(batched_inputs))
    print('median', torch.median(batched_inputs))
    from preprocess import util as putil
    putil.plotCDF_1Series(torch.flatten(batched_inputs).cpu(), bin=np.arange(-150, 150, 1), title=title)

if __name__=='__main__':
    # gd2dpg=GaussianDiff2dPG(3)
    # factors_batch=torch.randn(64,3)
    # point_set=torch.randn(25,2)
    # gd2dpg(factors_batch,point_set)

    coupled_loc=torch.randn(17,17,2,device=Const.device)
    point_set = torch.randn(17, 2, device=Const.device)
    gtcp=GTCPredictor(point_set,coupled_loc).to(Const.device)
    hidden=torch.randn(64,17,32,device=Const.device)
    factors=torch.randn(64,3,device=Const.device)
    wind=torch.randn(64,2,device=Const.device)
    y_hat,coupling=gtcp(hidden,factors,wind)
    print(y_hat.size())
    print(coupling.size())
    # bgdm=BatchedGaussianDiffModel2d(coupled_loc)
    # bgdm.set_sigma(torch.randn(64,2))
    # bgdm.set_q(torch.randn(64,17))
    # bgdm.forward(torch.randn(64,2))

    # print(a)
    # print(a.repeat(4,3))
    # a=torch.FloatTensor([1,4,3,6,5,7]).view(2,3)
    # print(a)
    # b=torch.FloatTensor([2,2,9,3,8,4]).view(2,3)
    # print(b)
    # print(a*b[0])