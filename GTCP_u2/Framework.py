import torch
import torch.nn as nn
import numpy as np
import random

from GTCP_u2 import Const,Model,util
from GTCP_u2 import MultivariateSeq2Seq as mvs2s

class Framework(nn.Module):
    def __init__(self,points_set,coupled_locations):
        '''
        :param points_set: shape=(node,2)
        :param coupled_locations: shape=(node,node,2)
        '''
        super(Framework,self).__init__()
        self.encoder=mvs2s.HomoEncoder(Const.framework_hidden_size,1,Const.node_count,share_params=False)
        # self.updater=mvs2s.HomoUpdater(Const.framework_hidden_size,Const.framework_hidden_size,
        #                 node_size=Const.node_count,share_params=False)
        self.updater = mvs2s.HomoUpdater(Const.framework_hidden_size, 1,
                        node_size=Const.node_count, share_params=False)
        # self.attn=mvs2s.HomoAttention(Const.framework_hidden_size,Const.framework_hidden_size,
        #                 4*Const.framework_hidden_size,Const.node_count,share_params=False)
        self.gtc=Model.GTCPredictor(points_set,coupled_locations)
        # self.bn=nn.BatchNorm1d(Const.node_count)
    def forward(self,input_tensor,target_tensor,factors_tensor,wind_vec_tensor,teacher_forcing_ratio):
        '''
        :param input_tensor: shape=(batch,node,in_len)
        :param target_tensor: shape=(batch,node,out_len)
        :param factors_tensor: shape=(batch,3,out_len)
        :param wind_vec_tensor: shape=(batch,2,out_len)
        :param teacher_forcing_ratio:
        :return dec_output: shape=(batch,node,out_len)
        :return coupling_matrices: shape=(batch,out_len,node,node)
        '''
        scaled_input=util.min_max_scale(input_tensor,Const.data_lowerbound,Const.data_upperbound)
        scaled_target=util.min_max_scale(target_tensor,Const.data_lowerbound,Const.data_upperbound)

        encoder_outputs, encoder_hidden = self.encoder(scaled_input.unsqueeze(-1))
        hidden_states=encoder_outputs[:,:,-1]
        dec_output=[]
        coupling_matrices=[]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_hidden=encoder_hidden
        all_sigma=[]
        all_q=[]
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(Const.target_time_window):
                scaled_y_hat, coupling_matrix,sigma,q=self.gtc(hidden_states,factors_tensor[:,:,di],wind_vec_tensor[:,:,di])
                y_hat=util.min_max_inv_scale(scaled_y_hat,Const.data_lowerbound,Const.data_upperbound)
                dec_output.append(y_hat)
                coupling_matrices.append(coupling_matrix)
                all_sigma.append(sigma)
                all_q.append(q)
                # context_group,attn_weight_group=self.attn(encoder_outputs.transpose(1,2).transpose(0,1),hidden_states)
                decoder_hidden=self.updater(scaled_target[:,:,di:di+1],decoder_hidden,None)
                hidden_states=decoder_hidden[0]
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(Const.target_time_window):
                scaled_y_hat, coupling_matrix,sigma,q=self.gtc(hidden_states,factors_tensor[:,:,di],wind_vec_tensor[:,:,di])
                y_hat = util.min_max_inv_scale(scaled_y_hat, Const.data_lowerbound, Const.data_upperbound)
                dec_output.append(y_hat)
                coupling_matrices.append(coupling_matrix)
                all_sigma.append(sigma)
                all_q.append(q)
                # scaled_y_hat=util.min_max_scale(y_hat.detach(),Const.data_lowerbound, Const.data_upperbound)
                # context_group, attn_weight_group = self.attn(encoder_outputs.transpose(1, 2).transpose(0, 1),
                #                                              hidden_states)
                decoder_hidden=self.updater(scaled_y_hat.detach().unsqueeze(-1),decoder_hidden,None)
                hidden_states=decoder_hidden[0]
        dec_output=torch.stack(dec_output,dim=-1)
        coupling_matrices=torch.stack(coupling_matrices,dim=1)
        all_sigma=torch.stack(all_sigma,dim=1) #(batch,out_len,2)
        all_q=torch.stack(all_q,dim=1) #(batch,out_len,node)
        return dec_output,coupling_matrices,all_sigma,all_q
    def evaluate(self,input_tensor,factors_tensor,wind_vec_tensor,prediction_length):
        self.eval()
        with torch.no_grad():
            scaled_input = util.min_max_scale(input_tensor, Const.data_lowerbound, Const.data_upperbound)
            encoder_outputs, encoder_hidden = self.encoder(scaled_input.unsqueeze(-1))
            hidden_states = encoder_outputs[:, :, -1]

            # for t in range(encoder_outputs.size(2)):
            #     test=encoder_outputs[0,:,t].cpu()
            #     test=np.round(test,3)
            #     util.plotOneHeatmap(test,title='step {}'.format(t))
            # test=hidden_states[1].cpu()
            # test=np.round(test,3)
            # util.plotOneHeatmap(test)
            # from preprocess import util as putil
            # c=hidden_states.flatten()
            # cp=c.cpu()
            # putil.plotCDF_1Series(cp,bin=np.arange(-1,1.1,0.01),title='encoder')

            dec_output = []
            coupling_matrices = []
            all_sigma = []
            all_q = []
            decoder_hidden = encoder_hidden
            for di in range(prediction_length):
                scaled_y_hat, coupling_matrix,sigma,q = self.gtc(hidden_states, factors_tensor[:, :, di], wind_vec_tensor[:, :, di])
                y_hat = util.min_max_inv_scale(scaled_y_hat, Const.data_lowerbound, Const.data_upperbound)
                dec_output.append(y_hat)
                coupling_matrices.append(coupling_matrix)
                all_sigma.append(sigma)
                all_q.append(q)
                # scaled_y_hat = util.min_max_scale(y_hat.detach(), Const.data_lowerbound, Const.data_upperbound)
                # context_group, attn_weight_group = self.attn(encoder_outputs.transpose(1, 2).transpose(0, 1),
                #                                              hidden_states)
                decoder_hidden = self.updater(scaled_y_hat.detach().unsqueeze(-1), decoder_hidden,None)
                hidden_states = decoder_hidden[0]

                # test = decoder_hidden[1][1].cpu()
                # test = np.round(test, 3)
                # util.plotOneHeatmap(test,title='step {}'.format(di))
                # putil.plotCDF_1Series(torch.flatten(hidden_states).cpu(), bin=np.arange(-1, 1.1, 0.01), title='step {}'.format(di))
            dec_output = torch.stack(dec_output, dim=-1)
            coupling_matrices = torch.stack(coupling_matrices, dim=1)
            all_sigma = torch.stack(all_sigma, dim=1)  # (batch,out_len,2)
            all_q = torch.stack(all_q, dim=1)  # (batch,out_len,node)
            return dec_output,coupling_matrices,all_sigma,all_q

class WeightedMSELoss(nn.MSELoss):
    def __init__(self,weight):
        super(WeightedMSELoss, self).__init__(reduction='none')
        self.weight=weight/weight.sum()
        self.weight_len=len(weight)
    def forward(self,pred,real):
        assert pred.is_same_size(real)
        assert pred.size(-1)==self.weight_len
        L=super(WeightedMSELoss, self).forward(pred, real)
        if pred.ndim==3:
            weighted_non_reduc=L*self.weight.repeat(L.size(0),L.size(1),1)
        elif pred.ndim==2:
            weighted_non_reduc = L * self.weight.repeat(L.size(0), 1)
        else:
            weighted_non_reduc=L
        weighted_non_reduc=weighted_non_reduc.sum(dim=-1)
        return weighted_non_reduc.mean()

if __name__=='__main__':
    # from GTCP import DataPreparation as dp
    # points_set=dp.load_locations(Const.dataset_name,file_name='points_set')
    # coupled_locations=dp.load_locations(Const.dataset_name)
    # model=Framework(points_set,coupled_locations).to(Const.device)
    # train_data = dp.make_data(Const.dataset_name,'test',shuffle=True)
    # for i, batch in enumerate(train_data):
    #     print(i)
    #     print(batch[0].size())
    #     print(batch[1].size())
    #     print(batch[2].size())
    #     print(batch[3].size())
    #     print(batch[4].size())
    #     print(batch[0].device)
    #     print('start calculation')
    #     dec_output,coupled_matrices=model(batch[1],batch[2],batch[3],batch[4],0)
    #     # dec_output, coupled_matrices = model.evaluate(batch[1], batch[3], batch[4], 11)
    #     print(dec_output.size())
    #     print(coupled_matrices.size())
    #     break
    loss=nn.MSELoss(reduction='mean')
    loss2=WeightedMSELoss(torch.FloatTensor([1, 4, 2]))
    a=torch.randn(2,2,3)
    b=torch.randn(2,2,3)
    print(loss(a,b))
    print(loss2(a,b))
    # torch.is_same_size()
    print(a.ndim)
    # print((a-b)**2)
    # print(torch.mean((a-b)**2))
    # print(a)
    # print(a.flatten())
    # print(loss(a,b))