import torch
import torch.nn as nn

from GTCP_u2 import Const
from GTCP_u2 import UnivariateSeq2Seq as uvs2s


class HomoEncoder(nn.Module):
    '''
    :arg hidden_size: LSTM hidden size
    :arg input_size: feature dimension in real world
    :arg share_params: boolean arg
    :arg node_size: need to be specified when share_params==True

    :parameter x shape=(batch,node_size,enc_seq_length,input_size) \n
    :returns output, (h_n,c_n) \n
    :return output shape=(batch,node_size,enc_seq_length,hidden_size) \n
    :return h_n shape=(batch,node_size,hidden_size) \n
    :return c_n shape same as h_n
    '''
    def __init__(self,hidden_size,input_size,node_size=1,share_params=True):
        super(HomoEncoder, self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.share_params=share_params
        if share_params:
            self.encoder=uvs2s.EncoderLSTM(hidden_size,input_size)
        else:
            self.node_size=node_size
            encoder=[]
            for i in range(node_size):
                encoder.append(uvs2s.EncoderLSTM(hidden_size,input_size))
                self.encoder=nn.ModuleList(encoder)

    def forward(self,x):
        node_size=x.size(1)
        # assert node_size==self.node_size
        output=[]
        hidden_h=[]
        hidden_c=[]
        for i in range(node_size):
            x_i = x[:, i]
            if self.share_params:
                output_i, hidden_cell_i = self.encoder(x_i)
            else:
                output_i, hidden_cell_i = self.encoder[i](x_i)
            output.append(output_i)
            hidden_h.append(hidden_cell_i[0])
            hidden_c.append(hidden_cell_i[1])
        output=torch.stack(output,dim=1)
        hidden_h=torch.stack(hidden_h,dim=1)
        hidden_c=torch.stack(hidden_c,dim=1)
        return output,(hidden_h,hidden_c)

class HomoUpdater(nn.Module):
    '''
    :arg hidden_size: LSTM hidden size
    :arg output_size: feature dimension in real world
    :arg share_params: boolean arg
    :arg node_size: need to be specified when share_params==True

    :parameter y shape=(batch,node_size,1) \n
    :parameter hidden_cell=(s_i,c_i) s_i.shape=(batch,node_size,hidden_size), c_i.shape same as s_i \n
    :parameter attn_context shape=(batch,node_size,enc_vec_size) if exist \n
    :returns (s_{i+1},c_{i+1})
    '''
    def __init__(self, hidden_size, output_size=1,node_size=1,share_params=True):
        super(HomoUpdater, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.share_params = share_params
        if share_params:
            self.updater=uvs2s.UpdaterLSTM(hidden_size,output_size)
        else:
            self.node_size=node_size
            updater=[]
            for i in range(node_size):
                updater.append(uvs2s.UpdaterLSTM(hidden_size,output_size))
                self.updater=nn.ModuleList(updater)

    def forward(self, y, hidden_cell, attn_context=None):
        node_size=y.size(1)
        hidden_h=[]
        hidden_c=[]
        for j in range(node_size):
            y_j=y[:,j]
            hi_j=hidden_cell[0][:,j]
            ci_j=hidden_cell[1][:,j]
            if attn_context!=None:
                attn_j=attn_context[:,j]
            else:
                attn_j=None
            if self.share_params:
                hidden_cell_next=self.updater(y_j,(hi_j,ci_j),attn_j)
            else:
                hidden_cell_next = self.updater[j](y_j, (hi_j, ci_j), attn_j)
            hidden_h.append(hidden_cell_next[0])
            hidden_c.append(hidden_cell_next[1])
        hidden_h = torch.stack(hidden_h, dim=1)
        hidden_c = torch.stack(hidden_c, dim=1)
        return (hidden_h,hidden_c)

class HomoAttention(nn.Module):
    def __init__(self,enc_vec_size,dec_vec_size,attn_hidden_size,node_size=1,share_params=True):
        super(HomoAttention, self).__init__()
        self.share_params = share_params
        if share_params:
            self.attn=uvs2s.Attention(enc_vec_size,dec_vec_size,attn_hidden_size)
        else:
            self.node_size=node_size
            attn=[]
            for i in range(node_size):
                attn.append(uvs2s.Attention(enc_vec_size,dec_vec_size,attn_hidden_size))
                self.attn=nn.ModuleList(attn)

    def forward(self,elements_group,compare_group):
        '''
        :param elements_group: shape=(enc_element_num,batch,node_size,enc_vec_size)
        :param compare_group: shape=(batch,node_size, dec_vec_size)
        :return: context_group: shape=(batch,node_size,enc_vec_size)
        :return: attn_weight_group: shape=(batch,node_size,enc_element_num)
        '''
        context_group = []
        attn_weight_group = []
        for j in range(self.node_size):
            elements=elements_group[:,:,j]
            compare=compare_group[:,j]
            if self.share_params:
                context, attn_weight=self.attn(elements,compare)
            else:
                context, attn_weight = self.attn[j](elements, compare)
            context_group.append(context)
            attn_weight_group.append(attn_weight)
        context_group=torch.stack(context_group,dim=1)
        attn_weight_group=torch.stack(attn_weight_group,dim=1)
        return context_group,attn_weight_group

if __name__=='__main__':
    # net=HomoEncoder(64,2,node_size=25,share_params=True).to(Const.device)
    # print(net)
    # x=torch.randn(50,25,72,2).to(Const.device)
    # out,hidden=net(x)
    # print(out.size())
    # print(hidden[0].size())
    # print(hidden[1].size())
    net=HomoUpdater(64,node_size=25,share_params=False).to(Const.device)
    print(net)
    y=torch.randn(50,25,1).to(Const.device)
    h=torch.randn(50,25,64).to(Const.device)
    c=torch.randn(50,25,64).to(Const.device)
    (hn,cn)=net(y,(h,c))
    print(hn.size())
    print(cn.size())