import torch
import torch.nn as nn
from torch.functional import F

from GTCP_u2 import Const

# 输入 x, hidden_cell=(h_0,c_0)
#   x.shape=(batch,enc_seq_length,input_size)
# 输出 output, hidden_cell=(h_n,c_n)
#  output.shape=(batch,enc_seq_length,hidden_size)
#  h_0.shape=(1,batch,hidden_size)
#  c_0同h_0
#  h_n.shape=(batch,hidden_size)
#  c_n同h_n
class EncoderLSTM(nn.Module):
    def __init__(self,hidden_size, input_size):
        super(EncoderLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size;
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
    def forward(self,x):
        batch_size=x.size(0)
        hidden_cell=self.initHiddenCell(batch_size)
        output,hidden_cell=self.lstm(x,hidden_cell)
        h=hidden_cell[0]
        c=hidden_cell[1]
        h=h[0]
        c=c[0]
        hidden_cell=(h,c)
        # print('enc_output size:',output.size())
        return output, hidden_cell
    def initHiddenCell(self,batch_size):
        return (torch.zeros(1,batch_size,self.hidden_size,device=Const.device),
                            torch.zeros(1,batch_size,self.hidden_size,device=Const.device))

# 输入y, hidden_cell=(s_i,c_i), attn_context
#   y.shape=(batch,1)
#   s_i.shape=(batch,hidden_size)
#   c_i同h_i
#   attn_context.shape=(batch,enc_vec_size)（若存在）
# 输出 hidden_cell=(s_{i+1},c_{i+1})
class UpdaterLSTM(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super(UpdaterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm=nn.LSTMCell(output_size, hidden_size)

    def forward(self, y, hidden_cell, attn_context=None):
        yc=y
        if attn_context!=None:
            yc = torch.cat((attn_context, y), dim=1)

        hidden_cell = self.lstm(yc, hidden_cell)
        return hidden_cell
        # y_hat=self.out(hidden_cell[0]).view(-1,self.output_size)
        # return y_hat, hidden_cell


# 输入encoder提供的全体elements(不一定是序列模型), decoder每个阶段被比较的向量compare
#    elements.shape=(enc_element_num,batch,enc_vec_size)
#    compare.shape=(batch, dec_vec_size)
# 输出elements的加权和context，权重列表attn_weight
#    context.shape=(batch,enc_vec_size)
#    attn_weight.shape=(batch,enc_element_num)
class Attention(nn.Module):
    def __init__(self,enc_vec_size,dec_vec_size,attn_hidden_size):
        super(Attention,self).__init__()
        self.attn_inner=nn.Linear(enc_vec_size+dec_vec_size,attn_hidden_size)
        self.attn_outer=nn.Linear(attn_hidden_size,1,bias=False)
    def forward(self,elements,compare):
        enc_element_num=elements.size(0)
        ss=compare.repeat(enc_element_num,1,1)
        sh=torch.cat([ss,elements],dim=2) # shape=(enc_element_num,batch,enc_vec_size+dec_vec_size)
        attn_middle = torch.tanh(self.attn_inner(sh))
        attn_weight = F.softmax(self.attn_outer(attn_middle), dim=0)
        # 转换attn_weight.shape为(batch,1,enc_element_num)，用于bmm
        attn_weight = attn_weight.transpose(0, 1).transpose(1, 2)
        # context.shape=(batch,1,enc_vec_size)
        context = torch.bmm(attn_weight, elements.transpose(0, 1))
        attn_weight = attn_weight.view(-1, enc_element_num) # shape=(batch,enc_element_num)
        context=context.squeeze(dim=1)
        return context,attn_weight

if __name__=='__main__':
    net=UpdaterLSTM(8)
    y=torch.randn(50,1)
    h0=torch.randn(50,8)
    c0 = torch.randn(50, 8)
    ret=net(y,(h0,c0))
    print(ret[0].size())
    print(ret[1].size())
    a=torch.randn(2,5,3)