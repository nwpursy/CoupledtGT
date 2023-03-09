import pandas as pd
import torch
pd.set_option('display.max_columns', None) # 展示所有列
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random

from GTCP import Const,util


def readCSVFiles(pollution_csv_name,weather_csv_name,stations_csv_name,parentPath='../resource/data/beijing'):
    pollution_df=pd.read_csv('{}/{}.csv'.format(parentPath,pollution_csv_name),index_col='datetime')
    stations_name=pollution_df.columns.tolist()
    stations_df=pd.read_csv('{}/{}.csv'.format(parentPath,stations_csv_name),index_col='监测点')
    selected_stations=stations_df.loc[stations_name][['x_km','y_km']]
    weather_df=pd.read_csv('{}/{}.csv'.format(parentPath,weather_csv_name),index_col='datetime_LMT',usecols=Const.use_weather_cols)
    return pollution_df,weather_df,selected_stations

'''
attributes:
    dataFrame: pandas.DataFrame 全体大气污染指标数据, shape=(rowNum,stationsNum)
    weatherFrame: pandas.DataFrame 全体气象数据, shape=(rowNum,5)
        前三列：温度（℃*10）、露点（℃*10）、大气压（百帕*10）；
        后两列风速风向：wind_x_kmph, wind_y_kmph
    stationsFrame: pandas.DataFrame 全体站点平面直角坐标（km）, shape=(stationsNum,2)
    rowNum: int
    stationsNum: int
'''
class DataAccess:
    def __init__(self, dataFrame,weatherFrame,stationsFrame):
        self.dataFrame = dataFrame
        self.weatherFrame=weatherFrame
        self.stationsFrame=stationsFrame
        self.rowNum = dataFrame.shape[0]
        self.stationsNum = dataFrame.shape[1]

    ''' 
        输入dataFrame.shape=(L,stationsNum),weatherFrame.shape=(L,5), stationsFrame.shape=(stationsNum,2)
        input长度为T, target长度为I，采样跳步seq_sample_step，相位offset
        特征维度，weatherFrame分解为non_wind_seq, wind_seq；dataFrame不变
        时间维度，两个Frame转化为s=ceil((L-T-I+1)/seq_sample_step)个(input,target)样本
        随后过滤非法样本（包含缺失值-9999的样本），过滤后，样本数为S
        输出: indices矩阵，存储了target矩阵第一个time step的索引 (τ+1), shape=(S)
              input矩阵, shape=(S,stationsNum,T)
              target矩阵, shape=(S,stationsNum,I)
              non_wind矩阵，存储了气温、露点、大气压，时间从τ开始到τ+p-1结束，p=I, shape=(S,3,I)
              wind矩阵，存储了风向分量，时间范围同上，shape=(S,2,I)
              points_set矩阵，存储了每个站点的地理坐标，shape=(stationsNum,2)
    '''
    def createInoutSequences(self, input_time_window, target_time_window, seq_sample_step,offset=0):
        indices_seq=[]
        input_seq = []
        target_seq = []
        non_wind_seq=[]
        wind_seq=[]
        L = self.rowNum
        T = input_time_window
        I = target_time_window
        non_wind_df=self.weatherFrame[Const.non_wind_cols]
        wind_df=self.weatherFrame[Const.wind_cols]
        total_cnt=0
        success_cnt=0
        for i in tqdm(range(T, L - I-offset + 1, seq_sample_step)):
            total_cnt+=1
            inp = torch.FloatTensor(self.dataFrame[i+offset - T:i+offset].values).transpose(0,1).to(Const.device)
            if -9999 in inp:
                continue
            target = torch.FloatTensor(self.dataFrame[i+offset:i+offset + I].values).transpose(0,1).to(Const.device)
            if -9999 in target:
                continue
            non_wind=torch.FloatTensor(non_wind_df[i+offset-1:i+offset+I-1].values).transpose(0,1).to(Const.device)
            if -9999 in non_wind:
                continue
            wind=torch.FloatTensor(wind_df[i+offset-1:i+offset+I-1].values).transpose(0,1).to(Const.device)
            if -9999 in wind:
                continue
            indices_seq.append(i+offset)
            input_seq.append(inp)
            target_seq.append(target)
            non_wind_seq.append(non_wind)
            wind_seq.append(wind)
            success_cnt+=1
        self.indices_seq=torch.IntTensor(indices_seq).to(Const.device)
        self.input_seq = torch.stack(input_seq)
        self.target_seq = torch.stack(target_seq)
        self.non_wind_seq=torch.stack(non_wind_seq)
        self.wind_seq = torch.stack(wind_seq)
        self.points_set=torch.FloatTensor(self.stationsFrame.values).to(Const.device)
        print('total:',total_cnt,'success:',success_cnt,'success_ratio:',success_cnt/total_cnt)
        print(self.indices_seq.size())
        print(self.input_seq.size())
        print(self.target_seq.size())
        print(self.non_wind_seq.size())
        print(self.wind_seq.size())

    '''
    输出: 
          {category}_indices_seq.shape=({category}_size)
          {category}_input_seq.shape=({category}_size,stationsNum,T)
          {category}_target_seq.shape=({category}_size,stationsNum,I)
          {category}_non_wind_seq.shape=({category}_size,3,I)
          {category}_wind_seq.shape=({category}_size,2,I)
          for category in {'train','dev','test'}
          points_set.shape=(stationsNum,2)
    '''
    def divideInoutSequences(self, dev_ratio, test_ratio, parentPath='../resource/preprocess/beijing'):
        sample_size = self.indices_seq.size(0)
        seed = np.arange(sample_size)
        np.random.shuffle(seed)
        options=['indices','input','target','non_wind','wind']
        for option in options:
            seq=self.__getattribute__('{}_seq'.format(option))
            # shuffle in all sequences
            self.__setattr__('{}_seq'.format(option),seq[seed])

        dev_size = int(sample_size * dev_ratio)
        test_size = int(sample_size * test_ratio)
        train_size = sample_size - dev_size - test_size
        # divide
        train_sample_ids=np.arange(train_size)
        dev_sample_ids=np.arange(train_size,train_size + dev_size)
        test_sample_ids=np.arange(start=-test_size,stop=0)
        id_dict={'train':train_sample_ids,'dev':dev_sample_ids,'test':test_sample_ids}

        categories=['train','dev','test']
        for category in categories:
            sample_ids =id_dict[category]
            to_save=dict()
            for option in options:
                seq = self.__getattribute__('{}_seq'.format(option))
                to_save[option]=seq[sample_ids]
                print(option,to_save[option].size())
            to_save['points_set']=self.points_set
            print('points_set',to_save['points_set'].size())
            torch.save(to_save,'{}/{}/{}set@in{}&out{}&step{}&offset{}.pt'.format(
                parentPath,Const.dataset_name,category,Const.input_time_window,Const.target_time_window,
                Const.seq_sample_step,Const.step_offset))
            print(category,'saved')
    def businessProcess(self):
        self.createInoutSequences(Const.input_time_window, Const.target_time_window,
                                  Const.seq_sample_step,Const.step_offset)
        self.divideInoutSequences(Const.dev_ratio, Const.test_ratio)

class MyDataset(Dataset):
    def __init__(self,data_dict,sample_ratio=1):
        if sample_ratio<1 and sample_ratio>0:
            sample_size = int(sample_ratio * len(data_dict['indices']))
            roll=random.sample(range(len(data_dict['indices'])),sample_size)
            self.indices_seq = data_dict['indices'][roll]
            self.input_seq = data_dict['input'][roll]
            self.target_seq = data_dict['target'][roll]
            self.non_wind_seq = data_dict['non_wind'][roll]
            self.wind_seq = data_dict['wind'][roll]
        else:
            self.indices_seq=data_dict['indices']
            self.input_seq=data_dict['input']
            self.target_seq=data_dict['target']
            self.non_wind_seq=data_dict['non_wind']
            self.wind_seq=data_dict['wind']
        self.points_set=data_dict['points_set']
    def __getitem__(self, index):
        return self.indices_seq[index],self.input_seq[index],self.target_seq[index],\
               self.non_wind_seq[index],self.wind_seq[index]
    def __len__(self):
        return len(self.indices_seq)

def make_data(dataset_name,category,parentPath='../resource/preprocess/beijing',shuffle=False,sample_ratio=1):
    data_dict = torch.load('{}/{}/{}set@in{}&out{}&step{}&offset{}.pt'.format(parentPath,dataset_name,
                    category,Const.input_time_window,Const.target_time_window,
                    Const.seq_sample_step,Const.step_offset),map_location=Const.device)
    dataset = MyDataset(data_dict, sample_ratio=sample_ratio)
    dataLoader = DataLoader(dataset=dataset, batch_size=Const.batch_size, shuffle=shuffle)
    return dataLoader

def process_coupled_locations(dataset_name,category,parentPath='../resource/preprocess/beijing'
                                    ,outName='coupledGEO_XY_vectors',outPath='../resource/preprocess/beijing'):
    data_dict = torch.load('{}/{}/{}set@in{}&out{}&step{}&offset{}.pt'.format(parentPath, dataset_name,
                                                                              category, Const.input_time_window,
                                                                              Const.target_time_window,
                                                                              Const.seq_sample_step, Const.step_offset))
    points_set=data_dict['points_set']
    coupled_locations=util.calculate_XY_vectors(points_set)
    print(coupled_locations.size())
    torch.save(coupled_locations,'{}/{}/coupled_location.pt'.format(parentPath,dataset_name))

def load_locations(dataset_name,file_name='coupled_location',parent_path='../resource/preprocess/beijing'):
    t=torch.load('{}/{}/{}.pt'.format(parent_path,dataset_name,file_name),map_location=Const.device)
    return t
def process_locations(dataset_name,category,parentPath='../resource/preprocess/beijing'):
    data_dict = torch.load('{}/{}/{}set@in{}&out{}&step{}&offset{}.pt'.format(parentPath, dataset_name,
                                                                              category, Const.input_time_window,
                                                                              Const.target_time_window,
                                                                              Const.seq_sample_step, Const.step_offset))
    points_set = data_dict['points_set']
    torch.save(points_set,'{}/{}/points_set.pt'.format(parentPath,dataset_name))

if __name__=='__main__':
    pollution_df,weather_df,selected_stations=readCSVFiles(Const.dataset_name,
                                                           'BEIJING2013-2021_processed','stations')
    da=DataAccess(pollution_df,weather_df,selected_stations)
    da.businessProcess()

    # train_data = make_data(Const.dataset_name,'test',shuffle=True)
    # for i, batch in enumerate(train_data):
    #     print(i)
    #     print(batch[0].size())
    #     print(batch[1].size())
    #     print(batch[2].size())
    #     print(batch[3].size())
    #     print(batch[4].size())
    #     print(batch[5].size())
    #     break

    # process_coupled_locations(Const.dataset_name,'test')
    # load_coupled_locations('AQI_processed')
    # process_locations(Const.dataset_name,'test')
    # load_locations(Const.dataset_name,file_name='points_set')