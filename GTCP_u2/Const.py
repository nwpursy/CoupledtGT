import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

dataset_name='PM2.5_processed'
# dataset_name='AQI_processed'
node_count = 17  # 随着输入数据修改

use_weather_cols=['datetime_LMT','Air Temperature','Dew Point Temperature','Sea Level Pressure','wind_x_kmph','wind_y_kmph']
non_wind_cols=['Air Temperature','Dew Point Temperature','Sea Level Pressure']
wind_cols=['wind_x_kmph','wind_y_kmph']
data_lowerbound=0
data_upperbound=200 #随着输入数据修改

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# data pre-process
dev_ratio = 0.2
test_ratio = 0.1
input_time_window = 24*7
target_time_window = 12 #可更改
seq_sample_step = 3
step_offset=0

# model
gaussian_PG_hidden_size=56
framework_hidden_size=18
intra_coupling_weight=0.04
ones_q=False

# run time
training_data_ratio=1
initial_teacher=0.5
# initial_teacher=0.1
step_teacher=0.005
# step_teacher=0.0
batch_size=256
epoch=80
log_path='../resource/result_models/beijing/{}/in{}&out{}&step{}&offset{}'.format(
    dataset_name,input_time_window,target_time_window,seq_sample_step,step_offset)
temporal_weight=torch.FloatTensor([15,15,14,11,11,11,5,5,5,2,2,2]).to(device)
# temporal_weight=torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,1]).to(device)
# temporal_weight=torch.FloatTensor([9,9,9,8,8,8,
#                                    6,6,5,4,4,3,
#                                    3,3,3,2,2,2,
#                                    1,1,1,1,1,1]).to(device)
