import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from GTCP_u2 import Const,Framework,util
from GTCP_u2 import DataPreparation as dp
from GTCP_u2.Train import train,test

def main(model_name,model_file_name=None):
    points_set = dp.load_locations(Const.dataset_name, file_name='points_set')
    coupled_locations = dp.load_locations(Const.dataset_name)
    model=Framework.Framework(points_set,coupled_locations).to(Const.device)

    if model_file_name!=None:
        print("Loading Saved Model")
        checkpoint = torch.load(model_file_name,map_location=Const.device)
        model_state_dict = checkpoint["model"]
        model.load_state_dict(model_state_dict)

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model).to(Const.device)
        #     model.load_state_dict(model_state_dict,False)
        # else:
        #     model = model.to(Const.device)
        #     # model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.item()}
        #     model.load_state_dict(model_state_dict)

    util.model_size(model)
    # cudnn.benchmark = True

    # criterion=nn.MSELoss()
    criterion=Framework.WeightedMSELoss(weight=Const.temporal_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, [15,30,45,60], gamma=0.316)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = MultiStepLR(optimizer, [100], gamma=0.316)

    train_loader = dp.make_data(Const.dataset_name, 'train', shuffle=True,sample_ratio=Const.training_data_ratio)
    dev_loader=dp.make_data(Const.dataset_name,'dev',shuffle=True)

    train(model, train_loader, dev_loader, criterion, optimizer, scheduler, model_name)
    # test_main(option.save_model + ".pkl", da, column, case=case)

def test_main(model_name,index_of_node=None):
    print("Loading Saved Model")
    dir = Const.log_path
    if index_of_node!=None:
        dir='{}/{}'.format(Const.log_path, model_name)
        # os.mkdir('{}/{}'.format(Const.log_path,model_name))
        model_name='finetune@node{}'.format(index_of_node)
    model_file_name = '{}/{}.pkl'.format(dir, model_name)

    checkpoint = torch.load(model_file_name,map_location=Const.device)
    model_state_dict = checkpoint["model"]

    points_set = dp.load_locations(Const.dataset_name, file_name='points_set')
    coupled_locations = dp.load_locations(Const.dataset_name)
    model = Framework.Framework(points_set, coupled_locations).to(Const.device)

    model.load_state_dict(model_state_dict)

    # if torch.cuda.device_count() > 1:
    #     model_structure = nn.DataParallel(model_structure).to(Const.device)
    #     model_structure.load_state_dict(model_state_dict,False)
    # else:
    #     model_structure = model_structure.to(Const.device)
    #     # model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.item()}
    #     model_structure.load_state_dict(model_state_dict)
    print("Loading Testing Dataset")
    # test_loader = dp.make_data('devset')
    test_loader = dp.make_data(Const.dataset_name,'test')
    # criterion = nn.MSELoss()
    criterion = Framework.WeightedMSELoss(weight=Const.temporal_weight)
    test(model, test_loader,criterion,index_of_node)
    # print("[ Results ]\n  - mae:{:2.8f}  - rmse:{:2.8f}".format(mae, rmse))
    # print("[ Results ]\n  - loss: {:2.8f}".format(loss))

def finetune_main(model_name,index_of_node):
    points_set = dp.load_locations(Const.dataset_name, file_name='points_set')
    coupled_locations = dp.load_locations(Const.dataset_name)
    model = Framework.Framework(points_set, coupled_locations).to(Const.device)

    model_file_name='{}/{}.pkl'.format(Const.log_path, model_name)
    print("Loading Saved Model")
    checkpoint = torch.load(model_file_name, map_location=Const.device)
    model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).to(Const.device)
    #     model.load_state_dict(model_state_dict,False)
    # else:
    #     model = model.to(Const.device)
    #     # model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.item()}
    #     model.load_state_dict(model_state_dict)

    util.model_size(model)
    # cudnn.benchmark = True

    # criterion = nn.MSELoss()
    criterion = Framework.WeightedMSELoss(weight=Const.temporal_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer, [60,100], gamma=0.316)

    train_loader = dp.make_data(Const.dataset_name, 'train', shuffle=True)
    dev_loader = dp.make_data(Const.dataset_name, 'dev', shuffle=True)

    train(model, train_loader, dev_loader, criterion, optimizer, scheduler, model_name,index_of_node)
    # test_main(option.save_model + ".pkl", da, column, case=case)
if __name__ == '__main__':
    # for i in [0.8,0.9]:
    #     Const.training_data_ratio=i
    #     model_name = 'gtcp@hs{}$dm{}$alpha{}unshare$ratio{}'.format(
    #         Const.framework_hidden_size, Const.gaussian_PG_hidden_size,
    #         Const.intra_coupling_weight,i)
    #     print('model name:', model_name)
    #     # dir = Const.log_path
    #     # model_path = '{}/{}.pkl'.format(dir, model_name)
    #     main(model_name)
    #     test_main(model_name)

    model_name = 'gtcp@hs{}$dm{}$alpha{}unshare'.format(
        Const.framework_hidden_size, Const.gaussian_PG_hidden_size,
        Const.intra_coupling_weight)
    print('model name:', model_name)
    # dir = Const.log_path
    # model_path = '{}/{}.pkl'.format(dir, model_name)
    main(model_name)
    test_main(model_name)

    # for sample in [0.9]:
    #     Const.training_data_ratio=sample
    #     model_name = 'gtcp@hs{}$dm{}$alpha{}$sample{}unshare'.format(
    #         Const.framework_hidden_size, Const.gaussian_PG_hidden_size,
    #         Const.intra_coupling_weight,Const.training_data_ratio)
    #     print('model name:', model_name)
    #     # dir = Const.log_path
    #     # model_path = '{}/{}.pkl'.format(dir, model_name)
    #     main(model_name)
    #     test_main(model_name)
        # finetune_main(model_name,index_of_node=15)
        # test_main(model_name,index_of_node=15)