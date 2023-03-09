import time
import torch
import numpy as np
import os

from GTCP_xi_an import Const,util
from GTCP_xi_an.EvaluationMetrics import Evaluation_Utils

def get_finetuned_performance(preds,targets):
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    # print('\t\t1-3h\t4-6h\t7-9h\t10-12h\t13-18h\t19-24h\tall')
    print('\t\t1-3h\t4-6h\t7-9h\t10-12h\tall')
    total_mae, total_rmse = get_performance_of_one_node(preds, targets, groups)
    output = 'Node \tMAE\t'
    output += ('\t'.join(total_mae))
    print(output)
    output = 'Node \tRMSE\t'
    output += ('\t'.join(total_rmse))
    print(output)

def get_performance(preds, targets):
    # groups=[[0,1,2],[3,4,5],[6,7,8],[9,10,11],
    #         [12,13,14,15,16,17],[18,19,20,21,22,23]]
    groups=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    # print('\t\t1-3h\t4-6h\t7-9h\t10-12h\t13-18h\t19-24h\tall')
    print('\t\t1-3h\t4-6h\t7-9h\t10-12h\tall')
    total_mae,total_rmse=get_performance_of_one_node(
        preds.view(-1,preds.size(-1)),targets.view(-1,targets.size(-1)),groups)
    output='All Node\tMAE\t'
    output+=('\t'.join(total_mae))
    print(output)
    output = 'All Node\tRMSE\t'
    output += ('\t'.join(total_rmse))
    print(output)
    for node in range(Const.node_count):
        preds_j=preds[:,node]
        targets_j=targets[:,node]
        total_mae, total_rmse = get_performance_of_one_node(preds_j,targets_j,groups)
        output = 'Node {}\tMAE\t'.format(node)
        output += ('\t'.join(total_mae))
        print(output)
        output = 'Node {}\tRMSE\t'.format(node)
        output += ('\t'.join(total_rmse))
        print(output)

def get_performance_of_one_node(preds,targets,groups):
    total_mae=[]
    total_rmse=[]
    for group in groups:
        pred=preds[:,group]
        target=targets[:,group]
        mae,rmse=Evaluation_Utils.total(pred.reshape(-1),target.reshape(-1))
        total_mae.append(str(mae))
        total_rmse.append(str(rmse))
    mae, rmse = Evaluation_Utils.total(preds.reshape(-1), targets.reshape(-1))
    total_mae.append(str(mae))
    total_rmse.append(str(rmse))
    return total_mae,total_rmse

def train_epoch(model, training_data, criterion, optimizer,teacher_forcing_ratio,index_of_node=None):
    total_loss = 0.0
    # i = 0
    print('共{}个train batch'.format(len(training_data)))
    for i,data in enumerate(training_data):
        # print('batch {},train开始'.format(i))
        # prepare data
        start = time.time()
        model.zero_grad()
        # forward
        preds, coupled_matrices,sigma,q = model(data[1], data[2], data[3], data[4], teacher_forcing_ratio)

        # backward
        if index_of_node!=None:
            loss=criterion(preds[:,index_of_node],data[2][:,index_of_node])
        else:
            loss = criterion(preds,data[2])
        loss.backward()

        # update parameters
        optimizer.step()
        # print('batch {} 运行时间: {}, 最大长度：{}'.format(i,time.time() - start,data[1].size(1)))
        # print('training loss:{}'.format(np.sqrt(loss.item())))

        total_loss += loss.item()*len(preds)
        # print("  - one batch data time: {:2.2f} min".format((time.time() - start)/60))
        # 清垃圾
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
    return model, np.sqrt(total_loss / len(training_data.dataset)),sigma,q

def eval_epoch(model, validation_data, criterion,index_of_node=None):
    total_loss = 0.0
    print('共{}个eval batch'.format(len(validation_data)))
    with torch.no_grad():
        for i,data in enumerate(validation_data):
            # print('batch {},validate开始'.format(i))
            # prepare data
            start=time.time()
            preds, coupled_matrices,sigma,q = model.evaluate(data[1], data[3], data[4], Const.target_time_window)
            if index_of_node != None:
                loss = criterion(preds[:, index_of_node], data[2][:, index_of_node])
            else:
                loss = criterion(preds, data[2])
            # print('模型时间: {}'.format(time.time() - start))
            total_loss += loss.item()*len(preds)

            # 清垃圾
            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()
        return np.sqrt(total_loss / len(validation_data.dataset)),sigma,q


def train(model,train_loader,dev_loader,criterion, optimizer, scheduler, model_name,index_of_node=None):
    dir=Const.log_path
    if not os.path.exists(dir):
        os.mkdir(dir)
    if index_of_node!=None:
        dir='{}/{}'.format(Const.log_path, model_name)
        if not os.path.exists(dir):
            os.mkdir('{}/{}'.format(Const.log_path,model_name))
        model_name='finetune@node{}'.format(index_of_node)

    log_train_file = '{}/{}_train.log'.format(dir,model_name)
    log_valid_file = '{}/{}_valid.log'.format(dir,model_name)

    print("[ INFO ] Training performance will be written to file\n {:s} and {:s}".format(
        log_train_file, log_valid_file))

    valid_losses = []
    train_losses=[]
    best_valid=0
    model_path = '{}/{}.pkl'.format(dir, model_name)
    teacher_forcing_ratio=Const.initial_teacher
    start=time.time()
    for each_epoch in range(Const.epoch):
        model.train()
        print("[ Epoch {:d} ]".format(each_epoch))

        start_time = time.time()

        model, train_loss,sigma,q = train_epoch(model, train_loader, criterion, optimizer,teacher_forcing_ratio,index_of_node)
        print("  - (Training) loss: {:2.8f},  elapse: {} second(s)".format(
            train_loss,
            (time.time() - start_time)))
        train_losses+=[train_loss]

        model.eval()
        start_time = time.time()
        eval_loss,sigma,q = eval_epoch(model, dev_loader, criterion,index_of_node)
        print("  - (Validation) loss: {:2.8f},  elapse: {} second(s)".format(
            eval_loss,
            (time.time() - start_time) ))
        valid_losses += [eval_loss]
        scheduler.step()

        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "epoch": each_epoch
        }
        if eval_loss <= min(valid_losses):
            best_valid=each_epoch
            torch.save(checkpoint, model_path)
            print("  - [ INFO ] The checkpoint file has been updated.")

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as train_file, open(log_valid_file, "a") as valid_file:
                train_file.write("{}, {:2.8f}\n".format(each_epoch, train_loss))
                valid_file.write("{}, {:2.8f}\n".format(each_epoch, eval_loss))
        # if (each_epoch+1)<5:
        #     globalAnalysis(total_truth, total_my, total_response_count, total_cdp, title='Epoch {}'.format(each_epoch+1))
        if (each_epoch+1)%20==0:
            print('best valid:',best_valid)
            util.plotLossCurve(np.array(train_losses), np.array(valid_losses),model_name,start_index=3)
            # globalAnalysis(total_truth,total_my,total_response_count,total_cdp,title='Epoch {}'.format(each_epoch+1))

        # dynamic teacher forcing
        if teacher_forcing_ratio > 0.01:
            teacher_forcing_ratio -= Const.step_teacher
    print('total training time:',time.time()-start)

def test(model, dataset, criterion,index_of_node=None):
    # TODO: recover data and then compute the prediction error
    total_loss = 0.0
    print('共{}个test batch'.format(len(dataset)))
    total_input=[]
    total_my=[]
    total_truth=[]
    total_coupling=[]
    total_sigma=[]
    total_q=[]
    with torch.no_grad():
        start=time.time()
        for data in dataset:
            # prepare data
            preds, coupled_matrices,sigma,q = model.evaluate(data[1], data[3], data[4], Const.target_time_window)
            if index_of_node != None:
                loss = criterion(preds[:, index_of_node], data[2][:, index_of_node])
            else:
                loss = criterion(preds, data[2])
            # print('模型时间: {}'.format(time.time() - start))
            total_loss += loss.item() * len(preds)
            total_input.append(data[1])
            total_truth.append(data[2])
            total_my.append(preds)
            total_coupling.append(coupled_matrices)
            total_sigma.append(sigma)
            total_q.append(q)
        total_truth=torch.cat(total_truth,dim=0).cpu()
        total_my = torch.cat(total_my, dim=0).cpu()
        total_input = torch.cat(total_input, dim=0).cpu()
        total_coupling=torch.cat(total_coupling,dim=0).cpu()
        total_sigma=torch.cat(total_sigma,dim=0).cpu()
        total_q=torch.cat(total_q,dim=0).cpu()
        # inv_scaled_total_my=total_my*(Const.data_upperbound-Const.data_lowerbound)

        ##case
        roll = np.random.randint(0, len(dataset.dataset), [3])
        # node_indices=[0,1,2,3,4,5]
        # node_indices = np.arange(start=6,stop=12)
        if index_of_node!=None:
            get_finetuned_performance(total_truth[:,index_of_node], total_my[:,index_of_node])
            caseStudy_finetune(total_input, total_truth, total_my, total_coupling,
                               total_sigma, total_q, roll,index_of_node)
        else:
            print('total test time:', time.time() - start)
            get_performance(total_truth,total_my)
            caseStudy(total_input, total_truth, total_my, total_coupling, total_sigma, total_q, roll)
        # return mae,rmse

def caseStudy(total_input,total_truth,total_my,total_coupling,total_sigma,total_q,roll):
    node_indices_group=[]
    node_indices_group.append(np.arange(start=0, stop=6))
    node_indices_group.append(np.arange(start=6, stop=12))
    # node_indices_group.append(np.arange(start=12, stop=18))
    # node_indices_group.append(np.arange(start=18, stop=24))
    for case in roll:
        title='case '+str(case)
        # print(total_input.size())
        my_chart = total_my[case]
        util.plotOneHeatmap(my_chart, title)
        coupling = total_coupling[case]
        util.plotManyHeatmap(4,3,coupling,title,np.arange(12))
        sigma=total_sigma[case]
        q=total_q[case]
        util.plotOneHeatmap(sigma,title+' :sigma')
        util.plotOneHeatmap(q, title + ' :q')
        for node_indices in node_indices_group:
            names=[]
            inp=total_input[case][node_indices,-24:]
            truth=total_truth[case][node_indices]
            my=total_my[case][node_indices]
            for node in node_indices:
                names.append('node ' + str(node))
            util.plotLineCharts(3,2,inp,truth,my,title,names)

def caseStudy_finetune(total_input,total_truth,total_my,total_coupling,total_sigma,total_q,roll,index_of_node):
    for case in roll:
        title='case {}, node {}'.format(case,index_of_node)
        coupling=total_coupling[case,:,index_of_node]
        util.plotOneHeatmap(coupling,title+' :coupling')
        sigma = total_sigma[case]
        q = total_q[case]
        util.plotOneHeatmap(sigma, title + ' :sigma')
        util.plotOneHeatmap(q, title + ' :q')
        inp=total_input[case,index_of_node,-24:]
        truth=total_truth[case,index_of_node]
        my=total_my[case,index_of_node]
        util.plotLineChart(inp,truth,my,title)


if __name__=='__main__':
    t=torch.randn(2,5,4)
    print(t[:,[4,1,3]])
