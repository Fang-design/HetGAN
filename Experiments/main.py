import torch
import copy
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F
from layers import GraphConvolution
from attention_layers import Attention
from utils import *
from graph import *
from parameter_ori import *
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import matplotlib as mpl
from pylab import cm
import os
import time
import argparse
torch.manual_seed(0)
parser = argparse.ArgumentParser()

parser.add_argument('--log_file', default='./logs/grid_search.log', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--epoch', default=100, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)

def init_para(model_):
    for name, param in model_.named_parameters():
        if name.find('.bias') != -1:
            param.data.fill_(0)
        elif name.find('.weight') != -1:
            nn.init.xavier_uniform_(param.data)

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2,bidirection=True):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=bidirection, batch_first = True) # rnn
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hidden,c) = self.rnn(x) # (seq, batch, hidden)
        out = x
        b, s, h = x.shape
        x = torch.mean(x,dim=1)
        return x,hidden
    
class Predict_Model(nn.Module):
    def __init__(self,input_size,output_size=1):
        super(Predict_Model, self).__init__()
        self.reg = nn.Linear(input_size, output_size)
    def forward(self,x):
        x = self.reg(x)
        return x
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid * 2)
        self.dropout = dropout
        self.fc = nn.Linear(nhid * 2, nclass)
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        x = self.fc(x)
        return x





def restruct_node(atten,out_conv,out_flow,neighbors_dict,conv_keys,flow_keys,neighbors_dis,out_path,mode):

    def compute_weight(distance):
        return math.exp(-float(distance)*attention_theta)



    out_flow_ = out_flow
    out_conv_ = out_conv
    dict_dis = {}
    for index in range(out_flow.shape[0]):
        dict_item = {}
        neighbors_node = neighbors_dict[flow_keys[index]]

        neighbors_distance = neighbors_dis[flow_keys[index]]

        neighbor_list = []
        dis_list = []
        for i in neighbors_node:
            if i!=None:
                i = int(i)
            if i in conv_keys:

                dis_list.append(compute_weight((neighbors_distance[i])))
                dict_item[i] = compute_weight((neighbors_distance[i]))
                index_node = conv_keys.index(i)
                neighbor_list.append(out_conv_[index_node])

            elif i in flow_keys:
                dis_list.append(compute_weight(float(neighbors_distance[i])))
                index_node = flow_keys.index(i)
                neighbor_list.append(out_flow_[index_node])
                dict_item[i] = compute_weight((neighbors_distance[i]))
        dict_dis[flow_keys[index]] = dict_item
        enc_vec = torch.stack(neighbor_list)
        dis_vec = torch.from_numpy(np.array(dis_list))
        dis_vec = dis_vec.unsqueeze(0)
        dis_vec = dis_vec.to(torch.float)
        if cuda:
            dis_vec = dis_vec.cuda()
        atten_socre = atten(out_flow_[index],enc_vec)

        atten_socre = torch.mul(atten_socre,dis_vec)
        enc_vec = enc_vec.unsqueeze(0)
        atten_socre = atten_socre.unsqueeze(0)
        atten_socre_  = atten_socre.detach()
        context = torch.bmm(atten_socre_ , enc_vec)
        context = context.squeeze(0)
        out_flow_[index] = context
    if mode=='eval':
        f = open(out_path+str(attention_theta)+'.txt','w')
        for k,v in dict_dis.items():
            f.writelines(str(k)+': '+str(v))
        f.close()
    return out_flow_

def minmaxscaler_flow(data_flow):
    data_flow = data_flow.reshape(439*365,1)
    max_x_flow = np.max(data_flow)
    min_x_flow = np.min(data_flow)
    results_flow = (data_flow - min_x_flow)/(max_x_flow - min_x_flow)
    results_flow = results_flow.reshape(439,365)
    return max_x_flow,min_x_flow,results_flow

def standardscaler_flow(data_flow):
    data_flow = data_flow.reshape(439*365,1)
    mean_x_flow = np.mean(data_flow)
    std_x_flow = np.std(data_flow)
    results_flow = (data_flow - mean_x_flow)/std_x_flow
    results_flow = results_flow.reshape(439,365)
    return mean_x_flow,std_x_flow,results_flow

def minmaxscaler_conv(data_conv):
    data_conv = data_conv.reshape(177*365,1)
    max_x_conv = np.max(data_conv)
    min_x_conv = np.min(data_conv)
    results_conv = (data_conv - min_x_conv)/(max_x_conv - min_x_conv)
    results_conv = results_conv.reshape(177,365)
    return max_x_conv,min_x_conv,results_conv

def standardscaler_conv(data_conv):
    data_conv = data_conv.reshape(177*365,1)
    mean_x_conv = np.mean(data_conv)
    std_x_conv = np.std(data_conv)
    results_conv = (data_conv - mean_x_conv)/std_x_conv
    results_conv = results_conv.reshape(177,365)
    return mean_x_conv,std_x_conv,results_conv

def reverse_minmax(results_flow,max_x_flow,min_x_flow):
    final = results_flow*(max_x_flow - min_x_flow) + min_x_flow
    return final

def reverse_standard(results_flow,mean_x_flow,std_x_flow):
    final = results_flow*std_x_flow + mean_x_flow
    return final


def single_train(flow_datasets,conv_datasets):
    print('start_training!')
    scaler_list_flow = []
    scaler_list_conv = []
    data_flow, adj_flow, flow_keys = flow_datasets
    
    data_conv, adj_conv, conv_keys = conv_datasets
    #before load_data, scaler data......
    flow_mean_x,flow_std_x,data_flow = standardscaler_flow(data_flow)
    conv_mean_x,conv_std_x,data_conv = standardscaler_conv(data_conv)

    train_size = 0.7
    data_total_len = data_flow.shape[1]
    data_flow_train = data_flow[:,0:int(data_total_len*train_size)]
    data_flow_test = data_flow[:,int(data_total_len*train_size):data_total_len]
    data_total_len = data_conv.shape[1]
    data_conv_train = data_conv[:, 0:int(data_total_len * train_size)]
    data_conv_test = data_conv[:, int(data_total_len * train_size):data_total_len]
    neighbors_info = load_neighbor('neighbors.txt')
    neighbors_dis = load_nei_dis('neighbors_distance.txt')

    node_number_flow,window_seq_len_flow_trian,final_data_x_flow_train, final_data_y_flow_train = load_datasets(data_flow_train,data_type='flow')
    node_number_flow, window_seq_len_flow_test, final_data_x_flow_test, final_data_y_flow_test = load_datasets(
        data_flow_test,data_type='flow')
    final_data_y_flow_test = final_data_y_flow_test.view(final_data_y_flow_test.shape[1],final_data_y_flow_test.shape[0])
    node_number_conv, window_seq_len_conv_train, final_data_x_conv_train, final_data_y_conv_train = load_datasets(data_conv_train,data_type='conv')
    node_number_conv, window_seq_len_conv_test, final_data_x_conv_test, final_data_y_conv_test = load_datasets(data_conv_test,data_type='conv')
    net = lstm_reg(input_size=embeding_size, hidden_size=hidden_size,bidirection=bidirection)
    net_conv = lstm_reg(input_size=embeding_size,hidden_size=hidden_size,bidirection=bidirection)
    gcn_model_output = GCN(nfeat=windows,
                           nhid=hidden_size*2,
                           nclass=gcn_output,
                           dropout=0.1) if bidirection else GCN(nfeat=windows,
                           nhid=hidden_size,
                           nclass=gcn_output,
                           dropout=0.1)
    gcn_model_output_conv = GCN(nfeat=windows,
                           nhid=hidden_size*2,
                           nclass=gcn_output,
                           dropout=0.1) if bidirection else GCN(nfeat=windows,
                           nhid=hidden_size,
                           nclass=gcn_output,
                           dropout=0.1)
    atten = Attention(enc_hid_dim=gcn_output, dec_hid_dim=gcn_output)
    criterion = nn.MSELoss()
    Predict_reg = Predict_Model(gcn_output+hidden_size *2,1)
    init_para(net)
    init_para(net_conv)
    init_para(gcn_model_output)
    init_para(gcn_model_output_conv)  
    init_para(atten)
    init_para(Predict_reg)
    parameters = list(net.parameters()) + list(net_conv.parameters()) + list(
        gcn_model_output.parameters()) + list(
        gcn_model_output_conv.parameters())+list(Predict_reg.parameters())+list(atten.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_list = []
    total_batch = 0
    net.train()
    net_conv.train()
    gcn_model_output.train()
    gcn_model_output_conv.train()
    Predict_reg.train()
    atten.train()
    mape_previous = np.inf
    patience = 10
    patience_count = 0
    stt_ = time.time()
    for e in range(epoch):
        for index in range(window_seq_len_flow_trian):
            if torch.cuda.is_available():
                var_x_flow = Variable(final_data_x_flow_train[index]).cuda()
                var_y_flow = Variable(final_data_y_flow_train[index]).cuda()
                var_x_conv = Variable(final_data_x_conv_train[index]).cuda()
                var_y_conv = Variable(final_data_y_conv_train[index]).cuda()
                adj_data_flow = torch.FloatTensor(adj_flow)
                adj_data_flow = normalize(adj_data_flow)
                adj_data_flow = adj_data_flow.cuda()
                adj_data_conv = torch.FloatTensor(adj_conv)
                adj_data_conv = normalize(adj_data_conv)
                adj_data_conv = get_laplacian(adj_data_conv)
                adj_data_conv = adj_data_conv.cuda()
                net.cuda()
                net_conv.cuda()
                gcn_model_output.cuda()
                gcn_model_output_conv.cuda()
                Predict_reg.cuda()
                atten.cuda()
            else:
                var_x_flow = Variable(final_data_x_flow_train[index])
                var_y_flow = Variable(final_data_y_flow_train[index])
                var_x_conv = Variable(final_data_x_conv_train[index])
                var_y_conv = Variable(final_data_y_conv_train[index])
                adj_data_flow = torch.FloatTensor(adj_flow)
                adj_data_flow = normalize(adj_data_flow)
                adj_data_conv = torch.FloatTensor(adj_conv)
                adj_data_conv = normalize(adj_data_conv)
                adj_data_conv = get_laplacian(adj_data_conv)

            out,_ = net(var_x_flow)
            out_conv,_ = net_conv(var_x_conv)
            gcn_out = gcn_model_output(var_x_flow.squeeze(2),adj_data_flow)
            out_conv_gcn = gcn_model_output_conv(var_x_conv.squeeze(2),adj_data_conv)
            result = restruct_node(atten=atten,out_conv=out_conv_gcn,out_flow=gcn_out,neighbors_dict=neighbors_info,conv_keys=conv_keys,flow_keys=flow_keys,neighbors_dis=neighbors_dis,out_path='image/weight',mode='train')
            result = torch.cat((out,result),dim=1)
            out = Predict_reg(result)
            out = out.squeeze(1)
            loss = criterion(out, var_y_flow)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            total_batch += 1

        pred_flow = []
        scores = []
        output_stations = [42428714,42748139,5328215082,42440721,1372888590,42870679]
        for i in range(window_seq_len_flow_test):
            if cuda:
                var_x_flow = Variable(final_data_x_flow_test[i]).cuda()
                var_y_flow = Variable(final_data_y_flow_test[i]).cuda()
                var_x_conv = Variable(final_data_x_conv_test[i]).cuda()
                var_y_conv = Variable(final_data_y_conv_test[i]).cuda()
            else:
                var_x_flow = Variable(final_data_x_flow_test[i])
                var_y_flow = Variable(final_data_y_flow_test[i])
                var_x_conv = Variable(final_data_x_conv_test[i])
                var_y_conv = Variable(final_data_y_conv_test[i])

            out_flow,_ = net(var_x_flow)
            out_conv,_ = net_conv(var_x_conv)
            out = out_flow.squeeze(1)
            gcn_out = gcn_model_output(var_x_flow.squeeze(2), adj_data_flow)
            out_conv = out_conv.squeeze(1)
            out_conv = gcn_model_output_conv(var_x_conv.squeeze(2), adj_data_conv)
            result = restruct_node(atten=atten,out_conv=out_conv, out_flow=gcn_out, neighbors_dict=neighbors_info, conv_keys=conv_keys,
                                    flow_keys=flow_keys,neighbors_dis=neighbors_dis,out_path='image/weight',mode='eval')
            result = torch.cat((out, result), dim=1)
            out = Predict_reg(result)
            pred_flow.append(out)
        pred_flow = torch.stack(pred_flow)
        #pred_flow = pred_flow.t()
        pred_flow = pred_flow.squeeze(2)
        pred_flow = pred_flow.view(pred_flow.size(1),pred_flow.size(0))
        final_data_y_flow_test_ = reverse_standard(final_data_y_flow_test,flow_mean_x,flow_std_x)
        pred_flow = reverse_standard(pred_flow,flow_mean_x,flow_std_x)
        mape_current = mape(final_data_y_flow_test_.cpu().data.numpy(), pred_flow.cpu().data.numpy())
        print('Epoch {} completed! Time cost: {}s'.format(e + 1, time.time() - stt_))
        stt_ = time.time()
        if not mape_current < mape_previous:
            patience_count +=1
            if patience_count > patience:
                print('Early stop at epoch {}'.format(e + 1))
                break
        else:
            mape_previous = mape_current
            patience_count = 0
            print('MAPE:{}'.format(mape_previous))
            np.save('pred_flow_value'+'.npy',pred_flow.cpu().data.numpy())
            np.save('final_data_y_flow_test'+'.npy',final_data_y_flow_test_.cpu().data.numpy())

            with open('image/result_all_test'  + '.txt', 'w') as f:
                f.writelines('r2_score:{}'.format(r2_score(final_data_y_flow_test_.cpu().data.numpy(), pred_flow.cpu().data.numpy())))
                f.writelines('\n')
                f.writelines('mape:{}'.format(mape_previous))
                f.writelines('\n')
                f.writelines('rmse_score:{}'.format(math.sqrt(mean_squared_error(final_data_y_flow_test_.cpu().data.numpy(),pred_flow.cpu().data.numpy())))

            for i in range(len(output_stations)):
                output_station = output_stations[i]
                output_index = flow_keys.index(output_station)
                pred_flow_value = pred_flow[output_index,:].cpu().data.numpy()
                truth_flow_value = final_data_y_flow_test_[output_index,:].cpu().data.numpy()
                np.save('pred_flow_value'+str(output_station)+'.npy',pred_flow[output_index,:].cpu().data.numpy())
                np.save('truth_flow_value'+str(output_station)+'.npy',final_data_y_flow_test_[output_index,:].cpu().data.numpy())

                with open('image/result_test'+str(output_station)+'.txt','w') as f:
                    f.writelines('r2_score:{}'.format(r2_score(truth_flow_value, pred_flow_value)))
                    f.writelines('mape:{}'.format(mape(truth_flow_value, pred_flow_value)))
                    f.writelines('rmse_score:{}'.format(math.sqrt(mean_squared_error(truth_flow_value,pred_flow_value))))




def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


if __name__ =='__main__':
    data_conv,adj_conv,conv_keys = get_data("coronavirus")
    data_flow,adj_flow,flow_keys = get_data("flow")
    flow_datasets = (data_flow,adj_flow,flow_keys)
    conv_datasets = (data_conv,adj_conv,conv_keys)
    single_train(flow_datasets=flow_datasets,conv_datasets=conv_datasets)
