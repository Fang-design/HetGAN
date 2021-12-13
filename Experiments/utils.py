import numpy as np
import pandas as pd
import torch
from graph import *
from parameter_ori import *
from sklearn import preprocessing
import json

def load_neighbor(path):
    dict = {}
    with open(path) as f:
        for i in f:
            index = i.find('[')
            left = i[0:index]
            right = i[index:-1]
            node_list = eval(right)
            node_cur = eval(left)
            dict[node_cur] = node_list
    return dict

def get_data(dataset):
    if dataset == 'coronavirus':
        data_frame = pd.read_csv('results_no_neg_modified_final.csv', index_col=None)
        adj_data = pd.read_csv('ZIP_code_connectivity.csv', index_col=None)
        adj_list = adj_data.values.tolist()
        adj_dict = {int(x[0]):x[1:] for x in adj_list}
        data_list = data_frame.values.tolist()
        data_dict = {int(x[0]):x[1:] for x in data_list}
    else:

        data_frame = pd.read_csv('results_vertified_day_new.csv',index_col=None)
        data_list = data_frame.drop(['Unnamed: 0'],axis=1)
        adj_data = pd.read_csv('station_distance.csv',index_col=None)

        adj_list = adj_data.values.tolist()
        adj_dict = {int(x[0]):x[1:] for x in adj_list}
        data_list = data_list.values.tolist()
        data_dict = {int(x[0]):x[1:] for x in data_list}
    for key in data_dict:
        assert key in adj_dict

    adj = []
    data = []
    for key in adj_dict:
        assert key in data_dict
        adj.append(adj_dict[key])
        if dataset == 'flow':
            data_item = data_dict[key]

        else:
            data_item = data_dict[key]
        data.append(data_item)
    adj = np.array(adj)
    data = np.array(data)
    key_list = list(adj_dict.keys())
    return data,adj,key_list

def filter_data(data,k=3):
    for index,value in enumerate(data):
        if index < len(data) - k:
            mean = np.mean(data[index:index+k-1])
            for i in range(k):
                if data[index+i] > k*mean or data[index+i] < mean // k:
                    data[index+i] = int(mean)
    return data


def create_dataset(dataset, windows=windows):
    dataX, dataY = [], []
    for i in range(len(dataset) - windows):
        a = dataset[i:(i + windows)]
        dataX.append(a)
        dataY.append(dataset[i + windows])
    return np.array(dataX), np.array(dataY)

def load_datasets(data,data_type='flow'):
    final_data_x = []
    final_data_y = []

    for index, value in enumerate(data):
        dataset_X, dataset_Y = create_dataset(value)
        final_data_x.append(dataset_X)
        final_data_y.append(dataset_Y)
    final_data_x = np.array(final_data_x)  
    final_data_y = np.array(final_data_y)  
    final_data_x = torch.from_numpy(final_data_x)
    final_data_y = torch.from_numpy(final_data_y)
    node_number = final_data_x.size(0) 
    window_seq_len = final_data_x.size(1) 
    final_data_x = final_data_x.unsqueeze(2)  
    final_data_y = final_data_y.unsqueeze(2)  
    final_data_y = final_data_y.to(torch.float32)
    final_data_x = final_data_x.to(torch.float32)
    final_data_x = final_data_x.view(window_seq_len, node_number, windows, embeding_size) 
    final_data_y = final_data_y.view(window_seq_len, node_number) 
    return node_number,window_seq_len,final_data_x,final_data_y


import demjson

def load_nei_dis(path):
    dict_val = {}
    with open(path) as f:
        for i in f:
            list_i = i.strip().split('{')
            key = eval(list_i[0])
            data = '{' + list_i[1]
            data = data.replace("'","")
            data = demjson.decode(data)
            dict_val[key] = data
    return dict_val

