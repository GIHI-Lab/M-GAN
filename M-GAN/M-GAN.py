#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Import
import os
# 在导入 tensorflow 之前设置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import ChebConv

# from torch_geometric.utils import dropout_adj, remove_self_loops, add_self_loops
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import add_self_loops

import argparse
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings('ignore')

# %% Args
def Args():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dev', default='cuda',
                        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--epochs', type=int, default=500,
                        choices=[1, 2, 10, 20, 50, 100, 500, 1000, 1500, 2500, 3000])
    parser_args = parser.parse_args()
    return parser_args


# %% load data
def load_datasets(args):
    """加载数据集"""
    try:
        # 加载预处理后的数据
        print("正在加载CPDB_processed.pkl数据集...")
        data = torch.load('./data/CPDB_processed.pkl')
        
        # 打印数据结构信息
        print("数据类型:", type(data))
        if isinstance(data, dict):
            print("数据键:", data.keys())
            print("features形状:", data['features'].shape if isinstance(data['features'], np.ndarray) else "非张量")
            print("metadata键:", data['metadata'].keys())
        
        # 确保特征是张量格式并移动到指定设备
        features = torch.tensor(data['features'], dtype=torch.float) if not torch.is_tensor(data['features']) else data['features']
        features = features.to(args.dev)
        
        # 构建KNN图（k=10）
        k = 10
        num_nodes = features.shape[0]
        edge_index = []
        
        # 计算每个节点与其他节点的相似度
        for i in range(num_nodes):
            # 计算当前节点与所有其他节点的欧氏距离
            distances = torch.norm(features - features[i], dim=1)
            # 获取最近的k个邻居（不包括自己）
            _, indices = torch.topk(distances, k+1, largest=False)
            # 添加边（跳过第一个，因为是节点自己）
            for j in indices[1:]:
                edge_index.append([i, j.item()])
                edge_index.append([j.item(), i])  # 添加反向边
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(args.dev)
        
        # 生成合成标签（基于特征的聚类）
        feature_norms = torch.norm(features, dim=1)
        y = torch.zeros(num_nodes, dtype=torch.float, device=args.dev)
        # 将特征范数较大的节点标记为正类
        threshold = torch.quantile(feature_norms, 0.8)  # 使用前20%作为正类
        y[feature_norms > threshold] = 1.0
        
        # 生成训练掩码（随机80%用于训练）
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=args.dev)
        train_size = int(0.8 * num_nodes)
        train_indices = torch.randperm(num_nodes)[:train_size]
        mask[train_indices] = True
        
        # 构建数据字典
        data_dict = {
            'x': features,  # 特征矩阵
            'edge_index': edge_index,  # 边索引（KNN图）
            'y': y,  # 合成标签
            'mask': mask  # 训练掩码
        }
        
        # 确保Y是正确的格式
        Y = data_dict['y'].reshape(-1, 1)
        
        # 加载k_sets
        k_sets = torch.load('./data/k_sets.pkl')
        
        print("数据加载完成!")
        print(f"特征维度: {data_dict['x'].shape}")
        print(f"边索引维度: {data_dict['edge_index'].shape}")
        print(f"标签维度: {Y.shape}")
        print(f"掩码维度: {data_dict['mask'].shape}")
        print(f"正类样本比例: {(Y == 1).float().mean().item():.4f}")
        
        return data_dict, Y, data_dict['mask'], k_sets, None, None
        
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        print(f"错误位置: {e.__traceback__.tb_lineno}")
        raise


# %% Model


class GNNEncoder(nn.Module):
    def __init__(self, nFeat, nHid, args, outAct=lambda x: x):
        super(GNNEncoder, self).__init__()
        # 这里，层数可以作为一个参数，待修改

        self.Dropout = nn.Dropout(0.5)

        self.convLs = nn.ModuleList()

        conv = nn.ModuleList([ChebConv(nFeat, nHid, K=2)])
        conv.append(nn.ReLU())
        conv.append(nn.Dropout(0.5))
        self.convLs.append(conv)


        conv = nn.ModuleList([ChebConv(nHid, nHid, K=2)])
        conv.append(nn.ReLU())
        conv.append(nn.Dropout(0.5))
        self.convLs.append(conv)
        
        self.outAct = outAct
        return

    def forward(self, x, edge_index):

        x = self.Dropout(x)

        xLs = []
        for iLayer in range(len(self.convLs)):
            edge_temp, _ = dropout_adj(edge_index, p=0.8,
                                       force_undirected=True,
                                       num_nodes=x.size()[0],
                                       training=self.training)
            x = self.convLs[iLayer][0](x, edge_temp)
            for subLayer in self.convLs[iLayer][1:]:
                x = subLayer(x)
                
            xLs.append(x)

        xOut = sum(xLs) / len(xLs)
        xOut = self.outAct(xOut)

        return xOut


class Generator(nn.Module):
    def __init__(self, args, features):
        super(Generator, self).__init__()
        n_modality = len(features)  # 直接使用 features 的长度
        nFeat = [temp.shape[1] for temp in features]
        self.args = args
        
        # 为每个模态添加注意力权重
        self.attention_weights = nn.Parameter(torch.ones(n_modality)/n_modality)
        self.encoder = nn.ModuleList(
            [GNNEncoder(nFeat[i], 256, args) for i in range(n_modality)]
        )
        
    def forward(self, features, edge_index):
        args = self.args
        # 使用softmax获得归一化的注意力权重
        weights = F.softmax(self.attention_weights, dim=0)
        
        # 加权融合多个模态的特征
        out = [self.encoder[i](features[i], edge_index) * weights[i]
               for i in range(len(features))]
        return out


class Discriminator(nn.Module):
    def __init__(self, args, outAct=lambda x: x):
        super(Discriminator, self).__init__()
        self.n_modality = n_modality = args.n_modality
        self.interactionWeight = torch.nn.Parameter(
            torch.randn(n_modality**2, 256, 256))
        self.outAct = outAct
        return

    def forward(self, x):
        output = []
        for i in range(len(x)):
            for j in range(len(x)):
                ind = i * self.n_modality + j
                out = torch.matmul(x[i], self.interactionWeight[ind])
                out = torch.mul(out, x[j]).sum(1)
                out = self.outAct(out)
                output.append(out)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, args, nFeat, nHid, nOut, outAct=lambda x: x):
        super(MultiLayerPerceptron, self).__init__()

        self.MLP = nn.ModuleList()
        self.MLP.append(nn.Linear(nFeat, nOut))
        self.outAct = outAct
        return

    def forward(self, x, edge_index):
        for layer in self.MLP:
            x = layer(x)
        out = self.outAct(x)
        return out


class Predictor(nn.Module):
    def __init__(self, args, nFeat, nHid, edge_index, outAct=lambda x: x):
        super(Predictor, self).__init__()

        self.args = args   
        pb, _ = remove_self_loops(edge_index)
        self.pb, _ = add_self_loops(pb)

        self.encoder = GNNEncoder(nFeat, nHid, args)
        self.MLPs = MultiLayerPerceptron(args, nHid, int(nHid/2), 1)
        
    def forward(self, x, edge_index):

        out = self.encoder(x, edge_index)
        out = self.MLPs(out, edge_index)
        
        return out


# %% Train

class Train(object):
    def __init__(self, args):
        self.args = args
        return
    
    def trainModel(self, data, Y, tr_mask, te_mask):
        print(f"Y type: {type(Y)}")
        print(f"Y shape: {Y.shape if hasattr(Y, 'shape') else 'No shape'}")
        print(f"tr_mask type: {type(tr_mask)}")
        print(f"tr_mask shape: {tr_mask.shape if hasattr(tr_mask, 'shape') else 'No shape'}")
        
        x = data['x']
        features = [x[:, i:i+16] for i in range(0,x.shape[1],16)]
        edge_index = data['edge_index']
        
        args = self.args
        
        device = args.dev
        epochs = args.epochs
        
        args.n_modality = n_modality = len(features)
        # 初始化共享生成器
        shared_G = Generator(args, features)
        shared_G = shared_G.to(device)
        optimizer_shared_G = torch.optim.Adam(
            shared_G.parameters(), 
            lr=0.001,  # 调小学习率
            betas=(0.9, 0.999),
            weight_decay=1e-4  # 增加正则化
        )

        # 添加学习率调度器(创新)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_shared_G, 
            mode='max',
            factor=0.5,
            patience=20,
            verbose=True
        )

        # 初始化特异性生成器
        specif_G = Generator(args, features)
        specif_G = specif_G.to(device)
        optimizer_specif_G = torch.optim.Adam(
            specif_G.parameters(), lr=0.0005, betas=(0.5, 0.999),
            weight_decay=5e-5)

        # 初始化鉴别器
        discri_M = Discriminator(args)
        discri_M = discri_M.to(device)
        optimizer_Discrimi = torch.optim.Adam(
            discri_M.parameters(), lr=0.005, betas=(0.5, 0.999),
            weight_decay=5e-5)

        ganLossFn = nn.BCEWithLogitsLoss()

        nFeat = 256 + len(features) * 256
        nHid = 256
        predModel = Predictor(args, nFeat, nHid, edge_index)
        predModel = predModel.to(device)
        optimizer_predict = torch.optim.Adam(
            predModel.parameters(), lr=0.005, betas=(0.5, 0.999),
            weight_decay=5e-5)
        predLossFn = nn.BCEWithLogitsLoss()

        # 初始化记录列表
        epoch_ls = []  # 记录轮数
        AUC__tr_ls = []  # 训练集AUC
        AUPR_tr_ls = []  # 训练集AUPR
        AUC__te_ls = []  # 测试集AUC
        AUPR_te_ls = []  # 测试集AUPR

        try:
            # 训练循环
            epoch = 1
            for epoch in range(1, epochs+1):
                # 设置模型状态
                shared_G.train()  # 共享生成器训练模式
                specif_G.train()  # 特异性生成器训练模式
                discri_M.eval()   # 判别器评估模式
                predModel.train() # 预测器训练模式

                # 生成器前向传播
                shared_out = shared_G(features, edge_index)  # 共享生成器输出
                shared_out_tr = [temp[tr_mask] for temp in shared_out]  # 获取训练集输出
                sharedDisOut_G = torch.cat(discri_M(shared_out_tr), dim=0)  # 判别器判别结果

                specif_out = specif_G(features, edge_index)  # 特异性生成器输出
                specif_out_tr = [temp[tr_mask] for temp in specif_out]  # 获取训练集输出
                specifDisOut_G = torch.cat(discri_M(specif_out_tr), dim=0)  # 判别器判别结果

                # 计算生成器损失
                valid = torch.ones_like(sharedDisOut_G, device=device)  # 真实标签
                fake = torch.zeros_like(sharedDisOut_G, device=device)  # 虚假标签

                sharedLoss_G = ganLossFn(sharedDisOut_G, valid)  # 共享生成器损失
                specifLoss_G = ganLossFn(specifDisOut_G, fake)  # 特异性生成器损失
                loss_G = (sharedLoss_G + specifLoss_G) / 2.0  # 总生成器损失

                # 计算预测损失
                shared_ave = sum(shared_out) / len(shared_out)  # 计算共享特征平均值
                featuresC = [shared_ave] + specif_out  # 组合特征
                generate_emb = torch.cat(featuresC, dim=1)  # 连接特征
                predictions = predModel(generate_emb, edge_index)  # 预测结果
                predLoss = predLossFn(predictions[tr_mask], Y[tr_mask])  # 预测损失

                # 优化生成器和预测器
                optimizer_shared_G.zero_grad()  # 清空梯度
                optimizer_specif_G.zero_grad()  # 清空梯度
                optimizer_predict.zero_grad()  # 清空梯度

                totalLoss_G = 0.3 * loss_G + 0.7 * predLoss  # 计算总损失
                totalLoss_G.backward()  # 反向传播

                optimizer_shared_G.step()  # 更新参数
                optimizer_specif_G.step()  # 更新参数
                optimizer_predict.step()  # 更新参数
                
                # 设置模型状态
                shared_G.eval()  # 共享生成器评估模式
                specif_G.eval()  # 特异性生成器评估模式
                discri_M.train() # 判别器训练模式
                predModel.train() # 预测器训练模式
                
                # 判别器训练
                shared_out = shared_G(features, edge_index)  # 生成共享特征
                sharedDisOut_D = torch.cat(
                    discri_M([temp.detach() for temp in shared_out_tr]), dim=0)  # 判别共享特征

                specif_out = specif_G(features, edge_index)  # 生成特异性特征
                specifDisOut_D = torch.cat(
                    discri_M([temp.detach() for temp in specif_out_tr]), dim=0)  # 判别特异性特征

                # 计算判别器损失
                sharedLoss_D = ganLossFn(sharedDisOut_D, fake)  # 共享特征判别损失
                specifLoss_D = ganLossFn(specifDisOut_D, valid)  # 特异性特征判别损失
                loss_D = (sharedLoss_D + specifLoss_D) / 2.0  # 总判别器损失

                # 特征融合和预测
                shared_ave = sum(shared_out) / len(shared_out)  # 计算共享特征平均值
                featuresC = [shared_ave] + specif_out  # 组合特征
                generate_emb = torch.cat(featuresC, dim=1)  # 连接特征
                predictions = predModel(generate_emb, edge_index)  # 预测结果
                predLoss = predLossFn(predictions[tr_mask], Y[tr_mask])  # 计算预测损失

                # 优化判别器和预测器
                optimizer_Discrimi.zero_grad()  # 清空梯度
                optimizer_predict.zero_grad()  # 清空梯度

                totalLoss_D = 0.3 * loss_D + 0.7 * predLoss  # 计算总损失
                totalLoss_D.backward()  # 反向传播

                optimizer_Discrimi.step()  # 更新参数
                optimizer_predict.step()  # 更新参数
                
                # 检查预测结果是否有NaN值
                if (True in np.isnan(predictions.detach().cpu().numpy().flatten())):
                    temp_x = 0
                    sys.exit(f'epoch: {epoch}, nan is in predictions')

                # 计算训练集性能指标
                AUC__tr = roc_auc_score(Y[tr_mask].cpu().numpy().flatten(),
                                        predictions[tr_mask].detach().cpu().numpy().flatten())
                AUPR_tr = average_precision_score(Y[tr_mask].cpu().numpy().flatten(),
                                                  predictions[tr_mask].detach().cpu().numpy().flatten())
                
                # 评估模式
                shared_G.eval()
                specif_G.eval()
                discri_M.eval()
                predModel.eval()

                # 在测试集上评估
                with torch.no_grad():
                    shared_out = shared_G(features, edge_index)
                    specif_out = specif_G(features, edge_index)

                    shared_ave = sum(shared_out) / len(shared_out)
                    featuresC = [shared_ave] + specif_out
                    generate_emb = torch.cat(featuresC, dim=1)

                    predictions = predModel(generate_emb, edge_index)
                    
                # 检查测试集预测结果
                if (True in np.isnan(predictions.cpu().numpy().flatten())):
                    temp_x = 0
                    sys.exit(f'epoch: {epoch}, nan is in predictions')

                # 计算测试集性能指标
                AUC__te = roc_auc_score(Y[te_mask].cpu().numpy().flatten(), 
                                        predictions[te_mask].detach().cpu().numpy().flatten())
                AUPR_te = average_precision_score(Y[te_mask].cpu().numpy().flatten(), 
                                                  predictions[te_mask].detach().cpu().numpy().flatten())       

                # 每10轮记录一次结果
                if (epoch == 1) or (epoch % 10 == 0):
                    epoch_ls.append(epoch)
                    AUC__tr_ls.append(AUC__tr)
                    AUPR_tr_ls.append(AUPR_tr)
                    AUC__te_ls.append(AUC__te)
                    AUPR_te_ls.append(AUPR_te)
                    
                    # 打印当前性能
                    print(f'epoch: {epoch}')
                    print(f'-train-\nAUC: {AUC__tr:.4f}, AUPR: {AUPR_tr:.4f}')
                    print(f'-test-\nAUC: {AUC__te:.4f}, AUPR: {AUPR_te:.4f}')
        except KeyboardInterrupt:
            pass
        
        # 保存训练记录
        self.epoch_ls = epoch_ls
        self.AUC__tr_ls = AUC__tr_ls
        self.AUPR_tr_ls = AUPR_tr_ls
        self.AUC__te_ls = AUC__te_ls
        self.AUPR_te_ls = AUPR_te_ls
        return
        

# %% Experiment

class Experiment(object):
    def __init__(self, args, data, Y, mask_all, k_sets):
        self.args = args
        self.runFold(data, Y, k_sets)
        return
              
    def subParaFun(self, data, Y, k_sets):
        args = self.args
        
        epochs = args.epochs

        numResult = math.floor(float(epochs)/10) + 1

        AUC__tr_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_tr_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_te_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_final = np.zeros((numResult, 1+10+2))
        AUPR_te_final = np.zeros((numResult, 1+10+2))
        
        columns_Run = ['epoch'] + [f'Run{iRun+1}'
                                   for iRun in range(10)] + ['mean', 'std']
        columns_fold = ['epoch'] + [f'fold{fold+1}'
                                   for fold in range(5)] + ['mean', 'std']
        
        iRun = 0
        for iRun in range(10):
            timeRun = time.time()
            kFold = 0
            for kFold in range(5):
                timeFold = time.time()
                print(f'Run: {iRun+1} \nFold: {kFold+1}')
                _, _, tr_mask, te_mask = k_sets[iRun][kFold]
                trainObj = Train(args)
                trainObj.trainModel(data, Y, tr_mask, te_mask)
                
                AUC__tr_ls = trainObj.AUC__tr_ls
                AUPR_tr_ls = trainObj.AUPR_tr_ls
                AUC__te_ls = trainObj.AUC__te_ls
                AUPR_te_ls = trainObj.AUPR_te_ls
                
                AUC__tr_Fold[iRun,:,kFold+1] = np.array(AUC__tr_ls)
                AUPR_tr_Fold[iRun,:,kFold+1] = np.array(AUPR_tr_ls)
                AUC__te_Fold[iRun,:,kFold+1] = np.array(AUC__te_ls)
                AUPR_te_Fold[iRun,:,kFold+1] = np.array(AUPR_te_ls)
                
                elapsedTime = round((time.time() - timeFold) / 60, 3)
                print(f'Fold time: {elapsedTime} minutes')
                
                break

            AUC__tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUC__tr_Fold[iRun,:,-2] = np.mean(AUC__tr_Fold[iRun,:,1:5+1], axis=1)
            AUC__tr_Fold[iRun,:,-1] = np.std(AUC__tr_Fold[iRun,:,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUPR_tr_Fold[iRun,:,-2] = np.mean(AUPR_tr_Fold[iRun,:,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:,-1] = np.std(AUPR_tr_Fold[iRun,:,1:5+1], axis=1)
            
            AUC__te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUC__te_Fold[iRun,:,-2] = np.mean(AUC__te_Fold[iRun,:,1:5+1], axis=1)
            AUC__te_Fold[iRun,:,-1] = np.std(AUC__te_Fold[iRun,:,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUPR_te_Fold[iRun,:,-2] = np.mean(AUPR_te_Fold[iRun,:,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:,-1] = np.std(AUPR_te_Fold[iRun,:,1:5+1], axis=1)
                
            elapsedTime = round((time.time() - timeRun) / 3600, 2)
            print(f'Run time: {elapsedTime} hours')
            break
            
        AUC__te_final[:, 0] = trainObj.epoch_ls
        AUC__te_final[:, iRun+1] = AUC__te_Fold[iRun, :, -2]
        AUC__te_final[:, -2] = np.mean(AUC__te_final[:, 1: 10+1], axis=1)
        AUC__te_final[:, -1] = np.std(AUC__te_final[:, 1: 10+1], axis=1)
        
        AUPR_te_final[:, 0] = trainObj.epoch_ls
        AUPR_te_final[:, iRun+1] = AUPR_te_Fold[iRun, :, -2]
        AUPR_te_final[:, -2] = np.mean(AUPR_te_final[:, 1: 10+1], axis=1)
        AUPR_te_final[:, -1] = np.std(AUPR_te_final[:, 1: 10+1], axis=1)
            
        # self.trainObj = trainObj
        self.AUC__tr_Fold = AUC__tr_Fold
        self.AUPR_tr_Fold = AUPR_tr_Fold
        self.AUC__te_Fold = AUC__te_Fold
        self.AUPR_te_Fold = AUPR_te_Fold
        self.AUC__te_final = AUC__te_final
        self.AUPR_te_final = AUPR_te_final
        return
        
    def runFold(self, data, Y, k_sets):
        args = self.args
        
        epochs = args.epochs
        numResult = len(range(1, epochs+1, 10))  # 每10轮记录一次

        AUC__tr_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_tr_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_te_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_final = np.zeros((numResult, 1+10+2))
        AUPR_te_final = np.zeros((numResult, 1+10+2))
        
        columns_Run = ['epoch'] + [f'Run{iRun+1}'
                                   for iRun in range(10)] + ['mean', 'std']
        columns_fold = ['epoch'] + [f'fold{fold+1}'
                                   for fold in range(5)] + ['mean', 'std']
        
        print("\n开始十次交叉验证...")
        for iRun in range(10):
            timeRun = time.time()
            print(f'\n=== 运行 {iRun+1}/10 ===')
            
            for kFold in range(5):
                timeFold = time.time()
                print(f'\n--- 折叠 {kFold+1}/5 ---')
                _, _, tr_mask, te_mask = k_sets[iRun][kFold]
                trainObj = Train(args)
                trainObj.trainModel(data, Y, tr_mask, te_mask)
                
                # 确保结果列表长度一致
                AUC__tr_ls = trainObj.AUC__tr_ls
                AUPR_tr_ls = trainObj.AUPR_tr_ls
                AUC__te_ls = trainObj.AUC__te_ls
                AUPR_te_ls = trainObj.AUPR_te_ls
                
                # 填充结果数组，确保维度匹配
                result_length = min(len(AUC__tr_ls), numResult)
                AUC__tr_Fold[iRun,:result_length,kFold+1] = np.array(AUC__tr_ls)[:result_length]
                AUPR_tr_Fold[iRun,:result_length,kFold+1] = np.array(AUPR_tr_ls)[:result_length]
                AUC__te_Fold[iRun,:result_length,kFold+1] = np.array(AUC__te_ls)[:result_length]
                AUPR_te_Fold[iRun,:result_length,kFold+1] = np.array(AUPR_te_ls)[:result_length]
                
                elapsedTime = round((time.time() - timeFold) / 60, 3)
                print(f'本折叠用时: {elapsedTime} 分钟')

            # 更新epoch列表和统计信息
            result_length = min(len(trainObj.epoch_ls), numResult)
            AUC__tr_Fold[iRun,:result_length,0] = np.array(trainObj.epoch_ls)[:result_length]
            AUC__tr_Fold[iRun,:result_length,-2] = np.mean(AUC__tr_Fold[iRun,:result_length,1:5+1], axis=1)
            AUC__tr_Fold[iRun,:result_length,-1] = np.std(AUC__tr_Fold[iRun,:result_length,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:result_length,0] = np.array(trainObj.epoch_ls)[:result_length]
            AUPR_tr_Fold[iRun,:result_length,-2] = np.mean(AUPR_tr_Fold[iRun,:result_length,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:result_length,-1] = np.std(AUPR_tr_Fold[iRun,:result_length,1:5+1], axis=1)
            
            AUC__te_Fold[iRun,:result_length,0] = np.array(trainObj.epoch_ls)[:result_length]
            AUC__te_Fold[iRun,:result_length,-2] = np.mean(AUC__te_Fold[iRun,:result_length,1:5+1], axis=1)
            AUC__te_Fold[iRun,:result_length,-1] = np.std(AUC__te_Fold[iRun,:result_length,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:result_length,0] = np.array(trainObj.epoch_ls)[:result_length]
            AUPR_te_Fold[iRun,:result_length,-2] = np.mean(AUPR_te_Fold[iRun,:result_length,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:result_length,-1] = np.std(AUPR_te_Fold[iRun,:result_length,1:5+1], axis=1)
                
            elapsedTime = round((time.time() - timeRun) / 3600, 2)
            print(f'本次运行用时: {elapsedTime} 小时')
            
            # 更新最终结果
            AUC__te_final[:result_length, 0] = np.array(trainObj.epoch_ls)[:result_length]
            AUC__te_final[:result_length, iRun+1] = AUC__te_Fold[iRun, :result_length, -2]
            AUC__te_final[:result_length, -2] = np.mean(AUC__te_final[:result_length, 1:iRun+2], axis=1)
            AUC__te_final[:result_length, -1] = np.std(AUC__te_final[:result_length, 1:iRun+2], axis=1)
            
            AUPR_te_final[:result_length, 0] = np.array(trainObj.epoch_ls)[:result_length]
            AUPR_te_final[:result_length, iRun+1] = AUPR_te_Fold[iRun, :result_length, -2]
            AUPR_te_final[:result_length, -2] = np.mean(AUPR_te_final[:result_length, 1:iRun+2], axis=1)
            AUPR_te_final[:result_length, -1] = np.std(AUPR_te_final[:result_length, 1:iRun+2], axis=1)
            
            # 打印当前运行的平均性能
            print(f"\n当前运行平均性能:")
            print(f"测试集 AUC: {AUC__te_final[result_length-1, -2]:.4f} ± {AUC__te_final[result_length-1, -1]:.4f}")
            print(f"测试集 AUPR: {AUPR_te_final[result_length-1, -2]:.4f} ± {AUPR_te_final[result_length-1, -1]:.4f}")
            
        print("\n交叉验证完成!")
        print("\n最终性能:")
        print(f"测试集 AUC: {AUC__te_final[result_length-1, -2]:.4f} ± {AUC__te_final[result_length-1, -1]:.4f}")
        print(f"测试集 AUPR: {AUPR_te_final[result_length-1, -2]:.4f} ± {AUPR_te_final[result_length-1, -1]:.4f}")
            
        self.AUC__tr_Fold = AUC__tr_Fold
        self.AUPR_tr_Fold = AUPR_tr_Fold
        self.AUC__te_Fold = AUC__te_Fold
        self.AUPR_te_Fold = AUPR_te_Fold
        self.AUC__te_final = AUC__te_final
        self.AUPR_te_final = AUPR_te_final
        return
    

# %% Main
if __name__ == '__main__':
    # 检查CUDA是否可用
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA设备数量:", torch.cuda.device_count())
        print("当前CUDA设备:", torch.cuda.current_device())
        print("CUDA设备名称:", torch.cuda.get_device_name(0))
    
    args = Args()
    print("使用设备:", args.dev)

    data, Y, mask_all, k_sets, pb, E = load_datasets(args)
    expObj = Experiment(args, data, Y, mask_all, k_sets)
    

def dynamic_sampling(X, y, model, threshold=0.5):
    # 根据当前模型预测的不确定性进行采样
    probs = model.predict_proba(X)
    uncertainty = np.abs(probs[:, 1] - threshold)
    sample_weights = 1 / (uncertainty + 1e-5)
    return sample_weights
    

class ContrastiveLearningEnsemble:
    def __init__(self, base_models):
        self.models = base_models
        
    def fit(self, X, y):
        # 对比学习损失
        def contrastive_loss(y_true, y_pred):
            positive_pairs =[] # ... 正样本对
            negative_pairs =[]# ... 负样本对
            return loss
    

# 1. 特征交叉
def create_interaction_features(df):
    # 数值特征交叉
    df['feat1_x_feat2'] = df['feat1'] * df['feat2']
    
    # 类别特征组合
    df['cat1_cat2'] = df['cat1'].astype(str) + '_' + df['cat2'].astype(str)
    return df

# 2. 时序特征提取
def add_time_features(df, date_column):
    df['hour'] = df[date_column].dt.hour
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['month'] = df[date_column].dt.month
    return df

# Stacking
estimators = [
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier())
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc'
)



# 自动计算权重
clf = LogisticRegression(class_weight='balanced')

# 手动设置权重
class_weights = {0: 1, 1: 5}  # 少数类给予更大的权重
clf = LogisticRegression(class_weight=class_weights)

# 使用平衡随机森林
brf = BalancedRandomForestClassifier(random_state=42)

# 使用EasyEnsemble
eec = EasyEnsembleClassifier(random_state=42)

def build_generator():
    # 构建GAN生成器
    pass

def build_discriminator():
    # 构建GAN判别器
    pass

class EnhancedAdaptiveFeature(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedAdaptiveFeature, self).__init__()
        
        # 改进通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )
        
        # 改进空间注意力
        self.spatial_attention = nn.Sequential(
            ChebConv(in_channels, in_channels * 2, K=4),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(),
            ChebConv(in_channels * 2, 1, K=3),
            nn.Sigmoid()
        )
        
        # 改进特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_channels * 4, in_channels * 2),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels)
        )
        
    def forward(self, x, edge_index):
        # 通道注意力
        channel_weights = self.channel_attention(torch.mean(x, dim=0))
        x = x * channel_weights
        
        # 空间注意力
        spatial_weights = self.spatial_attention(x, edge_index)
        x = x * spatial_weights
        
        # 特征增强
        enhanced = self.feature_enhancement(x)
        
        # 残差连接
        output = x + enhanced
        
        return output

class MultiScaleGraphAttention(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super(MultiScaleGraphAttention, self).__init__()
        
        self.scales = [2, 3, 4]  # 不同尺度的感受野
        
        # 多尺度图卷积
        self.conv_scales = nn.ModuleList([
            ChebConv(in_channels, in_channels, K=k) 
            for k in self.scales
        ])
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(in_channels * num_scales, num_scales),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(in_channels * num_scales, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index):
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.conv_scales:
            scale_feat = conv(x, edge_index)
            multi_scale_features.append(scale_feat)
            
        # 计算尺度注意力权重
        concat_features = torch.cat(multi_scale_features, dim=1)
        scale_weights = self.scale_attention(concat_features)
        
        # 加权融合
        weighted_features = []
        for i, features in enumerate(multi_scale_features):
            weighted = features * scale_weights[:, i].unsqueeze(1)
            weighted_features.append(weighted)
            
        # 特征融合
        fused = torch.cat(weighted_features, dim=1)
        output = self.fusion(fused)
        
        return output

class DynamicGraphLearning(nn.Module):
    def __init__(self, in_channels):
        super(DynamicGraphLearning, self).__init__()
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.graph_conv = ChebConv(in_channels, in_channels, K=3)
        
    def forward(self, x, edge_index):
        # 计算节点对特征
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        
        # 预测边权重
        edge_weights = self.edge_predictor(edge_features).squeeze()
        
        # 应用边权重
        weighted_features = self.graph_conv(x, edge_index, edge_weight=edge_weights)
        
        return weighted_features, edge_weights

class GraphStructureAdaptiveEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(GraphStructureAdaptiveEnhancement, self).__init__()
        
        # 结构重要性评估
        self.structure_scorer = nn.Sequential(
            nn.Linear(in_channels * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 自适应图卷积
        self.adaptive_conv = nn.ModuleList([
            ChebConv(in_channels, in_channels, K=k) 
            for k in [2, 3, 4]
        ])
        
        # 注意力融合
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 3, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, edge_index):
        # 评估边的重要性
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_scores = self.structure_scorer(edge_features)
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.adaptive_conv:
            features = conv(x, edge_index, edge_weight=edge_scores.squeeze())
            multi_scale_features.append(features)
            
        # 注意力加权融合
        concat_features = torch.cat(multi_scale_features, dim=1)
        attention_weights = self.attention(concat_features)
        
        # 加权组合
        output = sum([f * w.unsqueeze(1) for f, w in zip(multi_scale_features, attention_weights.t())])
        
        return output, edge_scores

class SelfSupervisedPretraining(nn.Module):
    def __init__(self, in_channels):
        super(SelfSupervisedPretraining, self).__init__()
        
        # 图结构预测
        self.structure_predictor = nn.Sequential(
            nn.Linear(in_channels * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # 节点特征重建
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LayerNorm(in_channels * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(in_channels * 2, in_channels)
        )
        
        # 对比学习头
        self.projection_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, x, edge_index):
        # 结构预测
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        structure_pred = self.structure_predictor(edge_features)
        
        # 特征重建
        reconstructed = self.feature_reconstructor(x)
        
        # 对比表示
        projected = self.projection_head(x)
        
        return {
            'structure_pred': structure_pred,
            'reconstructed': reconstructed,
            'projected': projected
        }

class DynamicEnsemblePrediction(nn.Module):
    def __init__(self, in_channels, num_experts=3):
        super(DynamicEnsemblePrediction, self).__init__()
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                ChebConv(in_channels, in_channels * 2, K=3),
                nn.LayerNorm(in_channels * 2),
                nn.GELU(),
                ChebConv(in_channels * 2, in_channels, K=2),
                nn.LayerNorm(in_channels)
            ) for _ in range(num_experts)
        ])
        
        # 动态权重生成
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index):
        # 专家预测
        expert_outputs = []
        for expert in self.experts:
            out = expert(x, edge_index)
            expert_outputs.append(out)
            
        # 生成集成权重
        ensemble_weights = self.weight_generator(x)
        
        # 加权集成
        weighted_sum = torch.zeros_like(x)
        for i, out in enumerate(expert_outputs):
            weighted_sum += out * ensemble_weights[:, i].unsqueeze(1)
            
        # 估计预测不确定性
        uncertainty = self.uncertainty_estimator(
            torch.cat([weighted_sum, x], dim=1)
        )
        
        return weighted_sum, uncertainty, ensemble_weights

class AdvancedPredictionFramework(nn.Module):
    def __init__(self, in_channels):
        super(AdvancedPredictionFramework, self).__init__()
        
        # 结构增强
        self.structure_enhancement = GraphStructureAdaptiveEnhancement(in_channels)
        
        # 自监督模块
        self.self_supervised = SelfSupervisedPretraining(in_channels)
        
        # 动态集成
        self.ensemble = DynamicEnsemblePrediction(in_channels)
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(in_channels, 1)
        )
        
    def forward(self, x, edge_index, phase='train'):
        # 结构增强
        enhanced_features, edge_importance = self.structure_enhancement(x, edge_index)
        
        if phase == 'train':
            # 自监督任务
            ssl_outputs = self.self_supervised(enhanced_features, edge_index)
        
        # 动态集成预测
        ensemble_features, uncertainty, weights = self.ensemble(enhanced_features, edge_index)
        
        # 特征融合
        final_features = torch.cat([enhanced_features, ensemble_features], dim=1)
        
        # 最终预测
        predictions = self.predictor(final_features)
        
        if phase == 'train':
            return predictions, {
                'ssl_outputs': ssl_outputs,
                'uncertainty': uncertainty,
                'ensemble_weights': weights,
                'edge_importance': edge_importance
            }
        else:
            return predictions

class TestEnsembleOptimization(nn.Module):
    def __init__(self, in_channels, num_models=3):
        super(TestEnsembleOptimization, self).__init__()
        
        # 多个预测模型
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(in_channels, 1)
            ) for _ in range(num_models)
        ])
        
        # 动态权重生成
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_models),
            nn.Softmax(dim=1)
        )
        
        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask):
        # 获取各个模型的预测
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        # 生成集成权重
        weights = self.weight_generator(x)
        
        # 计算不确定性
        uncertainty = self.uncertainty_estimator(x)
        
        # 加权集成
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * weights[:, i].unsqueeze(1)
            
        # 只优化测试集预测
        output = ensemble_pred.clone()
        output[mask] = output[mask] * (1 - uncertainty[mask])
        
        return output, uncertainty

def array_to_tuple(arr):
    try:
        return tuple(array_to_tuple(i) for i in arr)
    except TypeError:
        return arr

# 使用示例
array_2d = np.array([[1, 2], [3, 4]])
my_dict = {array_to_tuple(array_2d): 'value'}

class M_NoGAN(nn.Module):
    def __init__(self):
        super().__init__()
        # 保留多头注意力
        self.attention = MultiHeadAttention(num_heads=8)
        # 保留对比学习
        self.contrastive = ContrastiveLearning()
        # 特征转换层
        self.feature_transform = nn.Linear(input_dim, hidden_dim)

def feature_quality_analysis(features):
    # 计算特征区分度
    discriminative_score = calculate_discriminative(features)
    # 计算视图间相关性
    view_correlation = calculate_correlation(features)
    return discriminative_score, view_correlation

def create_ccg_tree():
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    G.add_node("74 novel CCGs", pos=(0.5, 1))
    
    # 添加主要分支节点
    branches = [(4, 0.2), (13, 0.4), (38, 0.6), (19, 0.8)]
    for count, x in branches:
        G.add_node(str(count), pos=(x, 0.7))
        G.add_edge("74 novel CCGs", str(count))
    
    # 添加Supporting evidence框
    G.add_node("Supporting evidence", pos=(0.3, 0.5))
    
    # 添加CancerMine分支
    G.add_node("14", pos=(0.8, 0.5))
    G.add_edge("19", "14")
    
    cancer_types = [("CancerMine\n(Driver)\n6", 0.7),
                   ("CancerMine\n(Oncogene)\n10", 0.8),
                   ("CancerMine\n(TSG)\n9", 0.9)]
    
    for label, x in cancer_types:
        G.add_node(label, pos=(x, 0.2))
        G.add_edge("14", label)
    
    return G

def plot_ccg_diagram():
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 生成网络
    G = create_ccg_tree()
    pos = nx.get_node_attributes(G, 'pos')
    
    # 绘制网络
    nx.draw(G, pos, with_labels=True, node_color='white', 
            node_size=3000, arrowsize=20, font_size=10,
            font_weight='bold', node_shape='s')
    
    # 添加Supporting evidence框
    plt.text(0.2, 0.5, "Supporting evidence\n\n□ OncoKB\n□ NCG candidate\n□ Bailey et al.\n■ Present □ Not",
             bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black'),
             ha='center', va='center', fontsize=8)
    
    # 添加颜色标记
    for i, (count, x) in enumerate([(4, 0.2), (13, 0.4)]):
        plt.gca().add_patch(plt.Rectangle((x-0.05, 0.65), 0.1, 0.1,
                                        facecolor='orange', alpha=0.3))
    
    plt.title("CCG Classification Tree")
    plt.axis('off')
    plt.tight_layout()
    
    return plt

# 生成图形
plt = plot_ccg_diagram()
plt.show()




