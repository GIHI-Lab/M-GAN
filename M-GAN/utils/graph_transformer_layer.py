import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    图注意力转换器层 (Graph Transformer Layer)
    实现了基于图结构的注意力机制，用于处理图数据
"""

"""
    工具函数
"""
def src_dot_dst(src_field, dst_field, out_field):
    """
    计算源节点和目标节点之间的点积注意力分数
    Args:
        src_field: 源节点特征字段
        dst_field: 目标节点特征字段
        out_field: 输出字段名
    """
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    """
    计算缩放指数，用于注意力分数的归一化
    Args:
        field: 输入字段
        scale_constant: 缩放常数
    """
    def func(edges):
        # 使用clamp确保数值稳定性
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func


"""
    多头注意力层实现
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        """
        初始化多头注意力层
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            num_heads: 注意力头数
            use_bias: 是否使用偏置
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # 初始化查询(Q)、键(K)、值(V)的线性变换层
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        """
        在图上传播注意力信息
        Args:
            g: 图对象
        """
        # 计算注意力分数
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # 将加权值发送到目标节点
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h):
        """
        前向传播
        Args:
            g: 图对象
            h: 节点特征
        Returns:
            多头注意力的输出
        """
        # 计算查询、键、值
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # 重塑为多头形式 [num_nodes, num_heads, feat_dim]
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        # 计算最终的注意力输出
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
    图转换器层
    包含多头注意力机制和前馈神经网络
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        """
        初始化图转换器层
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            num_heads: 注意力头数
            dropout: dropout比率
            layer_norm: 是否使用层归一化
            batch_norm: 是否使用批归一化
            residual: 是否使用残差连接
            use_bias: 是否使用偏置
        """
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        # 多头注意力层
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        # 输出投影层
        self.O = nn.Linear(out_dim, out_dim)

        # 归一化层
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # 前馈神经网络
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h):
        """
        前向传播
        Args:
            g: 图对象
            h: 节点特征
        Returns:
            转换后的节点特征
        """
        h_in1 = h # 第一个残差连接的输入
        
        # 多头注意力输出
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        # 第一个残差连接
        if self.residual:
            h = h_in1 + h
        
        # 第一个归一化层
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # 第二个残差连接的输入
        
        # 前馈神经网络
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        # 第二个残差连接
        if self.residual:
            h = h_in2 + h
        
        # 第二个归一化层
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        """
        返回层的字符串表示
        """
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)