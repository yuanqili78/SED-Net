import math
from audioop import bias
from turtle import forward, position

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from positional_encodings.torch_encodings import (PositionalEncoding1D,
                                                  PositionalEncoding2D,
                                                  PositionalEncoding3D, Summer)

from PointNet import (get_graph_feature, get_graph_feature_with_normals, knn,
                      knn_points_normals, my_get_graph_feature)


class DGCNNEncoderGn(nn.Module):
    def __init__(self, mode=0, input_channels=3, nn_nb=80, normal_metric_W=1.):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.mode = mode
        self.drop = 0.0
        self.input_channels = input_channels

        self.normal_metric_W = normal_metric_W

        if self.mode == 0 or self.mode == 5:
            self.bn1 = nn.GroupNorm(2, 64)
            self.bn2 = nn.GroupNorm(2, 64)
            self.bn3 = nn.GroupNorm(2, 128)
            self.bn4 = nn.GroupNorm(4, 256)
            self.bn5 = nn.GroupNorm(8, 1024)

            self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.mlp1 = nn.Conv1d(256, 1024, 1)
            self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.mode == 0 or self.mode == 1:
            # First edge conv
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))

            x4 = x.max(dim=2)[0]

            return x4, x_features

        if self.mode == 5:
            # First edge conv
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k, normal_metric_W = self.normal_metric_W)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]

            return x4, x_features


class PrimitivesEmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, emb_size=50, num_primitives=8, primitives=False, embedding=False, mode=0, num_channels=3,
                 loss_function=None, nn_nb=80, combine_label_prim=False, edge_module=False, late_fusion=False):
        super(PrimitivesEmbeddingDGCNGn, self).__init__()
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb)
        self.drop = 0.0
        self.loss_function = loss_function

        if self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 5 or self.mode == 6:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        elif self.mode == 1 or self.mode == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)

        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.combine_label_prim = combine_label_prim

        self.late_fusion = late_fusion

        self.edge_module = None
        if edge_module:
            self.edge_module = nn.Sequential(
                torch.nn.Conv1d(256, 128, 1),
                nn.GroupNorm(4, 128),
                torch.nn.Conv1d(128, 2, 1),
            )

        if self.combine_label_prim:
            self.asis = nn.Sequential(
                torch.nn.Conv1d(256, 256, 1),
                nn.GroupNorm(4, 256),
                nn.ReLU(True),
                nn.Dropout(0.0),   # 0.7
            )
        
        if self.embedding:
            '''
            in_channel = self.emb_size 
            if self.late_fusion:
                in_channel += num_primitives
                if self.edge_module:
                    in_channel += 2 '''
            self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points, labels=None, compute_loss=True):
        """
        edges: B x N long 0 or 1
        """
        batch_size = points.shape[0]
        num_points = points.shape[2]
        x, first_layer_features = self.encoder(points)  # 1024

        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)  # B x 1024 x N
        x = torch.cat([x, first_layer_features], 1)  # B x 1024+256 x N

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)   # 1024+256 -> 512
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)  # 512 -> 256

        if self.edge_module:
            edges_pred = self.edge_module(x_all)  # B x 2 x N

        x_type = None
        
        x_labels = None  # [B, 10, N]  softmax


        if self.primitives:
            x_type = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)  # 256 -> 256
            x = self.mlp_prim_prob2(x_type)  # 256 -> 10  10 prim
            if self.late_fusion:
                x_labels = torch.softmax(x, dim=1)
            primitives_log_prob = self.logsoftmax(x)

        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)  # 256 -> 256
            if self.combine_label_prim and self.primitives:
                x = self.asis(x_type) + x  # 256 -> 256

            embedding = self.mlp_seg_prob2(x)  # 256 -> 128

            if self.late_fusion:
                embedding = torch.cat([embedding, x_labels], dim=1)
                if self.edge_module:
                    embedding = torch.cat([embedding, torch.softmax(edges_pred, dim=1)], dim=1)

        if compute_loss:
            embed_loss = self.loss_function(embedding, labels.data.cpu().numpy())  # cluster instance
        else:
            embed_loss = torch.zeros(1).cuda()
        return embedding, primitives_log_prob, embed_loss, edges_pred


class SEDNet(nn.Module):
    def __init__(self, emb_size=50, num_primitives=8, primitives=False, embedding=False, mode=0, num_channels=3,
                 loss_function=None, nn_nb=80, combine_label_prim=False, edge_module=False, late_fusion=False, 
                 w_pos_enc=0.2, normal_metric_W=1., predict_normal=False):
        super(SEDNet, self).__init__()
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb, normal_metric_W=normal_metric_W)
        self.drop = 0.0
        self.loss_function = loss_function
        self.w_pos_enc = w_pos_enc

        if self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 5 or self.mode == 6:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)  # ================= default
        elif self.mode == 1 or self.mode == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)

        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.combine_label_prim = combine_label_prim

        self.late_fusion = late_fusion

        self.edge_module = None
        if edge_module:
            self.edge_module = nn.Sequential(
                torch.nn.Conv1d(256, 128, 1),
                nn.GroupNorm(4, 128),
                torch.nn.Conv1d(128, 2, 1),
            )

        if self.combine_label_prim:
            self.asis = nn.Sequential(
                torch.nn.Conv1d(256, 256, 1),
                nn.GroupNorm(4, 256),
                nn.ReLU(True),
                nn.Dropout(0.0),  
            )
        
        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)
        
        self.predict_normal = predict_normal


        if self.predict_normal:
            self.normal_predict_mlps = nn.Sequential(
                torch.nn.Conv1d(256, 128, 1),
                nn.GroupNorm(4, 128),
                torch.nn.Conv1d(128, 3, 1),
                # nn.nn.Tanh()   # ==========
            )            

        
        self.pos_enc = PositionalEncoding1D(256)

        self.prim_encoding = nn.Sequential(
            nn.Conv1d(8, 256, 1),
            nn.ReLU()
        )

    def forward(self, points, labels=None, compute_loss=False):
        """
        edges: B x N long 0 or 1
        """
        batch_size = points.shape[0]
        num_points = points.shape[2]
        x, first_layer_features = self.encoder(points)  # 1024

        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)  # B x 1024 x N
        x = torch.cat([x, first_layer_features], 1)  # B x 1024+256 x N

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)   # 1024+256 -> 512
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)  # 512 -> 256
        
        if self.predict_normal:
            normals_pred = torch.nn.functional.normalize(self.normal_predict_mlps(x_all), p=2, dim=1)  # B x 3 x N

        x_type = None

        if self.primitives:
            x_type = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)  # 256 -> 256
            type_logit = self.mlp_prim_prob2(x_type)  
            primitives_log_prob = self.logsoftmax(type_logit)

            if self.edge_module:
                edges_pred = self.edge_module(x_type)  # B x 2 x N

        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)  # 256 -> 256
            if self.combine_label_prim and self.primitives:
                x = self.w_pos_enc * self.asis(x_type) + x  # 256 -> 256

            if self.late_fusion:
                # x = x + self.inv_freq * self.gen_cls_enc(x, primitives_log_prob.detach())
                x = x + self.w_pos_enc * self.prim_encoding(torch.cat((type_logit.detach(), edges_pred.detach()), dim=1))  # 256 -> 256


            embedding = self.mlp_seg_prob2(x)  # 256 -> 128
                

        if compute_loss:
            embed_loss = self.loss_function(embedding, labels.data.cpu().numpy())  # cluster instance
        else:
            embed_loss = torch.zeros(1).cuda()

        ret_res = [embedding, primitives_log_prob, embed_loss,]
        if self.edge_module:
            ret_res.append(edges_pred)
        if self.predict_normal:
            ret_res.append(normals_pred)        
        return  ret_res

