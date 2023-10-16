"""
This scrip trains model to predict per point primitive type.
"""
import json
import logging
import nntplib
import os
import sys
from shutil import copyfile
from tabnanny import verbose

from torch import cosine_embedding_loss, index_put

from src.dataset_mix import my_mix_dataset


program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")

import os
from read_config import Config
config = Config(sys.argv[1])
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from read_config import Config

from src.SEDNet import SEDNet
from src.dataset_segments_my import my_simple_data
from src.dataset_segments import ori_simple_data

from src.segment_loss import (
    EmbeddingLoss,
    LabelSmoothingLoss,
    evaluate_miou,
    primitive_loss,
)
###
from src.My_edge_loss import compute_embedding_loss, edge_cls_loss, compute_edge_embedding_loss   # HPNet

model_name = config.model_path.format(
    config.batch_size,
    config.lr,
    config.mode,
    config.knn
)
print(model_name)


if not os.path.exists("trains/{}".format(model_name)):
    os.mkdir("trains/{}/".format(model_name))
    os.mkdir("trains/{}/config".format(model_name))
    os.mkdir("trains/{}/ckpts".format(model_name))

userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "trains/{}".format(model_name)+"/{}.log".format(model_name), mode="a"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "trains/{}/config".format(model_name)+"/config.json", "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "trains/{}/config".format(model_name)+"/{}".format(__file__.split("/")[-1])
copyfile(source_file, destination_file)
if_normals = config.normals

if_normal_noise = True

if_jitter_points = config.dataset == "noise"
if if_jitter_points:
    print("USE jitter NOISE!")

print("logs prepared!")


try:
    my_knn = config.knn
except:
    my_knn = 64
print("dgcnn knn {}".format(my_knn))

def on_load_checkpoint(model, state_dict) -> None:
        model_state_dict = model.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    '''
                    state_dict[k] = model_state_dict[k]
                    is_changed = True'''
            else:
                '''
                logger.info(f"Dropping parameter {k}")
                is_changed = True'''
                model_state_dict[k] = state_dict[k]
        return model_state_dict


Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

type_smoothCE_loss = LabelSmoothingLoss(smoothing=config.smooth)



model = SEDNet(
    embedding=True,
    emb_size=128,
    primitives=True,
    num_primitives=6,
    loss_function=Loss.triplet_loss,
    mode=5 if if_normals else 0,  
    num_channels= 6 if if_normals else 3,
    combine_label_prim=True,   # early fusion
    edge_module=True,  # add edge cls module
    late_fusion=True,    # ======================================
    nn_nb=my_knn, 
    predict_normal=False
)



print("model got!")
model = model.cuda()
if config.optim=="adam":
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
else:
    print("USE AdamW! L2 weight decay {}!".format(config.weight_decay))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)



if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

print("model to cuda!")
# ==== load ckpt 
if config.preload_model:
    print("loading from ckpt:", config.pretrain_model_path)

    state_dict = torch.load(config.pretrain_model_path)
    if torch.cuda.device_count() > 1:
        state_dict = {"module."+k: state_dict[k] for k in state_dict.keys()} if not list(state_dict.keys())[0].startswith("module.") else state_dict
    else:
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e: 
        print(e)
        print("load error!")
        new_dict = on_load_checkpoint(model, state_dict)
        model.load_state_dict(new_dict, strict=False)

if config.preload_model and config.pretrain_opti_path != "":
    print("loading from ckpt optimizer:", config.pretrain_opti_path)
    optimizer.load_state_dict(
        torch.load(config.pretrain_opti_path)
    )
    for g in optimizer.param_groups:
        g['lr'] = config.lr

print("model ckpt load!")


# origin ABC parsenet dataset + ours edge combined dataset for train

mix_train_dataset = my_mix_dataset(if_normals=if_normals, if_train=True, aug=False)  # ==== 

loader_train = torch.utils.data.DataLoader(
    mix_train_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True, drop_last=True, persistent_workers=True
)

print("get mixed train data")

# origin ABC parsenet dataset for test

mix_test_dataset = ori_simple_data(if_normals=if_normals, if_train=False)

loader_test = torch.utils.data.DataLoader(
    mix_test_dataset, batch_size=config.batch_size, num_workers=8, shuffle=False, drop_last=True, persistent_workers=True
)

print("get mixed test data")

cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
print("current LR: ", cur_lr)


if config.sche == "cos":
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=cur_lr / 20, verbose=True)
else:
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.patience, verbose=True, min_lr=5e-5
    )


prev_test_loss = 1e4
prev_inst_embed_loss = 1e4
prev_type_bce_loss = 1e4

eval_inter = config.eval_T

todebug = False

cur_inter = 0
for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_iou = [] 
    train_edgeBce = []
    train_losses = []
    train_edge_embed_loss = []
    model.train()

    num_iter = 1

    for train_b_id, data in enumerate(loader_train):   # ====================================> 1000 
        # ================== My ABC Edge train
        optimizer.zero_grad()
        losses = 0
        ious = 0
        p_losses = 0
        embed_losses = 0
        edge_cls_losses = 0
        edge_embed_losses = 0
        for _ in range(num_iter):
            points, labels, normals, primitives, edges, edges_W = data
            points, labels, normals, primitives, edges, edges_W = points.cuda(), labels.cuda(), normals.cuda(), primitives.cuda(), edges.cuda(), edges_W.cuda()
            aux_prim_logprob = None
            if if_normals:
                input = torch.cat([points, normals], 2).transpose(1,2)
                embedding, primitives_log_prob, _, edges_pred = model(points=input)
            else:
                embedding, primitives_log_prob, _, edges_pred = model(points.transpose(1,2))

            embed_loss = torch.mean(Loss.triplet_loss(embedding, labels.cpu().numpy()))
            
            primitives[(primitives==9) | (primitives==6) | (primitives==7)] = 0
            primitives[primitives==8] = 2
            edge_loss = edge_cls_loss(edges_pred, edges, edges_W)

            p_loss = type_smoothCE_loss(primitives_log_prob.transpose(1, 2).contiguous().view(-1, 6), primitives.contiguous().view(-1))  

            if aux_prim_logprob is not None:
                aux_p_loss = type_smoothCE_loss(aux_prim_logprob.transpose(1, 2).contiguous().view(-1, 6), primitives.contiguous().view(-1))  
            
            iou = 0
            
            edge_embed_loss = compute_edge_embedding_loss(edges_pred=edges_pred, pred_feat=embedding, 
                gt_label=labels,
                use_type=True, primitives=primitives, primitives_log_prob=primitives_log_prob
                ) 

            loss = embed_loss + p_loss + edge_loss +  0.25 * edge_embed_loss 

            if aux_prim_logprob is not None:
                loss += 0.5 * aux_p_loss
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            edge_cls_losses += edge_loss.data.cpu().numpy() / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter
            edge_embed_losses += aux_p_loss.data.cpu().numpy() / num_iter if aux_prim_logprob is not None else  0 

        optimizer.step()
        train_iou.append(ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        train_edgeBce.append(edge_cls_losses)
        train_edge_embed_loss.append(edge_embed_losses)

        cur_inter += 1
        print(
            "\rEpoch: {} iter: {}, prim loss: {}, emb loss: {}, iou: {}, edge_cls: {}, edge embed:{}".format(
                e, train_b_id, p_losses, embed_losses, iou, edge_cls_losses, edge_embed_losses
            ),
            end="",
        )    
        if cur_inter == eval_inter or todebug:
            todebug = False
            cur_inter = 0
            test_emb_losses = []
            test_prim_losses = []
            test_losses = []
            test_iou = []
            model.eval()

            for val_b_id, data in enumerate(loader_test):
                points, labels, normals, primitives, edges, edges_W = data
                points, labels, normals, primitives, edges, edges_W = points.cuda(), labels.cuda(), normals.cuda(), primitives.cuda(), edges.cuda(), edges_W.cuda()
                
                with torch.no_grad():
                    aux_prim_logprob = None
                    if if_normals:
                        input = torch.cat([points, normals], 2).transpose(1,2)
                        embedding, primitives_log_prob, _, edges_pred = model(input)

                    else:
                        embedding, primitives_log_prob, _, edges_pred = model(points.transpose(1,2))

                    embed_loss = torch.mean(compute_embedding_loss(embedding.transpose(1, 2), labels)[0])

                    primitives[(primitives==9) | (primitives==6) | (primitives==7)] = 0
                    primitives[primitives==8] = 2
                    p_loss = primitive_loss(primitives_log_prob, primitives)

                    loss = embed_loss + p_loss
                # 计算测试集 prim类别的iou
                iou = evaluate_miou(
                    primitives.data.cpu().numpy(),
                    primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
                )
                test_iou.append(iou)
                test_prim_losses.append(p_loss.data.cpu().numpy())
                test_emb_losses.append(embed_loss.data.cpu().numpy())
                test_losses.append(loss.data.cpu().numpy())

            # torch.cuda.empty_cache()
            print("\n")
            logger.info(
                "Epoch: {}/{} => TrL:{}, TsL:{}, TrP:{}, TsP:{}, TrE:{}, TsE:{}, TrI:{}, TsI:{}, TrEdgeCls {}, Tr EdgeEmbed {},".format(
                    e,
                    config.epochs,
                    np.mean(train_losses),
                    np.mean(test_losses),
                    np.mean(train_prim_losses),
                    np.mean(test_prim_losses),
                    np.mean(train_emb_losses),
                    np.mean(test_emb_losses),
                    np.mean(train_iou),
                    np.mean(test_iou),
                    np.mean(train_edgeBce),
                    np.mean(train_edge_embed_loss),
                )
            )

            my_crition = np.mean(test_emb_losses) + 0.15 * np.mean(test_prim_losses)

            test_emb_losses = np.mean(test_emb_losses)
            test_prim_losses = np.mean(test_prim_losses)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(my_crition)
            else:
                scheduler.step()

            
            if prev_test_loss > my_crition:
                logger.info("total improvement, saving model at epoch: {}".format(e))
                prev_test_loss = my_crition
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}.pth".format(model_name),
                )
            
            if prev_inst_embed_loss > test_emb_losses:
                logger.info("inst improvement, saving model at epoch: {}".format(e))
                prev_inst_embed_loss = test_emb_losses
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_InstBest.pth".format(model_name),
                )

            if prev_type_bce_loss > test_prim_losses:
                logger.info("type improvement, saving model at epoch: {}".format(e))
                prev_type_bce_loss = test_prim_losses
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_TypeBest.pth".format(model_name),
                )

            else:
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_latest.pth".format(model_name),
                )



