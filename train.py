import copy
import datetime
import gc
import os
import pickle
import statistics
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from torch_geometric import edge_index
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from models.model import MMAG, ContrastiveAttentionCompensation
import time
import logging

from utils import *
from args import args

use_agreed_label = args.use_agreed_label
predict_pairs = args.predict_pairs
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

train_losses = {
    "total_loss": [],
    "node_loss": [],
    "edge_loss": [],
    "edge_agreed_loss": [],
    "ou_dis_loss": []
}

eval_losses = {
    "total_loss": [],
    "node_loss": [],
    "edge_loss": [],
    "edge_agreed_loss": [],
    "ou_dis_loss": []
}

cache_path = "./cache/"


def run(task_name = "informative", alpha=0.2):
    print("running task:", task_name, 'use agreed label:', args.use_agreed_label)
    if use_agreed_label:
        train_file = f"crisis-agreed-{task_name}-tra-dev.pkl"
        test_file = f"crisis-agreed-{task_name}-test.pkl"
    else:
        train_file = f"crisis-disagreed-{task_name}-tra-dev.pkl"
        test_file = f"crisis-disagreed-{task_name}-test.pkl"
    crisis_file = os.path.join(cache_path, train_file)
    data, mappings, label_to_id = pickle.load(open(crisis_file, "rb"))
    test_data, test_mappings, test_label_to_id = pickle.load(open(os.path.join(cache_path, test_file), "rb"))
    labels = [k for k, v in sorted(label_to_id.items(), key=lambda kv: kv[1])]


    num_node_features = data.x.shape[1]

    hidden_features = 512
    num_classes = len(label_to_id)
    lr = args.lr
    device = args.device
    batch_size = args.batch_size
    temperature = args.temperature
    smoothing = args.smoothing
    gnn_out_dim = args.gnn_out_dim
    print("device:", device)
    gnn_type = args.gnn_type
    gnn_layers = args.gnn_num_layers
    ContrastiveAttentionFusion = ContrastiveAttentionCompensation(gnn_out_dim, num_classes, alpha).to(device)
    model = MMAG(num_node_features, hidden_features, num_classes, gnn_type, gnn_layers, gnn_out_dim, alpha).to(device)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'fusion' not in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'fusion' in n], 'lr': 8e-4}
    ]



    loss_func_bce = torch.nn.BCEWithLogitsLoss()
    weight_decay = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ConAttFu_optimizer = torch.optim.Adam(ContrastiveAttentionFusion.parameters(),lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)


    num_neighbors = [200, 20]

    sub_type = 'directional'
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=data.pairs_train,
        edge_label=data.pairs_train_y,
        batch_size=batch_size,
        replace=False,
        shuffle=True,
        # num_workers=num_workers,
        subgraph_type=sub_type,
        # directed=False
        drop_last=True,
    )

    dev_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=data.pairs_dev,
        edge_label=data.pairs_dev_y,
        batch_size=batch_size,
        replace=False,
        shuffle=True,
        # num_workers=num_workers,
        subgraph_type=sub_type,
        # directed=False,
        # drop_last=True,
    )

    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        edge_label_index=test_data.pairs_test,
        edge_label=test_data.pairs_test_y,
        batch_size=batch_size,
        # num_workers=num_workers,
        replace=False,
        shuffle=True,
        subgraph_type=sub_type,
        # directed=False,
        # drop_last=True,
    )

    ce_weight = None

    def smooth_loss(node_logit, node_y, pairs_index, pairs_y, node_mask, num_classes=8):
        pairs_node_y = node_y[pairs_index] # (2,E)
        mask = pairs_node_y != pairs_y # (2,E)
        rel_nodes = pairs_index.reshape(-1)
        return F.cross_entropy(node_logit[rel_nodes], node_y[rel_nodes])


    def train(model, epoch, loader):
        ContrastiveAttentionFusion.train()
        model.train()
        loss_list, node_acc_list, pairs_acc_list = [], [], []
        avg_loss, avg_node_acc, avg_edge_acc = 0.0, 0.0, 0.0

        batches = tqdm(loader, desc="Train Epoch {}".format(epoch))

        for batch in batches:
            batch = batch.to(device)
            pair_idx = batch.edge_label_index
            text_node, image_node = pair_idx[0], pair_idx[1]
            text, image, label = batch.x[text_node], batch.x[image_node], batch.edge_label

            text_label, img_label = batch.y[text_node], batch.y[image_node]
            match_label = (text_label[:,None] == img_label[None, :])
            inv_mask = ~match_label
            # train consattfusion modul
            h1, h2, attn = ContrastiveAttentionFusion(text, image)
            # attn = F.sigmoid(attn)
            attn = attn + inv_mask.float() * 0.5
            attn_input, match_label = attn.reshape(-1), match_label.clone().detach().to(torch.float).reshape(-1)
            attn_loss = loss_func_bce(attn_input, match_label)
            # match_label = match_label.clone().detach().to(torch.float)
            # attn_loss = loss_func_bce(torch.diagonal(attn), torch.diagonal(match_label)) * 0.5
            print(f'attn_loss: {attn_loss.item():.4f}')
            ConAttFu_optimizer.zero_grad()
            attn_loss.backward()
            ConAttFu_optimizer.step()
            #
            with torch.no_grad():
                h1, h2, _ = ContrastiveAttentionFusion(text, image)
            # h1, h2, _ = ContrastiveAttentionFusion(text, image)

            node_logit, pairs_logit, pairs_consistency_logit, x, gnn_x = model(batch, batch.edge_label_index, h1, h2)

            node_mask = batch.node_train_mask
            rel_node = batch.edge_label_index.reshape(-1)
            node_y = batch.y[rel_node]

            pairs_y = batch.edge_label

            node_loss = smooth_loss(node_logit, batch.y, batch.edge_label_index, pairs_y, node_mask, num_classes=num_classes)

            node_logit = node_logit[rel_node]

            pairs_loss = F.cross_entropy(pairs_logit, pairs_y, weight=ce_weight)

            if predict_pairs:
                loss = pairs_loss + node_loss * 0.1
            else:
                loss = node_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pairs_pred = torch.argmax(pairs_logit, dim=1).cpu().numpy()
            pairs_acc = metrics.accuracy_score(pairs_y.cpu().numpy(), pairs_pred)
            pairs_acc_list.append(pairs_acc)
            avg_edge_acc = sum(pairs_acc_list) / len(pairs_acc_list)

            node_pred = torch.argmax(node_logit, dim=1).cpu().numpy()
            node_acc = metrics.accuracy_score(node_y.cpu().numpy(), node_pred)
            node_acc_list.append(node_acc)
            avg_node_acc = sum(node_acc_list) / len(node_acc_list)

            loss_list.append(loss.item())
            avg_loss = sum(loss_list) / len(loss_list)

            train_losses["total_loss"].append(loss.item())
            train_losses["node_loss"].append(node_loss.item())

            batches.set_postfix(ordered_dict={"loss": "{:.4f}".format(avg_loss)}, refresh=True)

        return avg_loss, avg_node_acc, avg_edge_acc

    def evaluate(model, loader, epoch, test=False):
        model.eval()
        ContrastiveAttentionFusion.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_list, acc_list = [], []

        pairs_true_list, pairs_pred_list = [], []
        node_true_list, node_pred_list = [], []
        features_list = []
        node_type = []

        batches = tqdm(loader, desc="Eval  Epoch {}".format(epoch))
        for batch in batches:
            batch = batch.to(device)
            pair_idx = batch.edge_label_index
            text_node, image_node = pair_idx[0], pair_idx[1]
            text, image, label = batch.x[text_node], batch.x[image_node], batch.edge_label

            text_label, img_label = batch.y[text_node], batch.y[image_node]
            match_label = (text_label[:, None] == img_label[None, :])

            h1, h2, attn = ContrastiveAttentionFusion(text, image)

            attn_input, match_label = attn.reshape(-1), torch.tensor(match_label, dtype=torch.float).reshape(-1)
            attn_loss = loss_func_bce(attn_input, match_label)

            node_logit, pairs_logit, pairs_consistency_logit, x, gnn_x = model(batch, batch.edge_label_index,h1,h2, test=True)



            node_mask = batch.node_test_mask if test is True else batch.node_dev_mask

            rel_node = batch.edge_label_index.reshape(-1)
            node_y = batch.y[rel_node]


            pairs_y = batch.edge_label

            node_loss = smooth_loss(node_logit, batch.y, batch.edge_label_index, pairs_y, node_mask, num_classes=num_classes)

            node_logit = node_logit[rel_node]

            if predict_pairs:

                pairs_loss = F.cross_entropy(pairs_logit, pairs_y, weight=ce_weight)

                loss = pairs_loss

            else:
                loss = node_loss

            loss_list.append(loss.item())

            total_loss += loss.item()
            _, pair_predicted = torch.max(pairs_logit, 1)
            _, node_predicted = torch.max(node_logit, 1)

            if predict_pairs:
                total += pairs_y.size(0)
                correct += (pair_predicted == pairs_y).sum().item()
            else:
                total += node_y.size(0)
                correct += (node_predicted == node_y).sum().item()

            node_pred_list.extend(node_predicted.cpu().numpy())
            node_true_list.extend(node_y.cpu().numpy())

            pairs_pred_list.extend(pair_predicted.cpu().numpy())
            pairs_true_list.extend(pairs_y.cpu().numpy())

            eval_losses["total_loss"].append(loss.item())
            eval_losses["node_loss"].append(node_loss.item())

            avg_loss = sum(loss_list) / len(loss_list)
            postfix = {"loss": "{:.4f}".format(avg_loss)}
            batches.set_postfix(ordered_dict=postfix, refresh=True)

        node_report = metrics.classification_report(
            node_true_list, node_pred_list, zero_division=0.0, target_names=labels, digits=4)
        node_cm = metrics.confusion_matrix(node_true_list, node_pred_list)
        pair_report = metrics.classification_report(
            pairs_true_list, pairs_pred_list, zero_division=0.0, target_names=labels, digits=4)
        pair_cm = metrics.confusion_matrix(pairs_true_list, pairs_pred_list)
        acc = metrics.accuracy_score(node_true_list, node_pred_list)
        precision = metrics.precision_score(node_true_list, node_pred_list, average='weighted')
        recall = metrics.recall_score(node_true_list, node_pred_list, average='weighted')
        f1 = metrics.f1_score(node_true_list, node_pred_list, average='weighted')
        acc, precision, recall, f1 = round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)
        evaluation = (acc, precision, recall, f1)
        if test:
            print("node classification report:\n", node_report)
            print('node confusion matrix:\n', node_cm)
            print("pair classification report:\n", pair_report)
            print("pair confusion matrix:\n", pair_cm)
            print(f'node weighted avg: {acc} {precision} {recall} {f1}')

        if predict_pairs:
            return total_loss / len(batches), correct / total, loss_list, pair_report, pair_cm, evaluation
        else:
            return total_loss / len(batches), correct / total, loss_list, node_report, node_cm, evaluation

    best_loss = float("inf")

    best_epoch = 0
    max_try = args.early_step
    tries = 0


    mode_save_path = args.train_save_model_dir

    save_path = os.path.join(mode_save_path, f"{task_name}-model")
    os.makedirs(save_path, exist_ok=True)
    print('save path:', save_path)

    for epoch in range(args.epoch):

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, lr: {current_lr}")

        train_loss, train_node_acc, train_edge_acc = train(model, epoch, train_loader)
        logging.info(f'train epoch {epoch} loss: {train_loss:.4f}, train_node_acc: {train_node_acc:.4f}, train_edge_acc: {train_edge_acc:.4f}')
        eval_loss, eval_acc, eval_loss_list, _, _, e = evaluate(model, dev_loader, epoch, test=False)
        print(f'eval_loss:{eval_loss:.4f}, eval_acc:{eval_acc:.4f}')

        if best_loss > eval_loss:
            best_loss = eval_loss
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(save_path, f"best_{args.task}_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, f"best_optimizer_{args.task}.pth"))
        else:
            tries += 1

        if tries >= max_try:
            print("Early stopping, epoch {}".format(epoch))
            break

        print("\n")

    best_model_path = os.path.join(save_path, f"best_{args.task}_model.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=False), strict=True)
    print("Loaded best models from epoch {}, {}".format(best_epoch, best_model_path))

    print("Testing ")
    _, acc, _, report, cm, evaluation = evaluate(model, test_loader, 0, test=True)
    return acc, report, cm, evaluation


if __name__ == '__main__':
    acc, report, cm, evaluation = run()
    print(f'acc:{acc:.4f}')