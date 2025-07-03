import datetime
import gc
import os
import pickle
import statistics
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from torch_geometric import edge_index
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
# from models.sage import Sage
from models.model import UniMModel
import time
import logging

from utils import *
from args import args

use_agreed_label = False
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


def run(task_name = "informative", alpha=0.2, beta=0.5, gamma=0.2, l1=0.2):
    print("running task:", task_name, 'use agreed label:', use_agreed_label)
    mode = args.mode
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

    task_idx = 1
    num_nodes = data.x.shape[0]
    num_node_features = data.x.shape[1]
    test_num_nodes = test_data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    test_num_edges = test_data.edge_index.shape[1]
    hidden_features = 512
    num_classes = len(label_to_id)
    lr = 0.0001
    device = args.device
    batch_size = 3000
    temperature = args.temperature
    smoothing = args.smoothing
    gnn_out_dim = args.gnn_out_dim
    print("device:", device)
    gnn_type = args.gnn_type
    gnn_layers = args.gnn_num_layers

    model = UniMModel(num_node_features, hidden_features, num_classes, gnn_type, gnn_layers, gnn_out_dim=gnn_out_dim).to(device)

    weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    tra_dev = [i for i, m in enumerate(data.node_test_mask) if not m]
    test_node = [i for i, m in enumerate(test_data.node_test_mask) if m]

    num_neighbors = [50, 5]
    # num_neighbors = [5]
    num_workers = 1


    data = data.subgraph((data.node_type if mode == 'image' else ~data.node_type))
    test_data = test_data.subgraph(test_data.node_type if mode == 'image' else ~test_data.node_type)

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes = data.node_train_mask,
        batch_size=batch_size,
        replace=False,
        shuffle=True,
        # num_workers=num_workers,
        subgraph_type='directional', # induced bidirectional directional(default)
        # directed=False
    )

    dev_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes = data.node_dev_mask,
        batch_size=batch_size,
        replace=False,
        shuffle=True,
        # num_workers=num_workers,
        subgraph_type='directional',
        # directed=False,
    )

    test_loader = NeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        input_nodes = test_data.node_test_mask,
        batch_size=batch_size,
        # num_workers=num_workers,
        replace=False,
        shuffle=True,
        subgraph_type='directional',
        # directed=False,
    )

    def train(model, epoch, loader):
        model.train()
        loss_list, node_acc_list, pairs_acc_list = [], [], []
        avg_loss, avg_node_acc, avg_edge_acc = 0.0, 0.0, 0.0

        batches = tqdm(loader, desc="Train Epoch {}".format(epoch))

        for batch in batches:
            batch = batch.to(device)

            node_logit, x = model(batch)

            node_y = batch.y[:batch_size]

            node_logit = node_logit[:batch_size]
            node_loss = F.cross_entropy(node_logit, node_y)

            loss = node_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

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
        total_loss = 0.0
        correct = 0
        total = 0
        loss_list, acc_list = [], []
        avg_loss, avg_acc = 0.0, 0.0

        pairs_true_list, pairs_pred_list = [], []
        node_true_list, node_pred_list = [], []
        features_list = []

        batches = tqdm(loader, desc="Eval  Epoch {}".format(epoch))
        for batch in batches:
            batch = batch.to(device)

            node_logit, x = model(batch)

            if test:
                node_mask = batch.node_test_mask
                node_y = batch.y[:batch.batch_size]
                node_logit = node_logit[:batch.batch_size]
            else:
                node_mask = batch.node_dev_mask
                node_y = batch.y[node_mask]
                node_logit = node_logit[node_mask]
            if test:
                features_list.append(x[:batch.batch_size].cpu())

            node_y = batch.y[:batch.batch_size]
            node_logit = node_logit[:batch.batch_size]

            node_loss = F.cross_entropy(node_logit, node_y)

            loss = node_loss

            loss_list.append(loss.item())

            total_loss += loss.item()
            _, node_predicted = torch.max(node_logit, 1)


            total += node_y.size(0)
            correct += (node_predicted == node_y).sum().item()

            node_pred_list.extend(node_predicted.cpu().numpy())
            node_true_list.extend(node_y.cpu().numpy())


            eval_losses["total_loss"].append(loss.item())
            eval_losses["node_loss"].append(node_loss.item())

            avg_loss = sum(loss_list) / len(loss_list)
            postfix = {"loss": "{:.4f}".format(avg_loss)}
            batches.set_postfix(ordered_dict=postfix, refresh=True)

        node_report = metrics.classification_report(
            node_true_list, node_pred_list, zero_division=0.0, target_names=labels, digits=4)
        node_cm = metrics.confusion_matrix(node_true_list, node_pred_list)

        if test:
            print("node classification report:\n", node_report)
            print('node confusion matrix:\n', node_cm)

        return total_loss / len(batches), correct / total, loss_list, node_report, node_cm

    best_loss = float("inf")
    best_acc = 0
    best_epoch = 0
    max_try = args.early_step
    tries = 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode_save_path = args.train_save_model_dir
    # save_path = os.path.join(mode_save_path, f"{task_name}-model_{timestamp}")
    save_path = os.path.join(mode_save_path, f"{task_name}-model")
    os.makedirs(save_path, exist_ok=True)
    print('save path:', save_path)

    for epoch in range(args.epoch):

        train_loss, train_node_acc, train_edge_acc = train(model, epoch, train_loader)
        logging.info(f'train epoch {epoch} loss: {train_loss:.4f}, train_node_acc: {train_node_acc:.4f}, train_edge_acc: {train_edge_acc:.4f}')
        eval_loss, eval_acc, eval_loss_list, _, _ = evaluate(model, dev_loader, epoch, test=False)
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
    # best_model_path = os.path.join(cache_path, "models-acc-{:.4f}.pth".format(best_acc))
    model.load_state_dict(torch.load(best_model_path, weights_only=False), strict=True)
    print("Loaded best model from epoch {}, {}".format(best_epoch, best_model_path))

    print("Testing ")
    _, acc, _, report, cm = evaluate(model, test_loader, 0, test=True)
    return acc, report, cm

def objective_function(alpha, beta, gamma, task = args.task):
    acc, report, cm = run(task, alpha, beta, gamma)
    torch.cuda.empty_cache()
    gc.collect()
    return acc, report, cm


alpha, beta, gamma, _ = 0, 0, 0, 0

best_acc = 0
best_params = None
if __name__ == '__main__':
    set_seed(2357)

    print(f'alpha: {alpha}, beta: {beta}, gamma: {gamma}')
    acc_list = []
    num_exp = 1
    for i in range(num_exp):
        print(f'running {i+1}/{num_exp}')
        acc, _, _ = objective_function(alpha, beta, gamma)
        acc_list.append(acc)
    logging.info(f'use_agreed_label:{args.use_agreed_label},predict_pairs:{args.predict_pairs} task:{args.task}')
    print(f'acc_list:{acc_list}')
    mean_value = statistics.mean(acc_list)
    print(f'mean_value:{mean_value}')