# -*- coding: utf8 -*-
import csv
import os
import pickle
import re

import torch
from PIL import Image
from torch_geometric.data import Data

# from encoder_alt_clip import AltClip

# from encoder_align import EncoderAlign
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

from build_graph import build_attr_graph
from args import args

# dataset_dir = "D:\datasets"
dataset_dir = '/home/mnt/kliean/datasets/CrisisMMD'
data_image_dir = os.path.join(dataset_dir, "CrisisMMD_v2.0")
use_agreed_label = args.use_agreed_label

# sbert_model = "D:/models/transformers\clip-ViT-B-32"
sbert_model = "/opt/PLMs/clip-ViT-B-32"

cache_path = "./cache/"

task_names = ["informative", "humanitarian"]
# task_names = ["humanitarian"]
agreed_labels = {
    'Positive':1,
    'Negative':0
}

if use_agreed_label:
    data_splits_dir = os.path.join(dataset_dir, "crisismmd_datasplit_agreed_label")
    task_labels = {
        "humanitarian": [
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
        ],
        "informative": [
            "informative",
            "not_informative"
        ]
    }
    task_splits = {
        "humanitarian_dev": "task_humanitarian_text_img_agreed_lab_dev.tsv",
        "humanitarian_test": "task_humanitarian_text_img_agreed_lab_test.tsv",
        "humanitarian_train": "task_humanitarian_text_img_agreed_lab_train.tsv",

        "informative_dev": "task_informative_text_img_agreed_lab_dev.tsv",
        "informative_test": "task_informative_text_img_agreed_lab_test.tsv",
        "informative_train": "task_informative_text_img_agreed_lab_train.tsv",
    }
else:
    data_splits_dir = os.path.join(dataset_dir, "crisismmd_datasplit_all")
    task_labels = {
        "humanitarian": [
            "affected_individuals",
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",

            'vehicle_damage',
            'injured_or_dead_people',
            'missing_or_found_people',
        ],
        "informative": [
            "informative",
            "not_informative"
        ]
    }
    task_splits = {
        "humanitarian_dev": "task_humanitarian_text_img_dev.tsv",
        "humanitarian_test": "task_humanitarian_text_img_test.tsv",
        "humanitarian_train": "task_humanitarian_text_img_train.tsv",

        "informative_dev": "task_informative_text_img_dev.tsv",
        "informative_test": "task_informative_text_img_test.tsv",
        "informative_train": "task_informative_text_img_train.tsv",
    }


task_to_id = {t: i for i, t in enumerate(task_labels.keys())}
task_label_to_id = [{ll: i for i, ll in enumerate(task_labels[t])} for t in task_labels.keys()]

def load_tsv(tsv_path):
    data = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            it = {n: v for n, v in zip(header, row)}
            data.append(it)
    return data


def load_splits():
    return {
        name: load_tsv(os.path.join(data_splits_dir, split))
        for name, split in task_splits.items()
    }


def load_image(image_path):
    img_file = os.path.join(data_image_dir, image_path)
    img = Image.open(img_file).convert("RGB")
    return img


def clean_text(text):
    paten = re.compile(r'https?://[^ ]+', re.DOTALL)
    text = re.sub(paten, '', text)
    text = re.sub(r'[^a-zA-Z0-9$@$!%*?&#^-_. +]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub("#[A-Za-z0-9_]+", "", text) # Remove the hashtag from the text (including the # and what follows)
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = text.lower()
    return text


def encode(data_splits):
    os.makedirs(cache_path, exist_ok=True)
    embed_file = os.path.join(cache_path, "crisis-embed-v1.pkl")
    bert_file = os.path.join(cache_path, "crisis-bert.pkl")
    # load from cache
    if os.path.exists(embed_file):
        embed = pickle.load(open(embed_file, "rb"))
        return embed

    mm_encoder = SentenceTransformer(sbert_model)
    embed = {}
    bert_result = {}
    for split_name, split_data in data_splits.items():
        for it in split_data:
            if it['tweet_id'] not in embed:
                print("text", it['tweet_id'])
                # embed[it['tweet_id']] = mm_encoder.encode_text(clean_text(it['tweet_text']))
                tweet = it['tweet_text']
                hashtags = re.findall(r"#\w+", tweet)
                hts = [ht[1:].lower() for ht in hashtags]
                # if len(hts) > 0:
                #     embed_hts = mm_encoder.encode(hts).mean(axis=0)
                #     embed[it['tweet_id']] = mm_encoder.encode(clean_text(it['tweet_text'])) + embed_hts
                # else:
                #     embed[it['tweet_id']] = mm_encoder.encode(clean_text(it['tweet_text']))

                embed[it['tweet_id']] = mm_encoder.encode(clean_text(it['tweet_text']))


            if it['image_id'] not in embed:
                print("image", it['image_id'])
                # noinspection PyTypeChecker
                # embed[it['image_id']] = mm_encoder.encode_image(load_image(it['image']))
                embed[it['image_id']] = mm_encoder.encode(load_image(it['image']))

    for labels in task_labels:
        label_to_id = {label: i for i, label in enumerate(labels)}
        task_label_to_id.append(label_to_id)

    with open(embed_file, "wb") as f:
        pickle.dump(embed, f)


    with open(bert_file, "wb") as f:
        pickle.dump(bert_result, f)

    return embed

def dataset_to_graph(task, data_splits, embed, mode='tra-dev', edge_type='a', tau=0.8):
    os.makedirs(cache_path, exist_ok=True)
    if use_agreed_label:
        crisis_file = os.path.join(cache_path, f"crisis-agreed-{task}-{mode}.pkl")
    else:
        crisis_file = os.path.join(cache_path, f"crisis-disagreed-{task}-{mode}.pkl")

    mapping = {k: i for i, k in enumerate(embed.keys())}


    text_idx, img_idx = [], []
    for k, i in mapping.items():
        if '_' in k:
            img_idx.append(i)
        else:
            text_idx.append(i)

    x = [embed[k].tolist() for k in embed.keys()]
    x = torch.tensor(x, dtype=torch.float)

    tsk_id = task_to_id[task]

    norm_x = torch.nn.functional.normalize(x, dim=1)


    att_edges = torch.zeros((len(x), len(x)), dtype=torch.bool)

    similarity_matrix = torch.mm(norm_x, norm_x.t())

    use_selfloops = True
    directed = True

    if edge_type == 's':
        sem_edges = similarity_matrix > tau
        if ~use_selfloops:
            sem_edges.fill_diagonal_(False)

        edges = sem_edges
    elif edge_type == 'a':
        # 构造属性边
        all_items = [item for lst in data_splits.values() for item in lst]
        attr_edges_ = build_attr_graph(all_items, mapping, similarity_matrix)
        for e in attr_edges_:
            i, j = e
            att_edges[i, j] = True
            att_edges[j, i] = True
        if ~use_selfloops:
            att_edges.fill_diagonal_(False)

        edges = att_edges
    elif edge_type == 'a_s':
        # edges = torch.triu(similarity_matrix, diagonal=1) > tau
        sem_edges = similarity_matrix > tau

        all_items = [item for lst in data_splits.values() for item in lst]
        attr_edges_ = build_attr_graph(all_items, mapping, similarity_matrix)
        for e in attr_edges_:
            i, j = e
            att_edges[i, j] = True
            att_edges[j, i] = True # 构造为无向图.
        if ~use_selfloops:
            sem_edges.fill_diagonal_(False)
            att_edges.fill_diagonal_(False)
        edges = sem_edges | att_edges
    else:
        raise ValueError(f"Invalid edges_type: {edge_type}. Expected 'a', 's', or 'a_s'.")

    nodes_y = torch.zeros((len(embed)), dtype=torch.long)
    edges_y = torch.zeros((len(embed), len(embed)), dtype=torch.long)
    torch.fill_(edges_y, -1)

    node_train_mask = torch.zeros((len(embed)), dtype=torch.bool)
    node_dev_mask = torch.zeros((len(embed)), dtype=torch.bool)
    node_test_mask = torch.zeros((len(embed)), dtype=torch.bool)

    # textual node or image node, 0 indicates that the node is of text type, while 1 represents an image.
    node_type = torch.zeros(len(embed), dtype=torch.bool)

    pairs_train = torch.zeros((len(embed), len(embed)), dtype=torch.bool)
    pairs_test = torch.zeros((len(embed), len(embed)), dtype=torch.bool)
    pairs_dev = torch.zeros((len(embed), len(embed)), dtype=torch.bool)
    pairs_label = torch.zeros((len(embed), len(embed)), dtype=torch.long)
    num_agreed = 0
    bert_result = []
    for split_name, split_data in data_splits.items():
        if task not in split_name:
            continue

        for it in split_data:
            src = mapping[it['tweet_id']]
            trg = mapping[it['image_id']]

            if task in ["humanitarian", "informative"]:
                nodes_y[src] = task_label_to_id[tsk_id][it['label_text']]
                nodes_y[trg] = task_label_to_id[tsk_id][it['label_image']]

                edges_y[src, trg] = task_label_to_id[tsk_id][it['label']]
                pairs_label[src, trg] = task_label_to_id[tsk_id][it['label']]

                node_type[trg] = True

            if split_name.endswith("_train"):
                node_train_mask[src] = True
                node_train_mask[trg] = True

                pairs_train[src, trg] = True

            if split_name.endswith("_dev"):
                node_dev_mask[src] = True
                node_dev_mask[trg] = True

                pairs_dev[src, trg] = True

            if split_name.endswith("_test"):
                node_test_mask[src] = True
                node_test_mask[trg] = True

                pairs_test[src, trg] = True

            if edge_type != 's':
                edges[src, trg] = True
                edges[trg, src] = True
                num_agreed += 1

    if ~use_selfloops:
        edges.fill_diagonal_(False) # 去除自环

    if ~directed:
        edges = torch.triu(edges) # 仅保留上三角的元素
    edge_index = torch.nonzero(edges).t()
    edge_label = torch.masked_select(edges_y, edges)

    data = Data(x=x, y=nodes_y, edge_index=edge_index, edge_attr=edge_label)

    data.pairs_train = torch.nonzero(pairs_train).t()
    data.pairs_test = torch.nonzero(pairs_test).t()
    data.pairs_dev = torch.nonzero(pairs_dev).t()
    data.pairs_train_y = torch.masked_select(pairs_label, pairs_train)
    data.pairs_test_y = torch.masked_select(pairs_label, pairs_test)
    data.pairs_dev_y = torch.masked_select(pairs_label, pairs_dev)

    data.node_train_mask = node_train_mask
    data.node_dev_mask = node_dev_mask
    data.node_test_mask = node_test_mask

    data.node_type = node_type
    # data.pairs_idx = torch.tensor(pairs_idx, dtype=torch.long)

    label_to_id = task_label_to_id[tsk_id]

    with open(crisis_file, "wb") as f:
        pickle.dump((data, mapping, label_to_id), f)
        print('data saved:',crisis_file)


    return data, mapping, label_to_id


def run(edge_type='a', tau=0.8):
    data_splits = load_splits()
    embed = encode(data_splits)
    for task in task_names:
        # 只保留特定任务的数据
        filtered_data_splits = {k: v for k, v in data_splits.items() if k.startswith(f"{task}_")}
        # 筛选出该任务的特征
        task_embed = {}
        for split_name, split_data in filtered_data_splits.items():
            for it in split_data:
                if it['tweet_id'] not in task_embed:
                    task_embed[it['tweet_id']] = embed[it['tweet_id']]

                if it['image_id'] not in task_embed:
                    task_embed[it['image_id']] = embed[it['image_id']]

        print(f'构建任务: {task} 的图数据')
        # 构建两个图
        dataset_to_graph(task, {k:v for k, v in filtered_data_splits.items() if not k.endswith('_test')}, task_embed, 'tra-dev', edge_type=edge_type, tau=tau)
        dataset_to_graph(task, filtered_data_splits, task_embed, 'test', edge_type=edge_type, tau=tau)
        print(f'任务: {task} 的图数据构建完成！')


if __name__ == "__main__":
    edge_type = ['a', 's', 'a_s']
    tau = [0.8]
    run(edge_type[2], tau[0])

    print('Done!')

    exit(0)
