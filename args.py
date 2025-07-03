
import argparse

import torch

parser = argparse.ArgumentParser(description='**MH-GNN**')

parser.add_argument('--cache', default='./cache', type=str)
parser.add_argument('--train_save_model_dir', default='./checkpoints/train_models', type=str)
parser.add_argument('--pretrain_model_dir', default='./pretrain')
parser.add_argument('--use_agreed_label', type=bool, default=False, help='')
parser.add_argument('--predict_pairs', type=bool, default=False, help='predict text and image pairs')
parser.add_argument('--mode', type=str, default='text', help='text or image')
parser.add_argument('--task', dest='task', default='informative', type=str, help='informative,humanitarian')

parser.add_argument('--early_step', type=int, default=5, help='early step')
parser.add_argument('--gnn_out_dim', dest='gnn_out_dim', default=512, type=int)
parser.add_argument('--heads', type=int, default=4, help='num of gat heads')
parser.add_argument('--gnn_num_layers', type=int, default=2, help='num of gnn layers')
parser.add_argument('--gnn_type', type=str, default='sage', help='gat,sage,gcn')
parser.add_argument('--temperature', type=float, default=1, help='temperature')
parser.add_argument('--smoothing', type=float, default=0, help='')

parser.add_argument('--batch_size', dest='batch_size', default=1500, type=int, help='1000 if gat') # bs:1500 gat:1000
parser.add_argument('--lr', type=float, default=0.0004) # info 0.0004 hum 0.0002

parser.add_argument('--epoch', type=float, default=50)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

args = parser.parse_args()
