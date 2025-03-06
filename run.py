import torch
import numpy as np
import argparse
import sys
import os
import wandb as wb
from pprint import pprint
from inscd import listener
from inscd.datahub import DataHub
from inscd.models.static.neural import NCDM


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ncdm', type=str,
                    help='method')
parser.add_argument('--data_type', default='real', type=str, help='benchmark')
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark')
parser.add_argument('--epoch', type=int, help='epoch of method', default=10)
parser.add_argument('--seed', default=0, type=int, help='seed for exp')
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cpu', type=str, help='device for exp')
parser.add_argument('--latent_dim', type=int, help='dimension of hidden layer', default=32)
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=128)
parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
parser.add_argument('--if_type', type=str, help='interaction type')
parser.add_argument('--weight_decay', type=float)

config_dict = vars(parser.parse_args())

method_name = config_dict['method']
name = f"{method_name}-{config_dict['data_type']}-seed{config_dict['seed']}"
tags = [config_dict['method'], config_dict['data_type'], str(config_dict['seed'])]
config_dict['name'] = name
method = config_dict['method']
datatype = config_dict['data_type']

if config_dict.get('if_type', None) is None:
    config_dict['if_type'] = config_dict['method']


pprint(config_dict)
run = wb.init(project="over-estimate", name=name,
              tags=tags,
              config=config_dict)
config_dict['id'] = run.id


def main(config):
    def print_plus(tmp_dict, if_wandb=True):
        pprint(tmp_dict)
        if if_wandb:
            wb.log(tmp_dict)

    listener.update(print_plus)
    set_seeds(config['seed'])
    datahub = DataHub(f"datasets/{config['data_type']}")
    datahub.random_split(source="total", to=["train", "test"], seed=config['seed'],
                             slice_out=1 - config['test_size'])
    validate_metrics = ['auc','acc']
    print("Number of response logs {}".format(len(datahub)))
    ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
    ncdm.build(device=config['device'])
    ncdm.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],epoch=config['epoch'], weight_decay=0.0005, lr=2e-3)
    theta=ncdm.get_attribute("over_estimate")
    np.savetxt("test_theta_3.csv",theta,fmt="%f")

if __name__ == '__main__':
    sys.exit(main(config_dict))
