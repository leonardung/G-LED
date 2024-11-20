from main_seq_bfs import Args
import torch
import sys
import argparse
from torch.utils.data import DataLoader
import pdb

# Internal package
from util.utils import read_args_txt
from train_test_seq.test_seq import eval_seq_overall, test_plot_eval_truth_only
from data.data_bfs_preprocess import bfs_dataset
from transformer.sequentialModel import SequentialModel as transformer


class Args_seq_sample:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        """
		for training args txt
		"""
        self.parser.add_argument(
            "--train_args_txt",
            default="output/bfs_les_2024_11_20_10_49_56/logging/args.txt",
            help="load the args_train",
        )
        self.parser.add_argument(
            "--Nt_read", default=30, help="Which Nt model we need to read"
        )

        """
		for dataset
		"""
        self.parser.add_argument(
            "--trajec_max_len",
            default=500,
            help="max seq_length (per seq) to test the model",
        )
        self.parser.add_argument(
            "--start_n", default=0, help="the starting step of the data"
        )
        self.parser.add_argument(
            "--n_span",
            default=1000,
            help="the total step of the data from the staring step",
        )

        """
		for testing
		"""
        self.parser.add_argument(
            "--test_Nt", default=500, help="The length of forward propgatate"
        )
        self.parser.add_argument(
            "--batch_size",
            default=1,
            help="how many seqs you want to test together per bp",
        )
        self.parser.add_argument("--shuffle", default=False, help="shuffle the batch")
        self.parser.add_argument("--device", default="cuda:0")

    def update_args(self):
        args = self.parser.parse_args()
        args.experiment_path = None
        return args


if __name__ == "__main__":
    """
    Fetch args
    """
    args_sample = Args_seq_sample()
    args_sample = args_sample.update_args()
    args_train = read_args_txt(Args(), args_sample.train_args_txt)
    args_train.device = args_sample.device
    args_sample.experiment_path = args_train.experiment_path

    """
	Pre-check
	"""
    assert args_train.coarse_dim[0] * args_train.coarse_dim[1] * 2 == args_train.n_embd

    """
	Fetch dataset
	"""
    data_set = bfs_dataset(
        data_location=args_train.data_location,
        trajec_max_len=args_sample.trajec_max_len,
        start_n=args_sample.start_n,
        n_span=args_sample.n_span,
    )
    data_loader = DataLoader(
        dataset=data_set, shuffle=args_sample.shuffle, batch_size=args_sample.batch_size
    )

    """
	create loss function
	"""
    loss_func = torch.nn.MSELoss()

    """
	Eval
	"""
    down_sampler = torch.nn.Upsample(
        size=args_train.coarse_dim, mode=args_train.coarse_mode
    )
    test_plot_eval_truth_only(
        args=args_train,
        data_loader=data_loader,
        Nt=args_sample.test_Nt,
        down_sampler=down_sampler,
    )
    # eval_seq_overall(
    #     args_train=args_train,
    #     args_sample=args_sample,
    #     model=model,
    #     data_loader=data_loader,
    #     loss_func=loss_func,
    # )
