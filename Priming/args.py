import argparse
import os
import numpy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='StanfordCars',help = 'Name of target dataset, i.e. Flowers102, StanfordCars. Ensure that it matches the name in the Templates Folder.')
    parser.add_argument('--subset_path', type=str, default=None,help = 'Path to priming set.')
    parser.add_argument('--base_dataset', type=str, default='laion2b_s34b_b88k',help = 'Pretraining Dataset')
    parser.add_argument('--test_path', type=str, default='',help = 'Path to test dataset')
    parser.add_argument('--model', type=str, default='ViT-B-16',help = 'Model')
    parser.add_argument('--shots', type=int, default=1,help = 'Number of examples')
    parser.add_argument('--test_batches', type=int, default=np.inf,help = 'Number of test batches')
    parser.add_argument('--alpha', type=float, default=.9,help = 'Number of examples')
    parser.add_argument('--prime', action='store_true', help='Use the priming data')
    parser.add_argument('--retrain', action='store_true', help='Recache the image features')
    parser.add_argument('--cupl', action='store_true', help='Use the CuPL prompts to initialize the text classifier')
    parser.add_argument('--root', type=str, default='./',help = 'Base directory.')
    parser.add_argument('--val_path', type=str, default=None,help = 'Directory for custom evaluation data sets. For torchvision datasets leave as none and they will be automatically loaded.')
    parser.add_argument('--train_path', type=str, default=None,help = 'Directory for train data. Needed for non-torch vision datasets.')
    parser.add_argument('--batch_size', type=int, default=16, help = 'Adjust to use memory appropriately')
    parser.add_argument('--cache', type=bool, default=True, help = 'Cache image features for faster iteration.')
    parser.add_argument('--cuda', type=bool, default=True, help = 'Cache image features for faster iteration.')
    parser.add_argument('--cache_path', type=str, default='./cache/',help = 'Base directory.')
    parser.add_argument('--results_path', type=str, default='./results/',help = 'Base directory.')
    parser.add_argument('--custom_data', action='store_true', help = 'Whether to use a custom loader. Use for ImageNetv2, and other ImageNet variants as well as SUN.')
    parser.add_argument('--num_workers', type=int, default=8, help = 'Number of workers for each dataloader.')
    parsed_args = parser.parse_args()
    return parsed_args
