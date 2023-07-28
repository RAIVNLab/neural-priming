import torch
from PIL import Image
import open_clip
import torchvision
import numpy as np
import importlib
import os 
import json
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from data import ImageNetV2, CustomDataset
from util import zeroshot_classifier_gpt, zeroshot_classifier, centroid, create_exp_ID
from tqdm import tqdm
from args import parse_arguments

Shift_Datasets = ['ImageNet-V2', 'sketch','ImageNet-r', 'ImageNet-a']

#See args.py for list of arguments
args = parse_arguments()
template_folder = 'Templates'
cache_path = args.cache_path
dataset_name = args.dataset
results_path = args.results_path
experiment_ID = create_exp_ID(args)

if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

if dataset_name == 'ImageNet':
    f = open('imagenet.json',)
    data = json.load(f)
    labels = list(data.values())
    class_names = list(data.values())
    dataset_obj = importlib.import_module(template_folder + '.' + dataset_name)
    templates = dataset_obj.templates

elif dataset_name in ['ImageNet-V2', 'sketch', 'ImageNet-r', 'ImageNet-a']:
    f = open('imagenet.json',)
    data = json.load(f)
    dataset_obj = importlib.import_module(template_folder + '.' + 'ImageNet')
    labels = list(data.keys())
    class_names = list(data.values())
    templates = dataset_obj.templates

else:
    dataset_obj = importlib.import_module(template_folder + '.' + dataset_name)
    labels = dataset_obj.classes
    class_names = labels
    templates = dataset_obj.templates

print('Loading model')
dataset_type = args.base_dataset
model_type = args.model
model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=args.base_dataset)

if dataset_name in Shift_Datasets + ['ImageNet']:
    root = '/data/yfcc-tmp/data/imagenet/'
    split = 'val'
    test_set = CustomDataset(args.val_path, preprocess)
    train_set = CustomDataset(args.train_path, preprocess)

elif dataset_name == 'ImageNet-V2':
    root = '/data/yfcc-tmp/data/ImageNetV2-matched-frequency/'
    split = 'val'
    test_set = ImageNetV2(root, preprocess)
    train_set = ImageNetV2(root, preprocess)

elif dataset_name == 'sketch':
    root = '/data/yfcc-tmp/data/sketch/'
    split = 'val'
    test_set = torchvision.datasets.ImageFolder(root, preprocess)
    train_set = torchvision.datasets.ImageFolder(root, preprocess)

elif dataset_name == 'ImageNet-r':
    root = '/tmp/imagenet-r/'
    split = 'val'
    test_set = CustomDataset(root, preprocess)
    train_set = CustomDataset(root, preprocess)

elif dataset_name == 'ImageNet-a':
    root = '/tmp/imagenet-a/'
    split = 'val'
    test_set = CustomDataset(root, preprocess)
    train_set = CustomDataset(root, preprocess)

elif dataset_name == 'SUN397':        
    dataset = torchvision.datasets.__getattribute__(dataset_name)
    test_set = dataset(args.val_path, transform = preprocess, download = True)
    train_set = test_set
else: 
    root = './'
    test_split = 'test'
    if args.dataset == 'OxfordIIITPet':
        train_split = 'trainval'
    else:
        train_split = 'train'
    

    dataset = torchvision.datasets.__getattribute__(dataset_name)
    test_set = dataset(root, split = test_split, transform = preprocess, download = True)
    train_set = dataset(root, split = train_split, transform = preprocess, download = True)

if args.prime:
    #Use custom_data for SUN, ImageNet variants, and transductive datasets. 
    if args.custom_data:
        subset = CustomDataset(args.subset_path, transform = preprocess)
    else: 
        subset = torchvision.datasets.ImageFolder(args.subset_path, transform = preprocess)

if args.prime:
    cti = subset.class_to_idx
    keys = cti.keys()
    vals = cti.values()
    if args.dataset =='OxfordIIITPet':
        keys = [x.lower() for x in keys]
    cti = dict(zip(keys,vals))
    idx_map = dict()
    for i,j in enumerate(labels):
        if '/' in j:
            j = j.replace('/', 'or')
        try:
            index = cti[j]
            idx_map[index] = i
        except Exception as e:
            # print(e)
            pass

train_set = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
test_set = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
if args.prime:
    subset = DataLoader(subset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)


 

print('creating OpenAI text classifier')
model.eval()
if args.cuda:
    model.cuda()
tokenizer = open_clip.get_tokenizer(args.model)
text = tokenizer(class_names)
text_features = zeroshot_classifier(class_names, templates, model, tokenizer).cpu().numpy().T

if args.cupl:
    print('creating CuPL classifier')
    tokenizer = open_clip.get_tokenizer(args.model)
    text_features_cupl = zeroshot_classifier_gpt(class_names, model, tokenizer, dataset_name).cpu().numpy().T
    #average cupl with OpenAI prompts
    text_features = (text_features + text_features_cupl)/2


print('Handling train image features')
if args.retrain and dataset_name not in Shift_Datasets and args.shots > 0:
    train_set_cpu = None
    train_labels = None
    for x,y in tqdm(train_set):
        if args.cuda:
            x = x.cuda()
        feats = model.encode_image(x).detach().cpu()
        feats /= feats.norm(dim=-1, keepdim=True)
        feats = feats.squeeze(0).numpy()

        if train_set_cpu is None:
            train_set_cpu = feats
        else:
            train_set_cpu = np.concatenate((train_set_cpu, feats), axis = 0)
        if train_labels is None:
            train_labels = y
        else:
            train_labels = np.concatenate((train_labels, y), axis = 0)
    if args.cache:
        np.save(cache_path + 'train_feats_' + experiment_ID, train_set_cpu)
        np.save(cache_path + 'train_labels_'+ experiment_ID, train_labels)

if args.shots > 0 and args.cache:
    train_set_cpu = np.load(cache_path + 'train_feats_'+ experiment_ID+'.npy')
    train_labels = np.load(cache_path + 'train_labels_'+ experiment_ID+'.npy')
else: 
    train_set_cpu_sampled = []
    train_labels_sampled = []

print('Handling test image features')
if args.retrain:
    test_set_cpu = None
    test_labels = None
    i=0
    for x,y in tqdm(test_set):
        i+=1
        if i > args.test_batches:
            break
        if args.cuda:
            x = x.cuda()
        feats = model.encode_image(x).detach().cpu()
        feats /= feats.norm(dim=-1, keepdim=True)
        feats = feats.numpy()
        if test_set_cpu is None:
            test_set_cpu = feats
        else:
            test_set_cpu = np.concatenate((test_set_cpu, feats), axis = 0)
        if test_labels is None:
            test_labels = y
        else:
            test_labels = np.concatenate((test_labels, y), axis = 0)
    if args.cache:
        np.save(cache_path + 'test_feats_'+experiment_ID, test_set_cpu)
        np.save(cache_path + 'test_labels_'+experiment_ID, test_labels)
if args.cache:
    test_set_cpu = np.load(cache_path + 'test_feats_'+experiment_ID+'.npy')
    test_labels = np.load(cache_path + 'test_labels_'+experiment_ID+'.npy')

subset_cpu = []
subset_labels=[]
print('Creating priming image features')
if args.retrain and args.prime:
    subset_cpu = None
    subset_labels=None
    print('Parsing Subset, length {}'.format(len(subset)))

    for x,y in tqdm(subset):
        if args.cuda:
            x = x.cuda()
        feats = model.encode_image(x).detach().cpu()
        feats /= feats.norm(dim=-1, keepdim=True)
        feats = feats.numpy()

        if subset_cpu is None:
            subset_cpu = feats
        else:
            subset_cpu = np.concatenate((subset_cpu, feats), axis = 0)
        if subset_labels is None:
            subset_labels = y
        else:
            subset_labels = np.concatenate((subset_labels, y.detach().cpu()), axis = 0)
    if args.cache:
        np.save(cache_path + 'subset_feats_' + experiment_ID + os.path.split(args.subset_path)[-1], subset_cpu)
        np.save(cache_path + 'subset_labels_' + experiment_ID+os.path.split(args.subset_path)[-1], subset_labels)

if args.prime and args.cache:
    subset_cpu = np.load(cache_path + 'subset_feats_'+experiment_ID+os.path.split(args.subset_path)[-1]+'.npy')
    subset_labels = np.load(cache_path + 'subset_labels_'+experiment_ID+os.path.split(args.subset_path)[-1]+'.npy') 
    subset_labels = [idx_map[x] for x in subset_labels]

indices = []
shot = args.shots

means = []
if args.shots > 0:
    indices = []
    for i in range(0,len(labels)):
        idx = np.random.choice(np.where(train_labels==i)[0], size = shot, replace = False)
        indices += list(idx)
    train_set_cpu_sampled = list(train_set_cpu[indices])
    train_labels_sampled = list(train_labels[indices])

if args.prime:
    train_set_cpu_sampled += list(subset_cpu)
    train_labels_sampled += list(subset_labels)

if args.shots > 0 or args.prime:
    cents = centroid(train_set_cpu_sampled, train_labels_sampled).numpy()
    cents = np.nan_to_num(cents)
    alpha = args.alpha
    text_features_new = (alpha)*text_features + (1.0-alpha)*cents
else:
    text_features_new = text_features

text_probs = (test_set_cpu @ text_features_new.T).argmax(axis=-1)
acc = (test_labels == text_probs).mean()
print("text zero-shot: {}".format(acc))
np.save(results_path + 'accuracy' + '_' + experiment_ID + args.prime, acc)




