import numpy as np
import torch
import os
from PIL import Image
import json
from tqdm import tqdm

def zeroshot_classifier(classnames, templates, model, tokenizer):
	with torch.no_grad():
		zeroshot_weights = []
		for classname in tqdm(classnames):
			texts = [template.format(classname) for template in templates] #format with class
			texts = tokenizer(texts).cuda() #tokenize
			class_embeddings = model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
	return zeroshot_weights

def zeroshot_classifier_gpt(classnames, model, tokenizer, dataset = None, templates=None, use_both=False):

	# keys = [x.replace('(', '').replace(')', '') for x in gpt3_prompts.keys()]
	#keys = [x.lower() for x in gpt3_prompts.keys()]
	# values = gpt3_prompts.values()
	# gpt3_prompts = dict(zip(keys, values))

	if dataset in ['ImageNet', 'ImageNet-a','ImageNet-r', 'sketch', 'ImageNet-V2']:
		with open('CuPL_image_prompts.json') as f:
			gpt3_prompts = json.load(f)

	elif dataset == 'oxford-iiit-pet':
		with open('pets_prompts_full.json') as f:
			gpt3_prompts = json.load(f)

	elif dataset == 'SUN397':
		with open('sun_prompts_full.json') as f:
			gpt3_prompts = json.load(f)
		keys = [x.replace('(', '').replace(')', '') for x in gpt3_prompts.keys()]
		keys = [x.lower() for x in gpt3_prompts.keys()]
		values = gpt3_prompts.values()
		gpt3_prompts = dict(zip(keys, values))
		
	elif dataset == 'Flowers102':
		with open('flower_prompts_full.json') as f:
			gpt3_prompts = json.load(f)
	elif dataset == 'StanfordCars':
		with open('cars_prompts_full.json') as f:
			gpt3_prompts = json.load(f)

	elif dataset == 'Food101':
		with open('descriptors_food101.json') as f:
			gpt3_prompts = json.load(f)

	classnames = [x.replace(' or ', ' / ') for x in classnames]
	with torch.no_grad():
		zeroshot_weights = []
		for i in tqdm(range(len(classnames))):
			if use_both:
				texts = [template.format(classnames[i]) for template in templates]
			else:
				texts = []

			for t in gpt3_prompts[classnames[i]]:
				texts.append(t)
			texts = tokenizer(texts).cuda() #tokenize
			class_embeddings = model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm().cpu().detach()
			zeroshot_weights.append(class_embedding)

		zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
	return zeroshot_weights


def centroid(embeddings, labels):
    labels = torch.tensor(labels)
    embeddings = torch.tensor(np.array(embeddings))
    onehot = torch.zeros(labels.size(0), labels.max()+1)
    filled_onehot = onehot.scatter_(1, labels.unsqueeze(dim=1), 1)
    new_prototypes = torch.mm(filled_onehot.permute((1, 0)), embeddings)
    new_prototypes /= new_prototypes.norm(dim=-1, keepdim=True)
    return new_prototypes

def create_exp_ID(args):
	return args.model+ '_' + args.base_dataset 
	+ '_' + os.path.split(args.subset_path)[-1]

