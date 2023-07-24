from collections import defaultdict
from numpy.lib.format import open_memmap
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from utils.dataset import SafeImageFolder

import torchvision
import clip
import shutil
import pickle

import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import math
import os
import json
import random as r
import open_clip


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-path",
        type=Path,
        default=None,
        required=True,
        help="Path to retrieval reservoir (clean subset of LAION)",
    )
    parser.add_argument(
        "--transductive-path",
        type=Path,
        default=None,
        required=True,
        help="Path to transfer evaluation dataset (to be used in a transductive fashion)",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=None,
        help="Number of shots per class, only used in train-time data augmentation",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("/usr/aux/gen-datasets/cache"),
        help="Path to cache",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        required=True,
        help="Path to output directory (dataset in ImageFolder format)",
    )
    parser.add_argument(
        "--retrievals-per-image",
        default=10,
        type=int,
        help="Number of retrievals per image",
    )
    parser.add_argument(
        "--clip-filter",
        action="store_true",
        help="Filter using CLIP before transductive retrieval",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to prompt file, format classname to list of prompts",
    )
    parser.add_argument(
        "--dataset-type", type=str, default="ImageFolder", help="Type of dataset"
    )
    parser.add_argument(
        "--clip-score-filter",
        type=float,
        default=None,
        help="Filter using CLIP score, after clip classification filtering",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split to use, only applies to non-ImageFolder datasets",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="Model arch from open_clip to use for filtering"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="Pre-trained weights from open_clip to use for filtering. See open_clip repo for choices"
    )
    args = parser.parse_args()

    return args


def main():
    # Assume data retrieval set is the same structure transfer dataset

    # 1. Get data retrieval set, both metadata CSV + paths, and transfer dataset
    #   1a. Filter transfer dataset by classes which exist in the retrieval dataset
    # 2. Precompute all features, store in mmap arrays
    # 3. Precompute similarity matrix between retrieval set + data retrieval set
    # 4. For each element of the val set, get _k_ relevant images of the retrieval set.
    #    Hard label these retrieved images and "finetune" the zero-shot head.
    #   4a. Various fine-tuning options:
    #       i. NCM with zeroshot head
    #       ii. NCM + wise-ft zeroshot head + double-counting retrievals
    #       iii. Linear probe (maybe mask the irrelevant images)
    #       iv. Get _all_ images _k_ per class, then fine-tune
    #   4b. Test scale wrt _k_

    # Dummy args for testing
    args = get_args()

    args.cache_path.mkdir(exist_ok=True, parents=True)

    # Getting model
    print("=> Acquiring model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, args.pretrained, device="cuda"
    )

    # Getting retrieval dataset/loader
    print("=> Getting retrieval set")
    retrieval_set = SafeImageFolder(
        args.retrieval_path,
        transform=preprocess,
        is_valid_file=lambda x: Path(
            x
        ).is_file(),  # choose all subfiles, these get filtered later
    )
    retrieval_loader = DataLoader(
        retrieval_set, batch_size=256, shuffle=False, num_workers=16
    )
    print(f"---- Found {len(retrieval_set)} examples ----")

    tset_kwargs = dict(transform=preprocess)

    if args.dataset_type != "ImageFolder":
        tset_kwargs.update(**dict(split=args.split, download=True))

    print("=> Getting transductive set")
    transductive_set = getattr(torchvision.datasets, args.dataset_type)(
        args.transductive_path, **tset_kwargs
    )
    transductive_loader = DataLoader(
        transductive_set, batch_size=256, shuffle=False, num_workers=16
    )
    print(f"---- Found {len(transductive_set)} examples ----")

    print("=> Renormalizing retrieval set labels")
    if args.dataset_type != "ImageFolder":
        import templates

        # Fixing the retrieval dataset
        class_list = [
            c.replace("/", "or") for c in getattr(templates, args.dataset_type).classes
        ]
        class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    else:
        class_to_idx = transductive_set.class_to_idx
        class_list = transductive_set.classes

    imgs = []
    for path, label in retrieval_set.imgs:
        if retrieval_set.classes[label] not in class_to_idx:
            continue

        imgs.append((path, class_to_idx[retrieval_set.classes[label]]))

    retrieval_set.imgs = retrieval_set.samples = imgs
    retrieval_set.classes = class_list
    retrieval_set.class_to_idx = class_to_idx

    # Correcting datasets
    print("=> Filtering bad images from retrieval set")
    retrieval_set = filter_bad_images(
        retrieval_set,
        cache=args.cache_path / f"bad_{args.retrieval_path.stem}_images.npy",
    )
    print(f"---- Now {len(retrieval_set)} examples ----")

    # Feature extraction
    print("=> Doing feature extraction")
    transductive_features = extract_features(
        transductive_loader,
        model=model,
        memmap_file=args.cache_path
        / f"cache_{args.transductive_path.stem}_features.npy",
    )

    retrieval_features = extract_features(
        retrieval_loader,
        model=model,
        memmap_file=args.cache_path / f"cache_{args.retrieval_path.stem}_features.npy",
    )

    # Applying k-shot filtering and clip filtering
    logit_max_probs = None
    if args.clip_filter:
        print("=> Performing clip filtering")
        retrieval_features, retrieval_set, logit_max_probs = clip_filter(
            model,
            retrieval_features=retrieval_features,
            retrieval_set=retrieval_set,
            class_prompt_dict=json.load(args.prompt_file.open("r")),
            clip_score_filter=args.clip_score_filter,
        )
        print(f"=> Done clip filtering, {len(retrieval_set)} examples left")

    if args.k_shot is not None:
        print(f"=> Doing {args.k_shot}-shot filtering")
        if args.dataset_type != "ImageFolder":
            transductive_set, shot_indices = k_shot_generic(
                transductive_set, k=args.k_shot
            )

        else:
            transductive_set, shot_indices = k_shot_imagefolder(
                transductive_set, k=args.k_shot
            )

        transductive_features = transductive_features[shot_indices.astype(bool), :]

    # Computing sim matrix
    print("=> Computing batched inner products")
    sim_matrix = batched_inner_products(
        transductive_features,
        retrieval_features,
        batch_size=256,
        out=args.cache_path
        / f"cache_{args.retrieval_path.stem}_sim_shots={args.k_shot}_filter={args.clip_filter}_score={args.clip_score_filter}.npy",
    )

    print("=> Getting closest retrievals")
    paths, paths_by_image = get_closest_retrievals(
        sim_matrix=sim_matrix,
        dataset=retrieval_set,
        transductive_set=transductive_set,
        k=args.retrievals_per_image,
        allow_image_labels=args.k_shot is not None,
        logit_max_probs=logit_max_probs,
    )

    with open(args.cache_path / f"{args.retrieval_path.stem}_paths_by_image.pkl", "wb") as f:
        pickle.dump(paths_by_image, f)

    print(f"=> Copying images to output directory {args.out_dir}")

    for cname in class_list:
        (args.out_dir / cname).mkdir(exist_ok=True, parents=True)

    for path, label in tqdm.tqdm(paths):
        path = Path(path)

        class_dir = args.out_dir / class_list[label]
        out_path = class_dir / path.name
        shutil.copy(path, out_path)

    

    globals().update(locals())


def add_empty_folders(folder):
    folder = Path(folder)

    for subfolder in folder.iterdir():
        subfolder.mkdir(exist_ok=True)

    raise NotImplementedError()


@torch.no_grad()
def clip_filter(
    model, retrieval_features, retrieval_set, class_prompt_dict, clip_score_filter=0.0
):
    zs_head = compute_zero_shot_head(
        model, class_prompt_dict, classnames=retrieval_set.classes, device=0
    )

    imgs = []
    acc = 0
    total = 0
    indices = []
    logit_max_probs = []
    for image_features, batch_imgs in tqdm.tqdm(
        zip(
            batchify(retrieval_features, batch_size=256),
            batchify(retrieval_set.imgs, batch_size=256),
        ),
        total=len(retrieval_set) // 256 + 1,
        desc="CLIP filtering",
    ):
        img_paths, labels = zip(*batch_imgs)
        image_feature = torch.tensor(image_features).to(0)
        logits = image_feature @ zs_head.T

        for logit, img_path, label in zip(logits, img_paths, labels):
            score = (100 * logit).squeeze().softmax(dim=-1)[label].item()
            if label == logit.squeeze().argmax().item() and score >= clip_score_filter:
                imgs.append((img_path, label))
                acc += 1

                logit_max_probs.append(score)

                indices.append(total)

            total += 1

    retrieval_set.imgs = retrieval_set.samples = imgs

    print(f"Clip filter accuracy on retrieval set: {100*acc / total:0.2f})")

    return retrieval_features[np.array(indices)], retrieval_set, logit_max_probs


def get_closest_retrievals(
    sim_matrix,
    dataset: SafeImageFolder,
    transductive_set: SafeImageFolder,
    k,
    allow_image_labels=False,
    logit_max_probs=None,
):
    if allow_image_labels:
        labels_to_indices = defaultdict(lambda: np.zeros(len(dataset), dtype=np.uint8))

        for i, (_, label) in enumerate(dataset.imgs):
            labels_to_indices[label][i] = 1
    else:
        labels_to_indices = defaultdict(lambda: np.ones(len(dataset), dtype=np.uint8))

    # sim matrix is transductive example size x retrieval set size
    outs = []
    outs_by_image_path = {}

    print(sim_matrix.shape, len(transductive_set))
    for i in tqdm.tqdm(range(sim_matrix.shape[0])):
        label = transductive_set[i][1]

        # since sims are between -1 and 1, we add 10 to make sure that the retrieval indices
        # are only from the relevant class
        retrieval_indices = np.argpartition(
            (sim_matrix[i] + 10) * labels_to_indices[label], -k
        )[-k:]

        paths = [dataset.imgs[j] for j in retrieval_indices]
        if logit_max_probs is not None:
            logit_prob = [logit_max_probs[j] for j in retrieval_indices]
        else:
            logit_prob = [-1.0] * len(paths)

        sim_scores = sim_matrix[i][retrieval_indices]

        if hasattr(transductive_set, "imgs"):
            outs_by_image_path[transductive_set.imgs[i]] = list(
                zip(*zip(*paths), logit_prob, sim_scores)
            )

        outs.extend(paths)

    return set(list(outs)), outs_by_image_path


def k_shot_imagefolder(dataset: SafeImageFolder, k=10):
    """Returns an ImageFolder dataset that only contains k images per class."""
    old_imgs = dataset.imgs

    r.seed(0)
    classes_to_images = defaultdict(list)
    for i, (image, label) in enumerate(dataset.imgs):
        classes_to_images[label].append((image, label, i))

    new_dataset = []
    indices = np.zeros(len(old_imgs), dtype=np.uint8)
    for label, images in classes_to_images.items():
        for image, _, i in r.sample(images, k):
            new_dataset.append((image, label))
            indices[i] = 1

    dataset.imgs = dataset.samples = new_dataset

    return dataset, old_imgs, indices


def k_shot_generic(dataset, k=10):
    # wayyyyy slower, fix later with dataloader
    r.seed(0)
    classes_to_images = defaultdict(list)

    for i, (image, label) in enumerate(dataset):
        classes_to_images[label].append(i)

    indices = np.zeros(len(dataset), dtype=np.uint8)
    for label, images in classes_to_images.items():
        for i in r.sample(images, k):
            indices[i] = 1

    return SubsetDataset(dataset, np.where(indices == 1)), np.array(indices)


def maybe_cache_output(fn, cache_path, **kwargs):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    else:
        os.makedirs(cache_path.parent, exist_ok=True)
        output = fn(**kwargs)
        torch.save(output, cache_path)

        return output


def get_random_grid(image_list, transform, k=10):
    dataset = ImageListDataset(image_list, transform=transform)

    imgs = []
    for i in range(len(dataset)):
        imgs.append(dataset[i][0])

    imgs = torch.stack(imgs)
    grid = torchvision.utils.make_grid(imgs, nrow=int(math.sqrt(k)), normalize=True)

    return grid


def get_grids_from_loader(loader, transductive_set, synsets):
    loader_iter = iter(loader)
    count = 0
    while True:
        batch, labels = next(loader_iter)
        texts = [synsets[transductive_set.classes[i]] for i in labels]

        k = int(math.sqrt(len(batch)))
        grid = torchvision.utils.make_grid(batch, nrow=k, normalize=True)
        pil_grid = add_text_to_grid(grid, texts, grid_width=k)
        pil_grid.save(f"test_grids/{count}.jpg")
        count += 1


@torch.no_grad()
def compute_zero_shot_head(model, class_prompts_dict, classnames, device=0):
    zero_shot_head = []

    for i, classname in tqdm.tqdm(
        enumerate(classnames), total=len(classnames), desc="Computing zeroshot head"
    ):
        prompts = class_prompts_dict[classname]
        tokens = clip.tokenize(prompts).to(device)
        text_embeddings = model.encode_text(tokens).float()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        zero_shot_head.append(text_embeddings.mean(axis=0))

    zero_shot_head = torch.stack(zero_shot_head, dim=0)
    zero_shot_head = zero_shot_head / zero_shot_head.norm(dim=-1, keepdim=True)

    return zero_shot_head


def fine_tune_zero_shot_head(
    model,
    dataloader,
    val_loader,
    zero_shot_head,
    epochs,
    learning_rate,
    device="cuda:0",
    alpha=0.0,
):
    model.train()

    head = nn.Linear(
        in_features=zero_shot_head.shape[1], out_features=zero_shot_head.shape[0]
    ).to(device)

    zero_shot_head_copy = zero_shot_head.clone().cpu()
    head.weight.data = zero_shot_head.to(device).float()
    head.bias.data = torch.zeros_like(head.bias.data).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        head.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )

    # Add cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * epochs, eta_min=0
    )

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_features = F.normalize(
                    model.encode_image(images).float(), dim=1, p=2
                )

            output = head(image_features)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # Update learning rate using the scheduler
            scheduler.step()

            print(
                f"""Epoch: {epoch+1}/{epochs}, 
            Batch: {i+1}/{len(dataloader)}, 
            Loss: {loss.item():0.4f}, 
            LR: {scheduler.get_last_lr()[0]:.7f}"""
            )

        # todo: ensemble head
        with torch.no_grad():
            old_head_weight = head.weight.data.clone()
            old_head_bias = head.bias.data.clone()

            head.weight.data = (
                alpha * zero_shot_head_copy.to(device).float()
                + (1 - alpha) * head.weight.data
            )
            head.bias.data = head.bias.data * (1 - alpha)

            accuracy = compute_accuracy(model, head, val_loader, device)
            head.weight.data = old_head_weight
            head.bias.data = old_head_bias

        print(f"Epoch: {0}/{epochs}, Accuracy: {accuracy:.2f}%")

    return model


@torch.no_grad()
def compute_accuracy(model, head, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(model.encode_image(images).float(), dim=1, p=2)
            output = head(image_features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total


def image_grids_to_folders(
    paths_to_images, preprocess, transductive_set, retrieval_set, imagenet_to_classname
):
    save_path = Path("test_grids")
    save_path.mkdir(exist_ok=True)
    count = 0

    path_iter = iter(paths_to_images.items())

    while True:
        count += 1
        (img, img_label), batch = next(path_iter)

        batch = [(img, img_label, 0.0, 0.0)] + batch

        grid = get_random_grid([(a, b) for a, b, _, _ in batch], preprocess)
        classnames = [
            f"{imagenet_to_classname[transductive_set.classes[label]]} ({logit_prob:0.4f}) ({sim_score:0.4f}))"
            for _, label, logit_prob, sim_score in batch
        ]
        classnames[0] = f"GT: {imagenet_to_classname[retrieval_set.classes[img_label]]}"
        grid = add_text_to_grid(grid, classnames)

        (save_path / transductive_set.classes[img_label]).mkdir(exist_ok=True)

        grid.save(save_path / transductive_set.classes[img_label] / f"{count}.jpg")

        if count > 2000:
            break


def add_text_to_grid(grid, text_list, grid_width=3):
    # convert the tensor grid to a PIL image
    pil_grid = torchvision.transforms.ToPILImage()(grid)

    # loop through the list of texts and draw each text on the corresponding image
    for i, text in enumerate(text_list):
        x = i % grid_width  # calculate x coordinate based on index i
        y = i // grid_width  # calculate y coordinate based on index i
        draw = ImageDraw.Draw(pil_grid)

        draw.text(
            (x * 224 + 5 * x, y * 224 + 5 * y), text
        )  # adjust the position of text to your liking

    return pil_grid


def relabel_mismatch_classes(
    trans_set: SafeImageFolder, retrieval_set: SafeImageFolder
):
    # Assuming trans_set is the groundtruth, retrieval_set labels are recomputed wrt trans_set
    imgs = []
    for path, label in trans_set.imgs:
        class_name = trans_set.classes[label]
        if class_name in retrieval_set.class_to_idx:
            imgs.append((path, retrieval_set.class_to_idx[class_name]))

    trans_set.classes = retrieval_set.classes
    trans_set.class_to_idx = retrieval_set.class_to_idx
    trans_set.imgs = trans_set.samples = imgs

    return trans_set


def filter_bad_images(dataset: SafeImageFolder, cache=None):
    cache = Path(cache)

    if cache.exists():
        preload_cache = np.load(cache)
    else:
        preload_cache = np.ones(len(dataset))

        loader = DataLoader(
            BadImageCheckerDataset(dataset),
            batch_size=256,
            shuffle=False,
            num_workers=16,
        )
        pointer = 0

        for check in tqdm.tqdm(loader, desc="Filtering bad images"):
            preload_cache[pointer : pointer + len(check)] = check.float().numpy()
            pointer += len(check)

    dataset.imgs = [dataset.imgs[i] for i in np.where(preload_cache == 1)[0]]
    dataset.samples = dataset.imgs

    if not cache.exists():
        np.save(cache, preload_cache)

    return dataset


def batched_inner_products(m1, m2, out, batch_size=256):
    if os.path.exists(out):
        return np.load(out, mmap_mode="r")

    feature_vectors = open_memmap(
        out,
        dtype="float32",
        mode="w+",
        shape=(m1.shape[0], m2.shape[0]),
    )

    with torch.no_grad():
        count_m1 = 0

        for batch_m1 in tqdm.tqdm(
            batchify(m1, batch_size=batch_size), total=m1.shape[0] // batch_size + 1
        ):
            count_m2 = 0

            for batch_m2 in batchify(m2, batch_size=batch_size):
                inner_product = torch.einsum(
                    "ij,kj->ik",
                    torch.tensor(batch_m1).cuda(),
                    torch.tensor(batch_m2).cuda(),
                )

                feature_vectors[
                    count_m1 : count_m1 + batch_m1.shape[0],
                    count_m2 : count_m2 + batch_m2.shape[0],
                ] = inner_product.cpu().numpy()
                count_m2 += batch_m2.shape[0]

            count_m1 += batch_m1.shape[0]

    return feature_vectors


def extract_features(loader, model, memmap_file):
    if os.path.exists(memmap_file):
        return np.load(memmap_file, mmap_mode="r")

    # Create a numpy memmap to store the feature vectors
    feature_size = model.visual.output_dim
    feature_vectors = open_memmap(
        memmap_file,
        dtype="float32",
        mode="w+",
        shape=(len(loader.dataset), feature_size),
    )

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the images and extract the feature vectors
    count = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(loader), total=len(loader), ascii=True, desc="feature extraction"
        ):
            # Preprocess the image
            images = batch[0]
            images = images.to(0)

            # Pass the image through the model to get the feature vector
            feature_vector = (
                F.normalize(model.encode_image(images), p=2, dim=1).cpu().numpy()
            )

            # Store the feature vector in the memmap
            feature_vectors[count : count + len(images)] = feature_vector
            count += len(images)

    return feature_vectors


def batchify(iterable, batch_size=256):
    num_batches = math.ceil(len(iterable) / batch_size)

    for i in range(num_batches):
        yield iterable[i * batch_size : i * batch_size + batch_size]


class ImageListDataset(Dataset):
    def __init__(self, imgs, transform=None) -> None:
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = default_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class BadImageCheckerDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        try:
            entry = self.dataset[index]
            return True
        except:
            return False

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    main()
