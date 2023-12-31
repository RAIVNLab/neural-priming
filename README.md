# Neural Priming

Pytorch implementation of [Neural Priming for Sample-Efficient Adaptation](https://arxiv.org/pdf/2306.10191.pdf). Neural Priming improves transfer learning accuracy and robustness by recalling relevant data from the pretraining dataset. 



<img src='assets/teaser.jpg'>

## Getting Started

If you'd like a demo of the code see the [collab notebook](https://colab.research.google.com/drive/1FkwnkfHCwBjsxsfy_WdxF6W_m6US28ft?usp=sharing).

### Installation

1. Clone this repository to your local machine using the following command:
  ```bash
  git clone https://github.com/your-username/neural-priming.git
  ```
3. Navigate to the project directory:
```bash
cd neural-priming
```
5. (Optional) - Create a conda environment:
```bash
conda create -n neural-priming python=3.8
conda activate neural-priming
```
6. Install the required dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Downloading The Data

#### Priming Data (LAION-2B)
To get started quickly we provide the priming subsets of LAION-2B for each target task. The link to download the data from Google Drive can be found [here](https://drive.google.com/drive/folders/1yQfr6IYrG8_ViuQW7hOtHHr6yalrQ8d0?usp=sharing). If downloading to a headless server we recommend using [gdown](https://github.com/wkentaro/gdown). Once downloaded, unzip and place in the `/data` folder in the root directory. 

Alternatively, we provide code in the **Text Filtering and Downloading Data** section for creating your own priming subset.  

#### Evaluation Data 

- To download ImageNet-1k: Download from this Kaggle [link](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC). The validation and train set should be in ImageFolder format which looks like the following. 
```
├── ImageNet_Train
    ├── n01440764
        ├── img_1.jpeg
    ...
        ├── img_N.jpeg

    ...

    ├── n01632777
        ├── img_1.jpeg
    ...
        ├── img_N.jpeg
```


- Torchvision Datasets: The 6 transfer learning datasets (StanfordCars, FGVC Aircraft, Flowers102, OxfordPets, and SUN397) will automatically be downloaded upon running the training code. 

- The other datasets (ImageNetV2, ImageNet-a, r, and sketch) can be found on their respective webpages. For ImageNetV2 we use the *ImageNetV2-matched-frequency* version. 


## Train and Evaluate Models


### Zero-shot Priming
Example commands for priming and evaluating the model:
- ```python Priming/prime.py --dataset Flowers102 --shots 0 --alpha .7 --prime --subset_path ./data/Flowers102 --retrain```
- ```python Priming/prime.py --dataset StanfordCars --shots 0 --prime --subset_path ./data/StanfordCars --retrain```
  
Note: At this current time, StanfordCars is not available through torchvision. The dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).
- ```
  python Priming/prime.py --dataset ImageNet --shots 0 --prime --cupl --subset_path /data/ImageNet_subset --train_path ./data/ImageNet/train --val_path /data/ImageNet/val --retrain
  ```
To run the equivalent baselines, omit the `--prime` flag. For example:

```python priming/prime.py --dataset Flowers102 --shots 0  --subset_path ./data/Flowers102 --retrain```

Zero-shot Results:
|                         | ImageNet | Stanford Cars | FGVC Aircraft | Flowers102 | Food101 | Oxford Pets | SUN397 |
|-------------------------|----------|---------------|---------------|------------|---------|-------------|--------|
| CLIP Baseline            | 68.30    | 87.40         | 25.86         | 71.65      | 86.58   | 90.21       | 67.35  |
| CuPL             | 70.25    | 88.63         | 29.64         | 72.32      | 86.20   | 91.16       | 70.80  |
| Priming (Ours)          | 70.75    | 89.30         | 33.03         | 79.81      | 86.66   | 91.87   | 71.21  |
| Priming + CuPL (Ours)   | 71.38 | 90.23     | 36.00     | 80.04  | 86.86 | 91.85       | 72.35 |


### Few-shot Priming
Example commands for reproducing few-shot results. Note that alpha depends on the number of shots used. 

```bash
python priming/prime.py --dataset Flowers102 --shots 2 --alpha .58 --prime --subset_path ./data/Flowers102 --retrain
```

```bash
python priming/prime.py --dataset FGVCAircraft --shots 3 --alpha .55 --prime --subset_path ./data/FGVCAircraft --retrain
```

<div>
    <img src='assets/Cars_FSL.jpeg' alt='FGVC_FSL' style='float: left; width: 48%; margin-right: 1%;'>
    <img src='assets/Flowers_FSL.jpeg' alt='Flowers_FSL' style='float: left; width: 48%;'>
</div>


### Transductive Priming
Example commands for reproducing transductive results on the distribution shift datasets:


```bash
python prime.py --dataset ImageNet-V2 --shots 0 --prime --cupl  --subset_path ./data/ImageNetv2 --val_path ./data/ImageNetV2-matched-frequency --custom_data --retrain
```

|                                      | ImageNet-V2 | ImageNet-R | ImageNet Sketch | ImageNet-A |
|--------------------------------------|-------------|------------|-----------------|------------|
| CLIP                          | 62.61       | 64.57      | 57.05           | 35.95      |
| Transduct. Priming (Ours)            |64.23   | 79.37  | 59.97       | 38.20  |


Command line options: 

- `--prime` Use the priming subset to condition the model.
- `--retrain` Reprocess the image features from the train/val/subset datasets. If already cached, omit to avoid reprocessing. 
- `--text` Initialize the classifier with the text prompts from OpenAI for ensembling with image features.
- `--cupl` Initialize the classifier with text prompts from CuPL and OpenAI.
- `--cache` Whether to cache the image features of the train/test/priming subset. Set to true by default. Set to false if low on disk space. 
- `--alpha` Ensembling coefficient between text and image features. Depends on the size of the training/priming set.
- `--shots` Number of examples to be used from the target training set (not to be confused with the priming subset).
- `--model` Change the base model. The priming subsets provided above are from the B-16 model.
- `--subset_path` Path to the priming subset.
- `--val_path` Path to the evaluation dataset. Only required for ImageNet and the distribution shift datasets. 


For full list of command line options see `args.py`. 


## Creating Subsets from LAION-2B

### Text filtering and Downloading Images
To quickly filter through the LAION-2B dataset using text, we use SQLite in python. The data base parquets can be downloaded [here](https://drive.google.com/drive/folders/1yQfr6IYrG8_ViuQW7hOtHHr6yalrQ8d0?usp=sharing). We recommend [gdown](https://github.com/wkentaro/gdown) if downloading to a headless server. Once downloaded, place them in a `/parquets` folder. Each parquet is around 8 GB and all parquets are about 1 TB. If disk space is limited, you can download fewer parquets and filter on a subset of LAION-2B.  Also note that placing the parquets on SSD will significantly improve search speed. 
You'll need the sqlite package which can be installed with 
```pip install pysqlite3```.
Given the class names for a dataset, the code will filter for LAION-2B entries which have captions containing the class name and write the meta data to a json. Example to command to filter for ImageNet classes:

```bash
python ./DataFiltering/FilterData.py -o ./ImageNet_Filtered -q ImageNet \
 -d  /parquets/part-00{000..123}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.db --template
```
Adjust the line text `{000..123}` if using fewer parquets. To filter using with respect to your own custom dataset, places the class names in a .py file in templates and set it as the `--q` argument in the above command. 

Once the data is stored in the json, you can download the data from URLS using the following command:

```python DataFiltering/download_urls.py --r ./DataFiltering/ImageNet_Filtered/ --w ./data/ImageNet_filtered```

Note that links break over time. A signficantly smaller number of images may be actually downloaded.





### Transductive Filtering
Once the **Text Filtering and Downloading Images** has been completed, transductive filtering can be performed on the text-filtered subset. An example command would be the following:

Example command. Given the priming pool `ImageNet_filtered` and a path to a ground-truth dataset (`ImageNet`), this takes 10 dataset train images per class (`--k-shot=10`) and retrieves the 10 closest images for each of these from the priming pool (`--retrievals-per-image=10`): 

```bash
python ./DataFilering/TransFiltering.py \
        --dataset-type ImageNet \
        --retrieval-path "./data/ImageNet_filtered" \
        --transductive-path="/usr/data" \
        --cache-path="./data/kshot_cache_ImageNet" \
        --out-dir="/ImageNet" \
        --retrievals-per-image=10 \
        --prompt-file=./data/ImageNet.json \
        --k-shot=10 \
        --split="train"
```

This returns a dataset in the `ImageFolder` format, where each retrieved image is labeled by its given class in the priming pool. If you use `--split="test"`, this becomes the transductive setting discussed in the paper. You can add the `--clip-filter` command to apply a CLIP classifier to this new pool to further refine the dataset.

See the `DataFiltering/TransFiltering.py` file for more details. Below is a list of arguments and descriptions:

```bash
usage: TransFiltering.py [-h] --retrieval-path RETRIEVAL_PATH --transductive-path TRANSDUCTIVE_PATH [--k-shot K_SHOT]
                     [--cache-path CACHE_PATH] --out-dir OUT_DIR [--retrievals-per-image RETRIEVALS_PER_IMAGE] [--clip-filter]
                     [--prompt-file PROMPT_FILE] [--dataset-type DATASET_TYPE] [--clip-score-filter CLIP_SCORE_FILTER]
                     [--split SPLIT] [--model MODEL] [--pretrained PRETRAINED]

optional arguments:
  -h, --help            show this help message and exit
  --retrieval-path RETRIEVAL_PATH
                        Path to retrieval reservoir (clean subset of LAION)
  --transductive-path TRANSDUCTIVE_PATH
                        Path to transfer evaluation dataset (to be used in a transductive fashion)
  --k-shot K_SHOT       Number of shots per class, only used in train-time data augmentation
  --cache-path CACHE_PATH
                        Path to cache
  --out-dir OUT_DIR     Path to output directory (dataset in ImageFolder format)
  --retrievals-per-image RETRIEVALS_PER_IMAGE
                        Number of retrievals per image
  --clip-filter         Filter using CLIP before transductive retrieval
  --prompt-file PROMPT_FILE
                        Path to prompt file, format classname to list of prompts
  --dataset-type DATASET_TYPE
                        Type of dataset
  --clip-score-filter CLIP_SCORE_FILTER
                        Filter using CLIP score, after clip classification filtering
  --split SPLIT         Split to use, only applies to non-ImageFolder datasets
  --model MODEL         Model arch from open_clip to use for filtering
  --pretrained PRETRAINED
                        Pre-trained weights from open_clip to use for filtering. See open_clip repo for choices
```
