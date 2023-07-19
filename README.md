# Neural Priming

Pytorch implementation of [Neural Priming for Sample-Efficient Adaptation](https://arxiv.org/pdf/2306.10191.pdf).



<img src='assets/teaser.jpg'>

## Getting Started

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
conda create -n neural-priming-env python=3.8
```
6. Install the required dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Downloading The Data

#### Priming Data (LAION-2B)
To get started quickly we provide the priming subsets of LAION-2B for each target task. The link to download the data from Google Drive can be found here. If downloading to a headless server we recommend using [gdown](https://github.com/wkentaro/gdown). Once downloaded, unzip and place in the `/data` folder in the root directory. 

Alternatively, we provide code in the **Text Filtering and Downloading Data** section for creating your own priming subset.  

#### Evaluation Data 

To download ImageNet-1k: Download from this Kaggle [link](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC). The validation and train set should be in ImageFolder format which looks like the following. 
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


Torchvision Datasets: The 6 transfer learning datasets (StanfordCars, FGVC Aircraft, Flowers102, OxfordPets, and SUN397) will automatically be downloaded upon running the training code. 

The other datasets (ImageNetV2, ImageNet-a, r, and sketch) can be found on their respective webpages). For ImageNetV2 we use the *ImageNetV2-matched-frequency* version. 


## Train and Evaluate Models


### Zero-shot Priming
```bash
python prime.py --dataset Flowers102 --shots 0 --alpha .7 --text --prime --subset_path /data/Flowers102 --retrain
```

### Few-shot Priming


### Transductive Priming
To prime and evaluate the model on the distribution shift datasets. 


```bash
python prime.py --dataset ImageNet-V2 --shots 0 --text --prime --cupl  --subset_path /data/ImageNetv2 --val_path /data/ImageNetV2-matched-frequency --custom_data --retrain
```

Command line options: 

- `--prime` Use the priming subset to condition the model.
- `--text` Initialize the classifier with the text prompts from OpenAI for ensembling with image features.
- `--cupl` Initialize the classifier with text prompts from CuPL and OpenAI.
- `--cache` Whether to cache the image features of the train/test/priming subset. Set to true by default. Set to false if low on disk space. 
- `--alpha` Ensembling coefficient between text and image features. Depends on the size of the training/priming set.
- `--shots` Number of examples to be used from the target training set (not to be confused with the priming subset).
- `--model` Change the base model. The priming subsets provided above are from the B-16 model.
- `--subset_path` Path to the priming subset.
- `--val_path` Path to the evaluation dataset. Only needed for ImageNet and the distribution shift datasets. 


For further command line options see `args.py`. 


## Creating Custom Subsets from LAION-2B


### Text filtering and Downloading Images




### Transductive Filtering

