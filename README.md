# TAIM-GAN

Text-Assisted Image Manipulation

## Description

In this project we explore the idea of changing the images according to the textual captions through generative networks

## How to use

There's two primary ways in which you can use our project: use our publicly deployed version of TAIM-GAN or deploy it locally on your machine. Also, feel free to fork and modify the source code for latter reseach

### Hugging Face

Visit the [live demo](https://huggingface.co/spaces/ML701G7/taim-gan) of our project on Hugging Face! It has some interesting examples:

![image](https://user-images.githubusercontent.com/54360024/206266447-14af0ab7-cc2c-478a-a5fd-ac0f53496685.png)
![image](https://user-images.githubusercontent.com/54360024/206266487-4e7bc584-eb44-45b6-b820-f07bc9707b22.png)
![image](https://user-images.githubusercontent.com/54360024/206266506-f0c03fa0-5a18-4f54-a61c-8ec7398a4d29.png)

### Docker

To set up the project locally through docker, just do the following:
1. Clone our repository using `git clone https://github.com/Dmmc123/taim-gan.git`
2. Deploy the Gradio web application using `docker-compose up`

## Datasets

We used three datasets for training, evaluation, and deployment of TAIM-GAN:
1. [COCO](https://cocodataset.org/#download)
2. [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/)
3. [UTKFace](https://susanqq.github.io/UTKFace/) with generated captions from [BLIP](https://huggingface.co/spaces/Salesforce/BLIP)

## Train/evaluate

Example of training script:

```
train.py --data-dir data --split train --num-capt 5 --batch-size 16 --num-workers 8
```

Example of evaluation script:

```
compute_metrics.py --data-dir data --split test --num-capt 10 --batch-size 32 --num-workers 4
```

For additonal information about possible values of arguments and their meaning, please type `train.py --help` or `compute_metrics.py --help`

## Project Organization

Project uses template from [cookiecutter](https://drivendata.github.io/cookiecutter-data-science) for data science projects

------------

    ├── notebooks             <- Jupyter Notebooks for interim results of coding tasks
    │
    ├── references            <- Main papers we used as a source
    │
    ├── src                   <- Source files of the project
    |   | 
    │   ├── data              <- Code for preparing the datasets and vectorizing texts
    |   |   |
    │   |   ├── stubs         <- Folder for examples in the Gradio application
    |   |   |
    │   |   ├── collate.py    <- Function for preprocessing output of datasets for inference
    │   |   ├── datasets.py   <- Code for processing the raw data from datasets
    │   |   └── tokenizer.py  <- API for tokenizing text captions
    |   |
    │   ├── models            <- Code for model components and inference utility functions
    |   |   |
    |   |   ├── modules       <- All the atomic modules that constitute TAIM-GAN
    |   |   |
    │   |   ├── losses.py     <- Loss functions for Discriminator and Generator
    │   |   └── utils.py      <- Helper functions for loading/dumping weights of models and saving plots
    |   |
    |   └── config.py         <- Constants used throughout the project
    |
    ├── tests                 <- Code for integration and unit testing
    |   │
    |   ├── integration       <- Tests for checking the whole flow of TAIM-GAN
    |   │ 
    |   └── unit              <- Scripts for testing the mudules in atomic way
    |        
    ├── app.py                <- Code for Gradio web application
    ├── compute_metrics.py    <- Code for computing Inceptions Scores on datasets
    ├── docker-compose.yaml   <- Docker config for deploying the Gradio web app locally
    ├── requirements.txt      <- List of all dependencies of the project
    └── train.py              <- Code for training the model

--------

## References

The project is primarily based on [LWGAN](https://github.com/mrlibw/Lightweight-Manipulation/) research. Here is what we did differently:

* Removed the redundant modules from the source code
* Refactored from scratch the existing code by providing type hints for the most part of code-base
* Covered most parts of code with unit and intergrations tests
* Removed the deprecated functionality and upgraded the project according the latest version of PyTorch 1.12.1
* Collected new dataset with captions for facial pictures and finetuned TAIM-GAN with it

Also here are some other researches we used as an additional reference:

* [ManiGAN](https://github.com/mrlibw/ManiGAN)
* [StackGAN](https://github.com/hanzhanggit/StackGAN)
* [StackGAN++](https://github.com/hanzhanggit/StackGAN-v2)
* [AttnGAN](https://github.com/taoxugit/AttnGAN)
* [ControlGAN](https://github.com/mrlibw/ControlGAN)
