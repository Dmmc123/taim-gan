# TAIM-GAN

Text-Assisted Image Manipulation

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
    |   └── init              <- Scripts for testing the mudules in atomic way
    |        
    ├── app.py                <- Code for Gradio web application
    ├── compute_metrics.py    <- Code for computing Inceptions Scores on datasets
    ├── docker-compose.yaml   <- Docker config for deploying the Gradio web app locally
    ├── requirements.txt      <- List of all dependencies of the project
    └── train.py              <- Code for trainig the mdoel
--------
