# TAIM-GAN

Text-Assisted Image Manipulation

## Tentative Results
![2_fake](https://user-images.githubusercontent.com/30185369/203002443-c977d0d3-7667-4d33-94f4-16752417d220.png)
![5_generated](https://user-images.githubusercontent.com/30185369/203002454-cbad49df-f59f-4491-9fca-169e295dd33b.png)
![8](https://user-images.githubusercontent.com/30185369/203002461-88440789-853b-4ef3-b6be-e9aa4b417b42.png)

We are still trying to experiment with the losses and the training procedure to get better outputs from our GAN network. Currently it is managing to get the rough outline/shapes and most parts of the image but the colour seems to be washed out.

Update: We are now able to get some decent results, the epoch 50 results look like the following:

![3](https://user-images.githubusercontent.com/30185369/203063533-15e5e9fb-4b5a-4436-8188-3fe883db2c0a.png)
![3_2](https://user-images.githubusercontent.com/30185369/203063658-44043f3e-2e00-49ef-a8da-09559e3fa290.png)
![26_fake](https://user-images.githubusercontent.com/30185369/203063754-20ec572e-6920-401e-a249-76788996911e.png)

## Project Organization
------------

    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make clean` or `make test`
    ├── test_environment.py <- Python version check for env config
    ├── README.md           <- The top-level README for developers using this project.
    ├── data
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                 <- Source code for use in this project.
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to download or generate data
    │   │
    │   ├── features        <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   | 
    |   ├── test_project   <- Example project for testing
    |   |   ├── __init__.py
    │   │   └── example.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    |
    ├── tests               <- Code for integration and unit testing
    │   ├── __init__.py     <- Makes test a Python module
    │   │
    │   ├── integration     <- Scripts for integration testing
    |   |   └── __init__.py
    │   │ 
    │   └── init            <- Codes for unit testing of modules
    │       ├── __init__.py
    │       └── example_test.py
    │
    └── mypy.ini           <- MyPy typer configuration


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
