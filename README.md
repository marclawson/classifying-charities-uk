Classifying Charities in the UK
===============================

A project to predict International Classification of Non-Profit Organisations (ICNPO) classifications from charity attributes.

This project was an attempt to predict classifications of charities in terms of ICNPOs generated through several methods by the research team at the National Council for Voluntary Organisations (NCVO).  However, it was designed to be a proof-of-concept as to the capacity of machine learning to help categorise the voluntary sector in the UK.  As such the classifications have not been manually checked beforehand and were originally a mixture of keyword searches on the name of the charity and its charitable objects and application of Naive Bayesian techniques.

The code is ready to use but the data is not available for this project fir GitHub.  Please contact me via GitHub if you would like to know more about the data used.  There is an exploratory notebook with the last executed code blocks ready to view and this is partly to replace the 'predict_model.py' functionality, so this file is empty.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
