AutoGluon-Cloud: Train and Deploy AutoGluon on the Cloud
================================================================

.. |ReleaseVersion| image:: https://img.shields.io/badge/doc%20release-v0.1.0-blue
    :target: https://auto.gluon.ai/cloud/dev/versions.html
    :scale: 100%
.. |StableVersion| image:: https://img.shields.io/github/v/release/autogluon/autogluon-cloud?color=blue&label=stable%20release&sort=semver
    :target: https://auto.gluon.ai/cloud/stable/index.html
    :scale: 100%
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
    :target: https://pypi.org/project/autogluon.cloud/
.. |License| image:: https://img.shields.io/github/license/autogluon/autogluon-cloud?color=blue
    :target: https://github.com/autogluon/autogluon-cloud/blob/master/LICENSE

|ReleaseVersion| |StableVersion| |PythonVersion| |License|

AutoGluon-Cloud aims to provide user tools to train, fine-tune and deploy [AutoGluon](https://auto.gluon.ai/stable/index.html) backed models on the cloud. With just a few lines of codes, users could train a model and perform inference on the cloud without worrying about MLOps details such as resource management.

Currently, AutoGluon-Cloud supports [AWS SageMaker](https://aws.amazon.com/sagemaker/) as the cloud backend.

.. note::

    Example using AutoGluon-Cloud to train and deploy an AutoGluon backed model on AWS SageMaker:

    >>> from autogluon.cloud import TabularCloudPredictor
    >>> import pandas as pd
    >>> train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
    >>> test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
    >>> predictor_init_args = {"label": "class"}  # init args you would pass to AG TabularPredictor
    >>> predictor_fit_args = {"train_data": train_data, "time_limit": 120}  # fit args you would pass to AG TabularPredictor
    >>> cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH')
    >>> cloud_predictor.fit(predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args)
    >>> cloud_predictor.deploy()
    >>> result = cloud_predictor.predict_real_time(test_data)
    >>> cloud_predictor.cleanup_deployment()
    >>> # Batch inference
    >>> result = cloud_predictor.predict(test_data)

Installation
------------

.. note::

    >>> pip install -U pip
    >>> pip install -U setuptools wheel
    >>> pip install --pre autogluon.cloud  # You don't need to install autogluon itself locally

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/autogluon-cloud