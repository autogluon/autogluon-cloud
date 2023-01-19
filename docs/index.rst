AutoGluon-Cloud: Train and Deploy AutoGluon on the Cloud
================================================================

AutoGluon-Cloud aims to provide user tools to train, fine-tune and deploy [AutoGluon](https://auto.gluon.ai/stable/index.html) backed models on the cloud. With just a few lines of codes, users could train a model and perform inference on the cloud without worrying about MLOps details such as resource management.

Currently, AutoGluon-Cloud supports [AWS SageMaker](https://aws.amazon.com/sagemaker/) as the cloud backend.

.. note::

    Example using AutoGluon-Cloud to train and deploy AutoGluon backed model on AWS SageMaker:

    >>> predictor_init_args = {label='label'}  # init args you would pass to AG TabularPredictor
    >>> predictor_fit_args = {train_data, time_limit=120}  # fit args you would pass to AG TabularPredictor
    >>> cloud_predictor = TabularCloudPredictor(cloud_output_path='YOUR_S3_BUCKET_PATH').fit(predictor_init_args, predictor_fit_args)
    >>> cloud_predictor.deploy()
    >>> result = cloud_predictor.predict_real_time('test.csv')
    >>> cloud_predictor.cleanup_deployment()
    >>> result = cloud_predictor.predict('test.csv')

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