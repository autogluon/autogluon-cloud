# AutoGluon-Cloud Distributed Training
AutoGluon-Cloud currently supports distributed training for Tabular.

Tabular predictor trains multiple folds of models underneath and parallelize model training on a single machine. It is natural to expand this strategy to a cluster of machines.

With AutoGluon-Cloud, we help you to spin up the cluster, dispatch the jobs, and tear down the cluster. And it is not much different from how you would normally train a `TabularCloudPredictor``. All you need to do is specify `backend="ray_aws"` when you init the predictor

```python
cloud_predictor = TabularCloudPredictor(
    ...,
    backend="ray_aws"
)
```

And you would call fit as normal

```python
cloud_predictor.fit(predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args)
```

## How to Control Number of Instances in the Cluster
The default number of instances being launched will be introduced in the following section. You can control how many instances are created in the cluster by passing `instance_count` to `fit()`.
```python
cloud_predictor.fit(..., instance_count=4)
```

### General Strategy on How to Decide `instance_count`

#### Non-HPO
By default, this value will be determined by number of folds (`num_bag_folds` in `TabularPredictor`). We will launch as many instances as the number of folds, so each fold will be trained on a dedicated machine. The default value should work most of the time

You can of course lower this value to save budgets, but this will slow the training process as a single instance will need to train multiple folds in parallel and split its resources. Setting value larger than number of folds is meaningless, as we do not support distributed training of a single model.

#### HPO
When doing HPO, it's very hard to pre-determine how many instances to use. Therefore, we default to 1 instance, and you would need to specify the number of instances you want to use. The fastest option of course will be matching the number of instances and number of trials. However, this is likely impossible as HPO typically involves big number of trials.

In general, the recommendation would be try to make it so that (#vcpus_per_instance * #instances) is divisible by number of trials. We evenly distribute resources to tasks; therefore, a non-divisible value would results in some resources not being utilized.

To give a recommended example, suppose you want to do HPO with 128 trials. Choosing 8 `m5.2xlarge` (8 vcpus), would make the computing resources divisible by the number of trials: 128 / (8 * 8) = 2. This would results in two batches each containing 64 jobs being distributed on 64 vcpus.
