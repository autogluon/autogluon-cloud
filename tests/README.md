# Welcome to Contributing to AutoGluon-Cloud!

Most tests in this repo launch real SageMaker jobs and need credentials for the project's AWS
account, so CI cannot run them on pull requests from forks. On a fork PR only the lint check runs;
the test and doc jobs are skipped.

A maintainer needs to push your branch to this repository to get a full CI run against it. Ping the
maintainers on your PR once it is ready for review.

Pure unit tests (config, serializers, IAM, etc.) do not need AWS access and can be run locally:

```
pytest tests/unittests/general/
```
