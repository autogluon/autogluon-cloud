# Welcome to Contributing to AutoGluon-Cloud!

Most tests in this repo launch real SageMaker jobs and need credentials for the project's AWS
account, so CI cannot run them on pull requests from forks. On a fork PR only the lint check runs;
the test and doc jobs are skipped.

The full test and doc jobs only run when the PR's branch lives in this repository, not a fork. To
get a full CI run, ping the maintainers on your PR once it is ready for review — they will re-create
your branch in this repository and open a PR from it. Pushing to your fork's branch (even as a
maintainer) will not trigger the AWS jobs.

Pure unit tests (config, serializers, IAM, etc.) do not need AWS access and can be run locally:

```
pytest tests/unittests/general/
```
