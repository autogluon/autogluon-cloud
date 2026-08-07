# Contributing

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/autogluon/autogluon-cloud/issues) or [recently closed](https://github.com/autogluon/autogluon-cloud/issues?q=is%3Aissue+is%3Aclosed) issues to make sure somebody else hasn't already reported the issue.

## Contributing via Pull Requests

1. Work against the latest `master` branch.
2. Check existing issues and PRs to avoid duplicate work.
3. Open an issue to discuss any significant work before starting.

## Setup

1. Install the package in development mode:
```
pip install -e ".[tests]"
```

2. Install pre-commit hooks:
```
pip install pre-commit
pre-commit install
```

This will run ruff formatting and linting automatically on each commit.

## Running Tests

```
pytest tests/
```

To run a specific test:
```
pytest tests/path_to_file.py::test_name
```

Note: most tests require AWS credentials and will launch SageMaker jobs. CI cannot run them on pull requests from forks — only the lint check runs there, and the test and doc jobs are skipped. The full jobs only run when the PR's branch lives in this repository, not a fork. To get a full CI run, ping the maintainers on your PR once it is ready for review — they will re-create your branch in this repository and open a PR from it. Pushing to your fork's branch (even as a maintainer) will not trigger the AWS jobs. See [tests/README.md](tests/README.md) for details.

## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq).

## Licensing

See the [LICENSE](https://github.com/autogluon/autogluon-cloud/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
