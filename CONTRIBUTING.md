# Contributing

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

Note: most tests require AWS credentials and will launch SageMaker jobs. The CI will not run automatically for external contributors — ping a maintainer to tag your PR with `safe to test`.

## Pull Requests

1. Work against the latest `master` branch.
2. Check existing issues and PRs to avoid duplicate work.
3. Ensure `ruff check` and `ruff format --check` pass (the pre-commit hook handles this).
4. Open a PR and stay involved in the review.

## Security Issues

If you discover a potential security issue, please notify AWS/Amazon Security via the [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Do **not** create a public GitHub issue.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq).
