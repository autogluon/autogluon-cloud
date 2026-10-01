"""Delete SageMaker endpoints leaked by a CI run.

Tests delete their own endpoints, but a failed, timed-out or cancelled job can leave one running (and billing)
forever. tests/conftest.py tags every resource created in CI with the run id, so we sweep by that tag.

Usage: python3 cleanup_endpoints.py <github_run_id>
"""

import sys
import time

import boto3

# Must match tests/conftest.py
CI_RUN_TAG = "autogluon-cloud-ci-run"
# DeleteEndpoint is rejected while an endpoint is in one of these states, so wait for it to settle first.
TRANSITIONAL_STATUSES = {"Creating", "Updating", "SystemUpdating", "RollingBack"}
POLL_SECONDS = 30
TIMEOUT_SECONDS = 15 * 60


def find_run_endpoints(sm, run_id: str) -> list:
    endpoints = []
    for page in sm.get_paginator("list_endpoints").paginate(NameContains="ag-cloud"):
        for endpoint in page["Endpoints"]:
            tags = sm.list_tags(ResourceArn=endpoint["EndpointArn"])["Tags"]
            if {"Key": CI_RUN_TAG, "Value": run_id} in tags:
                endpoints.append(endpoint["EndpointName"])
    return endpoints


def delete_endpoint(sm, endpoint_name: str, endpoint_config_name: str) -> None:
    """Delete the endpoint together with its config and models, mirroring CloudPredictor.cleanup_deployment."""
    config = sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    sm.delete_endpoint(EndpointName=endpoint_name)
    sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    for variant in config["ProductionVariants"]:
        sm.delete_model(ModelName=variant["ModelName"])


def main(run_id: str) -> int:
    sm = boto3.client("sagemaker")
    pending = set(find_run_endpoints(sm, run_id))
    if not pending:
        print(f"No endpoints tagged with {CI_RUN_TAG}={run_id}")
        return 0

    for name in sorted(pending):
        print(f"::warning::Endpoint {name} was not cleaned up by the tests, deleting it")
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while pending:
        for name in sorted(pending):
            endpoint = sm.describe_endpoint(EndpointName=name)
            status = endpoint["EndpointStatus"]
            if status in TRANSITIONAL_STATUSES:
                continue
            pending.discard(name)
            if status != "Deleting":
                delete_endpoint(sm, name, endpoint["EndpointConfigName"])
                print(f"Deleted endpoint {name} (was {status})")
        if pending:
            if time.monotonic() > deadline:
                print(f"::error::Timed out waiting for endpoints to settle, delete manually: {sorted(pending)}")
                return 1
            time.sleep(POLL_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
