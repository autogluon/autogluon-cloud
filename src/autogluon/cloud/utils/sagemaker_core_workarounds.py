"""Temporary workarounds for sagemaker-core bugs. Delete each one once fixed upstream."""

import boto3
from sagemaker.core.utils import utils as core_utils
from sagemaker.core.utils.code_injection.shape_dag import SHAPE_DAG


def _register_acronym_field_names() -> None:
    # sagemaker-core serializes nested shapes with a naive snake_case -> PascalCase conversion, so e.g.
    # `memory_size_in_mb` is sent as `MemorySizeInMb` instead of `MemorySizeInMB`. Register the real API names.
    for shape in SHAPE_DAG.values():
        for member in shape.get("members") or []:
            snake = core_utils.pascal_to_snake(member["name"])
            if core_utils.snake_to_pascal(snake) != member["name"]:
                core_utils.SPECIAL_SNAKE_TO_PASCAL_MAPPINGS.setdefault(snake, member["name"])


_register_acronym_field_names()


def bind_core_session(boto_session: boto3.Session) -> None:
    """Make sagemaker-core's process-wide client cache use ``boto_session``.

    The cache ignores the ``session`` argument of resource methods once it exists, so rebuild it whenever a different
    session is requested. Not thread-safe across sessions.
    """
    current = core_utils.SingletonMeta._instances.get(core_utils.SageMakerClient)
    if current is not None and current.session is boto_session:
        return
    core_utils.SingletonMeta._instances.pop(core_utils.SageMakerClient, None)
    core_utils.SageMakerClient(session=boto_session, region_name=boto_session.region_name)
