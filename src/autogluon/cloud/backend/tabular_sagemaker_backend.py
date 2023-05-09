from .constant import TABULAR_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TabularSagemakerBackend(SagemakerBackend):
    name = TABULAR_SAGEMAKER
