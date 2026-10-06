VALID_ACCEPT = ["application/x-parquet", "text/csv", "application/json"]

LOCAL_MODE = "local"
LOCAL_MODE_GPU = "local_gpu"
MODEL_ARTIFACT_NAME = "model.tar.gz"

# AutoGluon container version used when `framework_version` is not specified. Bump it on each AutoGluon release.
DEFAULT_FRAMEWORK_VERSION = "1.6"

# Size in GB of the storage volume for training jobs (fit and batch predict). Small enough to fit the fixed local
# storage of instances like ml.g4dn.xlarge (125 GB), which caps the volume size.
DEFAULT_VOLUME_SIZE = 64

TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER = "ACCOUNT"
POLICY_ACCOUNT_PLACE_HOLDER = "ACCOUNT"
POLICY_BUCKET_PLACE_HOLDER = "CLOUD_BUCKET"
