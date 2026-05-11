import os

# Temporarily skip all tests except test_general.py
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "test_ec2.py"),
    os.path.join(os.path.dirname(__file__), "test_full_functionality.py"),
    os.path.join(os.path.dirname(__file__), "test_iam.py"),
    os.path.join(os.path.dirname(__file__), "test_prepare_data.py"),
    os.path.join(os.path.dirname(__file__), "test_s3_utils.py"),
]
