from .constants import POLICY_ACCOUNT_PLACE_HOLDER, POLICY_BUCKET_PLACE_HOLDER, TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER


def replace_trust_relationship_place_holder(trust_relationship_document, account_id):
    """Replace placeholder inside template with given values"""
    statements = trust_relationship_document.get("Statement", [])
    for statement in statements:
        for principal in statement["Principal"].keys():
            statement["Principal"][principal] = statement["Principal"][principal].replace(
                TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER, account_id
            )
    return trust_relationship_document


def replace_iam_policy_place_holder(policy_document, account_id=None, bucket=None):
    """Replace placeholder inside template with given values"""
    statements = policy_document.get("Statement", [])
    for statement in statements:
        resources = statement.get("Resource", None)
        if resources is not None:
            if account_id is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_ACCOUNT_PLACE_HOLDER, account_id) for resource in statement["Resource"]
                ]
            if bucket is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_BUCKET_PLACE_HOLDER, bucket) for resource in statement["Resource"]
                ]
    return policy_document
