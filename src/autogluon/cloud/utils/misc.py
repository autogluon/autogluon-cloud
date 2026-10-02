import secrets
import time
from collections import OrderedDict


def sagemaker_timestamp() -> str:
    """UTC timestamp with millisecond precision, e.g. ``2026-10-02-13-45-07-123``."""
    now = time.time()
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(now)) + f"-{int(now * 1000) % 1000:03d}"


def unique_name_from_base(base: str, max_length: int = 63) -> str:
    """Append a timestamp and a random suffix to ``base``, trimming it so the result fits in ``max_length``."""
    suffix = f"-{int(time.time())}-{secrets.token_hex(2)}"
    return base[: max_length - len(suffix)] + suffix


# https://stackoverflow.com/questions/9917178/last-element-in-ordereddict
class MostRecentInsertedOrderedDict(OrderedDict):
    @property
    def last(self):
        if len(self) > 0:
            return next(reversed(self))
        return None

    @property
    def last_value(self):
        return self.get(self.last, None)
