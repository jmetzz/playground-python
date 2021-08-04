import logging
import re

logger = logging.getLogger(__name__)


def underscore_to_camel(name: str) -> str:
    """
    Convert a name from underscore lower case convention to camel case convention.
    Args:
        name (str): name in underscore lowercase convention.
    Returns:
        Name in camel case convention.
    """
    under_pat = re.compile(r"_([a-z])")
    return under_pat.sub(lambda x: x.group(1).upper(), name)
