"""Public evaluation interfaces.

Controller code may import from this package.  It must not import from
``sealed_eval``.  The controller-facing surface consists only of the typed
Layer A view and inbox.
"""

from evaluation.firewall import ControllerEvaluationInbox, LayerBoundaryError
from evaluation.records import (
    CONTROLLER_SEARCH_FIELDS,
    ControllerSearchView,
    SearchEvaluationRecord,
)

__all__ = [
    "CONTROLLER_SEARCH_FIELDS",
    "ControllerEvaluationInbox",
    "ControllerSearchView",
    "LayerBoundaryError",
    "SearchEvaluationRecord",
]

