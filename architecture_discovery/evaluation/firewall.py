"""Runtime firewall for controller-visible evaluation records."""

from __future__ import annotations

from evaluation.records import ControllerSearchView, SearchEvaluationRecord


class LayerBoundaryError(RuntimeError):
    """Raised when sealed or untyped information is sent to a controller."""


class ControllerEvaluationInbox:
    """Accept exact Layer A records and retain only their restricted views."""

    def __init__(self) -> None:
        self._views: list[ControllerSearchView] = []

    def publish(self, record: object) -> ControllerSearchView:
        if type(record) is not SearchEvaluationRecord:
            raise LayerBoundaryError(
                "controllers accept only exact SearchEvaluationRecord instances"
            )
        view = record.controller_view()
        self._views.append(view)
        return view

    @property
    def views(self) -> tuple[ControllerSearchView, ...]:
        return tuple(self._views)

