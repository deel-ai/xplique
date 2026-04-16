"""Mixin providing output_as_list and output_as_tensor as inverse boolean properties."""


class OutputAsListMixin:
    """
    Mixin that exposes the output format as two inverse boolean properties.

    ``output_as_list`` and ``output_as_tensor`` are backed by a single
    ``_output_as_list`` field; setting either one automatically reflects in
    the other.
    """

    @staticmethod
    def _validate_bool(name: str, value: object) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {type(value).__name__!r}")

    @property
    def output_as_list(self) -> bool:
        """If True, outputs are returned as a list; if False, as a stacked tensor."""
        return self._output_as_list

    @output_as_list.setter
    def output_as_list(self, value: bool) -> None:
        self._validate_bool("output_as_list", value)
        self._output_as_list = value

    @property
    def output_as_tensor(self) -> bool:
        """If True, outputs are returned as a stacked tensor; if False, as a list."""
        return not self._output_as_list

    @output_as_tensor.setter
    def output_as_tensor(self, value: bool) -> None:
        self._validate_bool("output_as_tensor", value)
        self._output_as_list = not value
