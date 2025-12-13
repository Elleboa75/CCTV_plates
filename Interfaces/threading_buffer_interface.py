from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ThreadingBufferInterface(Protocol):
    """
    Abstract class for threading buffer interface.
    """
    def start(self) -> 'ThreadingBufferInterface':
        """
        Start the background thread.
        :return:
        """
        ...
    def stop(self) -> None:
        """
        Stop the background thread.
        :return:
        """
        ...
    def get(self) -> Optional[Any]:
        """
        Returns the latest data from the background thread.
        :return:
        """
        ...