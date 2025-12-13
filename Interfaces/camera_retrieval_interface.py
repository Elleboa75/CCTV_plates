from typing import Protocol, Any, Optional, runtime_checkable

@runtime_checkable
class CameraRetrievalProtocol(Protocol):
    """
    Camera Retrieval Protocol for video ingestion
    Can be overridden by users, for this project I'll be using hikvison nvr.
    """
    def connect(self) -> None:
        """
        Connect to the stream/camera/video ingestion mechanism.
        :return:
        """
        ...
    def disconnect(self) -> None:
        """
        Disconnect and free resources.
        :return:
        """
        ...

    def get_frame(self) -> Any:
        """
        Retrieve a single frame from the stream.
        :return: numpy.ndarray (the image) or None if failed to retrieve.
        """
        ...