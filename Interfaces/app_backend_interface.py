from typing import Any, Optional, Protocol, runtime_checkable
from Data.get_live_nvr_data import HikvisionCameraRetrieval
from Utils.threading_buffer import ThreadingBuffer
from Interfaces.camera_retrieval_interface import CameraRetrievalProtocol


@runtime_checkable
class App(Protocol):
    def start_system(camera: CameraRetrievalProtocol) -> None:
        """
        Driver Function which initialises all the utils and modules
        :return:
        """
        ...
    def stop_system(camera: CameraRetrievalProtocol) -> None:
        """
        Stops the app and frees resources
        """
        ...
    #TODO: function to get the output from the license plate recognizer

    #TODO: function to pass it to the recommender