import cv2
from Interfaces.camera_retrieval import CameraRetrievalProtocol
from typing import Any, Optional

class HikvisionCameraRetrieval:
    def __init__(self, rtsp_url: str) -> None:
        self.rtsp_url: str = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> None:
        print(f"Connecting to NVR: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise ConnectionError(f"Failed to open stream {self.rtsp_url}")

        ret, _ = self.cap.read()
        if not ret:
            raise ConnectionError(f"Stream open, failed data retrieval {self.rtsp_url}")
        print("Connected to NVR and verified data stream")

    def disconnect(self) -> None:
        if self.cap:
            self.cap.release()
            print(f"Disconnected from stream {self.rtsp_url}")

    def get_frame(self) -> Optional[Any]:
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame