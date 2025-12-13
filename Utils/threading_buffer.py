import cv2
import time
import threading
from typing import Any, Callable, Optional

class ThreadingBuffer():
    def __init__(self, source: Callable[[], Optional[Any]], sleep_time: float) -> None:
        """
        :param source: A function that returns the data from the specified video source.
        :param sleep_time: How long to sleep between frames
        """
        self.source: Callable[[], Optional[Any]] = source
        self.sleep_time: float = sleep_time

        self._latest_data: Optional[Any] = None

        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> 'ThreadingBuffer':
        """
        Start the threading buffer
        :return: Self for chaining
        """
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target = self._worker, daemon = True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """
        Stop the threading buffer
        :return: None
        """
        self._running = False
        if self._thread:
            self._thread.join()

    def _worker(self) -> None:
        """
        Loop running in the background thread
        """
        while self._running:
            try:
                data = self.source()
                if data is not None:
                    with self._lock:
                        self._latest_data = data

                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
            except Exception as e:
                print(f"Buffer Error: {e}")

    def get(self) -> Optional[Any]:
        """
        :return: Latest data or None if the buffer is empty
        """
        with self._lock:
            return self._latest_data