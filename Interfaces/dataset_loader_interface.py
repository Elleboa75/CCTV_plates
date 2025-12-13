from typing import Protocol, Any, Optional, runtime_checkable, Tuple, Union, List

@runtime_checkable
class DatasetLoader(Protocol):
    def load_data(self) -> Tuple[Any, Any, Any]:
        """
        Data loader file
        :return:
        """
        ...