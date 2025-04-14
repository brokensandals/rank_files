from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


class Document(ABC):
    @abstractmethod
    def read_text(self) -> str:
        ...
    
    @abstractmethod
    def read_bytes(self) -> bytes:
        ...

    @abstractmethod
    def cheap_sort_key(self) -> str:
        ...

    def __eq__(self, other: Self) -> bool:
        return self.read_bytes() == other.read_bytes()


class FileDocument(Document):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
    
    def read_text(self) -> str:
        return self.path.read_text("utf8")
    
    def read_bytes(self) -> bytes:
        return self.path.read_bytes()
    
    def cheap_sort_key(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return self.path.name


# Useful for unit tests
class StrDocument(Document):
    def __init__(self, text: str):
        super().__init__()
        self.text = text
    
    def read_text(self) -> str:
        return self.text
    
    def read_bytes(self) -> bytes:
        return self.text.encode("utf8")
    
    def cheap_sort_key(self) -> str:
        return self.read_text()
    
    def __str__(self) -> str:
        return self.text
