from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import sha256
from collections import defaultdict


class Document(ABC):
    def __init__(self) -> None:
        self.cached_hash = None

    @abstractmethod
    def read_text(self) -> str:
        ...
    
    @abstractmethod
    def read_bytes(self) -> bytes:
        ...
    
    def hash(self) -> str:
        if self.cached_hash is None:
            self.cached_hash = sha256(self.read_bytes()).hexdigest()
        return self.cached_hash


class FileDocument(Document):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
    
    def read_text(self) -> str:
        return self.path.read_text("utf8")
    
    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

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
    
    def __str__(self) -> str:
        return self.text


class DocumentSet:
    def __init__(self, paths: set[Path]):
        self.documents = [FileDocument(path) for path in sorted(paths)]

        by_hash = defaultdict(list)
        for doc in self.documents:
            by_hash[doc.hash()].append(doc)
        
        duplicates = [[doc.path for doc in dups] for dups in by_hash.values() if len(dups) > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Found the following sets of duplicate files; duplicates are not supported: {duplicates}")
