from pathlib import Path
from hashlib import sha256
from collections import defaultdict


class Document:
    def __init__(self, path: Path):
        self.path = path
        self.hash = sha256(path.read_bytes()).hexdigest()
    
    def __str__(self) -> str:
        return self.path.name


class DocumentSet:
    def __init__(self, paths: set[Path]):
        self.documents = [Document(path) for path in paths]

        by_hash = defaultdict(list)
        for doc in self.documents:
            by_hash[doc.hash].append(doc)
        
        duplicates = [[doc.path for doc in dups] for dups in by_hash.values() if len(dups) > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Found the following sets of duplicate files; duplicates are not supported: {duplicates}")
