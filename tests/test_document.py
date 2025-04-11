import pytest
from rank_files.document import DocumentSet


def test_duplicates(tmp_path):
    p1 = tmp_path / "file1.txt"
    p1.write_text("Hello", "utf-8")
    p2 = tmp_path / "file2.txt"
    p2.write_text("Hello", "utf-8")
    with pytest.raises(ValueError):
        DocumentSet({p1, p2})


def test_no_duplicates(tmp_path):
    p1 = tmp_path / "file1.txt"
    p1.write_text("Hello", "utf-8")
    p2 = tmp_path / "file2.txt"
    p2.write_text("Goodbye", "utf-8")
    docset = DocumentSet({p1, p2})
    assert len(docset.documents) == 2
    assert docset.documents[0].path == p1
    assert docset.documents[0].hash() == "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
    assert docset.documents[1].path == p2
    assert docset.documents[1].hash() == "c015ad6ddaf8bb50689d2d7cbf1539dff6dd84473582a08ed1d15d841f4254f4"
