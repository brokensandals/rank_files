from rank_files.document import FileDocument


def test_file_document(tmp_path):
    p1 = tmp_path / "file1.txt"
    p1.write_text("Hello", "utf-8")
    p2 = tmp_path / "file2.txt"
    p2.write_text("Goodbye", "utf-8")
    p3 = tmp_path / "file3.txt"
    p3.write_text("Hello", "utf-8")
    doc1, doc2, doc3 = [FileDocument(p) for p in [p1, p2, p3]]
    assert doc1.path == p1
    assert doc1.hash() == "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
    assert doc2.path == p2
    assert doc2.hash() == "c015ad6ddaf8bb50689d2d7cbf1539dff6dd84473582a08ed1d15d841f4254f4"
    assert doc3.path == p3
    assert doc3.hash() == "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
