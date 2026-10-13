import io
import zipfile
from pathlib import Path

import pytest

from resume_screener.ingest import pdf_text, read_resumes, unpack_pdfs

SAMPLE_ZIP = Path(__file__).resolve().parent.parent / "samples" / "sample_resumes.zip"
SAMPLE_PDF = zipfile.ZipFile(SAMPLE_ZIP).read("Arjun_Mehta.pdf")


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_keeps_only_real_pdfs():
    data = make_zip(
        {
            "batch/a.pdf": b"a",
            "batch/B.PDF": b"b",
            "notes.txt": b"x",
            "__MACOSX/batch/._a.pdf": b"x",
            "batch/.hidden.pdf": b"x",
        }
    )
    assert sorted(name for name, _ in unpack_pdfs(data)) == ["B.PDF", "a.pdf"]


def test_archive_paths_are_reduced_to_file_names():
    data = make_zip({"../../escape.pdf": b"x", "deep/nested/dir/cv.pdf": b"y", "win\\style\\path.pdf": b"z"})
    assert sorted(name for name, _ in unpack_pdfs(data)) == ["cv.pdf", "escape.pdf", "path.pdf"]


def test_rejects_too_many_pdfs():
    with pytest.raises(ValueError, match="at most 3"):
        unpack_pdfs(make_zip({f"r{i}.pdf": b"x" for i in range(5)}), max_files=3)


def test_rejects_oversized_archive():
    with pytest.raises(ValueError, match="more than"):
        unpack_pdfs(make_zip({"big.pdf": b"0" * 5000}), max_bytes=1000)


def test_rejects_non_zip():
    with pytest.raises(ValueError, match="not a valid ZIP"):
        unpack_pdfs(b"not a zip")


def test_pdf_text_extracts_and_survives_garbage():
    assert "Kafka" in pdf_text(SAMPLE_PDF)
    assert pdf_text(b"%PDF-1.4 broken") == ""


def test_read_resumes_skips_unreadable_and_disambiguates_names():
    resumes, skipped = read_resumes([("cv.pdf", SAMPLE_PDF), ("cv.pdf", SAMPLE_PDF), ("scan.pdf", b"garbage")])
    assert [r.name for r in resumes] == ["cv.pdf", "cv (2).pdf"]
    assert len({r.id for r in resumes}) == 2
    assert skipped == ["scan.pdf"]


def test_pdf_text_removes_control_characters(monkeypatch):
    class FakePage:
        def extract_text(self):
            return "Experience \x7f SDE at CloudNova\x00"

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]

    monkeypatch.setattr("resume_screener.ingest.PdfReader", FakeReader)
    assert pdf_text(b"x") == "Experience SDE at CloudNova"
