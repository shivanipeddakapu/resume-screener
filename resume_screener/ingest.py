"""Unpack an uploaded ZIP and read the text of every resume PDF in it."""

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from pypdf import PdfReader

from .config import MAX_PDFS, MAX_UNZIPPED_BYTES

log = logging.getLogger(__name__)

MIN_WORDS = 25  # fewer words than this => probably a scanned image without a text layer


@dataclass(frozen=True)
class Resume:
    id: str  # unique within one upload
    name: str  # file name shown in the UI
    text: str


def _is_resume_pdf(member: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(member.filename.replace("\\", "/"))
    return (
        not member.is_dir() and path.suffix.lower() == ".pdf" and not path.name.startswith(".") and "__MACOSX" not in path.parts
    )


def unpack_pdfs(zip_bytes: bytes, max_files: int = MAX_PDFS, max_bytes: int = MAX_UNZIPPED_BYTES) -> list[tuple[str, bytes]]:
    """Return (file name, bytes) for each PDF in the archive, read in memory.

    Nothing is written to disk, so archive paths like "../../x.pdf" cannot escape anywhere;
    only the base name is kept. Archives with too many PDFs or too much data are rejected.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError("That file is not a valid ZIP archive.") from e

    with archive:
        members = [m for m in archive.infolist() if _is_resume_pdf(m)]
        if len(members) > max_files:
            raise ValueError(f"Found {len(members)} PDFs in the ZIP; at most {max_files} are allowed.")
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError(f"The PDFs in this ZIP add up to more than {max_bytes // 2**20} MB.")
        return [(PurePosixPath(m.filename.replace("\\", "/")).name, archive.read(m)) for m in members]


def pdf_text(data: bytes) -> str:
    """Extract and tidy the text layer of a PDF; empty string if it cannot be parsed."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:  # pypdf raises many different error types for broken files
        log.warning("Could not parse PDF: %s", e)
        return ""
    text = "\n".join(pages)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)  # control characters some PDFs use as bullets
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_resumes(files: list[tuple[str, bytes]]) -> tuple[list[Resume], list[str]]:
    """Turn PDFs into Resume objects. Returns (resumes, names of files with no usable text)."""
    resumes: list[Resume] = []
    skipped: list[str] = []
    seen: dict[str, int] = {}
    for name, data in files:
        text = pdf_text(data)
        if len(re.findall(r"\w+", text)) < MIN_WORDS:
            skipped.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        display = name if seen[name] == 1 else f"{PurePosixPath(name).stem} ({seen[name]}).pdf"
        resumes.append(Resume(id=f"r{len(resumes):03d}", name=display, text=text))
    return resumes, skipped
