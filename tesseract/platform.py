"""The studios of the 4D Mind platform."""

from __future__ import annotations

from tesseract import __version__
from tesseract.codeblock import list_sculptures
from tesseract.life import SMEAR_KINDS
from tesseract.voice_text import CHARACTER_QUESTIONS

STUDIOS = (
    {
        "id": "sculpt",
        "title": "Sculpt",
        "goal": "Ambient 3D code sculptures",
        "blurb": "Hold Linux, Doom, Git, and other trees as solids you can orbit and cut.",
    },
    {
        "id": "scan",
        "title": "Scan",
        "goal": "Geometric code vulnerability predictor",
        "blurb": "Rank files and line ranges by where the next fix is likely to land.",
    },
    {
        "id": "resident",
        "title": "Resident",
        "goal": "A character who sees past and future at once",
        "blurb": "Stand at one instant. The resident answers from there, and from every other.",
    },
    {
        "id": "coder",
        "title": "Coder",
        "goal": "Ollama Cloud coder for the files behind the solid",
        "blurb": "Read the source the dots stand for and propose a fix with a cloud model.",
    },
    {
        "id": "smear",
        "title": "Smear",
        "goal": "Video and audio 4D smearing",
        "blurb": "Streak frames and sound along t until a clip is one block.",
    },
)


def catalog() -> dict:
    return {
        "name": "4D Mind",
        "version": __version__,
        "shell": "tauri-2",
        "studios": [dict(studio) for studio in STUDIOS],
        "sculptures": list_sculptures(),
        "smear_kinds": list(SMEAR_KINDS),
        "questions": list(CHARACTER_QUESTIONS),
    }
