"""Render the Streamlit app headlessly (no Ollama calls are made before a screening run)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_empty_state():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Screen candidates" in b.label for b in at.button)


def test_sample_button_fills_job_description():
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if "sample data" in b.label).click().run()
    assert not at.exception
    assert "Software Development Engineer" in at.text_area[0].value
    assert at.session_state.use_sample is True
