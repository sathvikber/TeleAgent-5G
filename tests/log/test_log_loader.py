import pytest

from teleagent.logs.log_loader import LogLoader


def test_load_text():

    loader = LogLoader()

    text = "AMF ERROR Registration Failed"

    assert loader.load_text(text) == text


def test_load_file(tmp_path):

    log_file = tmp_path / "sample.log"

    log_file.write_text(
        "2026-08-01 10:15:22 AMF ERROR Registration Failed"
    )

    loader = LogLoader()

    content = loader.load_file(str(log_file))

    assert "Registration Failed" in content


def test_load_missing_file():

    loader = LogLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_file("missing.log")
