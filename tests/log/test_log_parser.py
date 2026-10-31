from teleagent.logs.log_parser import LogParser


def test_parse_single_log():

    parser = LogParser()

    logs = (
        "2026-08-01 10:15:22 "
        "AMF ERROR Registration Failed"
    )

    entries = parser.parse(logs)

    assert len(entries) == 1

    assert entries[0].severity == "ERROR"

    assert entries[0].network_function == "AMF"


def test_parse_unknown_log():

    parser = LogParser()

    entries = parser.parse("Random Text")

    assert entries[0].severity == "UNKNOWN"


def test_empty_logs():

    parser = LogParser()

    assert parser.parse("") == []
