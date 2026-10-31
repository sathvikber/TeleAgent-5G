from teleagent.retrieval.query_processor import QueryProcessor


def test_strip_whitespace():
    processor = QueryProcessor()

    assert processor.process("   hello   ") == "hello"


def test_multiple_spaces():
    processor = QueryProcessor()

    assert (
        processor.process("UE     registration    failed")
        == "UE registration failed"
    )


def test_repeated_question_marks():
    processor = QueryProcessor()

    assert (
        processor.process("What is AMF????")
        == "What is AMF?"
    )


def test_repeated_periods():
    processor = QueryProcessor()

    assert (
        processor.process("AMF.....")
        == "AMF."
    )


def test_empty_string():
    processor = QueryProcessor()

    assert processor.process("") == ""


def test_none_query():
    processor = QueryProcessor()

    assert processor.process(None) == ""
