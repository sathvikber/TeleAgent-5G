from teleagent.logs.log_analysis_service import (
    LogAnalysisService
)


def test_analysis_pipeline():

    logs = """
2026-08-01 10:15:22 AMF ERROR Registration Failed
2026-08-01 10:15:23 AUSF WARNING Authentication Timeout
2026-08-01 10:15:24 SMF INFO Session Created
"""

    service = LogAnalysisService()

    result = service.analyze(logs)

    assert result.summary.total_entries == 3

    assert result.summary.error_count == 1

    assert result.summary.warning_count == 1

    assert len(result.important_entries) == 2

    assert "Registration" in result.event_categories


def test_prompt_generation():

    logs = """
2026-08-01 10:15:22 AMF ERROR Registration Failed
"""

    service = LogAnalysisService()

    prompt = service.build_prompt(

        "Why did registration fail?",

        logs,

    )

    assert "Registration Failed" in prompt

    assert "User Question" in prompt

    assert "Detected Patterns" in prompt
