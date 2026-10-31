class LogAgentError(Exception):
    """Base exception for Log Agent."""


class FileExtractionError(LogAgentError):
    """Raised when a file cannot be extracted."""


class UnsupportedFileError(LogAgentError):
    """Raised when a file type is unsupported."""


class LogParsingError(LogAgentError):
    """Raised when log parsing fails."""