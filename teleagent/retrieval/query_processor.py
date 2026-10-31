"""
Query Processor

Pre-processes user queries before retrieval.

Current Responsibilities
------------------------
- Strip leading/trailing whitespace
- Normalize whitespace
- Normalize punctuation

Future Responsibilities
-----------------------
- Acronym expansion
- Telecom terminology normalization
- Query rewriting using SLM
- Spell correction
- Synonym expansion
"""

import re


class QueryProcessor:
    """
    Performs lightweight preprocessing of user queries before retrieval.
    """

    def process(
        self,
        query: str,
    ) -> str:
        """
        Normalize a user query.

        Parameters
        ----------
        query : str
            Original user query.

        Returns
        -------
        str
            Processed query.
        """

        if query is None:
            return ""

        #
        # Remove leading/trailing whitespace
        #
        query = query.strip()

        #
        # Replace multiple whitespace characters with a single space
        #
        query = re.sub(r"\s+", " ", query)

        #
        # Collapse repeated punctuation
        #
        query = re.sub(r"\?{2,}", "?", query)
        query = re.sub(r"!{2,}", "!", query)
        query = re.sub(r"\.{2,}", ".", query)

        return query
