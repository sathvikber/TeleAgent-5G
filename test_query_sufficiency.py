from teleagent.retrieval.query_type_detector import (
    QueryTypeDetector,
)

from teleagent.retrieval.evidence_sufficiency_checker import (
    EvidenceSufficiencyChecker,
)


query = (
    "According to TS 33.501, what is AUSF?"
)

evidence = """
REQUESTED SPECIFICATION
TS 33.501

CANONICAL DEFINITIONS FOUND IN DEFINITION SECTIONS
AUSF: AUthentication Server Function
"""


print("=" * 80)
print("QUERY TYPE TEST")
print("=" * 80)

detector = QueryTypeDetector()

analysis = detector.detect(query)

print("Original Query              :", analysis.original_query)
print("Query Type                  :", analysis.query_type)
print("Target Entity               :", analysis.target_entity)
print(
    "Requires Explicit Definition:",
    analysis.requires_explicit_definition,
)

print()

print("=" * 80)
print("RETRIEVAL QUERY VARIANTS")
print("=" * 80)

queries = detector.build_retrieval_queries(
    analysis
)

for item in queries:
    print("-", item)

print()

print("=" * 80)
print("EVIDENCE SUFFICIENCY")
print("=" * 80)

checker = EvidenceSufficiencyChecker()

result = checker.check(
    analysis,
    evidence,
)

print("Sufficient :", result.is_sufficient)
print("Reason     :", result.reason)
print("Matched    :", result.matched_evidence)

print()

if not result.is_sufficient:

    print("=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)

    print(
        EvidenceSufficiencyChecker.FALLBACK_RESPONSE
    )
