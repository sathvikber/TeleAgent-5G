"""
RAG Prompt Builder

Constructs compact, strongly grounded prompts for
TeleAgent KnowledgeAgent.

The prompt is intentionally concise because the target SLM
(Qwen2.5-1.5B-Instruct) follows short, explicit constraints
more reliably than long instruction-heavy prompts.
"""


class RAGPromptBuilder:
    """
    Builds a strict evidence-grounded prompt.

    The SLM must answer only from retrieved evidence and
    must not invent acronym expansions, telecom entities,
    roles, procedures, interfaces, parameters, or facts.

    The prompt also requires the model to preserve entity
    types such as:

    - network functions
    - security entities
    - user equipment
    - parameters
    - authentication values
    - cryptographic keys
    """

    def build(
        self,
        query: str,
        retrieved_context: str,
    ) -> str:
        """
        Build the final RAG prompt.

        Parameters
        ----------
        query:
            Original user question.

        retrieved_context:
            Compact evidence produced by
            IntelligentRetrievalService.

        Returns
        -------
        str
            Prompt sent to the SLM.
        """

        query = (query or "").strip()

        retrieved_context = (
            retrieved_context or ""
        ).strip()

        # -----------------------------------------------------
        # Fallback when no retrieved evidence is available
        # -----------------------------------------------------

        if not retrieved_context:

            retrieved_context = (
                "No relevant information was retrieved "
                "from the current knowledge base."
            )

        # -----------------------------------------------------
        # Compact strict grounding prompt
        # -----------------------------------------------------

        return f"""
You are TeleAgent.

Answer the USER QUESTION using ONLY the EVIDENCE below.

STRICT RULES:

1. Use only facts explicitly supported by the EVIDENCE.

2. Do not use outside knowledge.

3. Do not invent or guess:
   - acronym expansions,
   - network functions,
   - security entities,
   - entity roles,
   - procedures,
   - interfaces,
   - protocols,
   - standards,
   - section numbers,
   - parameters,
   - causes,
   - recommendations.

4. Expand an acronym ONLY if its exact definition appears
   in the EVIDENCE.

5. If an acronym has no explicit definition in the EVIDENCE,
   keep the acronym unchanged.

6. Do not classify keys, authentication values, parameters,
   identifiers, messages, or authentication vectors as
   network functions.

7. When the question asks specifically for network functions:
   - do not label every participating entity as a network function,
   - preserve explicit entity types stated in the EVIDENCE,
   - if an entity is identified as a security entity, call it a
     security entity,
   - if the EVIDENCE does not explicitly establish an entity as a
     network function, do not claim that it is one,
   - distinguish participating entities from network functions.

8. If the EVIDENCE distinguishes two meanings of the same
   acronym, preserve that distinction.

9. For a procedure question:
   - identify only entities explicitly participating,
   - describe only actions explicitly stated in the EVIDENCE,
   - do not infer additional roles.

10. If a combined entity such as UDM/ARPF appears in the
    EVIDENCE, preserve it exactly unless the EVIDENCE
    separately describes its components.

11. Do not use speculative language such as:
    - likely,
    - probably,
    - typically,
    - generally,
    - may be.

12. Keep the answer concise and technically precise.

13. If the USER QUESTION asks for network functions but the
    EVIDENCE also contains other participating entity types:
    - answer the network-function part first,
    - then separately mention the other participating entities,
    - do not merge different entity types into one category.

14. If the EVIDENCE is insufficient to answer the question,
    answer exactly:

"The information is not available in the current knowledge base."


EVIDENCE
========

{retrieved_context}


USER QUESTION
=============

{query}


ANSWER
======
""".strip()
