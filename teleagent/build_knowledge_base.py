from teleagent.rag.knowledge_base_builder import KnowledgeBaseBuilder


def main():

    builder = KnowledgeBaseBuilder()

    builder.build(
        input_directory="/workspace/teleagent/data/raw_documents",
        output_directory="/workspace/teleagent/data/knowledge_base",
    )

    print("Knowledge base built successfully.")


if __name__ == "__main__":
    main()
