from langchain_core.documents import Document


def create_qa_prompt(chunk: Document) -> str:
    return f"""
        You are an AI assistant tasked with creating Q&A sets based on Emory's CS PhD student handbook. 
        Your goal is to generate high-quality question-answer pairs that can be used for training a retrieval-augmented chatbot.
        1. Read the following PhD handbook content carefully.
        2. Generate a question-answer pair based **only on the provided content**. 
        The pairs should be sufficient to cover the provided content adequately. Follow these guidelines: 
            a. Focus on creating questions that a human would naturally ask about the real concerns a PhD student might have.
            b. Ensure that all questions can be answered solely based on the information provided in the content.
            c. Avoid creating questions that require information not present in the given information.
            d. Create meaningful and insightful questions that provide valuable information about PhD programs including:
                - **Academic Policies** (e.g., coursework requirements, advisor selection)
                - **Funding & Financial Aid** (e.g., stipends, scholarships, deadlines)
                - **Graduation Requirements** (e.g., dissertation defense, time limits)
                - **Research & Ethics** (e.g., authorship, IRB approval)
                - **Student Life & Well-being** (e.g., mental health resources, work-life balance)
        3. The response should be a valid JSON object with the following structure:
            ```json
            {{
                "query": "...",
                "answer": "...",
            }}
            ```
            do NOT add anything to the response; your response must be valid for JSON parsing.
        
        --- Handbook Content Start ---
        {chunk.page_content}
        --- Handbook Content End ---
        
        Generate the QA JSON object, following all the guidelines and instructions provided above to provide a comprehensive representation of the Emory CS PhD handbook's content.
        """
