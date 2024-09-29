# Auto-Categorization of Transactions

This project involves building an auto-categorization system that can automatically categorize accounting application business transactions based on user preferences. The system adapts to individual users by learning from user feedback and generating personalized categories using Generative AI models (LLMs). The system uses pre-trained language models to generate categories based on transaction descriptions and past user behavior. For the Proof of Concept (POC), the data is stored in CSV format, and we use Streamlit to demonstrate the interface of the auto-categorization system.

## Key Technologies Used:
- **Langchain**: LangChain Framework for LLM orchestration and prompt management.
- **OpenAI GPT-4o API**: Provides the power to generate context-aware and personalized categories based on transaction data.
- **Prompt Engineering**: Used to improve response quality and accuracy.
- **Huggingface Transformers**: Used to load and interact with pre-trained language models (LLMs) for generating transaction categories.
- **CSV**: Used for storing and retrieving transaction and user preference data. Also, CSV Handling with Pandas for simulating a feedback database.
- **Few-shot Learning** and user feedback loop to refine categorization accuracy over time.
- **Streamlit**: A Python framework to build interactive web applications.