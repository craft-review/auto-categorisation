# Auto-Categorization of Transactions

This project involves building an POC for auto-categorization system that can automatically categorize accounting application business transactions based on user preferences. The system adapts to individual users by learning from user feedback and generating personalized categories using Generative AI models (LLMs). The system uses pre-trained language models and embeddings with vector index search to generate categories based on transaction descriptions and past user behavior. For the Proof of Concept (POC), the data is stored in CSV format, and we use Streamlit to demonstrate the interface of the auto-categorization system.

## Key Technologies Used:
- **LangChain**: LangChain Framework for LLM orchestration and prompt management.
- **LLM - OpenAI GPT-4o API**: Provides the power to generate context-aware and personalized categories based on transaction data.
- **Huggingface Transformers**: Used to load and interact with pre-trained language models embeddings for semantic search of transaction categories.
- **Prompt Engineering**: Used to improve response quality and accuracy.
- **Few-shot Learning** and user feedback loop to refine categorization accuracy over time.
- **CSV**: Used for storing and retrieving transaction and user preference data. Also, CSV Handling with Pandas for simulating a feedback database.
- **Streamlit**: A Python framework to build interactive web applications for POC.
- **Python**: For backend apps
- **Redis**: For storing metadata around user feedback like frequence of category changes and for storing user preferences
- **MySQL/SQLite**: For storing transactions feedback

## Steps to Start and Test the Streamlit App:

### 1. Clone the Repository:
Start by cloning the repository to your local machine:

```bash
git clone https://github.com/craft-review/auto-categorisation.git
cd auto-categorization
```

### 2. Install the Dependencies:
Ensure you have Python 3.8 or above installed. Install the required dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API:
You will need an OpenAI API key to use the GPT-3.5 model. Export your OpenAI API key as an environment variable or set it in the app directly:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 4. Run the Streamlit App:
Once everything is set up, run the Streamlit app by executing the following command:

```bash
streamlit run app.py
```

### 5. Test the App:
Open your browser and go to `http://localhost:8501`. You should see the Streamlit interface where you can upload a CSV file containing transactions, and the app will auto-categorize the transactions based on the GPT-4o model.

---

## Future Enhancements:
- Integration with a database (e.g., Postgres or NoSQL) for real-time storage and retrieval of user transactions and categories.
- Implementing fine-tuning based model on user corrections to improve category predictions over time.
- Moving FAISS vector index to scalable and managed vector database like Pinecone or Chroma DB
- Recon CSV's to be replaced with analytics DB for batching and reconcilliation purpose
- Adding user authentication to save preferences.
---

Feel free to fork this project and make your own enhancements! If you have any questions or issues, raise an issue or create a pull request.

Happy coding! 🚀