from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Example function to categorize a transaction
def generate_category(transaction_desc, user_categories=None, prompt=None, llm=None, model=None):
    if user_categories is None:
        user_categories = ""

    if model is None:
        model = "gpt-4o-mini"

    if llm is None:
        # Initialize OpenAI LLM via LangChain (Choose the desired model)
        # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=50)
        llm = ChatOpenAI(model=model, temperature=0, max_tokens=100, max_retries=2, verbose=True)

    if prompt is None:
        # Define a prompt template
        prompt_template = """
        You are CategorizePro - Categorize the following transaction based on the user's previous preferences:
        Transaction: {transaction}
        User preferences: {user_categories}
        Provide only the best category name for this transaction without any explanation. If nothing matches, then categorize it as 'Other'
        """

        # Set up LangChain's prompt template system
        prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | llm
    response = chain.invoke(
        {
            "output_language": "English",
            # "input": "",
            "transaction": transaction_desc,
            "user_categories": user_categories,
        }
    )
    if hasattr(response, 'content'):
        # AIMessage(content="J'adore la programmation.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 31, 'total_tokens': 36}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None}, id='run-63219b22-03e3-4561-8cc4-78b7c7c3a3ca-0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})
        content = response.content.strip()
    else:
        content = response.strip()
    
    # Assuming the response contains token usage information in the same format
    total_tokens = response.response_metadata['token_usage']['total_tokens']
    model_name = response.response_metadata['model_name']
    
    return content, total_tokens, model_name

