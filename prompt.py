# from transformers import pipeline
# from langchain import OpenAI

import os
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain import LLMChain
# from langchain_openai import OpenAI
# import openai
# from langchain_core.prompts import PromptTemplate


# Set up OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# Load the LLM model
# generator = pipeline('text-generation', model='gpt-3.5-turbo')  # Replace with appropriate LLM or API

# Define a prompt template
prompt_template = """
You are CategorizePro - Categorize and restrict the following transaction based on the user's previous preferences:
Transaction: {transaction}
User preferences: {user_categories}
Provide only the best category name for this transaction without any explanation. If nothing matches, then categorize it as 'Other'
"""

# Set up LangChain's prompt template system
prompt = ChatPromptTemplate.from_template(prompt_template)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )
# prompt = PromptTemplate(
#     input_variables=["transaction", "user_categories"],
#     template=prompt_template,
# )
# Initialize OpenAI LLM via LangChain (Choose the desired model)
# llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=50)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1, max_tokens=50, max_retries=1, verbose=True)

# Create a chain combining the prompt and LLM
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# Example function to categorize a transaction
def generate_category(transaction_desc, user_categories=None):
    if user_categories is None:
        user_categories = ""
    
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
        # print(response)
        # print(response.usage_metadata)
        # AIMessage(content="J'adore la programmation.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 31, 'total_tokens': 36}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None}, id='run-63219b22-03e3-4561-8cc4-78b7c7c3a3ca-0', usage_metadata={'input_tokens': 31, 'output_tokens': 5, 'total_tokens': 36})
        content = response.content.strip()
    else:
        content = response.strip()
    
    # Assuming the response contains token usage information in the same format
    total_tokens = response.response_metadata['token_usage']['total_tokens']
    model_name = response.response_metadata['model_name']
    
    return content, total_tokens, model_name
    # return response.strip()
    
    # Execute the LLM chain
    # response = llm_chain.run({
    #     "transaction": transaction,
    #     "user_categories": user_categories
    # })

    
    # Call the latest API format
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",  # Replace with "gpt-3.5-turbo" if you want to use GPT-3.5
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that categorizes business transactions."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.5
    # )

    # # Extract and return the assistant's reply
    # return response.choices[0].message['content'].strip()

# def generate_category(transaction, user_categories=None):
#     prompt = generate_prompt(transaction, user_categories)
#     result = generator(prompt, max_length=20)  # You can adjust the max_length based on your needs
#     return result[0]['generated_text'].strip()


# def generate_prompt(transaction, user_categories=None):
#     base_prompt = f"Please categorize the following transaction:\nTransaction: {transaction}\n"

#     if user_categories:
#         base_prompt += f"User Preferences:\n"
#         for category in user_categories:
#             base_prompt += f"- {category}\n"
#     # else:
#         # base_prompt += "No specific preferences for this user yet.\n"

#     base_prompt += "What would be the best category for this transaction?"
#     print(base_prompt)
    
#     return base_prompt

