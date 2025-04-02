from retrieval import retrieve_relevant_chunks
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
token = os.getenv("HF_TOKEN")
groq_api = os.getenv("chatbot")


def load_llm():
    return ChatGroq(model="llama3-70b-8192", api_key=groq_api)

template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
You are an AI chatbot answering questions based on retrieved documents. You should:
- Maintain conversation history for continuity.
- Prioritize document-based answers but use general knowledge when necessary.

### Previous Conversation ###
{chat_history}

### User's Question ###
{question}

### Document Context ###
{context}

### Guidelines for Response ###
- If the document contains the answer, base your response on it.
- If the document lacks the answer, state:
  "The answer is not in the document, but here is what I know from my knowledge."
- Keep answers **detailed yet concise**.
- Use a professional and informative tone.

### AI's Answer ###
"""
)

def generate_answer(question):
    """Generates a response based on document retrieval and LLM."""
    relevant_docs = retrieve_relevant_chunks(question)

    context = " ".join([doc.page_content for doc in relevant_docs])
    
    formatted_prompt = template.format(
        chat_history=memory.load_memory_variables({})["chat_history"],
        question=question,
        context=context
    )
    try:
        # Invoke the LLM and get the raw response
        response = load_llm().invoke(formatted_prompt)
        print("Raw Response Type:", type(response))
        print("Raw Response:", response)

        # Check if the response has the content attribute
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response  # If the response is already a string
        else:
            # Handle unexpected response type
            raise TypeError(f"Unexpected response type: {type(response)}")

        # Save the context and return the response text
        memory.save_context({"input": question}, {"output": response_text})
        return response_text

    except Exception as e:
        # Handle errors gracefully
        return f"Error during LLM invocation: {str(e)}"
