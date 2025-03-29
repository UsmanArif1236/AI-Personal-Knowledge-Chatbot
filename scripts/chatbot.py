from retrieval import retrieve_relevant_chunks
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_llm():
    """Loads Mistral 7B GGUF model to run on CPU"""
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="mistral",
        config={"max_new_tokens": 1024, "temperature": 0.5}
    )
    return llm

template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
You are an AI chatbot answering questions based on a document and maintaining conversation history.
Use the previous conversation context for better responses.

Previous Conversation:
{chat_history}

Now, answer the user's question using the provided document.
If the document does not contain the answer, say:
"The answer is not given in the document, but here is what I know based on my general knowledge."

Question: {question}

Document Context: {context}

Answer:
"""
)

def generate_answer(question):
    """Generates a response based on document retrieval and LLM."""
    relevant_docs = retrieve_relevant_chunks(question)

    context = " ".join([doc.page_content for doc in relevant_docs])
    print(f"Context: {context}")
    formatted_prompt = template.format(
        chat_history=memory.load_memory_variables({})["chat_history"],
        question=question,
        context=context
    )
    response = load_llm().invoke(formatted_prompt)
    memory.save_context({"input": question}, {"output": response})

    return response
