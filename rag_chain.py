from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt_template = """
You are a GitLab assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say:
"I don't have enough information from GitLab documentation."

Context:
{context}

Question:
{input}
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "input"]
)

def get_qa_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
    )
    
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain
