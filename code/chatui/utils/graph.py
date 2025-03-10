# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from typing_extensions import TypedDict
from typing import List

from chatui import ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.tools.tavily_search import TavilySearchResults

from chatui.utils import database, nim

### State

from typing import List
from langchain.schema import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    generator_model_id: str
    router_model_id: str
    retrieval_model_id: str
    hallucination_model_id: str
    answer_model_id: str
    prompt_generator: str
    prompt_router: str
    prompt_retrieval: str
    prompt_hallucination: str
    prompt_answer: str
    router_use_nim: bool
    retrieval_use_nim: bool
    generator_use_nim: bool
    hallucination_use_nim: bool
    answer_use_nim: bool
    nim_generator_ip: str
    nim_router_ip: str
    nim_retrieval_ip: str
    nim_hallucination_ip: str
    nim_answer_ip: str
    nim_generator_port: str
    nim_router_port: str
    nim_retrieval_port: str
    nim_hallucination_port: str
    nim_answer_port: str
    nim_generator_id: str
    nim_router_id: str
    nim_retrieval_id: str
    nim_hallucination_id: str
    nim_answer_id: str
    # New Ollama-specific state attributes
    use_ollama: bool
    ollama_server: str
    ollama_port: str
    ollama_model: str


from langchain.schema import Document

def node_to_document(node_with_score) -> Document:
    """ A helper function for converting a llamaindex node to a langchain document type. """
    return (Document(
        page_content=node_with_score.node.text,
        metadata={
            **node_with_score.node.metadata,
            "score": node_with_score.score
        }
    ), node_with_score.score)

def convert_nodes_to_documents(nodes_with_score) -> List[Document]:
    """ A helper function for converting llamaindex nodes to langchain documents. """
    return [node_to_document(node) for node in nodes_with_score]

def sort_and_filter(doc_score_list):
    # Filter out tuples with score 0.0
    filtered_list = filter(lambda x: x[1] != 0.0, doc_score_list)
    # Sort the list of tuples by the second element (score) in descending order
    sorted_list = sorted(filtered_list, key=lambda x: x[1], reverse=True)
    # Extract and return only the documents from the tuples
    return [doc for doc, score in sorted_list]

### Helper function to select the appropriate LLM based on settings
def get_llm(state, model_key, use_nim_key, nim_ip_key, nim_port_key, nim_id_key):
    """
    Helper function to get the appropriate LLM based on the state configuration.
    
    Args:
        state: The current state dictionary
        model_key: Key for the model ID in the state
        use_nim_key: Key for the flag indicating whether to use NIM
        nim_ip_key: Key for the NIM IP address
        nim_port_key: Key for the NIM port
        nim_id_key: Key for the NIM model ID
        
    Returns:
        An instance of the appropriate LLM class
    """
    # If Ollama is enabled, use that
    if state.get("use_ollama", False):
        return ollama.OllamaChatModel(
            ollama_server=state["ollama_server"],
            ollama_port=state["ollama_port"],
            model_name=state["ollama_model"],
            temperature=0.7
        )
    # Otherwise, use NIM or NVIDIA API as before
    elif state[use_nim_key]:
        return nim.CustomChatOpenAI(
            custom_endpoint=state[nim_ip_key], 
            port=state[nim_port_key] if len(state[nim_port_key]) > 0 else "8000",
            model_name=state[nim_id_key] if len(state[nim_id_key]) > 0 else "meta/llama3-8b-instruct",
            temperature=0.7
        )
    else:
        return ChatNVIDIA(model=state[model_key], temperature=0.7)

### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    webpages = []
    pdfs = []
    images = []

    # Retrieval
    
    if os.path.exists('/project/data/lancedb/web_collection.lance'):
        print("---RETRIEVING WEBPAGES---")
        web_retriever = database.get_webpage_retriever()
        webpages = web_retriever.similarity_search_with_score(question, k=3)
    if os.path.exists('/project/data/lancedb/pdf_collection.lance'):
        print("---RETRIEVING PDFS---")
        pdf_retriever = database.get_pdf_retriever()
        pdfs = pdf_retriever.similarity_search_with_score(question, k=3)
    if os.path.exists('/project/data/lancedb/text_img_collection.lance') and os.path.exists("/project/data/mixed_data/"):
        print("---RETRIEVING IMAGES AND VIDEO---")
        img_retriever = database.get_img_retriever()
        images = convert_nodes_to_documents(img_retriever.retrieve(question))
    
    print("---RERANKING RETRIEVED DOCUMENTS---")
    documents = sort_and_filter(webpages + pdfs + images)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    prompt = PromptTemplate(
        template=state["prompt_generator"],
        input_variables=["question", "document"],
    )
    
    llm = get_llm(
        state,
        "generator_model_id", 
        "generator_use_nim", 
        "nim_generator_ip", 
        "nim_generator_port", 
        "nim_generator_id"
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    prompt = PromptTemplate(
        template=state["prompt_retrieval"],
        input_variables=["question", "document"],
    )
    
    llm = get_llm(
        state,
        "retrieval_model_id", 
        "retrieval_use_nim", 
        "nim_retrieval_ip", 
        "nim_retrieval_port", 
        "nim_retrieval_id"
    )
    
    retrieval_grader = prompt | llm | JsonOutputParser()
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            continue
    # We set a flag to indicate that we want to run web search if insufficient relevant docs found
    web_search = "Yes" if len(filtered_docs) < 1 else "No"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    if len(question) > 4: # Tavily minimum search length is 5 characters
        web_search_tool = TavilySearchResults(k=3)
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
    return {"documents": documents, "question": question}


### Conditional edge


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    prompt = PromptTemplate(
        template=state["prompt_router"],
        input_variables=["question"],
    )
    
    llm = get_llm(
        state,
        "router_model_id", 
        "router_use_nim", 
        "nim_router_ip", 
        "nim_router_port", 
        "nim_router_id"
    )
    
    question_router = prompt | llm | JsonOutputParser()
    source = question_router.invoke({"question": question})
    print(source)
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    prompt = PromptTemplate(
        template=state["prompt_hallucination"],
        input_variables=["generation", "documents"],
    )
    
    llm_hallucination = get_llm(
        state,
        "hallucination_model_id", 
        "hallucination_use_nim", 
        "nim_hallucination_ip", 
        "nim_hallucination_port", 
        "nim_hallucination_id"
    )
    
    hallucination_grader = prompt | llm_hallucination | JsonOutputParser()

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    prompt = PromptTemplate(
        template=state["prompt_answer"],
        input_variables=["generation", "question"],
    )
    
    llm_answer = get_llm(
        state,
        "answer_model_id", 
        "answer_use_nim", 
        "nim_answer_ip", 
        "nim_answer_port", 
        "nim_answer_id"
    )
    
    answer_grader = prompt | llm_answer | JsonOutputParser()
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"