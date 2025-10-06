from .mytools import logging_print
from .settings import AgentState
from . import settings
from .agentic_rag import rag_invoke, robust_binary_grader, master_llm
import random
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

#Action node
def answer_generater(state: AgentState) -> AgentState:
    """ 
    Generate answer based on the retrieval documents.
    """
    query = state["query"]
    document = state["retrieved_doc"]
    logging_print(f"===Step {settings.STEP}===\n")    
    logging_print(f"Master_Agent: I am generating the answer based on the retrieved docs.\n")   
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions **only** using the retrieved documents.

        ## Instructions
        - Carefully read the retrieved documents below.
        - Write your answer in a **natural, conversational tone** as if explaining to a person.
        - Your answer must be **strictly based on the retrieved documents** — do not invent or assume information.
        - Do **not** mention or reference the documents explicitly (e.g., do not say "according to the document" or "based on the source").
        - If the documents do not provide enough information, respond with:
        "I'm sorry, the provided documents do not contain enough information to answer this question."

        ## Inputs
        Retrieved Documents:
        {document}

        User Query:
        {question}

        ## Output
        Provide a **clear, natural, and human-like** answer grounded only in the retrieved documents.
        Avoid formal or robotic phrasing, and never cite or reference the documents directly.
        """,
        input_variables=["document", "question"],
    )
    chain = prompt | master_llm | StrOutputParser()   
    raw = chain.invoke({"question": query, "document": document})
    logging_print(f"Real output: {raw}\n")        
    state["generation"] = str(raw)
    state["regenerate_counter"] += 1
    settings.STEP += 1
    return state 

#Action Node
def grade_greeting_query(state: AgentState) -> AgentState:
    """
    Determine whether a query is pure greeting.
    without relying on prior conversation context.
    """   
    query = state["query"]
    logging_print(f"===Step {settings.STEP}===\n")
    logging_print(f"""Master_Agent: Got a new query: "{query}"\nI will check if the query is pure greeting.\n""")     
   
    prompt = PromptTemplate(
        template="""
    You are a classifier that checks if a message is a **pure greeting**.

    Definition of pure greeting:
    - A short message whose sole purpose is to greet or say hello.
    - It may include polite questions or phrases like “How are you?”, “What’s up?”, 
    “Good to meet you”, “Nice to see you”, “How’s it going?”, etc.
    - It must NOT contain any request for information, task instructions, or other content.

    Your task:
    Given the user's input below, decide if it is a pure greeting.

    User input: {question}

    Return only a JSON object with a single key "score":
    - Output {{"score": "yes"}} if it is a pure greeting (even if it looks like a casual question such as “What’s up?”).
    - Output {{"score": "no"}} if it contains anything beyond a greeting.

    Do not add explanation or extra text.
    """,
        input_variables=["question"],
    )


    state["grade"] = robust_binary_grader(prompt=prompt, question=query)
    settings.STEP += 1
    return state

#Decision Node
def decide_greeting_query(state: AgentState) -> str:
    """ 
    If it's a greeting query, go to grader node for clinical checking.
    If it's not a greeting query, go to grader node for self-contained checking. 
    """
    if state['grade']["score"] == "yes":
        logging_print(f"Master_Agent: The query is just greeting. Greeting back.\n")
        return "greeting"
    else:
        logging_print(f"Master_Agent: The query is not greeting. Let's call RAG.\n")
        return "call_rag"
    
#Action Node
def greeting_back(state: AgentState) -> AgentState:
    """
    Greeting back    
    """
    query = state["query"]

    logging_print(f"===Step {settings.STEP}===\n")    
    logging_print(f"Master_Agent: I am greeting.\n")  

    greetings = [
        "Hey, how’s it going?",
        "What’s up?",
        "Good to see you!",
        "How have you been?",
        "Hi there!"
    ]    

    state["generation"] = random.choice(greetings)       
    settings.STEP = 1
    return state

#Action Node
def answer_normally(state: AgentState) -> AgentState:
    """
    Greeting back    
    """
    query = state["query"]

    logging_print(f"===Step {settings.STEP}===\n")    
    logging_print(f"Master_Agent: I am answering normal message.\n")    
     
    prompt = PromptTemplate(
        template="""You are a "Simple & Clear Answer" assistant.
    Your job: read the user's message and reply with a very simple, easy-to-understand answer.

    Rules:
    - Use plain English and common words.
    - Be direct. Answer the exact question first.
    - Keep it short: 1-2 short sentences maximum.
    - No jargon. If a term is necessary, add a 3-5 word explanation in parentheses.
    - No extra facts, no links, no emojis, no disclaimers, no small talk.
    - If the user asks for steps, give 3 or fewer bullet points, each 1 short line.
    - If the message is not a question (e.g., a statement or request), give a short, helpful response that matches the intent.

    User message: {message}

    Your entire reply must follow the rules above and contain only the answer.""",
        input_variables=["message"],
    )
    chain = prompt | master_llm | StrOutputParser()    
    raw = chain.invoke({"message": query})
    logging_print(f"Real output: {raw}\n")      
    state["generation"] = f"""{str(raw)}, but {state["retrieved_doc"]}"""
    settings.STEP = 1
    return state

#Action Node
def rag_calling(state: AgentState) -> AgentState:
    """
    Call Agentic RAG to retrieve relevant documents.   
    """   
    logging_print(f"===Step {settings.STEP}===\n")  
    logging_print(f"Master_Agent: I am calling RAG.\n") 
    state = rag_invoke(state)
    #logging_print(state)
    state["generation"] = state["retrieved_doc"] 
    settings.STEP += 1
    return state

#Decision Node
def decide_retrieve_success(state: AgentState) -> str:
    """ 
    If rag retrieved relevant docs, go to generate answer.
    If there is no retrieval docs, save to memory 
    """
    if state['grade']["score"] == "yes":
        logging_print(f"Master_Agent: Let's generate the answer.\n")
        return "answer_generater"
    else:
        logging_print(f"Master_Agent: There is no relevant docs from RAG.\n")          
        settings.STEP = 1 #reset step     
        return "answer_normally"
    
#Action Node
def grade_hallucination(state: AgentState) -> AgentState:
    """
    Determine whether the answer llm generated has hallucination.    
    """
    generation = state["generation"]    
    retrieval_doc = state["retrieved_doc"]

    logging_print(f"===Step {settings.STEP}===\n")  
    logging_print(f"Master_Agent: I am checking if the answer has hallucination.\n")
    prompt = PromptTemplate(
        template="""
    You are a classifier that checks if an **answer is fully supported by the provided document**.

    Definition of “fully supported”:
    - Every factual statement in the answer can be directly verified in the given document.
    - The answer contains **no hallucination**, speculation, or information that is absent from the document.
    - Minor rephrasing or summarizing of the document is acceptable as long as it remains faithful.

    Your task:
    Given the document and the answer below, decide if the answer is completely grounded in the document.

    Document:
    {document}

    Answer:
    {answer}

    Return only a JSON object with a single key "score":
    - Output {{"score": "yes"}} if the answer is fully supported by the document with no hallucination.
    - Output {{"score": "no"}} if any part of the answer is not supported by the document.

    Do not add explanation or extra text.
    """,
        input_variables=["document", "answer"],
    )
    state["grade"] = robust_binary_grader(prompt=prompt, answer=generation, document=retrieval_doc)
    settings.STEP += 1
    return state

#Decision Node
def decide_hallucination(state: AgentState) -> str:
    """ 
    If the answer has hallucination, re-generate the answer.
    If it ground the retrieval document, save to memory 
    """
    if state['grade']["score"] == "yes":
        logging_print(f"Master_Agent: The answer is good now.\n")
        return "save_node"
    elif state["regenerate_counter"] <= 3:
        logging_print(f"Master_Agent: The answer has hallucination. I will generate another one.\n")
        return "answer_generater"
    else:
        settings.STEP = 1 #reset step  
        return "end"    
    
#Action Node
def save_to_memory(state: AgentState) -> AgentState:
    """ 
    Before End, save user's query and final answer to memory 
    """   
    logging_print(f"===Step {settings.STEP}===\n")
    logging_print("Master_Agent: I am saving the user query and answer to memory.\n")      

    settings.SHORT_TERM_MEMORY.add_message(session_id=state["session_id"], message=state["query"], msg_type="human")
    settings.SHORT_TERM_MEMORY.add_message(session_id=state["session_id"], message=state["generation"], msg_type="ai")
    settings.STEP = 1 # Reset the STEP 
    return state  


medipal_graph = StateGraph(AgentState)
# Nodes
medipal_graph.add_node("grade_greeting_node", grade_greeting_query)

medipal_graph.add_node("decide_greeting_router", lambda state: state)

medipal_graph.add_node("greeting_node", greeting_back)

medipal_graph.add_node("call_rag_node", rag_calling)

medipal_graph.add_node("decide_retrieve_success_router", lambda state: state)

medipal_graph.add_node("generate_answer_node", answer_generater)

medipal_graph.add_node("grade_hallucination_node", grade_hallucination)

medipal_graph.add_node("decide_hallucination", lambda state: state)

medipal_graph.add_node("save_node", save_to_memory)

medipal_graph.add_node("answer_normally_node", answer_normally)

#Edges
medipal_graph.add_edge(START, "grade_greeting_node")

medipal_graph.add_edge("grade_greeting_node", "decide_greeting_router")

medipal_graph.add_conditional_edges(
    source="decide_greeting_router",
    path=decide_greeting_query,
    path_map={
        "greeting": "greeting_node",
        "call_rag": "call_rag_node"
    }
)

medipal_graph.add_edge("call_rag_node", "decide_retrieve_success_router")

medipal_graph.add_conditional_edges(
    source="decide_retrieve_success_router",
    path=decide_retrieve_success,
    path_map={
        "answer_generater": "generate_answer_node",
        "answer_normally": "answer_normally_node"
    }
)

medipal_graph.add_edge("generate_answer_node", "grade_hallucination_node")

medipal_graph.add_edge("grade_hallucination_node", "decide_hallucination")

medipal_graph.add_conditional_edges(
    source="decide_hallucination",
    path=decide_hallucination,
    path_map={
        "save_node": "save_node",
        "answer_generater": "generate_answer_node",
        "end": END
    }
)

medipal_graph.add_edge("answer_normally_node", END)
medipal_graph.add_edge("greeting_node", END)
medipal_graph.add_edge("save_node", END)

medipal_app = medipal_graph.compile()

def ask(query: str) -> str:
    state = AgentState(query=query, session_id=1,wiki_used=False,brave_used=False,rewrite_counter=0,regenerate_counter=0)
    result = medipal_app.invoke(state)
    logging_print(f"result:{result}")
    return result["generation"]

__all__ = ["ask"]

if __name__ == "__main__":
    questions = [    
    "Is there anything I can assist you with?",    
    "Can I help you in any way, next?",
    "Do you have any questions?",  
    "Are you looking for any particular information?",
    "I am a Medicine Agentic RAG. I can help you get medical and clinical documents. Just tell me what you need?"
    ]

    while True:
        user_input = input(random.choice(questions))
        if user_input.strip().lower() in ["end", "exit"]:
            break
        query = AgentState(query=user_input, session_id=1,wiki_used=False,brave_used=False,rewrite_counter=0,regenerate_counter=0)
        result = medipal_app.invoke(query)
        logging_print(f"result:{result}")