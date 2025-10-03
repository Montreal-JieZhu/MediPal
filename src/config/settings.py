from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # Short-term Memory
from typing import TypedDict

SESSION_ID: int = 0
SHORT_TERM_MEMORY = None
STEP: int = 0

#global variable

#Global Class
#Define a short-term memory
class Short_Term_Memory():
    def __init__(self) -> None: 
        """Initialize the message container and current session id """       
        self.session_store: dict[int,BaseChatMessageHistory] = {}
        self.current_session_id: int = 0

    def get_history(self, session_id: int) -> BaseChatMessageHistory:    
        """return history messages by sessionId"""    
        self.current_session_id = session_id
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]
    
    def get_current_history(self) -> BaseChatMessageHistory:
        """return history messages for current session"""
        return self.get_history(self.current_session_id)
    
    def add_message(self, session_id: int, message: str, msg_type: str) -> None:
        history_messages = self.get_history(session_id)     
        if msg_type == "ai": 
            history_messages.add_ai_message(message)
        else:
            history_messages.add_user_message(message)   

        if len(history_messages.messages) > 2: # Only keep the recent 2 messages
            del history_messages.messages[0] # Remove the first message     
            
    
    def delete_history(self, session_id: int) -> bool:
        """delete history messages by sessionId"""
        if session_id in self.session_store:
            deleted = self.session_store.pop(session_id)
            if deleted:
                return True
            else:
                return False
        return True
    
    def delete_current_history(self) -> bool:
        """delete history messages for current session"""
        return self.delete_history(self.current_session_id)

class AgentState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        session_id: current session id
        query: user's query or augmented query
        retrieved_doc: retrieval docment
        generation: the answer the llm generate    
        grade: keep the binary score for every router node to make decision   
        wiki_used: Flag whether it already used Wiki search
        brave_used: Flag whether it already used brave search
        rewrite_counter: count rewrite action, maximum 3 times
        regenerate_counter: count generate action, maximum 3 times
    """
    session_id: int
    query: str
    retrieved_doc: str
    generation: str
    grade: dict
    wiki_used: bool      # Avoid infinity loop in graph
    brave_used: bool     # Avoid infinity loop in graph
    rewrite_counter: int # Avoid infinity loop in graph
    regenerate_counter: int