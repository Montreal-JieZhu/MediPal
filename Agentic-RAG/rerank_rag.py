from mytools import timed, login_huggingface
import os
import json
import copy
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class Rerank_RAG():
    """
        Rerank_RAG class defines everything the RAG needs.
            Attributes:
                workspace_base_path: The current workspace.
                dataset_path: The path to the medicine dataset.                
                embedding_model_id: The name of the embedding model.
                cross_encoder_model_id: The name of crossEncoder model which is used to do reranking.
                embedding_model: A embedding model.
                retriever: It is a very important retriever who will similarity search the documents based on query.

            Functions:
                load_json_list: Load json file to json objects.
                login_huggingface: Login huggingface to gain the access to the LLMs
                build_medicine_retriever: Build a multi-vector db which contains vectorstore and docstore. Embedding hypothetical questions to vectorstore and Storing original documents to docstore.
                load_embedding_model: Load embedding model.
                load_crossencoder: Load cross encoder model.
                retrieve: Wrap retriever and reranker up to fetch top_k relevant documents.
    """
    def __init__(self) -> None:

        self.workspace_base_path = os.getcwd()
        self.dataset_path = os.path.join(self.workspace_base_path, "datasets", "medicine_data_hypotheticalquestions.json")  
        self.embedding_model_id = "sentence-transformers/embeddinggemma-300m-medical"
        self.cross_encoder_model_id = "ncbi/MedCPT-Cross-Encoder" 
        self.embedding_model = None
        self.retriever = None
        self.cross_encoder = None

    @timed
    def load_embedding_model(self):        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_id,
            model_kwargs = {'device': 'cpu'},
            # Normalizing helps cosine similarity behave better across models
            encode_kwargs={"normalize_embeddings": True},
        )        

    @timed
    def load_crossencoder(self):
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_id)

    @timed
    def load_json_list(self):    
        with open(self.dataset_path, mode = "r", encoding="utf-8") as f:
            return json.load(f)
        
    @timed    
    def build_medicine_retriever(self):        
        data = self.load_json_list()  
        login_huggingface()      
        self.load_embedding_model()
        self.load_crossencoder()
        docstore = InMemoryStore()
        id_key = "doc_id"

        # The vectorstore to use to index the questions
        vectorstore = Chroma(collection_name = "medicine_data", embedding_function = self.embedding_model)
        # The Multi-Vector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
        )

        doc_ids = list()
        questions = list()
        docs = list()
        for d in data[:50]:
            doc_id = d["doc_id"]
            doc_ids.append(doc_id)
            docs.append(Document(metadata={"doc_id": doc_id}, page_content=d["original_doc"]))
            for q in d["questions"]:
                questions.append(Document(metadata={"doc_id": doc_id}, page_content=q))

        self.retriever.vectorstore.add_documents(questions)
        self.retriever.docstore.mset(list(zip(doc_ids,docs)))

    @timed
    def retrieve(self, query: str, top_k: int=5):
        retrieved_docs = self.retriever.invoke(query, kwargs={"k":10})
        retrieved_docs = copy.deepcopy(retrieved_docs) # Avoid rerank changes original documents
        #Rerank part
        pairs = [[query, d.page_content] for d in retrieved_docs]
        scores = self.cross_encoder.predict(pairs, batch_size=32)
        for r_d, score in zip(retrieved_docs, scores):
            r_d.metadata["rerank_score"] = float(score)
        retrieved_docs.sort(key= lambda d: d.metadata["rerank_score"], reverse=True)
        #Rerank part
        return retrieved_docs[ :top_k]
    
__all__ = ["Rerank_RAG"]


if __name__ == "__main__":
    rag = Rerank_RAG()
    rag.build_medicine_retriever()
    print(rag.retrieve("My nasal is disconfort. Do you have a medicine to relieve sinus congestion and pressure?",top_k=2))