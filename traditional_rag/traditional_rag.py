"""Traditional RAG Pipeline using LangChain, OpenAI, and FAISS.
Modified to support multiple file types (txt, pdf, docx, md)
"""

import os
import time
from typing import List, Dict, Any
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Document Loaders for different file types
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)


class TraditionalRAG:
    """Traditional RAG system using vector similarity search with multi-file support."""

    # Supported file extensions and their loaders
    SUPPORTED_LOADERS = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader
    }

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Traditional RAG system.

        Args:
            openai_api_key: OpenAI API key
            model_name: LLM model to use
            embedding_model: Embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key
        )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=openai_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        self.vectorstore = None
        self.qa_chain = None

    def get_loader_for_file(self, file_path: Path):
        """
        Get the appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document loader instance
        """
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_LOADERS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(self.SUPPORTED_LOADERS.keys())}")
        
        loader_class = self.SUPPORTED_LOADERS[ext]
        return loader_class(str(file_path))

    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a single file (any supported type).

        Args:
            file_path: Path to the document file

        Returns:
            List of LangChain Documents
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get appropriate loader
        loader = self.get_loader_for_file(path)
        
        # Load documents
        raw_docs = loader.load()
        
        # Split into chunks
        documents = []
        for i, doc in enumerate(raw_docs):
            chunks = self.text_splitter.split_text(doc.page_content)
            for j, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(path.name),
                            "file_path": str(path),
                            "chunk_id": f"{i}_{j}",
                            "file_type": path.suffix.lower()
                        }
                    )
                )

        print(f"✓ Loaded {len(documents)} chunks from {path.name}")
        return documents

    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            List of all LangChain Documents combined
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_documents(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")
        
        print(f"\n📚 Total: {len(all_documents)} chunks from {len(file_paths)} files")
        return all_documents

    def load_folder(self, folder_path: str, recursive: bool = False) -> List[Document]:
        """
        Load all supported documents from a folder.

        Args:
            folder_path: Path to the folder
            recursive: Whether to search subfolders

        Returns:
            List of all LangChain Documents
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        all_documents = []
        supported_extensions = list(self.SUPPORTED_LOADERS.keys())
        
        # Get all files
        if recursive:
            files = [f for f in folder.rglob("*") if f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in folder.glob("*") if f.suffix.lower() in supported_extensions]
        
        if not files:
            print(f"⚠ No supported files found in {folder_path}")
            print(f"  Supported extensions: {supported_extensions}")
            return []
        
        print(f"\n📁 Found {len(files)} files in {folder_path}:")
        for f in files:
            print(f"   - {f.name} ({f.suffix})")
        print()
        
        # Load each file
        for file_path in files:
            try:
                docs = self.load_documents(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                print(f"✗ Failed to load {file_path.name}: {e}")
        
        print(f"\n📚 Total: {len(all_documents)} chunks from {len(files)} files")
        return all_documents

    def build_index(self, documents: List[Document]) -> None:
        """
        Build FAISS vector index from documents.

        Args:
            documents: List of LangChain Documents
        """
        if not documents:
            raise ValueError("No documents to index!")
            
        print("\n🔨 Building FAISS index...")
        start_time = time.time()

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        build_time = time.time() - start_time
        print(f"✓ FAISS index built in {build_time:.2f} seconds")
        print(f"  - Total vectors: {len(documents)}")

        # Create QA chain
        self._create_qa_chain()

    def _create_qa_chain(self) -> None:
        """Create the QA chain with custom prompt."""
        prompt_template = """You are a helpful AI assistant answering questions based on the provided documentation.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, say so - don't make up information.

Context:
{context}

Question: {question}

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User's question

        Returns:
            Dictionary with answer, source documents, and metrics
        """
        if not self.qa_chain:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"\n🔍 Querying Traditional RAG: {question}")
        start_time = time.time()

        # Execute query
        result = self.qa_chain.invoke({"query": question})

        query_time = time.time() - start_time

        # Extract results
        answer = result['result']
        source_docs = result['source_documents']

        # Get unique sources
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in source_docs]))

        # Calculate metrics
        num_tokens = len(answer.split())
        num_chunks = len(source_docs)

        return {
            "answer": answer,
            "source_documents": source_docs,
            "sources_used": sources,
            "metrics": {
                "query_time": query_time,
                "num_source_chunks": num_chunks,
                "answer_tokens": num_tokens,
                "retrieval_method": "vector_similarity"
            }
        }

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search without generation.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Index not built. Call build_index() first.")

        return self.vectorstore.similarity_search(query, k=k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if not self.vectorstore:
            return {"status": "No index built"}
        
        return {
            "status": "Index ready",
            "total_vectors": self.vectorstore.index.ntotal,
            "embedding_dimension": self.vectorstore.index.d
        }

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"✓ Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        self._create_qa_chain()
        print(f"✓ Index loaded from {path}")
