"""Knowledge Graph RAG Pipeline using Graphiti and Neo4j.
Modified to support multiple file types (txt, pdf, docx, md)
"""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

# Document Loaders for different file types
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeGraphRAG:
    """Knowledge Graph-based RAG system using Graphiti with multi-file support."""

    # Supported file extensions and their loaders
    SUPPORTED_LOADERS = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader
    }

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str,
        model_name: str = "gpt-4-turbo-preview",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Knowledge Graph RAG system.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
            model_name: LLM model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

        # Initialize Graphiti with new API (v0.3.6+)
        from graphiti_core.llm_client import OpenAIClient
        from graphiti_core.llm_client.config import LLMConfig

        llm_config = LLMConfig(
            api_key=openai_api_key,
            model=model_name,
            max_tokens=4096
        )
        llm_client = OpenAIClient(config=llm_config)

        self.graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=llm_client
        )

        # Initialize LLM for response generation
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=openai_api_key
        )

        print("✓ Knowledge Graph RAG initialized")

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

    def load_file(self, file_path: str) -> List[str]:
        """
        Load and chunk a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text chunks
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get appropriate loader
        loader = self.get_loader_for_file(path)
        
        # Load documents
        raw_docs = loader.load()
        
        # Split into chunks
        chunks = []
        for doc in raw_docs:
            doc_chunks = self.text_splitter.split_text(doc.page_content)
            chunks.extend(doc_chunks)
        
        print(f"✓ Loaded {len(chunks)} chunks from {path.name}")
        return chunks

    def load_multiple_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Load multiple files and return chunks organized by source.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary with source name as key and chunks as value
        """
        all_chunks = {}
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                chunks = self.load_file(file_path)
                source_name = path.stem  # filename without extension
                all_chunks[source_name] = chunks
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")
        
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        print(f"\n📚 Total: {total_chunks} chunks from {len(all_chunks)} files")
        return all_chunks

    def load_folder(self, folder_path: str, recursive: bool = False) -> Dict[str, List[str]]:
        """
        Load all supported files from a folder.
        
        Args:
            folder_path: Path to the folder
            recursive: Whether to search subfolders
            
        Returns:
            Dictionary with source name as key and chunks as value
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        supported_extensions = list(self.SUPPORTED_LOADERS.keys())
        
        # Get all files
        if recursive:
            files = [f for f in folder.rglob("*") if f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in folder.glob("*") if f.suffix.lower() in supported_extensions]
        
        if not files:
            print(f"⚠ No supported files found in {folder_path}")
            return {}
        
        print(f"\n📁 Found {len(files)} files in {folder_path}:")
        for f in files:
            print(f"   - {f.name} ({f.suffix})")
        print()
        
        return self.load_multiple_files([str(f) for f in files])

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Graph cleared")

    async def add_documents_to_graph(
        self,
        documents: List[str],
        source: str = "document"
    ) -> None:
        """
        Add documents to the knowledge graph.

        Args:
            documents: List of document chunks
            source: Source identifier for the documents
        """
        print(f"📊 Adding {len(documents)} chunks from '{source}' to knowledge graph...")
        start_time = time.time()

        for i, doc in enumerate(documents):
            # Add each document as an episode to Graphiti
            await self.graphiti.add_episode(
                name=f"{source}_chunk_{i}",
                episode_body=doc,
                source_description=f"Document chunk {i} from {source}",
                reference_time=datetime.now(),
                source=EpisodeType.text
            )

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(documents)} chunks...")

        build_time = time.time() - start_time
        print(f"✓ Added '{source}' to graph in {build_time:.2f} seconds")

    async def add_folder_to_graph(self, folder_path: str, recursive: bool = False) -> None:
        """
        Load all files from a folder and add them to the knowledge graph.
        
        Args:
            folder_path: Path to the folder
            recursive: Whether to search subfolders
        """
        # Load all files
        all_chunks = self.load_folder(folder_path, recursive)
        
        if not all_chunks:
            print("⚠ No documents to add to graph")
            return
        
        # Add each file's chunks to graph with its source name
        print("\n🔨 Building knowledge graph...")
        for source_name, chunks in all_chunks.items():
            await self.add_documents_to_graph(chunks, source=source_name)
        
        # Print final stats
        stats = self.get_graph_statistics()
        print(f"\n✓ Knowledge Graph Complete!")
        print(f"   - Total Nodes: {stats['total_nodes']}")
        print(f"   - Total Relationships: {stats['total_relationships']}")
        print(f"   - Entities: {stats['num_entities']}")
        print(f"   - Episodes: {stats['num_episodes']}")

    async def add_multiple_files_to_graph(self, file_paths: List[str]) -> None:
        """
        Load multiple specific files and add them to the knowledge graph.
        
        Args:
            file_paths: List of file paths
        """
        # Load all files
        all_chunks = self.load_multiple_files(file_paths)
        
        if not all_chunks:
            print("⚠ No documents to add to graph")
            return
        
        # Add each file's chunks to graph with its source name
        print("\n🔨 Building knowledge graph...")
        for source_name, chunks in all_chunks.items():
            await self.add_documents_to_graph(chunks, source=source_name)
        
        # Print final stats
        stats = self.get_graph_statistics()
        print(f"\n✓ Knowledge Graph Complete!")
        print(f"   - Total Nodes: {stats['total_nodes']}")
        print(f"   - Total Relationships: {stats['total_relationships']}")

    async def query(self, question: str, max_facts: int = 10) -> Dict[str, Any]:
        """
        Query the knowledge graph.

        Args:
            question: User's question
            max_facts: Maximum number of facts to retrieve

        Returns:
            Dictionary with answer, facts, and metrics
        """
        print(f"\n🔍 Querying Knowledge Graph: {question}")
        start_time = time.time()

        # Search the knowledge graph for relevant facts
        search_results = await self.graphiti.search(
            query=question,
            num_results=max_facts
        )

        retrieval_time = time.time() - start_time

        # Extract facts from search results
        facts = []
        entities = []
        relationships = []

        for result in search_results:
            if hasattr(result, 'fact'):
                facts.append(result.fact)
            if hasattr(result, 'content'):
                facts.append(result.content)

            if hasattr(result, 'nodes'):
                for node in result.nodes:
                    if hasattr(node, 'name'):
                        entities.append(node.name)

            if hasattr(result, 'edges'):
                for edge in result.edges:
                    if hasattr(edge, 'fact'):
                        relationships.append(edge.fact)

        # Build context from facts
        context = "\n\n".join(facts) if facts else "No relevant information found."

        # Generate answer using LLM
        generation_start = time.time()
        prompt = f"""You are a helpful AI assistant answering questions based on the provided documentation.

Use the following knowledge graph facts to answer the question. These facts represent relationships and entities extracted from the documentation.

Knowledge Graph Facts:
{context}

Question: {question}

Provide a comprehensive answer based on the knowledge graph facts. If the facts don't contain enough information, say so.

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        # Calculate metrics
        num_tokens = len(answer.split())
        num_facts = len(facts)
        num_entities = len(set(entities))
        num_relationships = len(relationships)

        return {
            "answer": answer,
            "facts": facts,
            "entities": list(set(entities)),
            "relationships": relationships,
            "metrics": {
                "query_time": total_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "num_facts": num_facts,
                "num_entities": num_entities,
                "num_relationships": num_relationships,
                "answer_tokens": num_tokens,
                "retrieval_method": "knowledge_graph"
            }
        }

    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific entity.

        Args:
            entity_name: Name of the entity

        Returns:
            List of relationships
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {name: $entity_name})-[r]->(target)
            RETURN e.name as source, type(r) as relationship, target.name as target
            UNION
            MATCH (source)-[r]->(e:Entity {name: $entity_name})
            RETURN source.name as source, type(r) as relationship, e.name as target
            """
            result = session.run(query, entity_name=entity_name)
            return [dict(record) for record in result]

    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            num_nodes = node_result.single()["count"]

            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            num_relationships = rel_result.single()["count"]

            # Count entities
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            num_entities = entity_result.single()["count"]

            # Count episodes
            episode_result = session.run("MATCH (n:Episode) RETURN count(n) as count")
            num_episodes = episode_result.single()["count"]

        return {
            "total_nodes": num_nodes,
            "total_relationships": num_relationships,
            "num_entities": num_entities,
            "num_episodes": num_episodes
        }

    def get_sources(self) -> List[str]:
        """Get all unique source names in the graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Episode) 
                WHERE n.name IS NOT NULL
                RETURN DISTINCT split(n.name, '_chunk_')[0] as source
            """)
            return [record["source"] for record in result]

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()
        print("✓ Neo4j connection closed")
