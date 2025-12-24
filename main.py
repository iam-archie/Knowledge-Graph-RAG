"""
Knowledge Graph vs Traditional RAG Demo
Modified to support multiple file types (txt, pdf, docx, md)

This script demonstrates the differences between Traditional RAG and Knowledge Graph-based RAG
using multiple document types as sample data.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from traditional_rag import TraditionalRAG
from knowledge_graph import KnowledgeGraphRAG
from comparison import compare_systems, run_comparison_suite, plot_comparison_metrics, visualize_graph

console = Console()


# Sample questions that highlight KG advantages
DEMO_QUESTIONS = [
    "How does the AuthenticationService relate to the UserManager?",
    "What services depend on the PermissionManager?",
    "Explain the file upload workflow and all the services involved.",
    "How are share links related to notifications?",
    "What is the relationship between QuotaManager and StorageManager?",
    "Which services interact with the FileManager?",
    "How does the search functionality work with permissions?"
]


def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()

    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        console.print(f"[bold red]Error: Missing required environment variables:[/bold red]")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\n[yellow]Please create a .env file based on .env.example[/yellow]")
        return False

    return True


def show_supported_formats():
    """Display supported file formats."""
    table = Table(title="Supported File Formats")
    table.add_column("Extension", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Loader", style="yellow")
    
    formats = [
        (".txt", "Plain Text", "TextLoader"),
        (".pdf", "PDF Document", "PyPDFLoader"),
        (".docx", "Word Document", "Docx2txtLoader"),
        (".md", "Markdown", "UnstructuredMarkdownLoader"),
    ]
    
    for ext, file_type, loader in formats:
        table.add_row(ext, file_type, loader)
    
    console.print(table)


async def initialize_systems():
    """Initialize both RAG systems with multi-file support."""
    console.print("\n[bold cyan]Initializing Systems...[/bold cyan]\n")

    # Load configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Initialize Traditional RAG
    console.print("[yellow]1. Initializing Traditional RAG...[/yellow]")
    rag_system = TraditionalRAG(
        openai_api_key=openai_api_key,
        model_name=model_name,
        embedding_model=embedding_model
    )

    # ============ MULTI-FILE SUPPORT ============
    data_folder = Path("sample_data")
    
    if not data_folder.exists():
        console.print(f"[bold red]Error: Sample data folder not found at {data_folder}[/bold red]")
        console.print("[yellow]Creating sample_data folder...[/yellow]")
        data_folder.mkdir(exist_ok=True)
        console.print("[yellow]Please add your documents to the sample_data folder and restart.[/yellow]")
        return None, None
    
    # Load all documents from folder
    all_documents = rag_system.load_folder(str(data_folder))
    
    if not all_documents:
        console.print("[bold red]Error: No documents found in sample_data folder[/bold red]")
        show_supported_formats()
        return None, None
    
    # Build index with all documents
    rag_system.build_index(all_documents)
    console.print("[green][OK] Traditional RAG initialized[/green]\n")

    # Initialize Knowledge Graph RAG
    console.print("[yellow]2. Initializing Knowledge Graph RAG...[/yellow]")
    kg_system = KnowledgeGraphRAG(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_username,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    # Build required Neo4j indexes and constraints
    try:
        await kg_system.graphiti.build_indices_and_constraints()
    except Exception as e:
        console.print(f"[yellow]Note: Index setup skipped (already exists): {type(e).__name__}[/yellow]")

    # Get graph statistics (ALWAYS do this after try-except)
    stats = kg_system.get_graph_statistics()
    
    # Check if we should rebuild the graph
    if stats['total_nodes'] > 0:
        console.print(f"[yellow]Found existing graph with {stats['total_nodes']} nodes[/yellow]")
        
        # Show existing sources
        sources = kg_system.get_sources()
        if sources:
            console.print(f"[yellow]Existing sources: {', '.join(sources)}[/yellow]")
        
        rebuild = Confirm.ask("Do you want to rebuild the knowledge graph?", default=False)
        if rebuild:
            kg_system.clear_graph()
            stats = kg_system.get_graph_statistics()

    # Build knowledge graph if needed
    if stats['total_nodes'] == 0:
        console.print("[yellow]Building knowledge graph (this may take a few minutes)...[/yellow]")
        
        # Add all files from folder to graph
        await kg_system.add_folder_to_graph(str(data_folder))

        stats = kg_system.get_graph_statistics()
        console.print(f"[green][OK] Knowledge Graph initialized[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")
    else:
        console.print(f"[green][OK] Using existing Knowledge Graph[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")

    return rag_system, kg_system


async def add_new_files(rag_system, kg_system):
    """Add new files to existing systems."""
    console.print("\n[bold cyan]Add New Files[/bold cyan]\n")
    
    show_supported_formats()
    
    console.print("\n[yellow]Options:[/yellow]")
    console.print("1. Add entire folder")
    console.print("2. Add specific files")
    
    choice = Prompt.ask("Select option", choices=["1", "2"])
    
    if choice == "1":
        folder_path = Prompt.ask("Enter folder path", default="sample_data")
        
        # Add to Traditional RAG
        new_docs = rag_system.load_folder(folder_path)
        if new_docs:
            rag_system.build_index(new_docs)
        
        # Add to Knowledge Graph
        await kg_system.add_folder_to_graph(folder_path)
        
    else:
        file_paths_input = Prompt.ask("Enter file paths (comma-separated)")
        file_paths = [p.strip() for p in file_paths_input.split(",")]
        
        # Add to Traditional RAG
        new_docs = rag_system.load_multiple_documents(file_paths)
        if new_docs:
            rag_system.build_index(new_docs)
        
        # Add to Knowledge Graph
        await kg_system.add_multiple_files_to_graph(file_paths)
    
    console.print("\n[green]✓ Files added successfully![/green]")


async def run_single_comparison(rag_system, kg_system):
    """Run a single question comparison."""
    console.print("\n[bold cyan]Single Question Comparison[/bold cyan]\n")

    # Show available demo questions
    console.print("[yellow]Suggested questions:[/yellow]")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"  {i}. {q}")

    console.print("\n[yellow]Enter a question number (1-7) or type your own question:[/yellow]")
    user_input = Prompt.ask("Question")

    # Parse input
    if user_input.isdigit() and 1 <= int(user_input) <= len(DEMO_QUESTIONS):
        question = DEMO_QUESTIONS[int(user_input) - 1]
    else:
        question = user_input

    # Run comparison
    await compare_systems(rag_system, kg_system, question, verbose=True)


async def run_full_comparison_suite(rag_system, kg_system):
    """Run the full comparison suite with all demo questions."""
    console.print("\n[bold cyan]Running Full Comparison Suite[/bold cyan]\n")
    console.print(f"This will test both systems with {len(DEMO_QUESTIONS)} predefined questions.\n")

    confirm = Confirm.ask("Continue?", default=True)
    if not confirm:
        return

    # Run suite
    results = await run_comparison_suite(rag_system, kg_system, DEMO_QUESTIONS)

    # Generate visualizations
    console.print("\n[yellow]Generating comparison visualizations...[/yellow]")
    plot_comparison_metrics(results, "comparison_metrics.png")
    console.print("[green][OK] Metrics plot saved to: comparison_metrics.png[/green]")


def visualize_knowledge_graph(kg_system):
    """Generate knowledge graph visualization."""
    console.print("\n[bold cyan]Generating Knowledge Graph Visualization[/bold cyan]\n")

    visualize_graph(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        output_file="knowledge_graph.html",
        max_nodes=100
    )

    console.print("[green][OK] Visualization saved to: knowledge_graph.html[/green]")
    console.print("[yellow]Open this file in a web browser to explore the graph interactively[/yellow]")


async def interactive_mode(rag_system, kg_system):
    """Run interactive question-answering mode."""
    console.print("\n[bold cyan]Interactive Mode[/bold cyan]\n")
    console.print("[yellow]Ask questions about your documents (type 'exit' to quit)[/yellow]\n")

    while True:
        question = Prompt.ask("\n[bold]Your question[/bold]")

        if question.lower() in ['exit', 'quit', 'q']:
            break

        await compare_systems(rag_system, kg_system, question, verbose=True)


def show_system_stats(rag_system, kg_system):
    """Show statistics for both systems."""
    console.print("\n[bold cyan]System Statistics[/bold cyan]\n")
    
    # Traditional RAG stats
    console.print("[yellow]Traditional RAG:[/yellow]")
    rag_stats = rag_system.get_index_stats()
    for key, value in rag_stats.items():
        console.print(f"  - {key}: {value}")
    
    # Knowledge Graph stats
    console.print("\n[yellow]Knowledge Graph RAG:[/yellow]")
    kg_stats = kg_system.get_graph_statistics()
    for key, value in kg_stats.items():
        console.print(f"  - {key}: {value}")
    
    # Show sources
    sources = kg_system.get_sources()
    if sources:
        console.print(f"\n[yellow]Indexed Sources:[/yellow]")
        for source in sources:
            console.print(f"  - {source}")


async def main():
    """Main demo function."""
    console.print(Panel.fit(
        "[bold green]Knowledge Graph vs Traditional RAG Demo[/bold green]\n"
        "Multi-File Support: txt, pdf, docx, md",
        border_style="green"
    ))

    # Setup
    if not setup_environment():
        return

    # Initialize systems
    rag_system, kg_system = await initialize_systems()
    if not rag_system or not kg_system:
        return

    # Main menu
    while True:
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]Demo Menu[/bold cyan]")
        console.print("=" * 80)
        console.print("1. Run single question comparison")
        console.print("2. Run full comparison suite (all demo questions)")
        console.print("3. Visualize knowledge graph")
        console.print("4. Interactive mode (ask your own questions)")
        console.print("5. View system statistics")
        console.print("6. Add new files")
        console.print("7. Show supported formats")
        console.print("8. Exit")

        choice = Prompt.ask("\n[bold]Select an option[/bold]", choices=["1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "1":
            await run_single_comparison(rag_system, kg_system)
        elif choice == "2":
            await run_full_comparison_suite(rag_system, kg_system)
        elif choice == "3":
            visualize_knowledge_graph(kg_system)
        elif choice == "4":
            await interactive_mode(rag_system, kg_system)
        elif choice == "5":
            show_system_stats(rag_system, kg_system)
        elif choice == "6":
            await add_new_files(rag_system, kg_system)
        elif choice == "7":
            show_supported_formats()
        elif choice == "8":
            console.print("\n[bold green]Thank you for using the demo![/bold green]")
            kg_system.close()
            break


if __name__ == "__main__":
    asyncio.run(main())