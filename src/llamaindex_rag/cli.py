"""Command-line interface for the LlamaIndex RAG system."""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import rich
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from llamaindex_rag import RagSystem, RagConfig
from llamaindex_rag.utils import load_dotenv

console = Console()
app = typer.Typer(
    name="llamarag",
    help="LlamaIndex RAG - A powerful Retrieval-Augmented Generation system",
    add_completion=False,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("llamaindex_rag")


def create_system() -> RagSystem:
    """Create and initialize the RAG system."""
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = RagConfig()
    
    # Initialize system
    return RagSystem(config)


@app.command()
def version():
    """Show version information."""
    from llamaindex_rag import __version__
    console.print(f"LlamaIndex RAG v{__version__}")


@app.command()
def index(
    input_dir: Path = typer.Argument(
        ..., 
        help="Directory containing documents to index",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    index_name: str = typer.Option(
        "default", 
        "--index-name", "-n", 
        help="Name of the index to use or create",
    ),
    file_extensions: Optional[List[str]] = typer.Option(
        None,
        "--file-ext", "-e",
        help="File extensions to include (e.g., pdf, txt)",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive", "-r/-nr",
        help="Recursively process subdirectories",
    ),
):
    """Index documents from a directory."""
    with console.status("[bold green]Indexing documents..."):
        system = create_system()
        try:
            system.index_documents(
                input_dir=input_dir,
                index_name=index_name,
                file_extns=file_extensions,
            )
            console.print(f"[bold green]✓[/] Documents indexed successfully to [bold]{index_name}[/]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            raise typer.Exit(code=1)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    index_name: str = typer.Option(
        "default", 
        "--index-name", "-n", 
        help="Name of the index to query",
    ),
    top_k: int = typer.Option(
        4, 
        "--top-k", "-k", 
        help="Number of documents to retrieve",
    ),
    mode: str = typer.Option(
        "default",
        "--mode", "-m",
        help="Query mode (default, tree, summarize)",
    ),
):
    """Query the RAG system."""
    with console.status("[bold green]Processing query..."):
        system = create_system()
        try:
            response = system.query(
                query_text=query_text,
                index_name=index_name,
                mode=mode,
                similarity_top_k=top_k,
            )
            
            console.print("\n[bold]Response:[/]")
            console.print(Panel(
                str(response),
                title="Answer",
                border_style="green",
            ))
            
            # Print source documents
            if hasattr(response, "source_nodes") and response.source_nodes:
                console.print("\n[bold]Sources:[/]")
                for i, node in enumerate(response.source_nodes):
                    metadata = node.node.metadata
                    source = metadata.get("file_name", "Unknown source")
                    console.print(f"[bold cyan]{i+1}.[/] [blue]{source}[/]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            raise typer.Exit(code=1)


@app.command()
def list(
    detailed: bool = typer.Option(
        False,
        "--detailed", "-d",
        help="Show detailed information about each index",
    ),
):
    """List all available indices."""
    with console.status("[bold green]Retrieving indices..."):
        system = create_system()
        indices = system.list_indices()
        
        if not indices:
            console.print("[yellow]No indices found.[/]")
            return
        
        if detailed:
            table = Table(title="Available Indices")
            table.add_column("Index Name", style="cyan")
            table.add_column("Path", style="green")
            table.add_column("Size", style="magenta")
            
            for index_name in indices:
                index_path = system.config.persist_dir / index_name
                size = sum(
                    f.stat().st_size for f in index_path.glob("**/*") if f.is_file()
                )
                size_str = f"{size / (1024*1024):.2f} MB"
                table.add_row(index_name, str(index_path), size_str)
            
            console.print(table)
        else:
            for index_name in indices:
                console.print(f"[cyan]● {index_name}[/]")


@app.command()
def delete(
    index_name: str = typer.Argument(..., help="Name of the index to delete"),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force deletion without confirmation",
    ),
):
    """Delete an index."""
    system = create_system()
    
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete index '{index_name}'?")
        if not confirm:
            console.print("[yellow]Deletion cancelled.[/]")
            return
    
    with console.status(f"[bold red]Deleting index '{index_name}'..."):
        success = system.delete_index(index_name)
        
        if success:
            console.print(f"[bold green]✓[/] Index '{index_name}' deleted successfully")
        else:
            console.print(f"[bold red]✗[/] Failed to delete index '{index_name}'")
            raise typer.Exit(code=1)


@app.command()
def webapp(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8501, "--port", "-p", help="Port to bind"),
):
    """Start the web UI."""
    try:
        import streamlit.web.cli as streamlit_cli
        
        # Find the webapp.py file
        import inspect
        import llamaindex_rag.webapp as webapp_module
        webapp_path = Path(inspect.getfile(webapp_module))
        
        # Use streamlit to run the webapp
        sys.argv = [
            "streamlit", "run", 
            str(webapp_path),
            "--server.address", host,
            "--server.port", str(port),
        ]
        
        console.print(f"[bold green]Starting web UI at http://{host}:{port}[/]")
        streamlit_cli.main()
    except ImportError:
        console.print("[bold red]Error:[/] Streamlit is required for the web UI.")
        console.print("Install with: [bold]pip install streamlit[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app() 