"""CLI entry point for zotqa."""

import argparse
import json
import sys
from pathlib import Path

from zotqa.extract import export_papers, get_papers
from zotqa.prompts import get_user_prompts_dir, init_user_prompts
from zotqa.rag import (
    QueryEngine,
    VectorIndex,
    detect_embedding_provider,
    format_answer,
    get_default_index_dir,
    get_embedding_adapter,
)


def cmd_export(args: argparse.Namespace) -> int:
    """Export Zotero library to RAG-friendly format."""
    if not args.db.exists():
        print(f"Error: Database not found: {args.db}")
        return 1

    papers = get_papers(
        args.db,
        args.storage,
        tags=args.tag,
        collections=args.collection,
    )

    if not papers:
        print("No papers found matching the criteria.")
        return 1

    export_papers(papers, args.output)
    print(f"Exported {len(papers)} papers to {args.output}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build vector index from exported corpus."""
    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}")
        return 1

    index_dir = Path(args.index_dir) if args.index_dir else get_default_index_dir()

    print(f"Building index from {corpus_dir}...")

    # Determine embedding provider
    provider = args.embedding_provider
    if provider == "auto":
        try:
            provider = detect_embedding_provider()
            print(f"Using embedding provider: {provider}")
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1

    try:
        embedding_adapter = get_embedding_adapter(provider)
    except Exception as e:
        print(f"Error initializing embedding provider '{provider}': {e}")
        return 1

    index = VectorIndex(index_dir)

    try:
        stats = index.build(
            corpus_dir,
            embedding_adapter,
            force_rebuild=args.force,
            include_pdf_body=not args.no_pdf,
        )
        print("Index built successfully:")
        print(f"  Papers: {stats.num_papers}")
        print(f"  Chunks: {stats.num_chunks}")
        print(f"  Embedding dim: {stats.embedding_dim}")
        print(f"  Saved to: {index_dir}")
        return 0
    except Exception as e:
        print(f"Error building index: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show index statistics."""
    index_dir = Path(args.index_dir) if args.index_dir else get_default_index_dir()

    index = VectorIndex(index_dir)
    if not index.load():
        print(f"No index found at {index_dir}")
        print("Run 'zotqa index <corpus_dir>' to build an index.")
        return 1

    stats = index.get_stats()
    if stats:
        print("Index Statistics:")
        print(f"  Papers: {stats.num_papers}")
        print(f"  Chunks: {stats.num_chunks}")
        print(f"  Embedding dim: {stats.embedding_dim}")
        print(f"  Indexed at: {stats.indexed_at}")
        print(f"  Corpus: {stats.corpus_path}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit chat UI."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit not installed. Install with: pip install zotqa[ui]")
        return 1

    import sys
    from pathlib import Path

    ui_path = Path(__file__).parent / "ui.py"
    sys.argv = ["streamlit", "run", str(ui_path), "--server.headless=true"]
    stcli.main()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Query the corpus with an LLM-generated answer."""
    index_dir = Path(args.index_dir) if args.index_dir else get_default_index_dir()

    # Check if index exists before trying to load embedding provider
    index = VectorIndex(index_dir)
    if not index.load():
        print(f"Error: No index found at {index_dir}. Run 'zotqa index' first.")
        return 1

    try:
        engine = QueryEngine(index_dir=index_dir)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    # Build paper filter if specified
    paper_filter = None
    if args.paper:
        paper_filter = set(args.paper)

    result = engine.query(
        question=args.question,
        max_chunks=args.max_chunks,
        max_tokens=args.max_tokens,
        paper_filter=paper_filter,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_answer(result, show_chunks=not args.quiet, show_tokens=not args.quiet))

    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Launch interactive terminal chat UI."""
    from zotqa.chat import run_chat

    index_dir = Path(args.index_dir) if args.index_dir else None
    paper_filter = set(args.paper) if args.paper else None

    return run_chat(
        index_dir=index_dir,
        max_chunks=args.max_chunks,
        max_tokens=args.max_tokens,
        paper_filter=paper_filter,
    )


def cmd_init_prompts(args: argparse.Namespace) -> int:
    """Copy default prompts to user config directory for customization."""
    prompts_dir = get_user_prompts_dir()

    try:
        result_dir = init_user_prompts(force=args.force)
        print(f"Prompts initialized at: {result_dir}")
        print("\nYou can now customize these prompts:")
        for prompt_file in sorted(result_dir.glob("*.md")):
            print(f"  - {prompt_file.name}")
        return 0
    except FileExistsError:
        print(f"Error: Prompts already exist at {prompts_dir}")
        print("Use --force to overwrite.")
        return 1
    except Exception as e:
        print(f"Error initializing prompts: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query your Zotero library with LLM assistance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export Zotero library to RAG format")
    export_parser.add_argument("output", type=Path, help="Output directory")
    export_parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / "Zotero" / "zotero.sqlite",
        help="Path to Zotero database",
    )
    export_parser.add_argument(
        "--storage",
        type=Path,
        default=Path.home() / "Zotero" / "storage",
        help="Path to Zotero storage directory",
    )
    export_parser.add_argument("--tag", action="append", help="Filter by tag (can specify multiple)")
    export_parser.add_argument("--collection", action="append", help="Filter by collection (can specify multiple)")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build vector index from exported corpus")
    index_parser.add_argument("corpus", help="Path to exported corpus directory")
    index_parser.add_argument("--index-dir", help="Directory to store index (default: ~/.zotqa/index)")
    index_parser.add_argument(
        "--embedding-provider",
        choices=["auto", "voyage", "openai"],
        default="auto",
        help="Embedding provider (default: auto-detect based on API keys)",
    )
    index_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force full rebuild (ignore incremental updates)",
    )
    index_parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF body text extraction (only index notes, abstract, metadata)",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show index statistics")
    info_parser.add_argument("--index-dir", help="Directory containing index (default: ~/.zotqa/index)")

    # UI command
    subparsers.add_parser("ui", help="Launch the chat UI")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Query your library")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--index-dir", help="Directory containing index (default: ~/.zotqa/index)")
    ask_parser.add_argument("--paper", action="append", help="Filter to specific paper ID(s)")
    ask_parser.add_argument("--max-chunks", type=int, default=10, help="Max chunks to include (default: 10)")
    ask_parser.add_argument("--max-tokens", type=int, default=2048, help="Max response tokens (default: 2048)")
    ask_parser.add_argument("--json", action="store_true", help="Output as JSON")
    ask_parser.add_argument("-q", "--quiet", action="store_true", help="Only show answer, no metadata")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat with your library")
    chat_parser.add_argument("--index-dir", help="Directory containing index (default: ~/.zotqa/index)")
    chat_parser.add_argument("--paper", action="append", help="Filter to specific paper ID(s)")
    chat_parser.add_argument("--max-chunks", type=int, default=10, help="Max chunks per query (default: 10)")
    chat_parser.add_argument("--max-tokens", type=int, default=2048, help="Max response tokens (default: 2048)")

    # Init prompts command
    init_prompts_parser = subparsers.add_parser(
        "init-prompts", help="Copy default prompts to user config directory for customization"
    )
    init_prompts_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing user prompts")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "export":
        return cmd_export(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "ui":
        return cmd_ui(args)
    elif args.command == "ask":
        return cmd_ask(args)
    elif args.command == "chat":
        return cmd_chat(args)
    elif args.command == "init-prompts":
        return cmd_init_prompts(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
