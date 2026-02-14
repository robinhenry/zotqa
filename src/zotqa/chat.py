"""Terminal UI for interactive chat with zotqa."""

import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from zotqa.prompts import get_system_prompt, get_user_prompt_template
from zotqa.rag.embed import detect_embedding_provider, get_embedding_adapter
from zotqa.rag.index import VectorIndex, get_default_index_dir
from zotqa.rag.llm import get_llm_adapter
from zotqa.rag.query import QueryEngine, _build_citation_index, _build_context
from zotqa.rag.retrieve import Retriever

console = Console()

HEADER = """[bold cyan]zotqa chat[/bold cyan] [dim]— ask questions about your Zotero library[/dim]
[dim]Type your question and press Enter. Use Ctrl+C or type /exit to quit.[/dim]"""


def _setup_readline_history() -> None:
    """Set up readline for input history if available."""
    try:
        import readline
        import atexit

        history_file = Path.home() / ".zotqa" / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        if history_file.exists():
            readline.read_history_file(str(history_file))

        readline.set_history_length(500)
        atexit.register(readline.write_history_file, str(history_file))
    except ImportError:
        pass


def _get_input(prompt: str) -> str:
    """Get input from the user with a styled prompt."""
    try:
        # Print the prompt without newline using rich, then use input()
        console.print(f"\n[bold green]You[/bold green]", end="")
        return input(" › ")
    except EOFError:
        return "/exit"


def _stream_response(client, model: str, messages: list[dict], system: str, max_tokens: int) -> tuple[str, int, int]:
    """Stream a response from Anthropic and print it in real time. Returns (content, input_tokens, output_tokens)."""
    import anthropic

    full_content = []
    input_tokens = 0
    output_tokens = 0

    console.print(f"\n[bold blue]Assistant[/bold blue] [dim]▸[/dim]")

    with console.status("[dim]Thinking...[/dim]", spinner="dots"):
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
        ) as stream:
            for text in stream.text_stream:
                full_content.append(text)

            final_msg = stream.get_final_message()
            input_tokens = final_msg.usage.input_tokens
            output_tokens = final_msg.usage.output_tokens

    content = "".join(full_content)
    console.print(Markdown(content))
    return content, input_tokens, output_tokens


def run_chat(
    index_dir: Path | None = None,
    max_chunks: int = 10,
    max_tokens: int = 2048,
    paper_filter: set[str] | None = None,
) -> int:
    """Run the interactive chat loop."""
    # Setup
    _setup_readline_history()
    index_dir = index_dir or get_default_index_dir()

    # Load index
    index = VectorIndex(index_dir)
    if not index.load():
        console.print(f"[red]Error:[/red] No index found at {index_dir}. Run 'zotqa index' first.")
        return 1

    # Init engine components
    try:
        provider = detect_embedding_provider()
        embedding_adapter = get_embedding_adapter(provider)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1

    retriever = Retriever(index_dir=index_dir, embedding_adapter=embedding_adapter)

    # Init Anthropic client directly for streaming
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error:[/red] ANTHROPIC_API_KEY environment variable is not set.")
            return 1
        client = anthropic.Anthropic(api_key=api_key)
        model = "claude-sonnet-4-20250514"
    except ImportError:
        console.print("[red]Error:[/red] anthropic package not installed.")
        return 1

    system_prompt = get_system_prompt()
    prompt_template = get_user_prompt_template()
    conversation: list[dict] = []
    total_input = 0
    total_output = 0

    # Print header
    console.print()
    console.print(Panel(HEADER, border_style="cyan", padding=(0, 1)))
    console.print()

    while True:
        try:
            question = _get_input("You")
        except KeyboardInterrupt:
            console.print("\n[dim]Bye![/dim]")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Bye![/dim]")
            break

        # Retrieve relevant chunks for this question
        try:
            retrieval = retriever.retrieve(
                query=question,
                top_k=max_chunks,
                paper_filter=paper_filter,
            )
        except Exception as e:
            console.print(f"[red]Retrieval error:[/red] {e}")
            continue

        if not retrieval.chunks:
            console.print("\n[yellow]No relevant content found in your library for this question.[/yellow]")
            continue

        # Build context
        corpus_path = Path(retriever.index.metadata.get("corpus_path", ""))
        citation_index = _build_citation_index(retrieval.paper_ids, corpus_path)
        context = _build_context(retrieval, citation_index)
        user_prompt = prompt_template.format(question=question, context=context)

        # Build messages (include history + current)
        messages = list(conversation) + [{"role": "user", "content": user_prompt}]

        # Stream response
        try:
            content, inp, out = _stream_response(client, model, messages, system_prompt, max_tokens)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            continue

        total_input += inp
        total_output += out

        # Show token info
        console.print(
            f"[dim]tokens: {inp} in / {out} out  |  session total: {total_input} in / {total_output} out[/dim]"
        )

        # Maintain conversation history (use the raw question for history, not the full prompt with context)
        conversation.append({"role": "user", "content": user_prompt})
        conversation.append({"role": "assistant", "content": content})

    return 0
