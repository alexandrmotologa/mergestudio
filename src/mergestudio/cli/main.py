"""Rich command line interface for MergeStudio built with Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from mergestudio.core.config import (
    DareConfig,
    DType,
    LayerSlice,
    MergeMethod,
    MergePlan,
    ModelSource,
    SlerpConfig,
    StreamingConfig,
    TiesConfig,
)
from mergestudio.streaming.engine import ModelIndex, StreamingMergeEngine

app = typer.Typer(
    name="mergestudio",
    help="MergeStudio: Visual model merging, layer surgery, and weight interpolation studio for local LLMs.",
    add_completion=False,
)
console = Console()


@app.command()
def merge(
    config: Path = typer.Option(..., "--config", "-c", help="Path to merge plan YAML or JSON file"),
) -> None:
    """Execute a model merge operation according to a configuration file."""
    if not config.exists():
        console.print(f"[red]Error: Config file not found at {config}[/red]")
        raise typer.Exit(1)

    try:
        if config.suffix.lower() in (".yaml", ".yml"):
            plan = MergePlan.load_yaml(config)
        else:
            with open(config, "r", encoding="utf-8") as f:
                plan = MergePlan.load_json(json.load(f))
    except Exception as exc:
        console.print(f"[red]Failed to parse configuration: {exc}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]MergeStudio Execution Engine[/bold cyan]\n"
            f"Plan: [green]{plan.name}[/green]\n"
            f"Method: [yellow]{plan.method.value}[/yellow]\n"
            f"Output: [magenta]{plan.output_dir}[/magenta]",
            border_style="cyan",
        )
    )

    _run_engine(plan)


@app.command()
def slerp(
    model_a: Path = typer.Option(..., "--model-a", "-a", help="Path to first model"),
    model_b: Path = typer.Option(..., "--model-b", "-b", help="Path to second model"),
    alpha: float = typer.Option(0.5, "--alpha", help="Interpolation parameter in [0.0, 1.0]"),
    output: Path = typer.Option(Path("./output/slerp_merged"), "--output", "-o", help="Output directory"),
    per_row: bool = typer.Option(False, "--per-row", help="Compute SLERP row wise"),
    preserve_norm: bool = typer.Option(True, "--preserve-norm/--no-preserve-norm", help="Preserve vector norms"),
) -> None:
    """Merge two models using Spherical Linear Interpolation (SLERP)."""
    plan = MergePlan(
        name="cli-slerp",
        method=MergeMethod.SLERP,
        models=[
            ModelSource(path=str(model_a), name="model_a"),
            ModelSource(path=str(model_b), name="model_b"),
        ],
        slerp_config=SlerpConfig(alpha=alpha, per_row=per_row, preserve_norm=preserve_norm),
        output_dir=str(output),
    )
    _run_engine(plan)


@app.command()
def ties(
    base: Path = typer.Option(..., "--base", help="Path to base anchor model"),
    donors: List[Path] = typer.Option(..., "--donor", "-d", help="Paths to fine tuned donor models"),
    negatives: Optional[List[Path]] = typer.Option(None, "--negative", "-n", help="Paths to models to unlearn / negate"),
    density: float = typer.Option(0.2, "--density", help="Fraction of top weights to retain (0.0 to 1.0)"),
    weight: float = typer.Option(1.0, "--weight", help="Global scaling lambda multiplier"),
    output: Path = typer.Option(Path("./output/ties_merged"), "--output", "-o", help="Output directory"),
) -> None:
    """Merge multiple fine-tuned models into a base anchor model using TIES-Merging with optional unlearning."""
    models = [ModelSource(path=str(base), name="base", is_base=True)]
    for i, d in enumerate(donors, start=1):
        models.append(ModelSource(path=str(d), name=f"donor_{i}", is_negative=False))
    if negatives:
        for j, neg in enumerate(negatives, start=1):
            models.append(ModelSource(path=str(neg), name=f"negative_{j}", is_negative=True))

    plan = MergePlan(
        name="cli-ties",
        method=MergeMethod.TIES,
        models=models,
        base_model="base",
        ties_config=TiesConfig(density=density, weight=weight),
        output_dir=str(output),
    )
    _run_engine(plan)


@app.command()
def dare(
    base: Path = typer.Option(..., "--base", help="Path to base anchor model"),
    donors: List[Path] = typer.Option(..., "--donor", "-d", help="Paths to fine tuned donor models"),
    negatives: Optional[List[Path]] = typer.Option(None, "--negative", "-n", help="Paths to models to unlearn / negate"),
    drop_rate: float = typer.Option(0.5, "--drop-rate", "-p", help="Delta drop probability"),
    rescale: bool = typer.Option(True, "--rescale/--no-rescale", help="Rescale surviving deltas"),
    output: Path = typer.Option(Path("./output/dare_merged"), "--output", "-o", help="Output directory"),
) -> None:
    """Merge models using Drop And REscale (DARE) with optional task vector negation."""
    models = [ModelSource(path=str(base), name="base", is_base=True)]
    for i, d in enumerate(donors, start=1):
        models.append(ModelSource(path=str(d), name=f"donor_{i}", is_negative=False))
    if negatives:
        for j, neg in enumerate(negatives, start=1):
            models.append(ModelSource(path=str(neg), name=f"negative_{j}", is_negative=True))

    plan = MergePlan(
        name="cli-dare",
        method=MergeMethod.DARE_TIES,
        models=models,
        base_model="base",
        dare_config=DareConfig(drop_rate=drop_rate, rescale=rescale),
        output_dir=str(output),
    )
    _run_engine(plan)


@app.command()
def analyze_divergence(
    model_a: Path = typer.Option(..., "--model-a", "-a", help="Path to reference or base model"),
    model_b: Path = typer.Option(..., "--model-b", "-b", help="Path to fine tuned or secondary model"),
) -> None:
    """Analyze tensor drift, cosine similarity, and calculate adaptive schedule curves across layers."""
    from mergestudio.analysis.divergence import analyze_layer_divergence

    if not model_a.exists() or not model_b.exists():
        console.print("[red]Error: Model paths must exist[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing layer divergence between {model_a.name} and {model_b.name}...[/cyan]")
    report = analyze_layer_divergence(model_a, model_b)

    table = Table(title=f"Layer Divergence Report: {report.model_a} vs {report.model_b}")
    table.add_column("Layer", style="cyan")
    table.add_column("Cosine Sim", style="green")
    table.add_column("Frobenius Drift", style="yellow")
    table.add_column("Angular Dist", style="magenta")
    table.add_column("Adaptive Weight", style="bold green")

    for m, w in zip(report.layer_metrics[:32], report.adaptive_schedule_weights[:32]):
        color = "green" if m.cosine_similarity > 0.98 else ("yellow" if m.cosine_similarity > 0.90 else "red")
        table.add_row(
            f"Layer {m.layer_idx}",
            f"[{color}]{m.cosine_similarity:.4f}[/{color}]",
            f"{m.frobenius_drift:.4f}",
            f"{m.angular_distance:.4f}",
            f"{w:.2f}",
        )

    console.print(table)
    console.print(
        Panel.fit(
            f"[bold]Mean Cosine Similarity:[/bold] {report.mean_cosine_similarity:.4f}\n"
            f"[bold]Highest Drift Layer:[/bold] Layer {report.max_divergence_layer}\n"
            f"[bold]Adaptive Weights Recommendation:[/bold] {report.adaptive_schedule_weights[:5]}...",
            border_style="cyan",
        )
    )


@app.command()
def export_ollama(
    model_dir: Path = typer.Argument(..., help="Path to merged model directory"),
) -> None:
    """Generate production ready Ollama Modelfile and GGUF quantization script."""
    from mergestudio.export.ollama import generate_gguf_conversion_script, generate_ollama_modelfile

    if not model_dir.exists():
        console.print(f"[red]Error: Directory does not exist: {model_dir}[/red]")
        raise typer.Exit(1)

    modelfile = generate_ollama_modelfile(model_dir)
    script = generate_gguf_conversion_script(model_dir)

    console.print(
        Panel.fit(
            f"[bold green]Ollama & GGUF Artifacts Generated[/bold green]\n"
            f"Modelfile: [cyan]{modelfile}[/cyan]\n"
            f"Quantization Script: [cyan]{script}[/cyan]\n\n"
            f"[yellow]To register in Ollama:[/yellow]\n"
            f"  ollama create my-merged-model -f {modelfile.name}",
            border_style="green",
        )
    )


@app.command()
def evaluate(
    model_dir: Path = typer.Argument(..., help="Path to merged model directory"),
) -> None:
    """Audit model parameter integrity, NaNs, infinities, and numerical stability."""
    from mergestudio.evaluation.smoke_test import evaluate_model_health

    if not model_dir.exists():
        console.print(f"[red]Error: Model directory does not exist: {model_dir}[/red]")
        raise typer.Exit(1)

    report = evaluate_model_health(model_dir)

    table = Table(title=f"Model Health Audit: {model_dir.name}")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")

    table.add_row("Total Tensors", str(report.total_tensors))
    table.add_row("Total Parameters", f"{report.total_parameters:,}")
    table.add_row("NaN Check", "[green]Passed (No NaNs)[/green]" if not report.has_nan else "[red]FAILED[/red]")
    table.add_row("Inf Check", "[green]Passed (No Infs)[/green]" if not report.has_inf else "[red]FAILED[/red]")
    table.add_row("Mean Weight Norm", f"{report.mean_weight_norm:.4f}")
    table.add_row("Overall Status", f"[bold green]{report.status}[/bold green]" if report.healthy else f"[bold red]{report.status}[/bold red]")

    console.print(table)
    for d in report.diagnostics:
        console.print(f"  • {d}")


@app.command()
def inspect(
    model_path: Path = typer.Argument(..., help="Path to local model directory"),
) -> None:
    """Inspect model structure, safetensors index, layers, and parameter statistics."""
    if not model_path.exists():
        console.print(f"[red]Error: Model path does not exist: {model_path}[/red]")
        raise typer.Exit(1)

    idx = ModelIndex(model_path)
    tensor_names = idx.tensor_names

    table = Table(title=f"Model Inspection: {model_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tensors", str(len(tensor_names)))

    cfg_file = model_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        table.add_row("Architecture", str(cfg.get("architectures", ["Unknown"])))
        table.add_row("Hidden Layers", str(cfg.get("num_hidden_layers", "N/A")))
        table.add_row("Hidden Size", str(cfg.get("hidden_size", "N/A")))
        table.add_row("Vocab Size", str(cfg.get("vocab_size", "N/A")))

    console.print(table)


@app.command()
def studio(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port number"),
) -> None:
    """Launch the interactive visual Web Studio."""
    from mergestudio.studio.app import create_app

    console.print(
        Panel.fit(
            f"[bold green]Starting MergeStudio Web UI[/bold green]\n"
            f"URL: [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
            f"Interactive layer surgery and weight interpolation active.",
            border_style="green",
        )
    )
    uvicorn.run(create_app(), host=host, port=port)


def _run_engine(plan: MergePlan) -> None:
    engine = StreamingMergeEngine(plan)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.fields[throughput]} MB/s"),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "Merging tensors...", total=100, throughput="0.0"
        )

        def on_progress(p):
            pct = (p.completed_tensors / max(1, p.total_tensors)) * 100
            progress_bar.update(
                task_id,
                completed=pct,
                description=f"Merging {p.current_tensor[:35]}...",
                throughput=f"{p.throughput_mb_s:.1f}",
            )

        out_path = engine.run(progress_callback=on_progress)

    console.print(f"[bold green]Success![/bold green] Merged model written to: [cyan]{out_path}[/cyan]")
