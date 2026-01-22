#!/usr/bin/env python
"""
Generate documentation files from source code.

Outputs:
    docs/TREE.md   - Repository tree with module descriptions
    docs/API_MAP.md - Public API reference
"""

import os
import ast
from pathlib import Path
from datetime import datetime


def get_module_docstring(filepath: Path) -> str:
    """Extract module docstring from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        docstring = ast.get_docstring(tree)
        return docstring.split('\n')[0] if docstring else ""
    except Exception:
        return ""


def generate_tree(root: Path, prefix: str = "", ignore_dirs: set = None) -> list:
    """Generate tree structure with descriptions."""
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.venv', 'venv', 'runs', '.claude'}
    
    lines = []
    items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        if item.name in ignore_dirs or item.name.startswith('.'):
            continue
        
        is_last = i == len(items) - 1
        connector = "`- " if is_last else "|- "
        
        if item.is_dir():
            # Check for __init__.py docstring
            init_file = item / "__init__.py"
            desc = get_module_docstring(init_file) if init_file.exists() else ""
            desc_str = f"  # {desc}" if desc else ""
            lines.append(f"{prefix}{connector}{item.name}/{desc_str}")
            
            extension = "   " if is_last else "|  "
            lines.extend(generate_tree(item, prefix + extension, ignore_dirs))
        else:
            if item.suffix == '.py':
                desc = get_module_docstring(item)
                desc_str = f"  # {desc}" if desc else ""
                lines.append(f"{prefix}{connector}{item.name}{desc_str}")
            elif item.suffix in {'.md', '.yaml', '.yml', '.txt'}:
                lines.append(f"{prefix}{connector}{item.name}")
    
    return lines


def generate_api_map(src_root: Path) -> str:
    """Generate API documentation."""
    lines = [
        "# API Map",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Public API",
        "",
        "```python",
        "from wendling_sim import simulate, optimize",
        "```",
        "",
        "### `simulate(sim_cfg, model_cfg, network_cfg=None, stim_cfg=None, noise_cfg=None, monitor_cfg=None) -> SimResult`",
        "",
        "Run forward simulation (single node or network) with specified configuration.",
        "",
        "**Returns:** `SimResult` with fields:",
        "- `t_s`: time array (seconds)",
        "- `lfp`: LFP proxy array (time-major, shape (T, N))", 
        "- `states`: monitored state variables (optional)",
        "- `meta`: simulation metadata dict",
        "",
        "### `optimize(opt_cfg, objective_cfg, target_psd, target_freqs=None) -> OptResult`",
        "",
        "Run nevergrad optimization to find parameters minimizing PSD loss.",
        "",
        "**Returns:** `OptResult` with fields:",
        "- `best_params`: optimal parameter dict",
        "- `best_loss`: final loss value",
        "- `history`: optimization history",
        "",
        "## Config Schemas",
        "",
        "See `configs/` directory for YAML examples.",
    ]
    return "\n".join(lines)


def main():
    project_root = Path(__file__).parent.parent
    src_root = project_root / "src" / "wendling_sim"
    docs_dir = project_root / "docs"
    
    docs_dir.mkdir(exist_ok=True)
    
    # Generate TREE.md
    tree_lines = [
        "# Repository Tree",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "```",
        "brainPy_modeling/",
    ]
    tree_lines.extend(generate_tree(project_root))
    tree_lines.append("```")
    
    tree_path = docs_dir / "TREE.md"
    tree_path.write_text("\n".join(tree_lines), encoding='utf-8')
    print(f"[generated] {tree_path}")
    
    # Generate API_MAP.md
    api_content = generate_api_map(src_root)
    api_path = docs_dir / "API_MAP.md"
    api_path.write_text(api_content, encoding='utf-8')
    print(f"[generated] {api_path}")


if __name__ == "__main__":
    main()
