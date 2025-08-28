# Code & Docstring Style (Team Repo)

## Docstrings
Use NumPy-style docstrings and **always document tensor shapes**.

**Example**
```python
def forward(x: torch.Tensor) -> torch.Tensor:
    """Apply MLP block.

    Parameters
    ----------
    x : Tensor, shape (B, T, D)
        B = batch size, T = sequence length, D = hidden size.

    Returns
    -------
    y : Tensor, shape (B, T, D)
    """
```

## Imports
- Library modules must not contain notebook or shell magics.
- Put install commands and experiments in notebooks under `notebooks/` or scripts under `examples/`.

## File layout
- `src/subliminal/` for importable modules.
- `examples/` for runnable scripts.
- `docs/` for design notes and READMEs.
