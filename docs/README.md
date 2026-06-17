# monte docs

This folder is the Quarto website root for `monte` examples and report outputs.

## Rendering the site

Quarto is distributed as a CLI tool rather than a normal Python package dependency. Install the Quarto CLI from <https://quarto.org/docs/get-started/> or through your system/tool manager, then render the site from the repository root:

```bash
uv run --group docs quarto render docs
```

The rendered website is written to `docs/_site/`.

The examples assume the project environment is available. If needed, run from an activated `uv` environment or install the package in editable mode for local documentation work.
