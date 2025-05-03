# Eval DSP

Goal of this project is to evaluate DSPy, observe its optimizations, prompts etc and compare it with a manually written prompt.

This project uses UV for dependency management and primarily thought to be a more command line approach.

## Prerequisites

- Python 3.11 or higher
- UV package installer

## Setup Instructions

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd eval-dsp
   ```

3. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

## Project Structure

- `hello.py` and `hello2.py`: Main Python scripts
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Lock file for dependency versions

## Dependencies

The project uses the following main dependencies:
- dspy >= 2.6.22

## Development

To add new dependencies:
1. Add them to `pyproject.toml`
2. Run `uv pip install -e .` to update the environment


# Executions

```sh
uv run hello2.py
```

A sample output is in `hello2_output.txt`