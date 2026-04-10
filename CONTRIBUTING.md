# Contributing to Fall Detection

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pkuCactus/fall_detection.git
   cd fall_detection
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   
   # Or using conda
   conda create -n fall_detection python=3.10
   conda activate fall_detection
   ```

3. **Install dependencies**
   ```bash
   # Recommended: Auto-detect CUDA version
   bash scripts/shell/install.sh
   
   # Or specify CUDA version
   pip install -e ".[torch-cu124,dev]"
   
   # Or CPU only
   pip install -e ".[torch-cpu,dev]"
   ```

4. **Verify installation**
   ```bash
   # Check Python version
   python --version  # Should be 3.10+
   
   # Check PyTorch
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   
   # Check project package
   python -c "import fall_detection; print('✓ Package installed')"
   ```

5. **Set up PYTHONPATH**
   ```bash
   # Add to your shell profile (~/.bashrc or ~/.zshrc)
   export PYTHONPATH=src
   ```

## Coding Standards

### Python Style

- **Line length**: Maximum 120 characters
- **Indentation**: 4 spaces
- **Quotes**: Use double quotes for strings, single quotes for internal keys
- **Imports**: Absolute imports using `from fall_detection.x import ...`

### Code Formatting

We use the following tools:
- **Ruff** for linting
- **Black** for formatting (optional)

Run linting before committing:
```bash
ruff check src/ scripts/ tests/
```

### Documentation

- Use docstrings for all public functions and classes
- Follow Google style for docstrings:
  ```python
  def function(arg1: str, arg2: int) -> bool:
      """Short description.
      
      Args:
          arg1: Description of arg1.
          arg2: Description of arg2.
      
      Returns:
          Description of return value.
      
      Raises:
          ValueError: If arg1 is empty.
      """
  ```

### Compact Code Style

As specified in `CLAUDE.md`:
- Write highly compact code
- Avoid unnecessary blank lines between statements
- Use one-liners (comprehensions, ternary operators) where appropriate
- Max line length: 120 characters

## Testing Guidelines

### Test-Driven Development (TDD)

This project follows TDD principles:
1. **Write tests first** before implementing features
2. **Run tests frequently** during development
3. **All tests must pass** before merging PRs

### Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run with coverage (minimum 90%)
PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-fail-under=90

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_detector.py -v

# Run only unit tests
PYTHONPATH=src pytest tests/unit/ -v

# Run only integration tests
PYTHONPATH=src pytest tests/integration/ -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use `pytest` framework with fixtures
- Mock external dependencies (YOLO models, CUDA, file I/O)
- Use `tmp_path` fixture for temporary files
- Use `mocker` fixture from pytest-mock for patching

### Coverage Requirements

- **Line coverage**: minimum 90%
- **Branch coverage**: minimum 85%
- **Critical paths** (pipeline, rules, fusion): 100%

## Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow coding standards
   - Write/update tests
   - Update documentation if needed

3. **Run all checks**
   ```bash
   # Run tests
   PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-fail-under=90
   
   # Run linter
   ruff check src/ scripts/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `refactor:` for code refactoring
   - `test:` for test updates
   - `chore:` for maintenance tasks

### Submitting the PR

1. Push your branch to GitHub
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub
   - Provide a clear title and description
   - Reference any related issues
   - Wait for review

### Review Process

- PRs require at least one approval
- All CI checks must pass
- Coverage must not decrease
- Reviewers may request changes

## Issue Reporting

### Bug Reports

When reporting bugs, please include:
- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, PyTorch version, OS
- **Logs**: Relevant error messages or logs

### Feature Requests

For feature requests, please include:
- **Use case**: Why this feature is needed
- **Proposed solution**: How it should work
- **Alternatives**: Other approaches considered

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Project Structure

Understanding the project structure helps you contribute effectively:

```
fall_detection/
├── src/fall_detection/    # Core source code
│   ├── core/              # Core components (detector, tracker, etc.)
│   ├── models/            # Model definitions
│   ├── data/              # Data processing
│   ├── pipeline/          # Pipeline orchestration
│   └── utils/             # Utility functions
├── scripts/               # Executable scripts
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   ├── demo/              # Demo scripts
│   └── tools/             # Utility tools
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── configs/               # Configuration files
│   ├── training/          # Training configs
│   ├── pipeline/          # Pipeline configs
│   └── tools/             # Tool configs
├── docs/                  # Documentation
└── data/                  # Data directory (not in Git)
```

## Additional Resources

- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [README.md](README.md) - Project overview and usage
- [CLAUDE.md](CLAUDE.md) - AI assistant integration guide
- [docs/design/](docs/design/) - Design documentation
- [docs/troubleshooting/](docs/troubleshooting/) - Troubleshooting guides

## Questions?

If you have questions, feel free to:
- Open an issue with the `question` label
- Check existing documentation
- Review closed issues for similar questions

Thank you for contributing!