# Installation Guide

## Prerequisites

- Mojo compiler (version 0.6.0 or later recommended)
- A system with support for Mojo compilation

## Installing Mojo

See the official Mojo installation guide: [Mojo Installation](https://docs.modular.com/stable/mojo/manual/install)

## Getting sklearn-mojo

### Clone the Repository

```bash
git clone https://github.com/MengAiDev/sklearn-mojo.git
cd sklearn-mojo
```

### Project Structure

After cloning, your directory should look like this:

```
sklearn-mojo/
├── sklearn_mojo/          # Main package
│   ├── __init__.mojo     # Package exports
|   ├── ...
├── docs/                  # Documentation
└── README.md             # Project overview
```

## Development Setup

### Optional: Virtual Environment

While not required, you may want to set up a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac

# Install dependency
pip install -r requirements.txt
```

## Troubleshooting

### Mojo Not Found

If `mojo` command is not found:

1. Check if Mojo is in your PATH
2. Verify installation: `~/.modular/bin/mojo --version`
3. Add to PATH: `export PATH="$HOME/.modular/bin:$PATH"`

### Compilation Errors

If you encounter compilation errors:

1. Ensure you're using a recent version of Mojo
2. Check that all files are present in the project directory
3. Verify the directory structure matches the expected layout

### Performance Issues

For better performance:

1. Ensure Mojo compiler is up to date
2. Use optimized build flags if available
3. Consider using release builds for production

## Next Steps

After successful installation:

1. Read the [Usage Guide](usage.md) for examples
2. Check the [API Reference](api_reference.md) for detailed function documentation
3. Explore the source code in `sklearn_mojo/` directory
