# ESI - Extended Simulation Interface

This project provides an interface for Isaac Sim robotics simulation with mission management and robot logging capabilities.

## Auto-Completion Setup

This project has been configured to provide better auto-completion support in your IDE. Here's what has been set up:

### 1. Project Configuration Files

- **`requirements.txt`**: Lists all Python dependencies
- **`pyproject.toml`**: Modern Python project configuration with development tools
- **`.vscode/settings.json`**: VS Code configuration for Python language server

### 2. Type Hints

All modules now include comprehensive type hints:
- `mission.py`: Mission management classes with full type annotations
- `robot_logger.py`: Robot trajectory logging with typed parameters
- `isaac_sim_stubs.py`: Type stubs for Isaac Sim modules

### 3. IDE Configuration

The project includes VS Code settings that enable:
- Auto-import completions
- Type checking
- Function parameter hints
- Return type hints
- Import organization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For development tools:
```bash
pip install -e ".[dev]"
```

## Testing Auto-Completion

Run the test script to verify everything works:
```bash
python3 test_autocomplete.py
```

## IDE Setup

### VS Code
The project includes `.vscode/settings.json` with optimal Python language server settings.

### Other IDEs
Make sure your IDE is configured to:
- Use Python 3.8+ 
- Enable type checking
- Use the project root as the Python path
- Enable auto-import completions

## Troubleshooting Auto-Completion

If you're still not getting auto-completion:

1. **Restart your IDE** after the configuration changes
2. **Reload the Python language server** (in VS Code: Ctrl+Shift+P → "Python: Restart Language Server")
3. **Check Python interpreter**: Make sure your IDE is using the correct Python interpreter
4. **Install language server**: Ensure you have a Python language server installed (Pylsp, Pyright, etc.)

## Project Structure

```
esi/
├── ESI.py                 # Main simulation class
├── mission.py             # Mission management (with type hints)
├── robot_logger.py        # Robot logging (with type hints)
├── esi_extension.py       # Isaac Sim extension
├── isaac_sim_stubs.py     # Type stubs for Isaac Sim
├── test_autocomplete.py   # Auto-completion test script
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
├── .vscode/
│   └── settings.json      # VS Code settings
└── missions/
    └── mission_1.yaml     # Mission configuration
```

## Usage

The main classes provide full auto-completion support:

```python
from mission import Mission, MissionType, Waypoint, StatusType
from robot_logger import RobotLogger

# Auto-completion will show available enum values
mission = Mission(1, MissionType.MOVE_TO_WAYPOINT, Waypoint(10, 20))

# Auto-completion will show method signatures
logger = RobotLogger(log_interval=0.1, stop_logging_time=15.0)
```

## Isaac Sim Integration

This project is designed to work with Isaac Sim. The `isaac_sim_stubs.py` file provides type information for Isaac Sim modules that may not be available in your development environment.
