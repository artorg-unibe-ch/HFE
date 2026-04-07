import sys
from pathlib import Path

# Add src directory to the Python path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
