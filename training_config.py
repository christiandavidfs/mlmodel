"""Auto-detected training configuration based on hardware."""
from setup_config import SetupConfig

# Initialize SetupConfig and auto-detect optimal configuration
_setup = SetupConfig()
TRAINING_CONFIG = _setup.detect_optimal_config()
