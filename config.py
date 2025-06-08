import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)
