from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("./data")
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEED = 123
