from dotenv import load_dotenv
load_dotenv()

import os

FASTAPI_URL = os.getenv('FASTAPI_URL')

BASE_URL = os.getenv("BASE_URL")
UDSA_API_KEY = os.getenv("USDA_API_KEY")