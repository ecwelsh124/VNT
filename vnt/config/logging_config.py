# vnt/config/logging_config.py

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler("vnt.log"),
        logging.StreamHandler()
    ]
)