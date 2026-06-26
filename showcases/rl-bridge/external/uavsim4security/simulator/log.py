import logging
from utils import config


# config logging
logging.basicConfig(filename=config.LOG_FILE,
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )

logger = logging.getLogger()
