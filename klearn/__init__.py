
def _setup_logging():
    import logging
    import sys

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False

_setup_logging()

from learners.ktica import ktICA
from learners.kpca import kPCA
