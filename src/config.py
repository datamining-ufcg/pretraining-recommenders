import multiprocessing

RANDOM_STATE = 42
EMBEDDING_SIZE = 128
LR = 0.001
REG = 0.04

EPOCHS = 100
PATIENCE = 5

MAX_THREADS = multiprocessing.cpu_count()
LOGGING_FORMAT = "(%(threadName)-9s) %(message)s"


def set_singlethreaded(args: dict):
    """
    Overrides the use of all available cores and uses a single core if
    args['--single-threaded'] is True.
    """
    global MAX_THREADS
    if args["--single-threaded"] == "True":
        MAX_THREADS = 1
