from enum import Enum

class WorkerInstruction(Enum):
    ADD_GRAPHS = 0
    EXIT = 2
    TRAIN = 3
    GET = 4
    SET = 5
    EXPLORE = 6
    GET_TRAIN_LOG = 7