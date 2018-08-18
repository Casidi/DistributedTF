from __future__ import print_function

import unittest
import os
import shutil

from training_worker import TrainingWorker
import mpi4py

class TrainingWorkerTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.isdir('savedata'):
            shutil.rmtree('savedata', ignore_errors=True)
        os.mkdir('savedata')

    def test_basic(self):
        worker = TrainingWorker()
        

unittest.main(verbosity=2)