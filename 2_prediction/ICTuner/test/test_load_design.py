import unittest
import os
import pathlib

from ictuner.log import get_logger
from ictuner import read_netlist

logger = get_logger()

class TestReadDesign(unittest.TestCase):
    def test_01(self):
        lef_file = '/local-disk/tools/TSMC65LP/tsmc/merged.lef'
        def_file = os.path.join(pathlib.Path.home(), 'ICCAD20', 'data', 'test', 'test.v.def')

        result = read_netlist(lef_file, def_file)

        self.assertIsNotNone(result)