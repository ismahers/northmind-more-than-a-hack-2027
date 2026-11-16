"""Synthetic-only tests with safe aggregate diagnostics; never read child logs."""
import importlib.util
import io
import traceback
import unittest
from possession.night.common import atomic_json
from possession.ball_attention.common import DEST,ROOT


def main():
    spec = importlib.util.spec_from_file_location("ball_attention_synthetic_tests",ROOT/"tests/test_ball_attention.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Results(unittest.TextTestResult):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.safe = []
        def addFailure(self,test,err):
            super().addFailure(test,err);self.record(test,err,"failure")
        def addError(self,test,err):
            super().addError(test,err);self.record(test,err,"error")
        def record(self,test,err,kind):
            # No messages, input tensors, paths from assets, or source context.
            self.safe.append({"test":test.id(),"kind":kind,"type":err[0].__name__,
                "locations":[{"function":f.name,"line":f.lineno} for f in traceback.extract_tb(err[2])]})
    result = unittest.TextTestRunner(stream=io.StringIO(),resultclass=Results,verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    atomic_json(DEST/"synthetic-tests.json",{"passed":result.wasSuccessful(),"tests":result.testsRun,"failures":result.safe,"diagnostics":module.DIAGNOSTICS})
    return 0 if result.wasSuccessful() else 1
