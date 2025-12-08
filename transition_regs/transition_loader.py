import os
import sys
import importlib.util

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

spec = importlib.util.spec_from_file_location("transitions", os.path.join(parent_dir, "transitions.py"))
transitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transitions)

sys.path.pop(0)
