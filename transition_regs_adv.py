import os
import sys
import importlib.util

def get_transitions():
    path = "transition_regs"
    abs_path = os.path.abspath(path)
    sys.path.insert(0, abs_path)  # allow imports within that directory

    result = {}

    for file in os.listdir(abs_path):
        if file.endswith(".py"):
            name = file[:-3]
            filepath = os.path.join(abs_path, file)

            spec = importlib.util.spec_from_file_location(name, filepath)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "TRANSITIONS"):
                    result[name] = getattr(module, "TRANSITIONS")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    sys.path.pop(0)  # optional cleanup
    return result
