import importlib.util
import os


def parse_class_from_file(s: str) -> object:
    f, c = s.split(":")
    path = f + ".py"
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location(
            os.path.dirname(path).replace(os.sep, "."),
            path,
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, c)

    return None
