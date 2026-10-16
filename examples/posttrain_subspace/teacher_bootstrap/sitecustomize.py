"""Scoped to teacher subprocesses; no Torch import in unrelated processes."""

import os

if os.environ.get("ORBIT_POSTTRAIN_TEACHER_PREFILL_T") == "0.7":
    import importlib.abc
    import importlib.machinery
    import sys

    class _Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec) if hasattr(self.original, "create_module") else None

        def exec_module(self, module):
            self.original.exec_module(module)
            from orbit.research.posttrain.teacher import install_prefill_temperature

            install_prefill_temperature(module, 0.7, os.environ["ORBIT_POSTTRAIN_TEACHER_AUDIT"], os.environ["ORBIT_POSTTRAIN_TEACHER_MODEL"])

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.layers.logits_processor":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
