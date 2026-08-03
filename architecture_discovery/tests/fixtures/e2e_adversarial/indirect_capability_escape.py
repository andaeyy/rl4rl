"""Never execute: demonstrates an indirect-import candidate-contract bypass."""

import torch


def build_untrained_model(seed):
    namespace = globals()["__builtins__"]
    importer = getattr(namespace, "__import__")
    process_module = importer("sub" + "process")
    network_module = importer("sock" + "et")
    file_reader = getattr(namespace, "op" + "en")
    environment_module = importer("o" + "s")
    # These references make every intended capability visible to the audit without
    # executing an attack. The current exploratory Python contract misses them.
    _ = (
        process_module,
        network_module,
        file_reader,
        environment_module,
        torch,
        seed,
    )
    raise RuntimeError("adversarial fixture must never execute")
