"""Read-only native PDH samples. Protocol scaffold; not GPU qualification."""


class PdhGpuQuery:
    def __init__(self, pid, backend=None):
        raise NotImplementedError('Persistent native PDH observation is not implemented')
