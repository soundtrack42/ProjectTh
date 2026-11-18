"""Minimal stand-in: MASA only uses TensorFlow for TensorBoard summaries, which these tests do not write."""
class _Summary:
    SummaryWriter = object
    def create_file_writer(self, *a, **k): raise RuntimeError("TensorBoard logging disabled (tensorflow stub)")
    def scalar(self, *a, **k): pass
    def histogram(self, *a, **k): pass
summary = _Summary()
class _IO:
    class gfile:
        @staticmethod
        def exists(p):
            import os; return os.path.exists(p)
io = _IO()
