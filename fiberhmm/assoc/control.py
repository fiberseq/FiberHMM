"""Cooperative association cancellation, local to the current call/thread/task."""
from contextvars import ContextVar
from functools import wraps
from inspect import Parameter, signature


class AssociationCancelled(RuntimeError):
    """Raised when an association cancel callback returns true."""


_control = ContextVar('association_control', default=(None, None))


def checkpoint(stage, completed=0, total=None):
    cancel, progress = _control.get()
    if cancel is not None and cancel():
        raise AssociationCancelled('association cancelled')
    if progress is not None:
        progress(stage, completed, total)


def cooperative(fn):
    """Add keyword-only cancel() and progress(stage, completed, total) callbacks.

    Nested analyses inherit callbacks. Tokens are reset even on exceptions;
    callbacks and checkpointing never consume the analysis random stream.
    """
    @wraps(fn)
    def run(*args, cancel=None, progress=None, **kwargs):
        old_cancel, old_progress = _control.get()
        token = _control.set((cancel if cancel is not None else old_cancel,
                              progress if progress is not None else old_progress))
        try:
            checkpoint(fn.__name__)
            result = fn(*args, **kwargs)
            checkpoint(fn.__name__, 1, 1)
            return result
        finally:
            _control.reset(token)
    sig = signature(fn)
    run.__signature__ = sig.replace(parameters=[*sig.parameters.values(),
        Parameter('cancel', Parameter.KEYWORD_ONLY, default=None),
        Parameter('progress', Parameter.KEYWORD_ONLY, default=None)])
    run.__doc__ = (fn.__doc__ or '') + ('\n\nOptional cancel() is checked between resampling batches and raises '
        'AssociationCancelled when true. progress(stage, completed, total) reports batch progress.')
    return run
