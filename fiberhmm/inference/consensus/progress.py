"""Optional structured work counters; legacy text callbacks remain supported.

Counters describe completed work, never estimated time or scientific confidence.
They belong to the live job, not to frozen model artifacts or evidence digests.
"""


def report(progress, stage, message, **work):
    if progress is None:
        return
    structured = getattr(progress, 'report', None)
    if structured is not None:
        structured(stage, message, **work)
    else:
        progress(stage, message)


def report_work(progress, message, **work):
    """For kernels with the historical one-argument callback."""
    if progress is None:
        return
    structured = getattr(progress, 'report', None)
    if structured is not None:
        structured(message, **work)
    else:
        progress(message)


def stage_progress(progress, stage, *, prefix='', **context):
    def callback(message):
        report(progress, stage, prefix+message if message is not None else None, **context)

    def structured(message, **work):
        report(progress, stage, prefix+message if message is not None else None, **(context | work))

    callback.report = structured
    return callback
