"""MA input-frame policy shared by consensus preparation and display.

Historical unmarked FiberHMM BAMs use stored SEQ coordinates. Molecular MA
producers declare coord=molecular in PG/CO; never infer it from alignment flags.
"""


def ma_annotation_frame(header):
    header = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
    texts = [str(c) for c in header.get('CO', [])]
    texts += [' '.join(str(v) for v in pg.values()) for pg in header.get('PG', [])]
    return 'molecular' if any('coord=molecular' in text.lower() for text in texts) else 'seq'
