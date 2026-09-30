# Citing FiberHMM

If you use FiberHMM in published work, please cite the software with the
version you used (`python -c "import fiberhmm; print(fiberhmm.__version__)"`)
and the repository:

```bibtex
@software{fiberhmm,
  title   = {FiberHMM: chromatin footprint calling from single-molecule
             Fiber-seq and DAF-seq data},
  author  = {{FiberHMM Authors}},
  version = {3.0.0},
  year    = {2026},
  url     = {https://github.com/fiberseq/FiberHMM}
}
```

The citation of the accompanying publication will be added here when it is
available.

Please also cite the tools FiberHMM builds on where relevant:
[fibertools](https://github.com/fiberseq/fibertools-rs) (FIRE, `ft`) and the
[Molecular-annotation specification](https://github.com/fiberseq/Molecular-annotation-spec)
for the `MA`/`AQ` tags.
