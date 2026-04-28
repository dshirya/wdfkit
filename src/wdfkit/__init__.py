"""Read Renishaw WiRE .wdf files (spectra and metadata)."""

from .reader import WDFReader, read_WDF

read_wdf = read_WDF

__all__ = ["WDFReader", "read_WDF", "read_wdf"]
