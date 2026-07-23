"""arc-llama — plug-and-play llama.cpp runtime for Intel Arc GPUs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arc-llama")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
