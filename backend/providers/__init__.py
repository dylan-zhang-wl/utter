"""Concrete speech and language backends, and only here.

铁律 6: no module outside this package may import mlx_whisper, faster_whisper or
any other concrete engine. MLX has a backend only on Apple Silicon, and the
author owns a machine that is not one — an import at the wrong level turns into
an ImportError on somebody else's laptop, at transcription time, with no way for
the author to reproduce it.

backend/tests/test_dependencies.py enforces this by walking the AST of every
module outside this directory.
"""
