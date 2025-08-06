"""
Generator package for creating expert conclusion reports.

This package contains functions to assemble a DOCX report based on the
results of handwriting analysis.  It uses python‑docx to build the
document according to the guidelines described in V.F. Orlova’s
handwriting expertise manuals and the development plan.  The module is
designed to be extended: additional sections, formatting rules and
attachments can be added as needed.

Module contents:

* ``docx_generator.py`` – the main interface for generating reports.
* ``templates`` – directory for storing Word templates (optional).
"""

from .docx_generator import generate_conclusion  # noqa: F401
