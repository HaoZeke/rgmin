#!/usr/bin/env python3

import os

project = "rgmin"
copyright = "2024--present, Rohit Goswami"
author = "Rohit Goswami"
html_logo = "_static/rgmin-notext-light.webp"

extensions = [
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx.ext.intersphinx",
]
try:
    import sphinxcontrib_rust  # noqa: F401
    import sphinx_rustdoc_postprocess  # noqa: F401

    extensions.extend(["sphinxcontrib_rust", "sphinx_rustdoc_postprocess"])
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_timeout = 5

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "rgpot": ("https://rgpot.rgoswami.me", None),
    "eon": ("https://eondocs.org", None),
}

rust_crates = {
    "rgmin": os.path.abspath("../../"),
}
rust_doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt = "rst"
rust_generate_mode = "always"

rustdoc_postprocess_toctree_target = "reference/rust-api.rst"
rustdoc_postprocess_toctree_rst = """
Rust API
--------

.. toctree::
   :maxdepth: 2

   ../crates/rgmin/lib
"""

html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/OmniPotentRPC/rgmin",
    "accent_color": "indigo",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "light_logo": "_static/rgmin-notext-light.webp",
    "dark_logo": "_static/rgmin-notext-dark.webp",
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "rgsaddle",
                    "url": "https://github.com/OmniPotentRPC/rgsaddle",
                    "summary": "Band, min-mode, and IRC sessions over rgmin",
                },
                {
                    "title": "rgpot",
                    "url": "https://rgpot.rgoswami.me",
                    "summary": "Potential-energy library and RPC server",
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Saddle-point search on potential energy surfaces",
                },
                {
                    "title": "readcon-core",
                    "url": "https://github.com/lode-org/readcon-core",
                    "summary": "CON frames; chemfiles import is a feature",
                },
                {
                    "title": "linkcell",
                    "url": "https://github.com/d-SEAMS/linkcell",
                    "summary": "Periodic MIC and linked-cell neighbours",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "rgmin",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_css_files = [
    "custom.css",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
