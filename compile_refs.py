#!/usr/bin/python3

import subprocess, sys

commands = [
    ['pdflatex', 'odsl_califa_paper.tex'],
    ['bibtex', 'odsl_califa_paper.aux'],
    ['pdflatex', 'odsl_califa_paper.tex'],
    ['pdflatex', 'odsl_califa_paper.tex']
]

for c in commands:
    subprocess.call(c)
