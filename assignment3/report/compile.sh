#!/bin/sh
pdflatex report.tex
bibtex report
# Yes, the pdflatex needs to be run two times in a row
pdflatex report.tex
pdflatex report.tex
rm -f report.aux report.bbl report.blg report.log report.out
