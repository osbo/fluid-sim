# latexmk (LaTeX Workshop): stable ACM + natbib + BibTeX build from a clean tree.
# Ensures pdflatex runs with nonstopmode so the first .aux includes \bibdata before bibtex.
# paper.tex uses xr-hyper to supplementary.tex: after a clean checkout run
#   latexmk supplementary.tex && latexmk paper.tex
# so supplementary.aux exists before the first pdflatex on paper.tex.
$pdf_mode = 1;
$dvi_mode = 0;
$postscript_mode = 0;
$bibtex_use = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
