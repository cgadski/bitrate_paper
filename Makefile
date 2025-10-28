latex/main.pdf: latex/*.tex \
	latex/content/*.tex \
	latex/capacity_paper.bib
	cd latex; pdflatex main.tex
	cd latex; bibtex main
	cd latex; pdflatex main.tex
	cd latex; pdflatex main.tex
