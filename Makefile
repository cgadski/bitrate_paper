FIGURES = \
	figures/storyline.pdf \
	figures/k_sweep.pdf \
	figures/eta_sweep.pdf

figures/%.pdf: project/graphs/%.py
	uv run $^

latex/main.pdf: latex/*.tex \
	latex/content/*.tex \
	latex/capacity_paper.bib \
	$(FIGURES)

	cd latex; pdflatex main.tex
	cd latex; bibtex main
	cd latex; pdflatex main.tex
	cd latex; pdflatex main.tex

.PHONY: upload

upload: latex/main.pdf
	rsync $^ root@cgad.ski:/www/math/information.pdf
