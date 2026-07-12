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

.PHONY: upload get-results pip-install

upload: latex/main.pdf
	rsync $^ root@cgad.ski:/www/math/information.pdf


get-results:
	scp -i ~/.ssh/laptop_new -P 35276 \
    	root@195.26.232.178:/workspace/bitrate_paper/results.tar.gz \
    	./results/eta_sweep_2.tar.gz
	rm -rf results/eta_sweep_2
	mkdir -p results/eta_sweep_2
	tar -xf ./results/eta_sweep_2.tar.gz -C results/eta_sweep_2

# for environments without uv
pip-install:
	pip install vandc pandas einops scipy simple-parsing
