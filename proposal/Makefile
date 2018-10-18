PAPER = proposal 
TEX = $(wildcard *.tex)
BIB = references.bib

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB)
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

