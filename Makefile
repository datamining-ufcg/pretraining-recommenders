.PHONY: catalog
catalog:
	cd scripts && python build_catalog.py

.PHONY: analyze
analyze:
	cd scripts && python leakage_analysis.py

.PHONY: plot
plot:
	cd scripts && bash plot_all.sh

.PHONY: analyze_and_plot
analyze_and_plot:
	make analyze && make plot

.PHONY: compile
compile:
	cd src/models && bash compile.sh