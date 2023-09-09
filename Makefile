

.PHONY: prepare-cpu
prepare-cpu:
	python3 -m venv ./autoencoders-venv && \
	. ./autoencoders-venv/bin/activate && \
	pip3 install -r requirements_base.txt && \
	pip3 install -r requirements_cpu.txt 
	
