setup-environment:
	@echo "Setup environment"
	@conda env create $(CONDA_ENV_CREATE_OPTS) -f environment.yaml

run-pipeline:
	@echo "Running the classification pipeline"
	@python classification/main.py
