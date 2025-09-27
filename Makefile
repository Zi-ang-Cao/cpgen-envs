PACKAGE_DIR=gearteleop/


.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-apidoc -o docs/source/ $(PACKAGE_DIR) --tocfile index --force --separate
	sphinx-autobuild -b html --watch $(PACKAGE_DIR) docs/source/ docs/build/


# `make api` will continuously monitor the python files in the package and generate .rst
# Run `make docs` first, and then `make api`, you'll be able to see the website constantly refreshed
.PHONY : api_ api
api_ :
	sphinx-apidoc -o docs/source/ $(PACKAGE_DIR) --tocfile index --force --separate

api:
	while true; do \
		find $(PACKAGE_DIR) -name '*.py' | entr -cd sh -c 'make api_ || exit 255'; \
		if [ $$? -eq 255 ]; then \
			echo "Error encountered. Stopping..."; \
			exit 1; \
		fi; \
		sleep 0.5 \
		echo "Waiting for changes..."; \
	done

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	# mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/

.PHONY : format
format :
	isort .
	black .
	ruff check . --ignore F401

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
