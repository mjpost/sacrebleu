.PHONY: test
test:
	ruff check .
	mypy sacrebleu scripts test
	python3 -m pytest
	bash test.sh

pip:
	python3 setup.py sdist bdist_wheel

publish: pip
	twine upload dist/*
