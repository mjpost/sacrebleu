.PHONY: test
test:
	mypy sacrebleu scripts test
	python3 -m pytest
	bash test.sh

pip:
	python3 -m build .

publish: pip
	twine upload dist/*
