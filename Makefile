test:
	pytest
	bash test.sh

pip:
	python3 setup.py sdist bdist_wheel

publish: pip
	twine upload dist/*
