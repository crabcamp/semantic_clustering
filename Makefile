
lint:
	flake8 --show-source keywords_similarity
	isort --check-only -rc keywords_similarity --diff

test:
	pytest tests

install-dev:
	pip install -U -r requirements/dev.txt

install-ci:
	pip install -U -r requirements/ci.txt
