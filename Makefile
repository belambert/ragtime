PROJECT_ID=llm-exp-405305
REGION=us

COMMIT=$(shell git rev-parse HEAD)
BASE="us-docker.pkg.dev/$(PROJECT_ID)/wikibot/main"
VERSION=$(BASE):$(COMMIT)
LATEST=$(BASE):latest


dockerbuild:
	docker build . --file Dockerfile --tag $(LATEST) --platform linux/amd64
	docker tag $(LATEST) $(VERSION)

dockerpush:
	docker push $(LATEST)
	docker push $(VERSION)

check:
	poetry run black --check --color wikibot tests
	poetry run isort --check wikibot tests
	poetry run mypy wikibot tests
	poetry run pylint wikibot tests

format:
	poetry run black wikibot tests
	poetry run isort wikibot tests

test:
	poetry run pytest tests
