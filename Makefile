PROJECT_ID=llm-exp-405305
REGION=us

COMMIT=$(shell git rev-parse HEAD)
BASE="us-docker.pkg.dev/$(PROJECT_ID)/ragtime/main"
VERSION=$(BASE):$(COMMIT)
LATEST=$(BASE):latest


dockerbuild:
	docker build . --file Dockerfile --tag $(LATEST) --platform linux/amd64
	docker tag $(LATEST) $(VERSION)

dockerpush:
	docker push $(LATEST)
	docker push $(VERSION)

check:
	poetry run black --check --color ragtime tests
	poetry run isort --check ragtime tests
	poetry run mypy ragtime tests
	poetry run pylint ragtime tests

format:
	poetry run black ragtime tests
	poetry run isort ragtime tests

test:
	poetry run pytest tests
