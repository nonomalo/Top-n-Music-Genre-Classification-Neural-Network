docker-build:
	docker build -t top-n-client:latest .

docker-run:
	docker run --rm --name top-n-service -p 8080:8080 docker.io/library/top-n-client:latest

deploy:
	gcloud config set project top-n-client
	gcloud app deploy
