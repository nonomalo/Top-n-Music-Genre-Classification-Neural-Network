docker-build:
	docker build -t top-n-server:latest .

docker-run:
	docker run --rm --name top-n-service -p 8080:8080 docker.io/library/top-n-server:latest

deploy:
	gcloud config set project top-n-server
	gcloud app deploy
