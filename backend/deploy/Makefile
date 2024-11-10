default: start

start:
	docker-compose --env-file .env up

# Target to perform a soft restart (without removing source images)
soft-restart:
	docker-compose stop
	docker-compose --env-file .env up

# Target to perform a hard restart (with removal of source images)
hard-restart: update
	docker-compose down
	docker rmi deploy-app
	docker rmi deploy-ml-app
	docker-compose --env-file .env up --build

# Target to perform a hard restart (with removal of source images, but without git pull)
restart:
	docker-compose down
	docker rmi deploy-app
	docker rmi deploy-ml-app
	docker-compose --env-file .env up --build

stop:
	docker-compose stop

clean:
	docker-compose down --rmi all

logs:
	docker-compose logs -f

update:
	git pull

.PHONY: default start soft-restart hard-restart stop clean logs update
