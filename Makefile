.PHONY: run stop clean

run:
	docker-compose up --build -d
	@echo "API running at http://localhost:8000"
	@echo "UI running at http://localhost:8501"

stop:
	docker-compose down

clean:
	docker-compose down -v --rmi all
