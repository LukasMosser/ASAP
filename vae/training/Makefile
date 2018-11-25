# This is your Editor pane. Write the Makefile here and 
# use the command line to execute commands
NAME = asap_train
default: build
build:
	docker build -t $(NAME) .
run:
	docker run --rm $(NAME)
release: build run
