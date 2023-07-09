# Makefile

install:
	pip install flask
	pip install openai
	pip install pandas
	pip install numpy
	pip install random
	pip install requests
	pip install scikit-learn
	pip install catboost
	pip install flask-session
	pip install flask-url-for
	pip install Flask

run: 
	python server.py
	go to http://127.0.0.1:5000
