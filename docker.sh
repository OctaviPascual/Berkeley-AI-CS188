#! /bin/sh

docker build -t berkeley_ai_cs188 . && docker run --rm -it berkeley_ai_cs188 $1 $2
