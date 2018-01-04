#! /bin/sh

docker build -t berkeley_ai_cs188 . && docker run --rm -ti berkeley_ai_cs188 $1 $2