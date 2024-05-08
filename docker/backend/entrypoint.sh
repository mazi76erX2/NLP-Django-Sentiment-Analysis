#!/bin/sh

# Chack if database has started
if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $DATABASE_HOST $DATABASE_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

until cd /app/nlp_sentiment_analysis
do
    echo "Waiting for server volume..."
done

set -e

python manage.py migrate  --noinput
exec "$@"
