FROM postgres:16-alpine

# Скрипт создания витрины выполняется при первой инициализации БД
COPY init.sql /docker-entrypoint-initdb.d/init.sql
