FROM python:3.10

# Allows docker to cache installed dependencies between builds
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . code
WORKDIR /code

EXPOSE 8000

COPY ./entrypoint.sh /
ENTRYPOINT ["sh","/entrypoint.sh"]

# ENTRYPOINT ["python","manage.py"] 
# CMD ["runserver", "0.0.0.0:8000"]