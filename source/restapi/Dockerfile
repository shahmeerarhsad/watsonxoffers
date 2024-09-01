# base image
FROM python:3.12.4

# working directory 
WORKDIR /rest_api

# copying local files to WORKDIR inside container
COPY . .

RUN pip install --upgrade pip


# needed packages specified in requirements.txt.
RUN pip install -r requirements.txt

# port 8000 
EXPOSE 8000

# run the application
CMD ["python", "myoffer.py"]