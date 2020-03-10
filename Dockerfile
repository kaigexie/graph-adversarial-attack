FROM ubuntu:latest
RUN apt update -y
RUN apt install -y python-pip
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get install -y python-tk
#ENTRYPOINT ['python3']
CMD ["./get_attack_result.sh"]
