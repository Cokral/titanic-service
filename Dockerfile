FROM python:3.7-stretch

RUN \
    apt-get -y update \
    && rm -rf /var/lib/apt/lists/*

RUN \
    pip install --upgrade pip \
    && pip install -U setuptools

WORKDIR /home/src/titanic_service

COPY requirements/common.txt requirements/common.txt
RUN pip3 install -r requirements/common.txt

COPY . .
RUN pip install -e .

EXPOSE 12000

CMD ["bash"]