FROM anibali/pytorch:cuda-9.0
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY train.py /opt/app/train.py
COPY ./asap /opt/app/asap
# Set the default command to python3
CMD ["/bin/bash"]