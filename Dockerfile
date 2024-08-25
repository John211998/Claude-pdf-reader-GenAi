FROM python:3.11

EXPOSE 8098

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./

CMD ["streamlit", "run", "main.py", "--server.port=8098", "--server.address=0.0.0.0"]
