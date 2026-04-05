FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
# Verify environment import works (required for grading)
RUN python -c "from helixdesk import HelixDeskEnv; print('HelixDeskEnv import OK')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]