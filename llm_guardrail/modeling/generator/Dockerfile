FROM luvx113274/llm_guardrail_base:latest

WORKDIR /app

COPY ./src ./src

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]