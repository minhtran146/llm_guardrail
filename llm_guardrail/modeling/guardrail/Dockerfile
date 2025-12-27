# Kế thừa từ base image đã có sẵn các thư viện
FROM luvx113274/llm_guardrail_base:latest

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn của dịch vụ vào container
COPY ./src ./src

# Mở cổng
EXPOSE 8000

# Lệnh chạy ứng dụng (vẫn chạy với user root mặc định)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]