#FROM public.ecr.aws/lambda/python:3.8
FROM svizor42/zoomcamp-dino-dragon-lambda:v2

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
RUN pip install Pillow

COPY dino_dragon_10_0.899.tflite .
#COPY dino-vs-dragon-v2.tflite .
COPY lambda_script.py .

CMD ["lambda_script.lambda_handler"]
