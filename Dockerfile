# CPU image for calibration, inference and measurement.
# (Training is done on Google Colab GPU; see models/Train_MaskRCNN_Colab.ipynb.)
FROM python:3.12-slim

# OpenCV runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first (avoids pulling the large CUDA build), then the rest
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENTRYPOINT ["python"]
# Example:
#   docker build -t xis-measure .
#   docker run --rm -v ${PWD}:/app xis-measure measurement/measure.py \
#       --image NEW.jpg --method model --weights models/weights/maskrcnn_best.pth
