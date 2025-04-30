pip install torchserve torch-model-archiver

pip install nvgpu

pip install einops

apt-get update

apt-get install -y default-jre

torch-model-archiver \
  --model-name grounded_sam2_florence2 \
  --version 1.0 \
  --handler model_store/handler.py \
  --export-path model_store \
  --extra-files model_store/model.py,sam2/,checkpoints/,utils/ \
  --force

torchserve --start --model-store model_store --models grounded_sam2_florence2=grounded_sam2_florence2.mar --no-config-snapshots

torchserve --start --model-store model_store --models grounded_sam2_florence2=grounded_sam2_florence2.mar --ts-config model_store/config.properties


8080:8080

docker build --no-cache -t grounded_sam2:1.0 .
❯ docker run --gpus all -p 8080:8080 -it --rm -v "/c/Users/Hasan/OneDrive/Desktop/Projects/Grounded-SAM-2:/home/appuser/Grounded-SAM-2" grounded_sam2:1.0


docker build --no-cache --build-arg USE_CUDA=1 --build-arg TORCH_ARCH="7.0;7.5;8.0;8.6+PTX" -t my_grounded_sam2:1.0 .