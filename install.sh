export NCCL_DEBUG=""
uv pip install git+https://github.com/t1101675/transformers@minillm
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install deepspeed
uv pip install numerize
uv pip install rouge-score
uv pip install torchtyping
uv pip install rich
uv pip install accelerate
uv pip install datasets
# uv pip install "peft==0.12.0" //0.17.1
uv pip install "peft"
uv pip install sentencepiece
uv pip install protobuf
uv pip install bitsandbytes 
uv pip install wandb       
