BASE_PATH="/home/minillm_ai_safety"

export TF_CPP_MIN_LOG_LEVEL=3

##DOLLY

# training and validation data

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/mix/0.9/llama3-dolly/full \
    --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 1000 \
    --model-type llama3-chat \
    --curriculum-mix-ratio 0.9

# # # test data

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/new/llama3/dolly/full \
#     --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type llama3-chat

# ###SELF INST
# # test data

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/self-inst/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/new/llama3/self-inst/full \
#     --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type llama3-chat

# ###SINST
# # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/sinst/11_/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/new/llama3/sinst/full \
#     --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type llama3-chat

# # ###UINST
# # # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/uinst/11_/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/new/llama3/uinst/full \
#     --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type llama3-chat


# # ###VICUNA
# # # test data
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/vicuna/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/new/llama3/vicuna/full \
#     --model-path /media/drive2/models/llama-models/Llama-3.1-8B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 512 \
#     --test \
#     --model-type llama3-chat 