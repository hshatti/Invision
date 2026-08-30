#!/bin/sh

# use first parameter, if null use FLUX.2-klein-4B
MODEL="${1:-FLUX.2-klein-4B}"

SCHEDULER_DIR="scheduler"
MODELSDIR="models"
REPO="${2:-black-forest-labs}"
ENCODER_DIR="text_encoder"
TOKENIZER_DIR="tokenizer"
TRANSFORMER_DIR="transformer"
VAE_DIR="vae"
HF_DOWNLOAD_ARG="?download=true"
DEFAULT_DIR="./models"
ReDownload=""

MODEL_JSON="model_index.json"

HTTP_URL="https://huggingface.co/%s/%s/resolve/main/%s%s"

if [ ! -d $MODELSDIR ]; then 
  mkdir $MODELSDIR 

fi

SCHEDULER_FILES=("scheduler_config.json")
ENCODER_FILES=("config.json" "generation_config.json" "model-00001-of-00002.safetensors" "model-00002-of-00002.safetensors" "model.safetensors.index.json")
TOKENIZER_FILES=("added_tokens.json" "chat_template.jinja" "merges.txt"	"special_tokens_map.json" "tokenizer.json" "tokenizer_config.json" "vocab.json")
TRANSFORMER_FILES=("config.json" "diffusion_pytorch_model.safetensors")
VAE_FILES=("config.json" "diffusion_pytorch_model.safetensors")

HTTP_ROOT=$(printf $HTTP_URL $REPO $MODEL $MODEL_JSON $HF_DOWNLOAD_ARG)
FN=$(printf "%s/%s/%s" $MODELSDIR $MODEL $MODEL_JSON)
FDIR=${FN%/*}
if [ ! -d $FDIR ];then 
    echo "Creating [$FDIR]"
    mkdir "${FDIR}"
fi
if [[ ! -e $FN || $ReDownload = "Y" ]]; then
    echo "downloading [$FN]"
    curl --header "Authorization: Bearer ${HF_TOKEN}" --output $FN --follow $HTTP_ROOT
fi


# use hf_download <folder> <array of filenames>
hf_download(){
  for F in ${@:2} ; do
    HTTP_ROOT=$(printf $HTTP_URL $REPO $MODEL "$1/$F" $HF_DOWNLOAD_ARG)
    FN=$(printf "%s/%s/%s" $MODELSDIR $MODEL "$1/$F")
    FDIR=${FN%/*}
    if [ ! -d $FDIR ];then 
      echo "Creating [$FDIR]"
      mkdir "${FDIR}"
    fi
    if [[ ! -e $FN  || $ReDownload = "Y" ]]; then
      echo "downloading [$FN]"
      curl --header "Authorization: Bearer ${HF_TOKEN}" --output $FN --follow $HTTP_ROOT
    fi
  done
}

hf_download $SCHEDULER_DIR ${SCHEDULER_FILES[@]}
hf_download $ENCODER_DIR ${ENCODER_FILES[@]}
hf_download $TOKENIZER_DIR ${TOKENIZER_FILES[@]}
hf_download $TRANSFORMER_DIR ${TRANSFORMER_FILES[@]}
hf_download $VAE_DIR ${VAE_FILES[@]}
