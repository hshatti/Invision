#!/bin/sh

MODELSDIR="./models"
MODEL="FLUX.2-klein-4B"

MODELSDIR="models"
ENCODER_DIR="text_encoder"
TOKENIZER_DIR="tokenizer"
TRANSFORMER_DIR="transformer"
VAE_DIR="vae"
HF_DOWNLOAD_ARG="?download=true"
DEFAULT_DIR="./models"
ReDownload=""
MODEL_JSON="model_index.json"


HTTP_URL="https://huggingface.co/black-forest-labs/%s/resolve/main/%s%s"

if [ ! -d $MODELSDIR ]; then 
  mkdir $MODELSDIR 

fi

HTTP_ROOT=$(printf $HTTP_URL $MODEL $MODEL_JSON $HF_DOWNLOAD_ARG)
FN=$(printf "%s/%s/%s" $MODELSDIR $MODEL $MODEL_JSON)

if [ -e $FN ] && [ "${REPLY^^}" != "Y" ]; then
  if [ ! -d "$MODELSDIR/$MODEL" ];then mkdir "$MODELSDIR/$MODEL"; fi
  echo "[$FN]" "already exists Re-Download? (Y,N)"
  read
  if [ "${REPLY^^}" = "Y" ]; then
    # echo $HTTP_ROOT
    # -H for header to send, -L to follow redirection, -o outfile instead of stdout
    curl -H "Authorization: Bearer $HF" --follow "$HTTP_ROOT" -o $FN
  fi
  exit
fi

if (-not (Test-Path $FN) -or ($ReDownload.ToUpper() -eq 'Y')) {
  echo "Downloading $FN"
  Invoke-WebRequest -Uri $HTTP_ROOT -headers $HEAD -OutFile $FN
}
  
$ReDownload = ""


$ENCODER_FILES  = @(
    "generation_config.json" ,
    "model-00001-of-00002.safetensors" ,
    "model-00002-of-00002.safetensors" ,
    "model.safetensors.index.json"
    )

foreach ($FILE in $ENCODER_FILES ) {
  $HTTP_ROOT       = "https://huggingface.co/black-forest-labs/{0}/resolve/main/{1}/{2}{3}" -f $MODEL, $ENCODER_DIR, $FILE, $HF_DOWNLOAD_ARG 
  $DN = "{0}/{1}/{2}" -f $MODELSDIR, $MODEL, $ENCODER_DIR
  $FN = $DN+"/"+$FILE
  if (-not (Test-Path $DN)) { md $DN }
  if ((Test-Path $FN) -and ($ReDownload.ToUpper() -ne 'Y')) {
    echo "[$FN]" "already exists Re-Download? (Y,N)"
    $ReDownload = Read-Host
  }
  if (-not (Test-Path $FN) -or ($ReDownload.ToUpper() -eq 'Y')) {
    echo "Downloading $FN"
    Invoke-WebRequest -Uri $HTTP_ROOT -headers $HEAD -OutFile $FN
  }
  
  $ReDownload = ""
}


$TOKENIZER_FILES  = @(
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"   
    )

foreach ($FILE in $TOKENIZER_FILES ) {
  $HTTP_ROOT       = "https://huggingface.co/black-forest-labs/{0}/resolve/main/{1}/{2}{3}" -f $MODEL, $TOKENIZER_DIR, $FILE, $HF_DOWNLOAD_ARG 
  $DN = "{0}/{1}/{2}" -f $MODELSDIR, $MODEL, $TOKENIZER_DIR
  $FN = $DN+"/"+$FILE
  if (-not (Test-Path $DN)) { md $DN }
  if ((Test-Path $FN) -and ($ReDownload.ToUpper() -ne 'Y')) {
    echo "[$FN]" "already exists Re-Download? (Y,N)"
    $ReDownload = Read-Host
  }
  if (-not (Test-Path $FN) -or ($ReDownload.ToUpper() -eq 'Y')) {
    echo "Downloading $FN"
    Invoke-WebRequest -Uri $HTTP_ROOT -headers $HEAD -OutFile $FN
  }
  
  $ReDownload = ""
}


$TRANSFORMER_FILES  = @(
    "config.json",
    "diffusion_pytorch_model.safetensors"  
  )

foreach ($FILE in $TRANSFORMER_FILES ) {
  $HTTP_ROOT       = "https://huggingface.co/black-forest-labs/{0}/resolve/main/{1}/{2}{3}" -f $MODEL, $TRANSFORMER_DIR, $FILE, $HF_DOWNLOAD_ARG 
  $DN = "{0}/{1}/{2}" -f $MODELSDIR, $MODEL, $TRANSFORMER_DIR
  $FN = $DN+"/"+$FILE
  if (-not (Test-Path $DN)) { md $DN }
  if ((Test-Path $FN) -and ($ReDownload.ToUpper() -ne 'Y')) {
    echo "[$FN]" "already exists Re-Download? (Y,N)"
    $ReDownload = Read-Host
  }
  if (-not (Test-Path $FN) -or ($ReDownload.ToUpper() -eq 'Y')) {
    echo "Downloading $FN"
    Invoke-WebRequest -Uri $HTTP_ROOT -headers $HEAD -OutFile $FN
  }
  
  $ReDownload = ""
}

$VAE_FILES  = @(
    "config.json",
    "diffusion_pytorch_model.safetensors"  
  )

foreach ($FILE in $VAE_FILES ) {
  $HTTP_ROOT       = "https://huggingface.co/black-forest-labs/{0}/resolve/main/{1}/{2}{3}" -f $MODEL, $VAE_DIR, $FILE, $HF_DOWNLOAD_ARG 
  $DN = "{0}/{1}/{2}" -f $MODELSDIR, $MODEL, $VAE_DIR
  $FN = $DN+"/"+$FILE
  if (-not (Test-Path $DN)) { md $DN }
  if ((Test-Path $FN) -and ($ReDownload.ToUpper() -ne 'Y')) {
    echo "[$FN]" "already exists Re-Download? (Y,N)"
    $ReDownload = Read-Host
  }
  if (-not (Test-Path $FN) -or ($ReDownload.ToUpper() -eq 'Y')) {
    echo "Downloading $FN"
    Invoke-WebRequest -Uri $HTTP_ROOT -headers $HEAD -OutFile $FN
  }
  
  $ReDownload = ""
}


