unit quicknn_downloader;

{$ifdef FPC}
{$mode Delphi}
{$endif}
{$Assertions On}

interface

uses
  Classes, SysUtils;

type

  { TFLUX4B }

  TFLUX4B= record
    const
      ENCODER_DIR    = 'text_encoder';
      TOKENIZER_DIR  = 'tokenizer';
      TRANSFORMER_DIR = 'transformer';
      VAE_DIR        = 'vae';
      HTTP_ROOT      = 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/';
      HF_DOWNLOAD_ARG = '?download=true';
    public
      class procedure download(const srcPath:string=''; dstPath:string='models');static;
  end;

implementation
uses nHttp, termesc;

const
  ERROR_CREATE_DIR = 'Cannot create model folder';


procedure MakeDir(const dirName:RawByteString);
begin
  if not DirectoryExists(dirName) then
    assert(CreateDir(dirName), ERROR_CREATE_DIR+ ' ['+dirName+']');
end;

{ TFLUX4B }

class procedure TFLUX4B.download(const srcPath: string; dstPath: string);
const
  ENCODER_FILES : array of string = [
    'generation_config.json' ,
    'model-00001-of-00002.safetensors' ,
    'model-00002-of-00002.safetensors' ,
    'model.safetensors.index.json'
  ];

  TOKENIZER_FILES : array of string = [
    'added_tokens.json',
    'chat_template.jinja',
    'merges.txt',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json'
  ];

  TRANSFORMER_FILES : array of string = [
    'config.json',
    'diffusion_pytorch_model.safetensor'
  ];

  VAE_FILES : array of string = [
    'config.json',
    'diffusion_pytorch_model.safetensor'
  ];


var curDir, fn :string;
    i : integer;
begin
  curDir := dstPath+PathDelim;
  MakeDir(dstPath);
  MakeDir(curDir + ENCODER_DIR);
  MakeDir(curDir + TOKENIZER_DIR);
  MakeDir(curDir + TRANSFORMER_DIR);
  MakeDir(curDir + VAE_DIR);
  cursorShow(false);
  for i := 0 to high(ENCODER_FILES) do begin
    if isConsole then
      writeln('Downloading ... ', ENCODER_FILES[i]);
    fn := ENCODER_FILES[i];
    http.Download(HTTP_ROOT+ENCODER_DIR+ '/' + fn + HF_DOWNLOAD_ARG, curDir+ENCODER_DIR + '/' + fn);
  end;

  for i := 0 to high(TOKENIZER_FILES) do begin
    if isConsole then
      writeln('Downloading ... ', TOKENIZER_FILES[i]);
    fn := TOKENIZER_FILES[i];
    http.Download(HTTP_ROOT+TOKENIZER_DIR+ '/' + fn + HF_DOWNLOAD_ARG, curDir+TOKENIZER_DIR+'/' + fn);
  end;

  for i := 0 to high(TRANSFORMER_FILES) do begin
    if isConsole then
      writeln('Downloading ... ', TRANSFORMER_FILES[i]);
    fn := TRANSFORMER_FILES[i];
    http.Download(HTTP_ROOT+TRANSFORMER_DIR+ '/' + fn + HF_DOWNLOAD_ARG, curDir+TRANSFORMER_DIR+'/' + fn);
  end;

  for i := 0 to high(ENCODER_FILES) do begin
    if isConsole then
      writeln('Downloading ... ', ENCODER_FILES[i]);
    fn := ENCODER_FILES[i];
    http.Download(HTTP_ROOT+VAE_DIR+ '/' + fn + HF_DOWNLOAD_ARG, curDir+VAE_DIR+'/' + fn);
  end;
  cursorShow(True);

end;


initialization
end.

