unit quicknn_downloader;

{$ifdef FPC}
{$mode Delphi}
{$endif}
{$Assertions On}

interface

uses
  Classes, SysUtils

  ;

type

  { TFLUX4B }

  TFLUX4B= record
    const
      ENCODER_DIR     = 'text_encoder';
      TOKENIZER_DIR   = 'tokenizer';
      TRANSFORMER_DIR = 'transformer';
      VAE_DIR         = 'vae';
      DST_PATH        = 'FLUX.2-klein-4B';
      HTTP_ROOT       = 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/';
      HF_DOWNLOAD_ARG = '?download=true';
    class var
      stage:longint;
      currentFile : string;
    public
      class procedure download(const modelsPath: string = 'models'; srcPath: string=''); static;
  end;

implementation
uses nHttp, termesc;

const
  ERROR_CREATE_DIR = 'Cannot create model folder';


procedure MakeDir(const dirName:string);
begin
  if not DirectoryExists(dirName) then
    assert(CreateDir(dirName), ERROR_CREATE_DIR+ ' ['+dirName+']');
end;

{ TFLUX4B }

class procedure TFLUX4B.download(const modelsPath: string; srcPath: string);
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
    'diffusion_pytorch_model.safetensors'
  ];

  VAE_FILES : array of string = [
    'config.json',
    'diffusion_pytorch_model.safetensors'
  ];


var curDir :string;
    i : integer;
begin

  curDir := modelsPath + PathDelim + DST_PATH + PathDelim;
  if modelsPath<>'' then begin
    curDir := modelsPath + PathDelim + DST_PATH + PathDelim;
    MakeDir(modelsPath);
  end else begin
    curDir := DST_PATH + PathDelim;
  end;
  makeDir(curDir);
  MakeDir(curDir + ENCODER_DIR);
  MakeDir(curDir + TOKENIZER_DIR);
  MakeDir(curDir + TRANSFORMER_DIR);
  MakeDir(curDir + VAE_DIR);
  if isConsole then cursorShow(false);
  for i := 0 to high(ENCODER_FILES) do begin
    currentFile := ENCODER_FILES[i];

    if fileExists(curDir+ENCODER_DIR+PathDelim + currentFile) then continue;
    if isConsole then
      writeln('Downloading ... ', ENCODER_FILES[i]);
    http.Download(HTTP_ROOT+ENCODER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+ENCODER_DIR + '/' + currentFile);
  end;

  for i := 0 to high(TOKENIZER_FILES) do begin
    currentFile := TOKENIZER_FILES[i];
    if fileExists(curDir+TOKENIZER_DIR+PathDelim + currentFile) then continue;
    if isConsole then
      writeln('Downloading ... ', TOKENIZER_FILES[i]);
    http.Download(HTTP_ROOT+TOKENIZER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+TOKENIZER_DIR+'/' + currentFile);
  end;

  for i := 0 to high(TRANSFORMER_FILES) do begin
    currentFile := TRANSFORMER_FILES[i];
    if fileExists(curDir+TRANSFORMER_DIR+PathDelim + currentFile) then continue;
    if isConsole then
      writeln('Downloading ... ', TRANSFORMER_FILES[i]);
    http.Download(HTTP_ROOT+TRANSFORMER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+TRANSFORMER_DIR+'/' + currentFile);
  end;

  for i := 0 to high(VAE_FILES) do begin
    currentFile := VAE_FILES[i];
    if fileExists(curDir+VAE_DIR+PathDelim + currentFile) then continue;
    if isConsole then
      writeln('Downloading ... ', VAE_FILES[i]);
    http.Download(HTTP_ROOT+VAE_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+VAE_DIR+'/' + currentFile);
  end;
  if isConsole then cursorShow(True);

end;


initialization

  ExecuteProcess('powershell', ['-Command', 'Invoke-WebRequest', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/text_encoder/model-00001-of-00002.safetensors?download=true', '-OutFile', './model-00001-of-00002.safetensors']);

end.

