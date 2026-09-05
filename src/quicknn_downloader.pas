unit quicknn_downloader;

{$ifdef FPC}
{$mode Delphi}
{$endif}
{$Assertions On}

interface

uses
  Classes, SysUtils, nHttp
  {$if not defined(FPC) and defined(MSWINDOWS)}
  , ShellApi
  {$endif}
  ;

type


  { TFLUX4BDownloader }

  TFLUX4BDownloader= record
  const
     ENCODER_DIR     = 'text_encoder';
     TOKENIZER_DIR   = 'tokenizer';
     TRANSFORMER_DIR = 'transformer';
     VAE_DIR         = 'vae';
     DST_PATH        = 'FLUX.2-klein-4B';
     HTTP_ROOT       = 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/';
     HF_DOWNLOAD_ARG = '?download=true';
   var
     FHttp : TNHttp;
     stage:longint;
     currentFile : string;
     //FCanceled : boolean;
     FSubReceived, FSubTotal, FReceived, FTotal:int64;
     OnProgress : procedure (const subProg, subTotal, progress, total:int64) of object;
   private
     procedure OnReceive(sender:TObject; const received, total:int64);
   public
     procedure Cancel();
     procedure download(modelsPath: string = 'models'; srcPath: string='');
  end;

  procedure getOpenBlas();

implementation
uses termesc;

const
  ERROR_CREATE_DIR = 'Cannot create model folder';


procedure MakeDir(const dirName:string);
begin
  if not DirectoryExists(dirName) then
    assert(CreateDir(dirName), ERROR_CREATE_DIR+ ' ['+dirName+']');
end;

{ TFLUX4B }

function getFileSize(const filename: TFileName):int64;
var f:TFileStream;
begin
  f := nil;
  result := 0;
  try
    f := TFilestream.Create(filename, fmOpenRead);
    result := f.Size;
  finally
    freeandnil(f)
  end;
end;

procedure getOpenBlas();
var
    FHttp : TNHttp;
    appPath : string;
begin
  {$ifdef MSWINDOWS}
  AppPath := ExtractFilePath(ParamStr(0));
  assert(not fileExists(appPath+'libopenblas.dll'), 'OpenBLAS already exists, Ignore to continue');
  try
    FHttp := TNHttp.Create;
    FHttp.Download('https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip', AppPath+'OpenBLAS-0.3.34-x64.zip');

  finally
    freeAndNil(FHTTP);
  end;
  unzip(AppPath+'OpenBLAS-0.3.34-x64.zip', 'bin/libopenblas.dll', appPath+'libopenblas.dll')
  {$else}
  {$endif}
end;

procedure TFLUX4BDownloader.OnReceive(sender: TObject; const received,
  total: int64);
begin
  FSubReceived:=received;
  FSubTotal:=total;
  if assigned(OnProgress) then OnProgress(FSubReceived, FSubTotal, FReceived, FTotal);
end;

procedure TFLUX4BDownloader.Cancel();
begin
  {$ifdef FPC}
  FHTTP.FHTTP.Terminate;
  {$else}

  {$endif}

end;

procedure TFLUX4BDownloader.download(modelsPath: string; srcPath: string);
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
    i, dsize, fSize : int64;
    //sl : TStringList;
begin
  //sl := TStringList.Create;
  //FCanceled:=false;
  FSubtotal := 0;
  FSubReceived:=0;
  FReceived:=0;
  FTotal:=4;
  FHttp := TNHttp.Create;
  FHttp.OnReceive:=OnReceive;
  try
    if isConsole then cursorShow(false);
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
    if not fileExists(curDir+'model_index.json') then
      fhttp.Download('https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/model_index.json?download=true', curDir+'model_index.json');
    for i := 0 to high(ENCODER_FILES) do begin
        if FHTTP.FHTTP.Terminated then abort;
      currentFile := ENCODER_FILES[i];
      if fileExists(curDir+ENCODER_DIR+PathDelim + currentFile) then continue;
      //fSize := getfileSize(curDir+ENCODER_DIR + '/' + currentFile);
      if isConsole then
        writeln('Downloading ... ', ENCODER_FILES[i]);
      Fhttp.Download(HTTP_ROOT+ENCODER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+ENCODER_DIR + '/' + currentFile);
    end;

    inc(FReceived);
    if assigned(OnProgress) then OnProgress(FSubReceived, FSubTotal, FReceived, FTotal);

    for i := 0 to high(TOKENIZER_FILES) do begin
      if FHTTP.FHTTP.Terminated then abort;
      currentFile := TOKENIZER_FILES[i];
      if fileExists(curDir+TOKENIZER_DIR+PathDelim + currentFile) then continue;
      if isConsole then
        writeln('Downloading ... ', TOKENIZER_FILES[i]);
      Fhttp.Download(HTTP_ROOT+TOKENIZER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+TOKENIZER_DIR+'/' + currentFile);
    end;

    inc(FReceived);
    if assigned(OnProgress) then OnProgress(FSubReceived, FSubTotal, FReceived, FTotal);

    for i := 0 to high(TRANSFORMER_FILES) do begin
      if FHTTP.FHTTP.Terminated then abort;
      currentFile := TRANSFORMER_FILES[i];
      if fileExists(curDir+TRANSFORMER_DIR+PathDelim + currentFile) then continue;
      if isConsole then
        writeln('Downloading ... ', TRANSFORMER_FILES[i]);
      Fhttp.Download(HTTP_ROOT+TRANSFORMER_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+TRANSFORMER_DIR+'/' + currentFile);
    end;

    inc(FReceived);
    if assigned(OnProgress) then OnProgress(FSubReceived, FSubTotal, FReceived, FTotal);

    for i := 0 to high(VAE_FILES) do begin
      if FHTTP.FHTTP.Terminated then abort;
      currentFile := VAE_FILES[i];
      if fileExists(curDir+VAE_DIR+PathDelim + currentFile) then continue;
      if isConsole then
        writeln('Downloading ... ', VAE_FILES[i]);
      Fhttp.Download(HTTP_ROOT+VAE_DIR+ '/' + currentFile + HF_DOWNLOAD_ARG, curDir+VAE_DIR+'/' + currentFile);
    end;
    inc(FReceived);
    if assigned(OnProgress) then OnProgress(FSubReceived, FSubTotal, FReceived, FTotal);
  finally
    // this will execute even when abort
    freeAndNil(Fhttp);
    if isConsole then cursorShow(True);
  end;
  //sl.free;

end;


initialization
  //TFLUX4B.download();
  //ExecuteProcess('cmd',[]);
   //ExecuteProcess('powershell', ['-Command', 'Invoke-WebRequest', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/text_encoder/model-00001-of-00002.safetensors?download=true', '-resume', '-OutFile', './model-00001-of-00002.safetensors']);

end.

