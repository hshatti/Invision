unit mainUnit;
{$ifdef FPC}
{$mode Delphi}
{$endif}
{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls, Buttons, ComCtrls, Spin
  , quicknn_flux, quicknn_zimage, quicknn_downloader, nhttp
  ;

type

  { TMainForm }

  TMainForm = class(TForm)
    BitBtn1: TBitBtn;
    BitBtn2: TBitBtn;
    BitBtn3: TBitBtn;
    BitBtn4: TBitBtn;
    ComboBox1: TComboBox;
    Edit1: TEdit;
    Image1: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Memo1: TMemo;
    Notebook1: TNotebook;
    Panel2: TPanel;
    Panel3: TPanel;
    Panel4: TPanel;
    ProgressBar1: TProgressBar;
    SpinEdit1: TSpinEdit;
    Splitter1: TSplitter;
    txt2img: TPage;
    img2mg: TPage;
    Panel1: TPanel;
    procedure BitBtn1Click(Sender: TObject);
    procedure BitBtn2Click(Sender: TObject);
    procedure BitBtn4Click(Sender: TObject);
    procedure ComboBox1DropDown(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    procedure checkExistingModels;
    procedure OnDataReceive(sender:TObject; const aReadCount, aContentLength:Int64);
  public

  end;

var
  MainForm: TMainForm;

implementation

{$R *.lfm}

const MODELs_DIR = 'models';

var AppPath : RawByteString;
{ TMainForm }

procedure TMainForm.BitBtn1Click(Sender: TObject);
begin
  txt2img.Show;
end;

procedure TMainForm.BitBtn2Click(Sender: TObject);
begin
  img2mg.Show;
end;

procedure TMainForm.BitBtn4Click(Sender: TObject);
begin
  TFLUX4B.download(MODELs_DIR);
end;

procedure TMainForm.ComboBox1DropDown(Sender: TObject);
begin
  checkExistingModels;
end;

procedure TMainForm.FormCreate(Sender: TObject);
begin
  http.FHTTP.OnDataReceived := OnDataReceive;
end;

procedure TMainForm.checkExistingModels;
var sr : TSearchRec;
  r:longint;
  path:RawByteString;
begin
  path := AppPath+MODELs_DIR;
  if not DirectoryExists(Path) then
    CreateDir(Path);

  ComboBox1.Items.Clear;;
  try
    r := FindFirst(path +'\*.*', faDirectory, sr);
    while r=0 do begin
      if (sr.name<>'.') and (sr.name<>'..') then ComboBox1.Items.Add(sr.Name);
      r := FindNext(sr)
    end;
  finally
    FindClose(sr);
  end;
end;

procedure TMainForm.OnDataReceive(sender: TObject; const aReadCount, aContentLength: Int64);
var c:int64;
begin
  if AContentLength>0 then
    c := AContentLength
  else begin
    c := http.FDownloadSize;
    ProgressBar1.Max:=C;
  end;
  ProgressBar1.Caption:=TFLUX4B.currentFile;
  if (AContentLength<=0) and (http.FDownloadSize=0) then exit;
  if AReadCount>=c then
    c := AReadCount;

  if aReadCount=c then // if file download is complete
  if aReadCount=http.FDownloadSize then
    ProgressBar1.StepIt;
  if ProgressBar1.step = 15 then ProgressBar1.Hide;
end;

initialization
  AppPath := ExtractFilePath(ParamStr(0));

end.

