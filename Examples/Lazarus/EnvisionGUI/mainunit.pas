unit mainUnit;
{$ifdef FPC}
{$mode Delphi}
{$endif}
{$pointermath on}
{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls,
  Buttons, ComCtrls, Spin, ExtDlgs, quicknn_transformers, quicknn_common,
  quicknn_vae, quicknn_flux, quicknn_zimage
  , Types;

type

  { TGeneratThread }

  TGenerateThread = class(TThread)
    flux : TQNNFlux;
    zi   : TQNNZImage;
    procedure Execute; override;
    procedure DoTerminate; override;

  end;

  THSProgressStyle = (psNormal, psMarquee, psStrips, psPulse, psPie, psDonut);

  { THSProgressBar }

  THSProgressBar = class(TGraphicControl)
    procedure paint;                         override;
  private
    FBarColor: TColor;
    FBarShowText: boolean;
    FMax: integer;
    FMin: integer;
    FOrientation: TProgressBarOrientation;
    FPosition: integer;
    FStep: integer;
    FStyle: THSProgressStyle;
    procedure SetBarColor(AValue: TColor);
    procedure SetBarShowText(AValue: boolean);
    procedure SetMax(AValue: integer);
    procedure SetMin(AValue: integer);
    procedure SetOrientation(AValue: TProgressBarOrientation);
    procedure SetPosition(AValue: integer);
    procedure SetStep(AValue: integer);
    procedure SetStyle(AValue: THSProgressStyle);
  public
    procedure StepIt();
    constructor Create(AOwner:TComponent);   override;
    destructor Destroy();                    override;
  published
    property Min:integer read FMin write SetMin;
    property Max:integer read FMax write SetMax;
    property Position:integer read FPosition write SetPosition;
    property Step:integer read FStep write SetStep;
    property Orientation : TProgressBarOrientation read FOrientation write SetOrientation;
    property Style : THSProgressStyle read FStyle write SetStyle;
    property BarShowText: boolean read FBarShowText write SetBarShowText;
    property BarColor : TColor read FBarColor write SetBarColor;

  end;

  { TMainForm }

  TMainForm = class(TForm)
    btnGenerate: TBitBtn;
    btnGenerate1: TBitBtn;
    btnImg2Img: TBitBtn;
    btnTxt2Img: TBitBtn;
    cmbModels: TComboBox;
    edtSeed: TEdit;
    edtSeed1: TEdit;
    Image1: TImage;
    Image2: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    memoPrompt: TMemo;
    memoPrompt1: TMemo;
    Notebook1: TNotebook;
    dlgImage: TOpenPictureDialog;
    Panel2: TPanel;
    Panel3: TPanel;
    Panel4: TPanel;
    Panel5: TPanel;
    Panel6: TPanel;
    btnLoadPic: TSpeedButton;
    Splitter2: TSplitter;
    spnSteps: TSpinEdit;
    Splitter1: TSplitter;
    spnSteps1: TSpinEdit;
    txt2img: TPage;
    img2mg: TPage;
    Panel1: TPanel;
    ProgressBar1:THSProgressBar;
    procedure btnGenerateClick(Sender: TObject);
    procedure btnTxt2ImgClick(Sender: TObject);
    procedure btnImg2ImgClick(Sender: TObject);
    procedure cmbModelsDropDown(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure btnLoadPicClick(Sender: TObject);
    procedure applyHsButton;
  private
    procedure checkExistingModels;
  public
    generateThread : TGenerateThread
  end;

var
  MainForm: TMainForm;

implementation
uses
  HSButton, HSButtons
{$ifdef MSWINDOWS}
  , DwmApi
{$endif};

{$R *.lfm}

{$ifdef MSWINDOWS}

function IsWindows10OrGreater(BuildNumber: Integer): Boolean;
begin
  Result := (Win32MajorVersion >= 10) and (Win32BuildNumber >= BuildNumber);
end;

procedure SetDarkModeTitleBar(AForm: TForm; Active: longbool);
const
  DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19;
  DWMWA_USE_IMMERSIVE_DARK_MODE = 20;
var
  Attr: DWord;
begin
  Attr := DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1;
  if IsWindows10OrGreater(18985) then Attr := DWMWA_USE_IMMERSIVE_DARK_MODE;

  DwmSetWindowAttribute(AForm.Handle, Attr, @Active, SizeOf(Active));
end;
{$endif}

{$if defined(MacOS) or defined(DARWIN)}
const MODELs_DIR = '../../../../../models';
{$else}
const MODELs_DIR = '../../models';
{$endif}

var AppPath : RawByteString;

{ TGeneratThread }

procedure OnStepCallback(const msg:TSubstepType; const step, steps:longint);
begin
  MainForm.ProgressBar1.StepIt;
  MainForm.ProgressBar1.caption := format('%d/%d', [MainForm.ProgressBar1.Position, MainForm.ProgressBar1.Max]);
  MainForm.generateThread.Synchronize(MainForm.Update);
  //MainForm.Update;
  if MainForm.generateThread.CheckTerminated then
    abort
end;

procedure OnTextStepCallback(const step, steps:longint);
begin
  MainForm.ProgressBar1.StepIt;
  MainForm.generateThread.Synchronize(MainForm.Update);
  if MainForm.generateThread.CheckTerminated then
    abort
end;

procedure OnPreviewCallback(const step, steps:longint; const latent:TMemoryBlock);
type TRGB = packed record r, g, b:byte end;
     PRGB = ^TRGB;
var img:TQNNImage;
  bmp:Graphics.TBitmap;
  y, x:longint;
  rgb : PRGB;
begin
  img := PVAE(vae_ptr).preview(latent, flux2_latent_rgb_proj, latent.height(), latent.width(), latent.height(), 1, flux2_latent_rgb_bias, 2).resize(128, 128);
  bmp := Graphics.TBitmap.Create;
  bmp.PixelFormat:=pf24bit;
  bmp.SetSize(img.width, img.height);
  bmp.BeginUpdate();
  for y:=0 to img.Height-1 do begin
    RGB := bmp.ScanLine[y];
    for x :=0 to img.width-1 do begin
      rgb[x].r := img.data[y*img.width*3 + x*3+2];
      rgb[x].g := img.data[y*img.width*3 + x*3+1];
      rgb[x].b := img.data[y*img.width*3 + x*3];
    end;
  end;
  bmp.EndUpdate();
  if MainForm.Notebook1.PageIndex=0 then begin
    mainform.Image1.Picture.Graphic:= bmp;
    //MainForm.generateThread.Synchronize(MainForm.Image1.Repaint);
  end;
  if MainForm.Notebook1.PageIndex=1 then begin
    mainform.Image2.Picture.Graphic:= bmp;
    //MainForm.generateThread.Synchronize(MainForm.Image2.Repaint);
  end;
  MainForm.generateThread.Synchronize(MainForm.Repaint);
  img.free;
  freeAndNil(bmp)
end;

procedure TGenerateThread.Execute;
var
  params:TGenerateParams;
  img:TQNNImage;
  fn : TFileName;
begin
  with MainForm do begin
    if Notebook1.PageIndex=0 then try // txt2Img
      params := default(TGenerateParams);
      params.num_steps := spnSteps.Value;
      params.guidance  := 0;
      params.powerAlpha:= 2.0;
      if (trim(edtSeed.Text)='') or (strToInt(edtSeed.text)<=0) then
        params.seed := random(Int64.MaxValue)
      else
        params.seed := strToInt(edtSeed.text);

      MainForm.generateThread.Synchronize(MainForm.Update);

      substep_callback:=OnStepCallback;
      text_progress_callback:= OnTextStepCallback;
      vae_progress_callback:= OnTextStepCallback;
      flux := TQNNFlux.load(AppPath+MODELS_DIR+'/'+cmbModels.Text);
      ProgressBar1.Max:= 27 + params.num_steps*(5 + 20 + 2) + 16;  // text_encode_steps + steps*transformer_blocks + ve_decoder_steps
      ProgressBar1.Position := 0;
      flux.use_mmap:=true;
      img := flux.generate(memoPrompt.Text, params, OnPreviewCallback);
      fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';
      img.saveToFile(fn);
      image1.Picture.LoadFromFile(fn);

    finally
      flux.free();
      img.free
    end;

    if Notebook1.PageIndex=1 then try // img2Img
      if not fileExists(dlgImage.FileName) then begin
        ShowMessage('Load an image 1st.');
        exit
      end;

      params := default(TGenerateParams);
      params.num_steps := spnSteps1.Value;
      params.guidance  := 0;
      params.powerAlpha:= 2.0;
      if (trim(edtSeed1.Text)='') or (strToInt(edtSeed1.text)<=0) then
        params.seed := random(Int64.MaxValue)
      else
        params.seed := strToInt(edtSeed1.text);

      MainForm.generateThread.Synchronize(MainForm.Update);

      img := TQNNImage.loadFromFile(dlgImage.FileName);
      substep_callback:=OnStepCallback;
      text_progress_callback:= OnTextStepCallback;
      vae_progress_callback:= OnTextStepCallback;
      flux := TQNNFlux.load(MODELs_DIR+'/'+cmbModels.Text);
      ProgressBar1.Max:= 27 + params.num_steps*(5 + 20 + 2) + 16;  // text_encode_steps + steps*transformer_blocks + ve_decoder_steps
      ProgressBar1.Position := 0;
      flux.use_mmap:=true;
      img := flux.generate(memoPrompt1.Text, params, img, OnPreviewCallback);
      fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';
      img.saveToFile(fn);
      image2.Picture.LoadFromFile(fn);

    finally
      flux.free();
      img.free
    end;
  end;
end;

procedure TGenerateThread.DoTerminate;
begin
  with MainForm do begin
    ProgressBar1.Position := 0;
    if Notebook1.PageIndex=0 then
       TControl(btnGenerate).Caption := 'Generate';
    if Notebook1.PageIndex=1 then
       TControl(btnGenerate1).Caption := 'Generate';
    TControl(btnTxt2Img).Enabled:=True;
    TControl(btnImg2Img).Enabled:=True;
    generateThread.Synchronize(MainForm.Repaint);
  end;
  inherited DoTerminate;
end;

{ THSProgressBar }
{$Assertions on}
constructor THSProgressBar.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FMin := 0;
  FMax := 10;
  FStep:= 1;
  FBarColor := clLime;
end;

destructor THSProgressBar.Destroy();
begin
  inherited Destroy();
end;

procedure THSProgressBar.paint;
var ts:TTextStyle;
begin
  inherited paint;
  with canvas do begin
    Pen.Style := psClear;
    Brush.Style := bsSolid;
    Brush.Color := Color;
    FillRect(Canvas.ClipRect);
    Brush.Style := bsSolid;
    Brush.Color := FBarColor;
    case FOrientation of
      pbVertical:
        RoundRect(0, 0, ClientWidth, round(ClientHeight*FPosition/FMax), 4, 4);
      pbHorizontal:
        RoundRect(0, 0, round(ClientWidth*FPosition/FMax), ClientHeight, 4, 4);
      pbTopDown:
        RoundRect(0, round(ClientHeight*FPosition/FMax), ClientWidth, ClientHeight, 4, 4);
      pbRightToLeft:
        RoundRect(round(ClientWidth*FPosition/FMax), 0, ClientWidth, ClientHeight, 4, 4);
    end;
    if FBarShowText then begin
      font.assign(self.font);
      brush.Style:=bsClear;
      ts:= TextStyle;
      ts.Alignment := taCenter;
      ts.Layout := tlCenter;
      TextStyle := ts;
      TextRect(ClientRect, 0, 0, Caption);
    end;
  end;
end;

procedure THSProgressBar.SetBarShowText(AValue: boolean);
begin
  if FBarShowText=AValue then Exit;
  FBarShowText:=AValue;
end;

procedure THSProgressBar.SetBarColor(AValue: TColor);
begin
  if FBarColor=AValue then Exit;
  FBarColor:=AValue;
end;

procedure THSProgressBar.SetMax(AValue: integer);
begin
  if FMax=AValue then Exit;
  FMax:=AValue;
end;

procedure THSProgressBar.SetMin(AValue: integer);
begin
  if FMin=AValue then Exit;
  FMin:=AValue;
end;

procedure THSProgressBar.SetOrientation(AValue: TProgressBarOrientation);
begin
  if FOrientation=AValue then Exit;
  FOrientation:=AValue;
end;

procedure THSProgressBar.SetPosition(AValue: integer);
begin
  if FPosition=AValue then Exit;
  assert((AValue>=FMin) and (AValue<=FMax),'[ProgressBar] Position is out of range!');
  FPosition := AValue;
  Invalidate;
end;

procedure THSProgressBar.SetStep(AValue: integer);
begin
  if FStep=AValue then Exit;
  FStep:=AValue;
end;

procedure THSProgressBar.SetStyle(AValue: THSProgressStyle);
begin
  if FStyle=AValue then Exit;
  FStyle:=AValue;
  Invalidate;
end;

procedure THSProgressBar.StepIt();
begin
  inc(FPosition, FStep);
  Invalidate;
end;

{ TMainForm }

procedure TMainForm.btnTxt2ImgClick(Sender: TObject);
begin
  txt2img.Show;
end;

procedure TMainForm.btnGenerateClick(Sender: TObject);
var params : TGenerateParams;
begin
  if Notebook1.PageIndex=0 then begin
    if (btnGenerate.Caption = 'Generate') then begin
      TControl(btnGenerate).Caption := 'Stop';
      TControl(btnTxt2Img).Enabled:=false;
      TControl(btnImg2Img).Enabled:=false;
      generateThread := TGenerateThread.Create(false);
    end
    else begin
      if not generateThread.Finished then generateThread.Terminate;
      //btnGenerate.Caption := 'Generate';
    end;

  end;
  if Notebook1.PageIndex=1 then begin
    if btnGenerate1.Caption = 'Generate' then begin
      TControl(btnGenerate1).Caption := 'Stop';
      TControl(btnTxt2Img).Enabled:=false;
      TControl(btnImg2Img).Enabled:=false;
      generateThread := TGenerateThread.Create(false);
    end
    else begin
      if not generateThread.Finished then generateThread.Terminate;
      //btnGenerate.Caption := 'Generate';
    end;

  end;

end;

procedure TMainForm.btnImg2ImgClick(Sender: TObject);
begin
  img2mg.Show;
end;

procedure TMainForm.cmbModelsDropDown(Sender: TObject);
begin
  checkExistingModels;
end;

procedure TMainForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if assigned(generateThread) and not generateThread.Finished then begin
    generateThread.Terminate;
    generateThread.WaitFor;
  end;

end;

procedure TMainForm.FormCreate(Sender: TObject);
begin
  checkExistingModels;
  if cmbModels.Items.count>0 then
    cmbModels.ItemIndex:=0;
  progressBar1 := THSProgressBar.Create(Self);
  ProgressBar1.Parent := Self;
  ProgressBar1.Height := 12;
  ProgressBar1.Align  := alBottom;
  {$ifdef MSWindows}
  SetDarkModeTitleBar(self, true);
  {$endif}
  applyHsButton;
end;

procedure TMainForm.btnLoadPicClick(Sender: TObject);
type TRGB = packed record r, g, b:byte end;
     PRGB = ^TRGB;
var img:TQNNImage;
  bmp:Graphics.TBitmap;
  y, x:longint;
  rgb:PRGB;
begin
  if not dlgImage.Execute  then exit;
  img := TQNNImage.loadFromFile(dlgImage.FileName);
  bmp := Graphics.TBitmap.Create;
  bmp.PixelFormat:=pf24bit;
  bmp.SetSize(img.width, img.height);
  bmp.BeginUpdate();
  for y:=0 to img.Height-1 do begin
    RGB := bmp.ScanLine[y];
    for x :=0 to img.width-1 do begin
      rgb[x].r := img.data[y*img.width*3 + x*3+2];
      rgb[x].g := img.data[y*img.width*3 + x*3+1];
      rgb[x].b := img.data[y*img.width*3 + x*3];
    end;
  end;
  bmp.EndUpdate();
  mainform.Image2.Picture.Graphic:= bmp;
  MainForm.generateThread.Synchronize(MainForm.Update);
  img.free;
  freeAndNil(bmp)
end;


procedure TMainForm.applyHsButton;
var i:longint;
  ctrl : THSButton;
  src1 : TBitBtn;

  n,c : string;
begin
  //for i:=0 to ComponentCount-1 do
  i :=0;
  while i< ComponentCount do begin
    if (Components[i] is TBitBtn) then begin
      src1 := TBitBtn(Components[i]);
      ctrl := THSButton.Create(Self); // the ownwer will free the button so no memory leaks in theory
      ctrl.SetBounds(src1.Left, src1.Top, src1.Width, src1.Height);
      ctrl.Visible := src1.Visible;
      ctrl.Parent  := src1.Parent;
      ctrl.anchors  := src1.anchors;
      ctrl.Align   := src1.Align;
      ctrl.OnClick := src1.OnClick;
      ctrl.Glyph   := src1.Glyph;
      ctrl.BorderSpacing.assign(src1.BorderSpacing);
      //ctrl.Caption := src1.Caption;
      ctrl.Layout  := src1.Layout;
      ctrl.Color   := src1.Color;
      ctrl.Smooth  := 0;
      ctrl.Style   := THSButtonStyle.bsRounded;
      ctrl.Border  := 4;
      ctrl.TabOrder:= src1.TabOrder;
      ctrl.SlowDecease:= sdEnterLeave;
      ctrl.SlowDeceaseSpeed:=16;
      ctrl.DrawFocus:=false;
      ctrl.Hint:=src1.Hint;
      ctrl.ShowHint := src1.ShowHint;
      n := src1.Name;// can do? no we cant
      c := src1.Caption;
      Components[i].Free;
      ctrl.Name := n;
      ctrl.Caption:= c;
    end
    //else if (Components[i] is TProgressBar) then begin
    //  src2 := TProgressBar(Components[i]);
    //  bar := THSProgressBar.Create(Self); // the ownwer will free the button so no memory leaks in theory
    //  bar.SetBounds(src2.Left, src2.Top, src2.Width, src2.Height);
    //  bar.Parent  := src2.Parent;
    //  bar.Align   := src2.Align;
    //  bar.Visible := src2.Visible;
    //  bar.Caption := src2.Caption;
    //  bar.Color   := src2.Color;
    //  bar.Step    := src2.Step;
    //  bar.Min     := src2.Min;
    //  bar.Max     := src2.Max;
    //  bar.Position:= src2.Position;
    //  n := src2.Name;
    //  Components[i].Free;
    //  bar.Name := n;
    //end
    else
      inc(i);
  end;
end;

procedure TMainForm.checkExistingModels;
var sr : TSearchRec;
  r:longint;
  path:RawByteString;
begin
  path := AppPath+MODELs_DIR;
  if not DirectoryExists(Path) then
    CreateDir(Path);

  cmbModels.Items.Clear;;
  try
    r := FindFirst(path +'/*', faDirectory, sr);
    while r=0 do begin
      if (sr.name<>'.') and (sr.name<>'..') then cmbModels.Items.Add(sr.Name);
      r := FindNext(sr)
    end;
  finally
    FindClose(sr);
  end;
end;

initialization
  AppPath := ExtractFilePath(ParamStr(0));

end.

