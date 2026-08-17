//{$apptype console}
unit mainUnit;
{$ifdef FPC}
{$mode Delphi}
{$endif}
{$pointermath on}
{$assertions on}
{$H+}

interface

uses
  Classes, SysUtils, Types, LCLType, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls,
  Buttons, ComCtrls, Spin, ExtDlgs, MaskEdit, quicknn_transformers,
  quicknn_common, quicknn_vae, quicknn_flux, quicknn_zimage;

type

  { TGeneratThread }

  TGenerateThread = class(TThread)
    flux : TQNNFlux;
    zi   : TQNNZImage;
    procedure Execute; override;
    procedure DoTerminate; override;

  end;

  { THSEdit }

  THSEdit = class(TCustomControl)
  const BTN_WIDTH = 20;
  type
    TBtn=class(TCustomControl)
      published
        Property OnMouseEnter;
        Property OnMouseLeave;
        Property OnMouseDown;
        Property OnMouseUp;
    end;
  private
    FItemIndex: integer;
    FItems: TStrings;
    FText: string;
    FEdit : TCustomEdit;
    FBtn: TBtn;
    FList :TCustomListBox;
    procedure FOnBtnPaint(Sender:TObject);
    procedure FOnItemsChange(Sender:TObject);
    procedure FOnListSelectionChange(Sender:TObject;user:boolean);
    procedure FOnListClick(sender:TObject);
    procedure FOnListKeyDown(sender:TObject; var key:word; shift:TShiftState);
    procedure FOnEditChange(Sender:TObject);
    procedure FOnEditKeyDown(sender:TObject; var key:word; shift:TShiftState);
    procedure FOnListExit(Sender:TObject);
    procedure FBtnEnter(Sender:TObject);
    procedure FBtnLeave(Sender:TObject);
    procedure FBtnClick(Sender:TObject);
    procedure FOnListBoxSetVisible(Sender:TObject);
    procedure SetItemIndex(AValue: integer);
    procedure SetItems(AValue: TStrings);
    procedure SetText(AValue: string);
    // gets the left and top reltive to the top form
    function getLocation():TRect;
  protected
    procedure SetColor(Value: TColor); override;
  public
    constructor Create(AOwner: TComponent); override;
    procedure paint;override;
    destructor Destroy; override;
    procedure FOnItemChange(Sender:TObject);
  published
    property ItemIndex:integer read FItemIndex write SetItemIndex;
    property Text : string read FText write SetText;
    property Items:TStrings read FItems write SetItems;

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
    btnGenerate: TSpeedButton;
    btnGenerate1: TSpeedButton;
    btnImg2Img: TSpeedButton;
    btnTxt2Img: TSpeedButton;
    cmbModels: TComboBox;
    cmbImgRes: TComboBox;
    edtSeed: TEdit;
    edtSeed1: TEdit;
    Image1: TImage;
    Image2: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
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
    img2img: TPage;
    Panel1: TPanel;
    ProgressBar1:THSProgressBar;
    procedure btnGenerateClick(Sender: TObject);
    procedure btnTxt2ImgClick(Sender: TObject);
    procedure btnImg2ImgClick(Sender: TObject);
    procedure cmbModelsDropDown(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure btnLoadPicClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
  private
    procedure checkExistingModels;
  public
    modeledit1, imgRes:THSEdit;
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
  MainForm.ProgressBar1.caption := format('%d/%d', [MainForm.ProgressBar1.Position, MainForm.ProgressBar1.Max]);
  MainForm.generateThread.Synchronize(MainForm.Update);
  if MainForm.generateThread.CheckTerminated then
    abort
end;

procedure OnPreviewCallback(const step, steps:longint; const latent:TMemoryBlock);
type

     TRGB = packed record r, g, b, a:byte end;
     PRGB = ^TRGB;
var img:TQNNImage;
  bmp:Graphics.TBitmap;
  y, x:longint;
  rgb : PRGB;
begin
  img := PVAE(vae_ptr).preview(latent, flux2_latent_rgb_proj, latent.height(), latent.width(), latent.height(), 1, flux2_latent_rgb_bias, 2).resize(128, 128);
  bmp := Graphics.TBitmap.Create;
{$ifdef MSWINDOWS}
  bmp.PixelFormat:=pf32bit;
{$else}
  bmp.PixelFormat:=pf24bit;
{$endif}
  bmp.SetSize(img.width, img.height);
  bmp.BeginUpdate();
  for y:=0 to img.Height-1 do begin
    RGB := bmp.ScanLine[y];
    for x :=0 to img.width-1 do begin
      rgb[x].r := img.data[y*img.width*3 + x*3+2];
      rgb[x].g := img.data[y*img.width*3 + x*3+1];
      rgb[x].b := img.data[y*img.width*3 + x*3];
{$ifdef MSWINDOWS}
      rgb[x].a := 255;
{$endif}
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
  strRes : array of string;
  imWidth, imHeight:longInt;
begin
  strRes := string(MainForm.cmbImgRes.Text).Split(' ');

  assert((length(strRes)=3) and (TryStrToInt(strRes[0], imWidth)) and (TryStrToInt(strRes[2], imHeight)), 'Incorrect image dimensions!');
  with MainForm do begin
    if Notebook1.PageIndex=0 then try // txt2Img
      params := default(TGenerateParams);
      params.width:=imWidth;
      params.height:=imHeight;
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
      ProgressBar1.Position := 0;
      if LowerCase(cmbModels.Text).Contains('flux') then begin
        flux := TQNNFlux.load(AppPath+MODELS_DIR+'/'+cmbModels.Text);
        ProgressBar1.Max:= 27 + params.num_steps*(5 + 20 + 2) + 16;  // text_encode_steps + steps*transformer_blocks + ve_decoder_steps
        flux.use_mmap:=true;
        img := flux.generate(memoPrompt.Text, params, OnPreviewCallback);
      end;
      if lowerCase(cmbModels.Text).Contains('z-image') then begin
        zi := TQNNZImage.load(AppPath+MODELS_DIR+'/'+cmbModels.Text);
        ProgressBar1.Max:= 35 + params.num_steps*(2 + 2 + 44) + 15;  // text_encode_steps + steps*transformer_blocks + ve_decoder_step;
        zi.use_mmap:=true;
        img := zi.generate(memoPrompt.Text, params{, OnPreviewCallback});
      end;
      fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';
      img.saveToFile(fn);
      image1.Picture.LoadFromFile(fn);

    finally
      if flux.transformer.isLoaded() then flux.free();
      if zi.transformer.isLoaded() then zi.free();
      img.free
    end;

    if Notebook1.PageIndex=1 then try // img2Img

      params := default(TGenerateParams);
      params.num_steps := spnSteps1.Value;
      params.guidance  := 0;
      params.width:=imWidth;
      params.height:=imHeight;

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

{ THSEdit }

procedure THSEdit.FOnBtnPaint(Sender: TObject);
var ts:TTextStyle;
begin
  FBtn.Canvas.Pen.Color:=$c0c0c0;
  FBtn.Canvas.Pen.Cosmetic:=false;
  FBtn.Canvas.Pen.Width:=1;
  if FBtn.MouseInClient then
    FBtn.Canvas.Brush.Color:=ColorBright(FBtn.Color, 20)
  else
    FBtn.Canvas.Brush.Color:=FBtn.Color;
  FBtn.Canvas.RoundRect(FBtn.ClientRect, 0, 0);
  FBtn.Font.Color:=parent.Font.Color;
  with TCustomControl(Sender).Canvas do begin
    ts := TextStyle;
    ts.Alignment:=taCenter;
    ts.Layout:=tlCenter;
    TextStyle:=ts;
    TextRect(FBtn.ClientRect, 0, 0, FBtn.Caption);
  end
end;

procedure THSEdit.FOnItemsChange(Sender: TObject);
begin
  FList.Items.Text:=Items.Text;
end;

procedure THSEdit.FOnListSelectionChange(Sender: TObject; user: boolean);
begin
  if FList.ItemIndex>=0 then begin
    FEdit.Text := FList.Items[FList.ItemIndex];
  end;
  //FList.Hide
end;

procedure THSEdit.FOnListClick(sender: TObject);
begin
  //FList.Hide;
end;

procedure THSEdit.FOnListKeyDown(sender: TObject; var key: word;
  shift: TShiftState);
begin
  case key of
    VK_RETURN:
      FList.Click;
    //VK_UP: if FList.ItemIndex>0 then
    //  FList.ItemIndex := FList.ItemIndex -1;
    //VK_DOWN: if FList.ItemIndex<FList.Items.Count-1 then
    //  FList.ItemIndex := FList.ItemIndex +1;

  end;
end;

procedure THSEdit.FOnEditChange(Sender: TObject);
begin
  Text:=FEdit.Text;
end;

procedure THSEdit.FOnEditKeyDown(sender: TObject; var key: word;
  shift: TShiftState);
begin
  case key of
    VK_RETURN: FList.Click;
    VK_UP:begin
      if FList.ItemIndex>0 then
        FList.ItemIndex := FList.ItemIndex -1;
      key := 0
    end;
    VK_DOWN: begin
      if FList.ItemIndex<FList.Items.Count-1 then
        FList.ItemIndex := FList.ItemIndex +1;
      key := 0;
    end;
    VK_ESCAPE:
      FList.Hide;
  end;
end;

procedure THSEdit.FOnListExit(Sender: TObject);
begin
  //Flist.Hide
end;

procedure THSEdit.FBtnEnter(Sender: TObject);
begin
  FBtn.Color:=ColorBright(FBtn.Color, 20);
end;

procedure THSEdit.FBtnLeave(Sender: TObject);
begin
  FBtn.Color:=ColorBright(FBtn.Color, -20);
end;

procedure THSEdit.FBtnClick(Sender: TObject);
begin
  //FList.Items.Text := Items.Text;
  FList.Visible := not FList.Visible;
  if FList.Visible and FList.CanSetFocus then begin
    FList.Update;
    FList.SetFocus();
  end;

end;

procedure THSEdit.FOnListBoxSetVisible(Sender: TObject);
var
  r:TRect;
begin
  if not Visible then exit;
  FList.Parent := TWinControl(GetTopParent);
  R := getLocation();
  FList.Left:= FEdit.Left  + R.Left;
  FList.Top:=  FEdit.Top + FEdit.height + R.Top;

  Flist.Width  := FEdit.Width;
end;

procedure THSEdit.SetItemIndex(AValue: integer);
begin
  if FItemIndex=AValue then Exit;
  FItemIndex:=AValue;
  FList.ItemIndex:=AValue;
end;

procedure THSEdit.SetItems(AValue: TStrings);
begin
  assert(assigned(AValue), 'Items cannot be nil!');
  if FItems=AValue then Exit;
  if assigned(FItems) then FItems.free;
  FItems:=AValue;
end;

procedure THSEdit.SetText(AValue: string);
begin
  if FText=AValue then Exit;
  FText:=AValue;
end;

function THSEdit.getLocation(): TRect;
var control:TWinControl;
begin
  result := Default(TRect);
  result.Width  :=Width;
  result.Height :=Height;
  control:=Self;
  while control.parent<>nil do begin
    result.Left := result.Left + control.left;
    result.Top := result.Top + control.Top;
    control := control.parent;
  end;
end;

procedure THSEdit.SetColor(Value: TColor);
begin
  inherited SetColor(Value);
  FEdit.Color:=ColorBright(Value, -20);
end;


constructor THSEdit.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  width := 100;
  height:=20;
  FItems := TStringList.Create;
  FItemIndex:=-1;
  FEdit :=TCustomEdit.Create(Self);
  FEdit.Parent:=Self;
  FEdit.Color := Color;

  FEdit.BorderStyle:=bsNone;
  FEdit.SetBounds(1, 1, Width-BTN_WIDTH, Height-2);
  FEdit.Anchors := [akLeft, akTop, akRight, akbottom];
  //FEdit.Alignment:=taVerticalCenter;

  FList := TCustomListBox.Create(self);
  FList.BorderStyle:=bsNone;
  FList.Color := ColorBright(Color, -20);
  //FList.Font.Color:=Font.Color;
  FList.OnSelectionChange:=FOnListSelectionChange;
  FEdit.OnKeyDown:=FOnEditKeyDown;
  FList.OnClick:=FOnListClick;
  FList.OnKeyDown:=FOnListKeyDown;
  FList.AddHandlerOnVisibleChanged(FOnListBoxSetVisible);
  TStringList(FItems).OnChange:=FOnItemsChange;
  FList.Visible := false;
  Flist.Height := 100;

  FBtn  :=TBtn.Create(Self);
  FBtn.Color := $cc7700;
  FBtn.Parent := Self;
  FBtn.SetBounds(width - BTN_WIDTH, 1, BTN_WIDTH, height-2);
  FBtn.Anchors:=[akTop, akRight, akBottom];
  FBtn.OnPaint:=FOnBtnPaint;
  FBtn.OnMouseEnter:=FBtnEnter;
  FBtn.OnMouseLeave:=FBtnLeave;
  FBtn.OnClick:=FBtnClick;

  FBtn.Caption:='▼';


  //FBtn.SetBounds(width - 16, 2, 16, height - 4);

end;

procedure THSEdit.paint;
begin
  inherited paint;
end;

destructor THSEdit.Destroy;
begin
  Fedit.free;
  Flist.free;
  Fbtn .free;
  FItems.Free;
  inherited Destroy;
end;

procedure THSEdit.FOnItemChange(Sender: TObject);
begin
  FList.Items.Text:=FItems.Text;
end;

{ THSProgressBar }
{$Assertions on}
constructor THSProgressBar.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FMin := 0;
  FMax := 10;
  FStep:= 1;
  FBarColor := $bb6600;
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
    if not fileExists(dlgImage.FileName) then begin
      ShowMessage('Load an image 1st.');
      exit
    end;
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
  img2img.Show;
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
  cmbModels.hide;
  cmbImgRes.hide;

  modelEdit1:= THSEdit.Create(Self);
  modeledit1.Parent:=Panel4;
  modelEdit1.SetBounds(cmbModels.Left, cmbModels.Top, cmbModels.Width, cmbModels.Height);
  modelEdit1.Anchors:=[akLeft, akTop, akRight];
  modeledit1.Items.Text:=cmbModels.Items.Text;
  if modeledit1.Items.Count>0 then
    modeledit1.ItemIndex:=0;

  imgRes := THSEdit.Create(Self);
  imgRes.parent := panel4;
  imgRes.SetBounds(cmbImgRes.Left, cmbImgRes.Top, cmbImgRes.Width, cmbImgRes.Height);
  modelEdit1.Anchors:=[akLeft, akTop];
  imgRes.items.Text := cmbImgRes.items.text;
  if imgRes.Items.Count>0 then
    imgRes.ItemIndex:=0;

  progressBar1 := THSProgressBar.Create(Self);
  progressBar1.BarShowText:=true;
  ProgressBar1.Parent := Self;
  ProgressBar1.Height := 12;
  ProgressBar1.Align  := alBottom;
  {$ifdef MSWindows}
  SetDarkModeTitleBar(self, true);
  {$endif}
end;

procedure TMainForm.btnLoadPicClick(Sender: TObject);
type

     TRGB = packed record r, g, b, a:byte end;
     PRGB = ^TRGB;
var img:TQNNImage;
  bmp:Graphics.TBitmap;
  y, x:longint;
  rgb:PRGB;
begin
  if not dlgImage.Execute  then exit;
  img := TQNNImage.loadFromFile(dlgImage.FileName);
  bmp := Graphics.TBitmap.Create;
  {$ifdef MSWINDOWS}
    bmp.PixelFormat:=pf32bit;
  {$else}
    bmp.PixelFormat:=pf24bit;
  {$endif}
  bmp.SetSize(img.width, img.height);
  bmp.BeginUpdate();
  for y:=0 to img.Height-1 do begin
    RGB := bmp.ScanLine[y];
    for x :=0 to img.width-1 do begin
      rgb[x].r := img.data[y*img.width*3 + x*3+2];
      rgb[x].g := img.data[y*img.width*3 + x*3+1];
      rgb[x].b := img.data[y*img.width*3 + x*3];
{$ifdef MSWINDOWS}
      rgb[x].a := 255;
{$endif}
    end;
  end;
  bmp.EndUpdate();
  mainform.Image2.Picture.Graphic:= bmp;
  MainForm.generateThread.Synchronize(MainForm.Update);
  img.free;
  freeAndNil(bmp)
end;


procedure TMainForm.FormShow(Sender: TObject);
begin
  //modeledit1.parent := self;
  //
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

