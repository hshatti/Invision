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
  Buttons, ComCtrls, Spin, ExtDlgs, Menus, quicknn_transformers,
  quicknn_common, quicknn_vae, quicknn_flux, quicknn_zimage, quicknn_downloader;

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
    FOnChange : TNotifyEvent;
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
    function GetText: string;
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
    property Text : string read GetText write SetText;
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
    //FMin: integer;
    FOrientation: TProgressBarOrientation;
    FPosition: integer;
    FStep: integer;
    FStyle: THSProgressStyle;
    procedure SetBarColor(AValue: TColor);
    procedure SetBarShowText(AValue: boolean);
    procedure SetMax(AValue: integer);
    //procedure SetMin(AValue: integer);
    procedure SetOrientation(AValue: TProgressBarOrientation);
    procedure SetPosition(AValue: integer);
    procedure SetStep(AValue: integer);
    procedure SetStyle(AValue: THSProgressStyle);
  public
    procedure StepIt();
    constructor Create(AOwner:TComponent);   override;
    destructor Destroy();                    override;
  published
    //property Min:integer read FMin write SetMin;
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
    btnImg2Img: TSpeedButton;
    btnLoadPic1: TLabel;
    btnTxt2Img: TSpeedButton;
    cmbImgRes: TComboBox;
    cmbModels: TComboBox;
    cmbSchedular: TComboBox;
    edtCFG: TEdit;
    edtPowerAlpha: TEdit;
    edtSeed: TEdit;
    Image1: TImage;
    Image2: TImage;
    ImageList1: TImageList;
    Label1: TLabel;
    btnLoadPic: TLabel;
    Label10: TLabel;
    lblDownload: TLabel;
    lblMore: TLabel;
    lblSetAsRef: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    dlgImage: TOpenPictureDialog;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    memoPrompt: TMemo;
    memoNegPrompt: TMemo;
    MenuItem1: TMenuItem;
    MenuItem2: TMenuItem;
    mnuDLOpenBLAS: TMenuItem;
    mnuDL1: TMenuItem;
    mnuDL2: TMenuItem;
    Panel2: TPanel;
    Panel3: TPanel;
    Panel4: TPanel;
    Panel1: TPanel;
    Panel5: TPanel;
    Panel6: TPanel;
    Panel7: TPanel;
    pnlDL: TPanel;
    pnlPrompt: TPanel;
    pnlNegative: TPanel;
    PopupMenu1: TPopupMenu;
    Separator1: TMenuItem;
    Separator2: TMenuItem;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    Splitter3: TSplitter;
    spnSteps: TSpinEdit;
    prog, subDL, totalDL :THSProgressBar;
    procedure btnGenerateClick(Sender: TObject);
    procedure btnLoadPic1Click(Sender: TObject);
    procedure btnTxt2ImgClick(Sender: TObject);
    procedure btnImg2ImgClick(Sender: TObject);
    procedure cmbModelsDropDown(Sender: TObject);
    procedure edtCFGChange(Sender: TObject);
    procedure edtCFGMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure btnLoadPicClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure btnLoadPicMouseEnter(Sender: TObject);
    procedure btnLoadPicMouseLeave(Sender: TObject);
    procedure Image1DblClick(Sender: TObject);
    procedure Label10Click(Sender: TObject);
    procedure lblMoreClick(Sender: TObject);
    procedure lblMoreMouseEnter(Sender: TObject);
    procedure lblMoreMouseLeave(Sender: TObject);
    procedure lblSetAsRefClick(Sender: TObject);
    procedure MenuItem2Click(Sender: TObject);
    procedure mnuDL1Click(Sender: TObject);
    procedure mnuDLOpenBLASClick(Sender: TObject);
  private
    procedure checkExistingModels;
    procedure OnDownload(const subReceived, subTotal, received, total:int64);
  public
    modeledit1, imgRes, schedularEdit1:THSEdit;
    generateThread : TGenerateThread
  end;

var
  MainForm: TMainForm;
  gotImage:boolean = false;
  gotImage2:boolean = false;
  lastGenerated : string = '';

  procedure QNNImageToBitmap(const img:TQNNImage; var bmp:TBitmap);
  const DL_Symbols : array of rawbytestring = ['   ', '  .', ' ..', '...', '.. ', '.  '];
implementation
uses
  math
  ,unitAbout
{$ifdef MSWINDOWS}
  ,DwmApi
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
const MODELS_DIR = '../../../../../models';
{$else}
const MODELS_DIR = '../../models';
{$endif}

var AppPath : RawByteString;

{ TGeneratThread }

procedure OnStepCallback(const msg:TSubstepType; const step, steps:longint);
begin
  MainForm.prog.StepIt;
  MainForm.prog.caption := format('%d/%d', [MainForm.Prog.Position, MainForm.Prog.Max]);
  MainForm.generateThread.Synchronize(MainForm.Update);
  //MainForm.Update;
  if MainForm.generateThread.CheckTerminated then
    abort
end;

procedure OnTextStepCallback(const step, steps:longint);
begin
  MainForm.Prog.StepIt;
  MainForm.Prog.caption := format('%d/%d', [MainForm.Prog.Position, MainForm.Prog.Max]);
  MainForm.generateThread.Synchronize(MainForm.Update);
  if MainForm.generateThread.CheckTerminated then
    abort
end;

procedure OnPreviewCallback(const step, steps:longint; const latent:TMemoryBlock);
var
  img:TQNNImage;
  bmp:Graphics.TBitmap;
begin
  img := PVAE(vae_ptr).preview(latent, flux2_latent_rgb_proj, latent.height(), latent.width(), latent.height(), 1, flux2_latent_rgb_bias, 2).resize(128, 128);
  bmp := Graphics.TBitmap.Create;
  QNNImageToBitmap(img, bmp);
  if MainForm.btnTxt2Img.Down then begin
    mainform.Image1.Picture.Graphic:= bmp;
    //MainForm.generateThread.Synchronize(MainForm.Image1.Repaint);
  end;
  if MainForm.btnImg2Img.Down then begin
    mainform.Image2.Picture.Graphic:= bmp;
    //MainForm.generateThread.Synchronize(MainForm.Image2.Repaint);
  end;
  MainForm.generateThread.Synchronize(MainForm.Repaint);
  img.free;
  freeAndNil(bmp)
end;

function Limit(V,Min,Max:Integer):Integer;
begin
  if V>Max then
    Result:=Max
  else if V<Min then
    Result:=Min
  else Result:=V

end;

function ColorBright(C:TColor;Brightness:Integer):TColor;
begin
//  C:=ColorToRGB(C);
  Result:=RGBToColor(EnsureRange(red(c)+Brightness, 0, 255), EnsureRange(green(c)+Brightness, 0, 255), EnsureRange(blue(c)+Brightness, 0, 255))
end;

procedure QNNImageToBitmap(const img: TQNNImage; var bmp: TBitmap);
type
  TRGB = packed record r, g, b, a:byte end;
  PRGB = ^TRGB;
var
  y, x:longint;
  rgb:PRGB;
begin
    assert(assigned(bmp), 'ERROR [QNNImageToBMP]: target cannot be nil');
    assert(assigned(img.data), 'ERROR [QNNImageToBMP]: source cannot be nil');
    bmp.Clear;
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
end;

var
  params:TGenerateParams;
const
  strMeta = 'EnvisionGUI, a (txt2img/img2img) generator example written in Object Pascal (Delphi and FPC)';
procedure TGenerateThread.Execute;
var
  img:TQNNImage;
  fn : TFileName;
  strRes : array of string;
  s:string;
  imWidth, imHeight:longInt;
begin
  strRes := string(MainForm.imgRes.Text).Split(' ');

  assert((length(strRes)=3) and (TryStrToInt(strRes[0], imWidth)) and (TryStrToInt(strRes[2], imHeight)), 'Incorrect image dimensions!');


  with MainForm do begin
    params := default(TGenerateParams);
    params.width:=imWidth;
    params.height:=imHeight;
    params.num_steps := spnSteps.Value;
    params.guidance  := strToFloat(edtCFG.Text);
    params.powerAlpha:= strToFloat(edtPowerAlpha.Text);
    params.schedule  := TQNNSchedule(schedularEdit1.ItemIndex);
    if (trim(edtSeed.Text)='') or (strToInt(edtSeed.text)<=0) then
      params.seed := random(Int64.MaxValue)
    else
      params.seed := strToInt(edtSeed.text);
    str(params.schedule, s);
    if btnTxt2Img.Down then try // txt2Img

      MainForm.generateThread.Synchronize(MainForm.Update);

      substep_callback:=OnStepCallback;
      text_progress_callback:= OnTextStepCallback;
      vae_progress_callback:= OnTextStepCallback;
      prog.position := 0;
      if LowerCase(ModelEdit1.Text).Contains('flux') then begin
        flux := TQNNFlux.load(AppPath+MODELS_DIR+'/'+modeledit1.Text);
        params.model_name := flux.model_name;
        prog.max:= 27 + params.num_steps*(5 + 20 + 2) + 16+100;  // text_encode_steps + steps*transformer_blocks + ve_decoder_steps
        flux.use_mmap:=true;
        img := flux.generate(memoPrompt.Text, memoNegPrompt.Text, params, OnPreviewCallback);
      end;
      if lowerCase(ModelEdit1.Text).Contains('z-image') then begin
        zi := TQNNZImage.load(AppPath+MODELS_DIR+'/'+modeledit1.Text);
        params.model_name := zi.model_name;
        prog.max:= 35 + params.num_steps*(2 + 2 + 44) + 15+100;  // text_encode_steps + steps*transformer_blocks + ve_decoder_step;
        zi.use_mmap:=true;
        img := zi.generate(memoPrompt.Text, params{, OnPreviewCallback});
      end;
      fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';
      lastGenerated := fn;
      img.saveToFile(fn, 'program', strMeta);
      img.addPngMeta(fn, 'json',
                              '{"model" : "'+params.model_name+'"'+
                              ', "prompt" : "'+StringReplace(memoPrompt.Text, '"', '\"', [rfReplaceAll])+
                              '", "seed" : '+IntToStr(params.seed)+
                              ', "steps" : '+intToStr(params.num_steps)+
                              ', "schedueler" : "'+copy(s, 5)+'"}');
      image1.Picture.LoadFromFile(fn);
      dlgImage.FileName := fn;
      gotImage := true;
    finally
      if flux.transformer.isLoaded() then flux.free();
      if zi.transformer.isLoaded() then zi.free();
      img.free
    end;

    if btnImg2Img.Down then try // img2Img

      MainForm.generateThread.Synchronize(MainForm.Update);

      img := TQNNImage.loadFromFile(dlgImage.FileName);
      substep_callback:=OnStepCallback;
      text_progress_callback:= OnTextStepCallback;
      vae_progress_callback:= OnTextStepCallback;
      flux := TQNNFlux.load(MODELs_DIR+'/'+modeledit1.Text);
      prog.max:= 27 + params.num_steps*(5 + 20 + 2) + 16 + 100;  // text_encode_steps + steps*transformer_blocks + ve_decoder_steps
      prog.position := 0;
      flux.use_mmap:=true;
      params.model_name := flux.model_name;
      img := flux.generate(memoPrompt.Text, memoNegPrompt.Text, params, img, OnPreviewCallback);
      fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';
      lastGenerated := fn;
      img.saveToFile(fn, 'program', strMeta);
      img.addPngMeta(fn, 'json',
                              '{"model" : "'+params.model_name+'"'+
                              ', "prompt" : "'+StringReplace(memoPrompt.Text, '"', '\"', [rfReplaceAll])+
                              '", "seed" : '+IntToStr(params.seed)+
                              ', "steps" : '+intToStr(params.num_steps)+
                              ', "schedueler" : "'+copy(s, 5)+'"}');

      image2.Picture.LoadFromFile(fn);
      gotImage2 := true;
      lblSetAsRef.Show;
    finally
      flux.free();
      img.free
    end;
  end;
end;

procedure TGenerateThread.DoTerminate;
begin
  with MainForm do begin
    prog.position := 0;
    btnGenerate.Caption := 'Generate';
    btnTxt2Img.Enabled:=True;
    btnImg2Img.Enabled:=True;
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
    if FEdit.Text<>FList.Items[FList.ItemIndex] then
      FEdit.Text := FList.Items[FList.ItemIndex];
  end;
  //FList.Hide
end;

procedure THSEdit.FOnListClick(sender: TObject);
begin
  FList.Hide;
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
var i : longint;
begin
  i := FList.items.IndexOf(FEdit.Text);
  if ItemIndex <> i then
    ItemIndex := i;
  if assigned(FOnChange) then FOnChange(Self)
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

function THSEdit.GetText: string;
begin
  result := FEdit.Text;
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
  FEdit.text := text;
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
  FEdit.OnKeyDown:=FOnEditKeyDown;
  FEdit.OnChange:=FOnEditChange;

  //FEdit.Alignment:=taVerticalCenter;

  FList := TCustomListBox.Create(self);
  FList.BorderStyle:=bsNone;
  FList.Color := ColorBright(Color, -20);
  //FList.Font.Color:=Font.Color;
  FList.OnSelectionChange:=FOnListSelectionChange;
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
  //FMin := 0;
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
    Brush.Color := ColorBright(Color, -$10);
    FillRect(Canvas.ClipRect);

    Brush.Style := bsSolid;
    Brush.Color := FBarColor;
    if FMax>0 then
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
    if BarShowText then begin
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
  Invalidate;
end;

procedure THSProgressBar.SetBarColor(AValue: TColor);
begin
  if FBarColor=AValue then Exit;
  FBarColor:=AValue;
  Invalidate;
end;

procedure THSProgressBar.SetMax(AValue: integer);
begin
  if FMax=AValue then Exit;
  FMax:=AValue;
  Invalidate;
end;

//procedure THSProgressBar.SetMin(AValue: integer);
//begin
//  if FMin=AValue then Exit;
//  FMin:=AValue;
//  Invalidate;
//end;

procedure THSProgressBar.SetOrientation(AValue: TProgressBarOrientation);
begin
  if FOrientation=AValue then Exit;
  FOrientation:=AValue;
  Invalidate;
end;

procedure THSProgressBar.SetPosition(AValue: integer);
begin
  if FPosition=AValue then Exit;
  assert({(AValue>=FMin) and} (AValue<=FMax),'[ProgressBar] Position is out of range!');
  FPosition := AValue;
  Invalidate;
  Update;
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
  image2.Hide;
  btnLoadPic.Hide;
end;

procedure TMainForm.btnGenerateClick(Sender: TObject);
begin
  if btnTxt2Img.Down then begin
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
  if btnImg2Img.Down then begin
    if not fileExists(dlgImage.FileName) then begin
      ShowMessage('Load an image 1st.');
      exit
    end;
    if btnGenerate.Caption = 'Generate' then begin
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

end;

procedure TMainForm.btnLoadPic1Click(Sender: TObject);
var fn: string;
begin
  fn := AppPath+FormatDateTime('yyyymmdd_hhnnss', now())+'.png';

  if not ((btnTxt2Img.Down and gotImage) or (btnImg2Img.Down and gotImage2)) then begin
    ShowMessage('No output image yet!');
    exit
  end;

  if PromptForFileName(fn, '*.png|*.png|*.bmp|*.bmp|*.jpg|*.jpg|*.ico|*.ico|*.*|*.*', 'png', '', '', true) then begin
    if FileExists(fn) then
      if MessageDlg('Overwrite ['+fn+'] ?', mtConfirmation, mbYesNo, 0)<>mrYes then exit;
    if btnTxt2Img.Down then begin
      image1.Picture.SaveToFile(fn);
      TQNNImage.addPngMeta(fn, 'program', strMeta);
      TQNNImage.addPngMeta(fn, 'json',
                              '{"model" : "'+modeledit1.text+'"'+
                              ', "prompt" : "'+StringReplace(memoPrompt.text, '"', '\"', [rfReplaceAll])+
                              '", "seed" : '+edtSeed.text+
                              ', "steps" : '+spnSteps.text);

    end
    else begin
      image2.Picture.SaveToFile(fn);

      TQNNImage.addPngMeta(fn,'program', strMeta);
      TQNNImage.addPngMeta(fn, 'json',
                              '{"model" : "'+modeledit1.text+'"'+
                              ', "prompt" : "'+StringReplace(memoPrompt.text, '"', '\"', [rfReplaceAll])+
                              '", "seed" : '+edtSeed.text+
                              ', "steps" : '+spnSteps.text);

    end;
  end;
end;

procedure TMainForm.btnImg2ImgClick(Sender: TObject);
begin
  Image2.Show;
  btnLoadPic.Show;
  image2.width := image2.parent.width div 2
end;

procedure TMainForm.cmbModelsDropDown(Sender: TObject);
begin
  checkExistingModels;
end;

procedure TMainForm.edtCFGChange(Sender: TObject);
var s:Currency;
begin
  if not TryStrToCurr(edtCFG.Text, s) then exit;
  pnlNegative.Visible := StrToFloat(edtCFG.Text) <> 1.0;
  if pnlNegative.Visible then
    pnlNegative.Width := pnlNegative.parent.Width div 2;
end;

procedure TMainForm.edtCFGMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
var s:currency;
begin
  if TryStrToCurr(TEdit(sender).text, s) then
    if (s>0) or (WheelDelta>0) then
      TEdit(Sender).Text := CurrToStr(s + (2*ord(WheelDelta>0)-1)*0.1);
end;

procedure TMainForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if assigned(generateThread) and not generateThread.Finished then begin
    generateThread.Terminate;
    generateThread.WaitFor;
  end;

end;

procedure TMainForm.FormCreate(Sender: TObject);
var i:TQNNSchedule; s:ansistring;
begin
  DefaultFormatSettings.DecimalSeparator:='.';
  checkExistingModels;
  if cmbModels.Items.count>0 then
    cmbModels.ItemIndex:=0;
  cmbSchedular.Items.Clear;

  for i := low(TQNNSchedule) to high(TQNNSchedule) do begin
    str(i, s);
    cmbSchedular.Items.add(copy(s,14));
  end;
  if cmbSchedular.items.count>0 then
    cmbSchedular.ItemIndex:=0;


  cmbModels.hide;
  cmbImgRes.hide;
  cmbSchedular.hide;

  modelEdit1:= THSEdit.Create(Self);
  modeledit1.Parent:=cmbModels.Parent;
  modelEdit1.SetBounds(cmbModels.Left, cmbModels.Top, cmbModels.Width, cmbModels.Height);
  modelEdit1.Anchors:=[akLeft, akTop];
  modeledit1.Items.Text:=cmbModels.Items.Text;
  if modeledit1.Items.Count>0 then
    modeledit1.ItemIndex:=0;

  imgRes := THSEdit.Create(Self);
  imgRes.parent := cmbImgRes.parent;
  imgRes.SetBounds(cmbImgRes.Left, cmbImgRes.Top, cmbImgRes.Width, cmbImgRes.Height);
  imgRes.Anchors:=[akRight, akTop];
  imgRes.items.Text := cmbImgRes.items.text;
  if imgRes.Items.Count>0 then
    imgRes.ItemIndex:=0;

  schedularEdit1 := THSEdit.Create(Self);
  schedularEdit1.parent := cmbSchedular.parent;
  schedularEdit1.SetBounds(cmbSchedular.Left, cmbSchedular.Top, cmbSchedular.Width, cmbSchedular.Height);
  schedularEdit1.Anchors:=[akRight, akTop];
  schedularEdit1.items.Text := cmbSchedular.items.text;
  if schedularEdit1.Items.Count>0 then
    schedularEdit1.ItemIndex:=0;


  spnSteps.BorderStyle := bsNone;

  prog             := THSProgressBar.Create(Self);
  prog.BarShowText :=true;
  prog.parent      := Self;
  prog.height      := 12;
  prog.align       := alBottom;

  totalDL := THSProgressBar.Create(self);
  totalDL.parent := pnlDl;
  totalDL.SetBounds(lblDownload.width+2, 4, 120, 12);
  totalDL.Anchors := [akLeft, akTop, akRight];
  totalDL.position := totalDL.max div 2;
  totalDL.BarShowText:= true;
  //totalDL.Font.Size := totalDL.Font.Size - 4;

  subDL := THSProgressBar.Create(self);
  subDL.parent := pnlDl;
  subDL.SetBounds(lblDownload.width+2, totalDL.Height+8, 120, 12);
  subDL.Anchors :=  [akLeft, akTop, akRight];
  subDL.position := subDL.max div 2;
  //subDl.BorderSpacing.top:=2;
  subDL.BarShowText:= true;
  //subDL.Font.Size := subDL.Font.Size - 4;


  {$ifdef MSWindows}
  SetDarkModeTitleBar(self, true);
  {$endif}
end;

procedure TMainForm.btnLoadPicClick(Sender: TObject);
var
  img:TQNNImage;
  bmp:Graphics.TBitmap;

  i, w, h : longint;
  dims : string;
begin
  dlgImage.FileName:='';
  if not dlgImage.Execute  then exit;
  img := TQNNImage.loadFromFile(dlgImage.FileName);
  bmp := Graphics.TBitmap.Create;
  QNNImageToBitmap(img, bmp);
  mainform.Image1.Picture.Graphic:= bmp;
  for i:= 1 to 8 do begin
    w := trunc(i*0.25*bmp.width);
    h := trunc(w * bmp.Height / bmp.width);
    if (w>=QNN_VAE_MAX_DIM) or (h>=QNN_VAE_MAX_DIM) then break;
    dims := format('%d X %d', [w, h]);
    if imgRes.Items.IndexOf(dims)<0 then
      imgRes.Items.add(dims);
    if i=1 then imgRes.ItemIndex:=imgRes.Items.IndexOf(dims);
  end;
  MainForm.generateThread.Synchronize(MainForm.Update);
  img.free;
  freeAndNil(bmp)
end;


procedure TMainForm.FormShow(Sender: TObject);
begin
  //modeledit1.parent := self;
  //
end;

procedure TMainForm.btnLoadPicMouseEnter(Sender: TObject);
begin
  TControl(Sender).font.Style := TControl(Sender).font.Style + [fsUnderline]
end;

procedure TMainForm.btnLoadPicMouseLeave(Sender: TObject);
begin
  TControl(Sender).font.Style := TControl(Sender).font.Style - [fsUnderline]
end;

procedure TMainForm.Image1DblClick(Sender: TObject);
begin
  if btnImg2Img.Down then
    btnLoadPicClick(btnLoadPic);
end;

var model:TFLUX4BDownloader;
procedure TMainForm.Label10Click(Sender: TObject);
begin
  if Assigned(model.FHttp) then begin
    model.FHttp.FHTTP.Terminate;
  end;
  pnlDL.Hide;
end;

procedure TMainForm.lblMoreClick(Sender: TObject);
begin
  PopupMenu1.PopUp(lblMore.ClientOrigin.X, lblMore.ClientOrigin.Y+lblMore.Height);
end;

procedure TMainForm.lblMoreMouseEnter(Sender: TObject);
begin
  lblMore.ParentColor:=false;
  lblMore.Color:=clBlack;
end;

procedure TMainForm.lblMoreMouseLeave(Sender: TObject);
begin
  lblMore.ParentColor:=true;
end;

procedure TMainForm.lblSetAsRefClick(Sender: TObject);
begin
  Image1.Picture.Graphic := Image2.Picture.Graphic;
  dlgImage.FileName := lastGenerated;
end;

procedure TMainForm.MenuItem2Click(Sender: TObject);
begin
  frmAbout.ShowModal;
end;

procedure TMainForm.mnuDL1Click(Sender: TObject);

begin
  try
    subDL.Position:=0;
    totalDL.Position:=0;
    pnlDL.Show;
    Application.ProcessMessages;
    model.OnProgress:=OnDownload;
    model.download(appPath+'../../models');
    pnlDL.Hide;
  finally

    pnlDL.Hide;
  end;
end;

procedure TMainForm.mnuDLOpenBLASClick(Sender: TObject);
begin
  {$if defined(MSWINDOWS)}
  if MessageDlg('Download [Open Basic Linear Algebra]?'#13'This will improve the generation speed by ~X2', mtConfirmation, mbYesNo, 0)<>mrYes then exit;
  getOpenBlas();
  ShowMessage('OpenBLAS installed successfully, re-launch this program.');
  {$elseif defined(DARWIN) or defined(MACOS)}
  ShowMessage('Already using Apple''s native "Accelerate" library, no need for OpenBLAS!');
  {$else}
  ShowMessage('Install OpenBLAS from your package manager, e.g :'#13'"sudo apt install openblas"');
  {$endif}
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
      r := FindNext(sr)            ;
    end;
  finally
    FindClose(sr);
  end;
end;

const i: integer = 0;
procedure TMainForm.OnDownload(const subReceived, subTotal, received, total: int64);
begin
  subDL.Max:=ceil(subTotal / $4FFFFF); // 4MB
  subDL.Position:=subReceived div $4FFFFF;
  i := subDL.Position mod length(DL_Symbols);
  subDL.Caption := DL_Symbols[i];
  totalDL.Max:=Total;
  totalDL.Position:=Received;
  totalDL.Caption:=format('%d/%d', [totalDL.Position, totalDL.Max]);

  Application.ProcessMessages;
end;

initialization
  AppPath := ExtractFilePath(ParamStr(0));

end.

