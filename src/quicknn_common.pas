unit quicknn_common;

{$ifdef FPC}
  {$PackRecords C}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$modeswitch nestedprocvars}
  {$ifdef CPUX64}
    {$asmmode intel}
  {$endif}
{$endif}
{$TYPEINFO ON} // include typeinfo
{$C+} // enable assertions
{$H+}
{$pointermath on}
{$T+}
{$Z4}  // enums ar aligned to 4 bytes to match C API

{$define _USE_CALLOC}

interface
uses typinfo, generics.Collections

  ;

//{$if not declared(FP16)}
type
  PPSingle = ^PSingle;
  BF16 = type word;
  PBF16 = ^BF16;

  FP16  = type word;
  PFP16 = ^FP16;
//{$endif}

{$if not defined(USE_MULTITHREADING)}
type
  {$ifdef FPC}
  TThreadProcNested = procedure(idx: IntPtr; ptr: Pointer) is nested;
  TGroupProcNested=procedure(const _start,_end:IntPtr; const params:Pointer)is nested;
  {$else} //delphi compiler
  TThreadProcNested = reference to procedure(idx: IntPtr; ptr: Pointer);
  TGroupProcNested= reference to procedure(const _start,_end:IntPtr; const params:Pointer);  // must be [register]?
  {$endif}
  TThreadProc       = procedure(idx: IntPtr; ptr: Pointer);
  TGroupProc    = procedure(const _start,_end:IntPtr;const params:Pointer);
{$endif}

const EPSILON = 1e-6;

type
  blasint = longint;
  CBLAS_Layout = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_ORDER = CBLAS_Layout;
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114);

  TSingleArray = TArray<Single>;
  PFloatarray = ^TFloatArray;
  TFloatArray = array[0..MaxLongint div 32] of single;
  TInterpolation = ( iNearest, iLinear, iCubic, iLanczos);

  INT4 = -8..7;
  TINT4Array = packed array[0..MaxInt-1] of INT4;
  PINT4Array = ^TINT4Array;
  PINT4 = ^INT4;

  TQNNDeviceType = (dvCPU, dvCUDA, dvVULKAN, dvOPENCL, dvROCM);

  TQNNDevice = record
    deviceType : TQNNDeviceType;
    mem : Pointer;
  end;

  TQNNDataType =( // matching oneDNN
        dtUndef = 0,
        dtF16 = 1,
        dtBf16 = 2,
        dtF32 = 3,
        dtS32 = 4,
        dtS8 = 5,
        dtU8 = 6,
        dtF64 = 7,
        dtBoolean = 8,
        dtF8_e5m2 = 9,
        dtF8_e4m3 = 10,
        dtS4 = 11,
        dtU4 = 12,
        dtE8m0 = 13,
        dtF4_e2m1 = 14,
        dtF4_e3m0 = 15,
        dtS16 = 16,
        dtU16//,
        //dtData_type_max = $7fff
  );

const
  dtI4  = dtS4;
  dtI8  = dtS8;
  dtI16 = dtS16;
  dtI32 = dtS32;

type
  PPQNNFloat = ^PQNNFloat;
  PQNNFloat = PSingle;//^QNNFloat;
  QNNFloat = single;


const
  DATATYPE_BITS : array[low(TQNNDataType)..high(TQNNDataType)] of byte=
        (0, 16, 16, 32, 32, 8, 8, 64, 1, 8, 8, 4, 4, 8, 4, 4, 16, 16) ;
  QNN_DATATYPE = dtF32;

type

  TTensorPrintStyle = (
                    psValues,
                    psGray5{5 ascii shades},
                    psGray24{24 half char shades},
                    psGray {256 half char shades},
                    psColor8 {256 colors},
                    psColor24 {~16M colors},
                    psSIXELGray,
                    psSIXEL,
                    psSIXELDithered
                  );

  TQNNSchedule = (
    QNN_SCHEDULE_DEFAULT   = 0,
    QNN_SCHEDULE_LINEAR    = 1,
    QNN_SCHEDULE_POWER     = 2,
    QNN_SCHEDULE_SIGMOID   = 3,  (* Flux shifted sigmoid *)
    QNN_SCHEDULE_FLOWMATCH = 4  (* Z-Image FlowMatch Euler *)
  );

  TGenerateParams = record
    width, height, num_steps :longint;
    seed : Int64;
    guidance :single;
    schedule : TQNNSchedule;
    powerAlpha : single;
  end;
// todo link TMemoryBlock to IMemoryArena for allocations

  { TMemoryBlock }
  PMemoryBlock = ^TMemoryBlock;
  TMemoryBlock = record
      DataType : TQNNDataType;
      shape    : TArray<Int64>;
      offset   : NativeInt;
      DataPtr  : pointer;
      size     : NativeInt;
{$ifdef USE_CALLOC}
      Data32   : PLongWord;
      Data16   : PSmallInt;
      Data8    : PByte;
      Data4    : PInt4;
{$else}
      Data32   : TArray<LongWord>;
      Data16   : TArray<WORD>;
      Data8    : TArray<Byte>;
      Data4    : TArray<INT4>;
{$endif}
      device   : TQNNDevice;
      name : string;
  const
      ERRSTR_CAST_ARRAY = 'MemoryBlock with non zero offset cannot be casted to an Array';
      ERRSTR_CAST_TYPE = 'ERROR : Data is not of ';
  private
      constructor Create(const aSize:NativeInt; const aName:string; const dType:TQNNDataType = QNN_DATATYPE; const src : pointer =nil);           overload;// do not use
  public
      constructor Create(const aShape : TArray<Int64>; const aName:string ; const dType:TQNNDataType = QNN_DATATYPE; const aData : pointer =nil);  overload;
      procedure reSize(const aSize:NativeInt);  overload;
      procedure reSize(const aShape:TArray<Int64>);  overload;
      procedure free();
      function count:NativeInt;

      function channels():longint;
      function height():longint;
      function width():longint;

      function isAssigned():boolean;
      function isAllocated():boolean;
      procedure printStat;
      procedure assignPtr(const ptr:PSingle; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PBF16; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PFP16; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PLongint; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PSmallInt; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PShortInt; const aShape:TArray<int64>); overload;
      procedure assignPtr(const ptr:PInt4; const aShape:TArray<int64>); overload;


      //class operator implicit(const val:Pointer ):TMemoryBlock;

{$ifndef USE_CALLOC}
      class operator implicit(const val:TArray<longint> ):TMemoryBlock;
      class operator implicit(const val:TArray<single>  ):TMemoryBlock;
      class operator implicit(const val:TArray<BF16>    ):TMemoryBlock;
      class operator implicit(const val:TArray<FP16>    ):TMemoryBlock;
      class operator implicit(const val:TArray<smallint>):TMemoryBlock;
      class operator implicit(const val:TArray<shortint>):TMemoryBlock;
      class operator implicit(const val: TArray<INT4>   ):TMemoryBlock;

      class operator implicit(const val:TMemoryBlock):TArray<longint> ;
      class operator implicit(const val:TMemoryBlock):TArray<single>  ;
      class operator implicit(const val:TMemoryBlock):TArray<BF16>    ;
      class operator implicit(const val:TMemoryBlock):TArray<FP16>    ;
      class operator implicit(const val:TMemoryBlock):TArray<smallint>;
      class operator implicit(const val:TMemoryBlock):TArray<shortint>;
      class operator implicit(const val:TMemoryBlock):TArray<INT4>;
{$endif}
(*
      class operator implicit(const val:PLongint ):TMemoryBlock;
      class operator implicit(const val:PSingle  ):TMemoryBlock;
      class operator implicit(const val:PBF16    ):TMemoryBlock;
      class operator implicit(const val:PFP16    ):TMemoryBlock;
      class operator implicit(const val:Psmallint):TMemoryBlock;
      class operator implicit(const val:PShortint):TMemoryBlock;
      class operator implicit(const val:PINT4    ):TMemoryBlock;
*)

      class operator implicit(const val:TMemoryBlock):Plongint  ;
      class operator implicit(const val:TMemoryBlock):PSingle  ;
      class operator implicit(const val:TMemoryBlock):PBF16    ;
      class operator implicit(const val:TMemoryBlock):PFP16    ;
      class operator implicit(const val:TMemoryBlock):PSmallint;
      class operator implicit(const val:TMemoryBlock):PShortint;
      class operator implicit(const val:TMemoryBlock):PINT4    ;
      class operator implicit(const val:TMemoryBlock):boolean ;
      class operator add(const src:TMemoryBlock; const aOffset:longint):TMemoryBlock;
      class operator add(const src:TMemoryBlock; const aOffset:Int64):TMemoryBlock;
      procedure printCompare(const src:TMemoryBlock; const isSumSqrDiff:boolean =false);

      function TypeName():string;
      function print(const consolePixel: TTensorPrintStyle = psGray; tile: Integer = 1; minVal: double = 0; maxVal: double = 0): TArray<NativeInt>; overload;
      class operator Initialize({$ifdef FPC}var{$else}out{$endif} val:TMemoryBlock);
  end;
  // old and new are memory sizes in bytes
  TOnMemoryUpdate = procedure(const status:string; const old, New:IntPtr; const mem:TMemoryBlock);

  TQNNPixelOrder = (poHWC, poCHW);


  PQNNImage = ^TQNNImage;

  { TQNNImage }

  TQNNImage = record
  type
      TPngChunk = record
        tag : array[0..3] of ansichar;
        data : TArray<Byte>;
        crc : longword;
      end;
  var
      width : longint;
      height : longint;
      channels : longint;// RGB
      data : TArray<byte>;      (* Row-major, channel-interleaved *)
      class function loadFromFile(const fileName : string; resizeWidth: longint =0; resizeHeight: longint=0):TQNNImage;static;
      class procedure addPngMeta(const filename:string; const keyword, meta:ansistring);static;
      class procedure computePngCRCTable(); static;
      class function calcPngCRC(const buf :PByte; const len : NativeInt; const initCRC : longword = $ffffffff): longword;static;
      function resize(const w, h:longint):TQNNImage;
      constructor Create(const aWidth, aHeight:longint; const aChannels : longint =3; const aData:PQNNFloat =nil);
      function asMemoryBlock(const aDataType:TQNNDataType=QNN_DATATYPE; const aPixelOrder: TQNNPixelOrder= poCHW):TMemoryBlock;
      procedure saveToFile(const filename:string; const tagKey:string = ''; const tagDesc :string ='');
      procedure printSixel();
      procedure free();
  end;

  { TImageRef }
  PImageRef = ^TImageRef;

  TImageRef = record
    w, h, t_offset : longint;
    latent : TMemoryBlock;
  end;

  { IMemoryArena }

  IMemoryArena = interface
    function Allocate(const aSize:NativeUInt):Pointer;
    procedure ReSize(const aSize:NativeUInt);
    function ArenaSize:NativeUInt;
  end;

  { TMemoryArena }
  {$UnDef DEBUG}
  TMemoryArena = class(TInterfacedObject, IMemoryArena)
  private
    FData    : Pointer;
    FSize    : NativeUInt; // in bytes
    FMemPos  : NativeUInt;
  public
    constructor Create(const aPool: NativeUInt);
    function ArenaSize:NativeUInt;
    procedure ReSize(const aSize:NativeUInt);
    function Allocate(const aSize:NativeUInt):pointer;
    destructor Destroy; override;
  end;

  TVocabs = TDictionary<rawbytestring, longint>;

  TCHMult = array[0..3] of longint;

const
  TAB = #9;
  LF  = #10;
  CR  = #13;
  SP  = ' ';
  WHITE_SPACE = [TAB, LF, CR, SP];
  PUNCTUATIONS = ['!'..'/', ':'..'@', '['..'`', '{'..'~'];
  ALPHABETS    = ['A'..'Z', 'a'..'z'];
  NUMERICS     = ['0'..'9'];
  WORD_SEPERATORS = WHITE_SPACE+PUNCTUATIONS;



  QNN_LATENT_CHANNELS    = 128  ;(* Flux: 32*2*2, Z-Image: 16*2*2=64 *)

  (* VAE architecture *)
  QNN_VAE_Z_CHANNELS     = 32   ;(* Flux default; Z-Image uses 16 *)
  QNN_VAE_BASE_CH        = 128  ;
  QNN_VAE_CH_MULT_0      = 1    ;
  QNN_VAE_CH_MULT_1      = 2    ;
  QNN_VAE_CH_MULT_2      = 4    ;
  QNN_VAE_CH_MULT_3      = 4    ;
  QNN_VAE_CH_MULT        : TCHMult = (QNN_VAE_CH_MULT_0, QNN_VAE_CH_MULT_1, QNN_VAE_CH_MULT_2, QNN_VAE_CH_MULT_3);
  QNN_VAE_NUM_RES        = 2    ;
  QNN_VAE_GROUPS         = 32   ;
  QNN_VAE_MAX_DIM        = 1792 ; (* Max image dimension for VAE *)

  (* Tokenizer *)
  QNN_MAX_SEQ_LEN        = 512  ;
  QNN_VOCAB_HASH_SIZE    = 150001 ;

  (* Sampling *)
  QNN_MAX_STEPS          = 256     ;


//const
  (* ========================================================================
   * Generation Parameters
   * ======================================================================== *)

  (* Schedule type: 0 = model default (sigmoid for Flux, flowmatch for Z-Image) *)

  //QNN_SCHEDULE_DEFAULT   = 0;
  //QNN_SCHEDULE_LINEAR    = 1;
  //QNN_SCHEDULE_POWER     = 2;
  //QNN_SCHEDULE_SIGMOID   = 3;  (* Flux shifted sigmoid *)
  //QNN_SCHEDULE_FLOWMATCH = 4;  (* Z-Image FlowMatch Euler *)

type
  TQNNParams = record
      width : longint;              (* Output width (default: 256) *)
      height : longint;             (* Output height (default: 256) *)
      num_steps : longint;          (* Inference steps (default: 4 distilled, 50 base) *)
      seed : int64;                 (* Random seed (-1 for random) *)
      guidance : QNNFloat;          (* CFG guidance scale (0 = auto from model type) *)
      schedule : longint;                 (* Schedule type (SCHEDULE_*)
      power_alpha : QNNFloat;            (* Exponent for power schedule (default: 2.0) *)
  end;

const
  (* Default parameters *)
  QNN_DEFAULT_WIDTH  = 256   ;
  QNN_DEFAULT_HEIGHT = 256   ;
  QNN_PARAMS_DEFAULT : TQNNParams = (
     width     : QNN_DEFAULT_WIDTH;
     height    : QNN_DEFAULT_HEIGHT;
     num_steps : 0;
     seed      : -1;
     guidance  : 0.0;
     schedule  : 0;//QNN_SCHEDULE_DEFAULT;
     power_alpha : 2.0
  );


function UTF8CharSize(const str:ansichar):NativeInt;
function UTF8Length(const str:RawByteString):NativeInt;


procedure BF16ToSingle(const N: NativeInt; const src: PBF16; dst: PSingle);
procedure SingleToBF16(const N: NativeInt; const src: PSingle; dst: PBF16);

procedure FP16ToSingle(const N: NativeInt; const src: PFP16; dst: PSingle);
procedure SingleToFP16(const N: NativeInt; const src: PSingle; dst: PFP16);

function ifthen(const cond:boolean; const ifTrue, ifFalse:longint):longint; overload;inline;
function ifthen(const cond:boolean; const ifTrue, ifFalse:int64):int64;     overload;inline;
function ifthen(const cond:boolean; const ifTrue, ifFalse:single):single;   overload;inline;
function ifthen(const cond:boolean; const ifTrue, ifFalse:double):double;   overload;inline;
function ifthen(const cond:boolean; const ifTrue, ifFalse:string):string;   overload;inline;
function LCase(const c:ansichar):ansichar;
function UCase(const c:ansichar):ansichar;
function product(const ar:TArray<int64>):int64;

function readInt(var f:file):longint;
function readSingles(var f:file; const count:longint):TMemoryBlock;

{$if defined(MSWINDOWS)}
function calloc(count, size:UIntPtr):pointer;WINAPI;                external 'msvcrt.dll';
function malloc(size:UIntPtr):pointer;WINAPI;                       external 'msvcrt.dll';
function realloc(mem:pointer; size:UIntPtr):pointer;WINAPI;         external 'msvcrt.dll';
procedure free(mem:pointer);WINAPI;                                 external 'msvcrt.dll';
procedure printf(const fmt:ansistring);winapi;varargs;              external 'msvcrt.dll';
//procedure sprintf(out a; const fmt:ansistring);winapi;varargs; external 'msvcrt.dll';
{$else}
function calloc(count, size:UIntPtr):pointer;WINAPI;external;
function malloc(size:UIntPtr):pointer;WINAPI;external;
function realloc(mem:pointer; size:UIntPtr):pointer;WINAPI;external;
procedure free(mem:pointer);WINAPI;external;
function mmap(addr:pointer; len:UintPtr; prot, flag, fd:longint; offset: IntPtr):pointer;WINAPI;external;
function munmap(addr:pointer; len:UIntPtr):longint;WINAPI;external;
procedure printf(const fmt:ansistring);winapi;varargs;external;

const
  PROT_NONE   = $00;
  PROT_READ   = $01;
  PROT_WRITE  = $02;
  PROT_EXEC   = $04;

  MAP_SHARED  = 0001;
  MAP_PRIVATE = 0002;
  MAP_COPY    = MAP_PRIVATE;

  MAP_FIXED   = $0010;
  MAP_RENAME  = $0020;

  MAP_NOCACHE = $0400;
  MAP_FAILED  = pointer(-1);

  MAP_FILE      = $0000;
  MAP_ANON      = $1000;
  MAP_ANONYMOUS = MAP_ANON ;
{$endif}

type
  TSubstepType = (
    SUBSTEP_DOUBLE_BLOCK,  // Double-stream block completed
    SUBSTEP_SINGLE_BLOCK,  // Single-stream block completed
    SUBSTEP_FINAL_LAYER    // Final layer completed
  ) ;

  (*
   * Substep callback - called during transformer forward pass.
   * type: which operation completed
   * index: 0-based index of this substep within its type
   * total: total count for this substep type
   *)
  TSubstepCallback = procedure(const &type :TSubstepType; const index, total: longint);

  (*
   * Step callback - called at sampling step boundaries.
   * step: current step (1-based), or 0 to indicate sampling is starting
   * total: total number of steps
   *)
  TStepCallback = procedure (const step, total:longint);
  TProgressCallback = procedure (const step, total:longint; const outTensor : TMemoryBlock);


  (*
   * Phase callback - called at major phase boundaries.
   * phase: descriptive name ("encoding text", "decoding image", etc.)
   * done: 0 when starting, 1 when finished
   *)
  TPhaseCallback = procedure (const phase :string; const done:boolean);

  (*
   * Step image callback - called after each denoising step with decoded image.
   * step: current step (1-based)
   * total: total number of steps
   * img: decoded image at this step (caller must NOT free)
   *
   * To use: set both step_image_callback and step_image_vae before
   * calling the sampling function. The callback is only invoked when both are set.
   *)
  TStepImageCallback = procedure (const step, total: longint; const img : TQNNimage(* todo maybe its an array? *));

  (*
   * Text encoder progress callback - called once per Qwen3 layer.
   * layer: current layer (0-based)
   * total: total number of layers (36)
   *)
  TTextProgressCallback = procedure (const layer, total : longint);

  (*
   * VAE progress callback - called once per resblock/attention block.
   * block: current block (0-based)
   * total: total number of blocks (11 for encoder, 15 for decoder)
   *)
  TVAEProgressCallback = procedure (const block, total: longint );

var
  substep_callback : TSubstepCallback;
  step_callback    : TStepCallback;
  phase_callback   : TPhaseCallBack;
  step_image_callback : TStepImageCallback;
  vae_ptr : pointer;
  text_progress_callback : TTextProgressCallback;
  vae_progress_callback  : TVAEProgressCallback;
  onMemoryUpdate : TOnMemoryUpdate;

  function IndexOf(const needle:string; const haystack:TArray<string>):nativeInt;  overload;

implementation
uses SysUtils, Math
  {$ifdef MSWINDOWS}
  , Windows
  {$endif}
  {$ifdef FPC}
  , FPCanvas
  , FPWriteBMP
  , FPWriteJPEG
  , FPWritePNG
  , FPReadBMP
  , FPReadJPEG
  , FPReadPNG
  , FPImage
  {$else}
  ,UITypes, fmx.Types, fmx.Graphics
  {$endif}
  , quicknncpu, termesc, sixel;

type
  PSingle = System.PSingle; // fix delphi incompatible PSingle between System and Windows units

var
  PNG_CRC_TABLE : array[0..255] of longword;

function LCase(const c:ansichar):ansichar;
begin
  result := C;
  if result in ['A'..'Z'] then inc(result, $20)

end;

function UCase(const c: ansichar): ansichar;
begin
  result := C;
  if result in ['a'..'z'] then dec(result, $20)
end;


function product(const ar: TArray<int64>): int64;
var
  i: Integer;
begin
  //if not assigned(ar) then exit(0);
  result := 1;
  for i:=0 to high(ar) do
    result := result * ar[i]
end;

function readInt(var f:file):longint;
begin
    blockread(f, result, sizeof(longint))
end;

function readSingles(var f:file; const count:longint):TMemoryBlock;
begin
    result := TMemoryBlock.Create(count, 'readSingles '+ TGUID.NewGuid.ToString());
    blockread(f, PSingle(result)^, count*sizeof(single))
end;

function IndexOf(const needle: string; const haystack: TArray<string>): nativeInt;
var i:nativeint;
begin
  for i:=0 to high(haystack) do
    if needle=haystack[i] then exit(i);
  result := -1;
end;

{ TQNNImage }

function TQNNImage.resize(const w, h: longint): TQNNImage;
var y, x, x0, x1, y0, y1, c :longint;
  scale_x, scale_y, v, v00, v01, v10, v11, wx, wy, src_x, src_y : single;
begin
  result := TQNNImage.create(w, h, channels);

  scale_x := width / w;
  scale_y := height / h;

  for y := 0 to h-1 do begin
      for x := 0 to w-1 do begin
          src_x := (x + 0.5) * scale_x - 0.5;
          src_y := (y + 0.5) * scale_y - 0.5;

          x0 := floor(src_x);
          y0 := floor(src_y);
          x1 := x0 + 1;
          y1 := y0 + 1;

          wx := src_x - x0;
          wy := src_y - y0;

          if x0 < 0 then x0 := 0 else if x0 >= width then x0 := width - 1 ;
          if x1 < 0 then x1 := 0 else if x1 >= width then x1 := width - 1 ;
          if y0 < 0 then y0 := 0 else if y0 >= height then y0 := height - 1;
          if y1 < 0 then y1 := 0 else if y1 >= height then y1 := height - 1;

          for c := 0 to channels -1 do begin
              v00 := data[(y0 * width + x0) * channels + c];
              v01 := data[(y0 * width + x1) * channels + c];
              v10 := data[(y1 * width + x0) * channels + c];
              v11 := data[(y1 * width + x1) * channels + c];

              v := v00 * (1 - wx) * (1 - wy) +
                   v01 * wx * (1 - wy) +
                   v10 * (1 - wx) * wy +
                   v11 * wx * wy;

              result.data[(y * w + x) * channels + c] := round(v);
          end
      end
  end;
end;

constructor TQNNImage.Create(const aWidth, aHeight: longint; const aChannels: longint; const aData: PQNNFloat);
var y, x, ch : longint; val: QNNFloat;
begin
  //data := nil;
  width := aWidth;
  height := aHeight;
  channels := aChannels;
  setLength(data, width*height*channels);
  if assigned(adata) then
    for y := 0 to height -1 do
      for x := 0 to width -1 do
        for ch := 0 to channels -1 do
          begin
              val := aData[(ch* height + y)*width + x];
              val := (val+1.0) * 0.5;
              val := val * 255.0;
              if val < 0 then
                  val := 0;
              if val > 255 then
                  val := 255;
              data[(y*width + x)*3 + ch] := round(val)
          end;
end;

function TQNNImage.asMemoryBlock(const aDataType: TQNNDataType; const aPixelOrder: TQNNPixelOrder): TMemoryBlock;
var c, y, x, idx : longint;
  single_ptr : PSingle;
begin
  assert(aDataType=dtF32, '[TQNNImage.asMemoryBlock]: ERROR DataType not implemented.');
  case aPixelOrder of
    poHWC : begin
      result := TMemoryBlock.Create([Height, Width, Channels], '',aDataType);
      case aDataType of
        dtF32 :
          begin
            single_ptr := result;
            for y:=0 to height-1 do
              for x := 0 to width-1 do
                for c :=0 to channels-1 do begin
                  idx := (y*width + x)*channels + c;
                  single_ptr[idx] := data[idx]*2.0/255.0 - 1.0; // result range must be between -1.0 and +1.0
                end;
          end;

      end;
    end;
    poCHW : begin
      result := TMemoryBlock.Create([channels, Height, Width], intToStr(Random($fffffff)), aDataType);
      case aDataType of
        dtF32 :
          begin
            single_ptr := result;
            for c :=0 to channels-1 do
              for y:=0 to height-1 do
                for x := 0 to width-1 do begin
                  idx := (y*width + x)*channels + c;
                  single_ptr[(c*height + y)*width + x] := data[idx]*2.0/255.0 - 1.0; // result range must be between -1.0 and +1.0
                end;
          end;
      end
    end
  else
    assert(false, '[TQNNImage.asMemoryBlock]: ERROR PixelOrder not implemented.');
  end;



end;

procedure TQNNImage.saveToFile(const filename: string; const tagKey: string; const tagDesc: string);
var
  y, x, w, h : longint;
  {$if defined(fpc)}
  clr:TFPColor;
  bmp : TFPMemoryImage;
  {$else}
  bmp : TBitmap;
  bmpData : TBitmapData;
  {$endif}
  D : PByte;
  ext : string;
begin
  w := width;
  h := height;
  {$ifdef fpc}
  bmp := TFPMemoryImage.Create(width, height);

  for y := 0 to height-1 do
    for x :=0 to width-1 do begin
        d := @data[(y*w + x)*3];
        clr.red   := d[0] shl 8;
        clr.Green := d[1] shl 8;
        clr.Blue  := d[2] shl 8;
        clr.Alpha:= $FF00;
        bmp.Colors[x, y] := clr;
    end;
  bmp.SaveToFile(fileName);
  bmp.free;
  {$else}
  bmp := TBitmap.Create();
  bmp.setSize(width, height);
  bmp.Map(TMapAccess.Write, bmpData);
  for y := 0 to height-1 do begin
    d := bmpData.GetScanline(y);
    for x :=0 to width-1 do begin
      {$ifdef MSWINDOWS}
      d[0] := data[(y*width + x)*3 + 2];
      d[1] := data[(y*width + x)*3 + 1];
      d[2] := data[(y*width + x)*3    ];
      d[3] := $FF;
      {$else}
      // todo check posix color order on delphi
      d[0] := $FF;
      d[1] := data[(y*width + x)*3    ];
      d[2] := data[(y*width + x)*3 + 1];
      d[3] := data[(y*width + x)*3 + 2];
      {$endif}
      inc(d, 4);
    end;
  end;
  bmp.Unmap(bmpData);
  bmp.SaveToFile(fileName);
  bmp.free;
  {$endif}
  ext := LowerCase(ExtractFileExt(filename));
  if (ext='.png') and (tagKey<>'') then begin
    addPngMeta(filename, tagKey, tagDesc);
  end;
end;

class function TQNNImage.loadFromFile(const fileName: string;
  resizeWidth: longint; resizeHeight: longint): TQNNImage;
{$ifdef FPC}
var
  img : TFPMemoryImage;
  P:TFPColor;
  x, y, imSize, _h, _w : SizeInt;
  d:PByte;
begin
  result := default(TQNNImage);
  img := TFPMemoryImage.Create(0, 0);
  try
      img.LoadFromFile(fileName);
      if resizeWidth<=0 then resizeWidth    := img.Width;
      if resizeHeight<=0 then resizeHeight  := img.Height;
      _w := img.Width;
      _h := img.Height;
      imSize := resizeHeight*resizeWidth;
      setLength(result.Data, 3*imSize);
      for y := 0 to resizeHeight-1 do
        for x := 0 to resizeWidth-1 do begin
           p := img.Colors[ round(_w * x / resizeWidth),  round(_h * y / resizeHeight)];  // nearst neighor
           d := @result.data[3*(y*resizeWidth + x)];
           d[0]:= p.red   shr 8;
           d[1]:= p.green shr 8;
           d[2]:= p.blue  shr 8;
        end;
      result.width  := resizeWidth;
      result.height := resizeHeight;
      result.channels := 3;
  finally
    freeAndNil(img)
  end;
end;

{$else}
var
  img:TBitmap;
  bmpData : TBitmapData;
  y, x, _w, _h, imSize : longint;
  d : PByte;
  p : TAlphaColor;
begin
  result := default(TQNNImage);
  img := TBitmap.Create;
  try
    img.LoadFromFile(fileName);
    if resizeWidth<=0 then resizeWidth    := img.Width;
    if resizeHeight<=0 then resizeHeight  := img.Height;
    _w := img.Width;
    _h := img.Height;
    img.Map(TMapAccess.Read, bmpData);
    imSize := resizeHeight*resizeWidth;
    setLength(result.Data, 3*imSize);
    for y := 0 to resizeHeight-1 do begin
      for x := 0 to resizeWidth-1 do begin
         p := bmpData.GetPixel(round(_w * x/resizeWidth), round(_h * y/resizeHeight));  // nearst neighor
         d := @result.data[3*(y*resizeWidth + x)];
         d[0] := (p shr 16) and $FF;
         d[1] := (p shr 8) and $FF;
         d[2] :=  p and $FF;

      end;
    end;
    result.width  := resizeWidth;
    result.height := resizeHeight;
    result.channels := 3;
  finally
    if assigned(bmpData.Data) then
      img.Unmap(bmpData);
    img.Free
  end;
end;

function SwapEndian(const x:longword):longword;
begin
  result := (x shl 24) or ((x shl 8) and $00ff0000) or ((x shr 8) and $0000ff00) or (x shr 24)
end;
{$endif}


function _FileSize(var f:file):Int64; // a work around Delphi FileSize function bug returning incorrect file size on large files
var fl :longword;
begin
{$if defined(FPC)}
  result := FileSize(f);
{$elseif defined(MSWINDOWS)}
  result :=0;
  fl := GetFileSize(TFileRec(f).handle, @result);
  result := fl or (result shl 32)
{$else}
  result := FileSize(f); // fallback to default
{$endif}
end;

class function TQNNImage.calcPngCRC(const buf: PByte; const len: NativeInt; const initCRC: longword): longword;
var i:NativeInt;
begin
  result := initCRC;
  for i := 0 to len-1 do
      result := PNG_CRC_TABLE[(result xor buf[i]) and $ff] xor (result shr 8);
  result := result xor $ffffffff; // isn't this the same as not()?
end;

class procedure TQNNImage.addPngMeta(const filename: string; const keyword, meta: ansistring);
const PNG_SIGNATURE :Uint64 = $0A1A0A0D474E5089;
var
  f:File;
  content:rawbytestring;
  fsize, fpos: NativeInt;
  pngSig: uint64;
  chunkSize, r, i:longword;
  pngChunk : ^TPngChunk;
  pngchunks : TArray<TPngChunk>;
  len, crc:longword;
  strData : ansistring;
  tag : array[0..3] of ansichar;
begin
  if (keyword='') or (meta='') then exit;
  try
    AssignFile(f, filename);
    reset(f, 1);
    _FileSize(f);
    fSize := _FileSize(f);
    setLength(content, fSize);
    blockread(f, pngSig, sizeof(pngSig));
    if pngSig=PNG_SIGNATURE then begin
      while not EOF(f) do begin
        blockRead(f, chunkSize, sizeOf(chunkSize), r);
        chunkSize := SwapEndian(chunkSize);
        blockread(f, tag, sizeof(tag));
        //if (FilePos(f)=fSize) or (tag='IEND') then break;
        //fpos := FilePos(f);
        //Seek(f, fpos + chunksize); // move to the next Chunk (data + CRC value)
        setLength(pngChunks, length(pngChunks)+1);
        pngChunk := @pngchunks[high(pngchunks)];
        move(tag, pngChunk.tag, sizeof(tag));
        setLength(pngChunk.data, chunkSize);
        if assigned(pngChunk.data) then
          BlockRead(f, pngChunk.data[0], chunkSize);
        blockRead(f, pngChunk.crc, sizeOf(pngChunk.crc));
        pngChunk.crc := SwapEndian(pngChunk.crc);
      end;
      insert(default(TPngChunk), pngChunks, 1);
      pngChunk := @pngchunks[1];
      pngChunk.tag := 'tEXt';
      setLength(pngChunk.data, length(keyword)+length(meta)+1);
      move(keyword[1], pngChunk.data[0], length(keyword));
      move(meta[1], pngChunk.data[length(keyword)+1], length(meta));
      strData := pngChunk.tag + keyword+#0+meta;
      pngChunk.crc := calcPngCRC(Pointer(strData), length(strData));
      reWrite(f, 1);
      blockWrite(f, PNG_SIGNATURE, sizeof(PNG_SIGNATURE));
      for i:= 0 to high(pngChunks) do begin
        len := swapEndian(longword(length(pngChunks[i].data)));
        BlockWrite(f, len, sizeof(longword));
        blockWrite(f, pngChunks[i].tag, sizeof(tag));
        if assigned(pngChunks[i].data) then begin
          blockWrite(f, pngChunks[i].data[0], length(pngchunks[i].data));
        end;
        crc := swapEndian(pngChunks[i].crc);
        blockWrite(f, crc, sizeOf(pngchunks[i].crc));
      end;
    end;
  finally
    closeFile(f)
  end;
end;

class procedure TQNNImage.computePngCRCTable();
var n, k:longint; c: longword;
begin
  for n := 0 to high(PNG_CRC_TABLE) do begin
      c := n;
      for k := 0 to 7 do
          if boolean(c and 1) then
              c := $edb88320 xor (c shr 1)
          else
              c := c shr 1;
      PNG_CRC_TABLE[n] := c;
  end;
end;
procedure TQNNImage.printSixel;
begin
  sixel.printSixel(Data, width, height, true);
end;

procedure TQNNImage.free();
begin
  if assigned(data) then setLength(Data, 0);
  self := default(TQNNImage)
end;

{ TMemoryBlock }

constructor TMemoryBlock.Create(const aSize: NativeInt;
  const aName:string; const dType: TQNNDataType; const src: pointer);

begin
  self := default(TMemoryBlock);
  DataType := dType;
  name := aName;
  if name='' then begin

    name := TGUID.NewGuid().ToString();
  end;
  case DATATYPE_BITS[dType] of
    //1 : setlength(Data1, aSize);
    4  : begin
      size := aSize;
      {$ifdef USE_CALLOC}
      Data4 := nil;
      Data4 := malloc((aSize*DATATYPE_BITS[dtype]+4) div 8);
      {$else}
      setLength(Data4 , aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('new', 0, (aSize*DATATYPE_BITS[dtype]+4) div 8, self);
      assert((aSize<>0) and assigned(data4), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if assigned(src) then
        move(src^, pointer(Data4) , (aSize*DATATYPE_BITS[dType]) div 8)
    end;

    8  : begin
      size := aSize;
      {$ifdef USE_CALLOC}
      Data8 := nil;
      Data8 := malloc((aSize*DATATYPE_BITS[dtype]) div 8);
      {$else}
      setLength(Data8 , aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('new', 0, (aSize*DATATYPE_BITS[dtype]) div 8, self);
      assert((aSize<>0) and assigned(data8), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if assigned(src) then
        move(src^, pointer(Data8) , (aSize*DATATYPE_BITS[dType]) div 8)
    end;

    16 : begin
      size := aSize;
      {$ifdef USE_CALLOC}
      Data16 := nil;
      Data16 := malloc((aSize*DATATYPE_BITS[dtype]) div 8);
      {$else}
      setLength(Data16, aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('new', 0, (aSize*DATATYPE_BITS[dtype]) div 8, self);
      assert((aSize<>0) and assigned(data16), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if assigned(src) then
        move(src^, pointer(Data16), (aSize*DATATYPE_BITS[dType]) div 8)
    end;

    32 : begin
      size := aSize;
      {$ifdef USE_CALLOC}
      Data32 := nil;
      Data32 := malloc((aSize*DATATYPE_BITS[dtype]) div 8);
      //Data32 := calloc(aSize, sizeOf(longint));
      {$else}
      setLength(Data32, aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('new', 0, (aSize*DATATYPE_BITS[dtype]) div 8, self);
      assert((aSize<>0) and assigned(data32), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if assigned(src) then
        move(src^, pointer(Data32), (aSize*DATATYPE_BITS[dType]) div 8)
    end;
  else
    assert(false, 'ERROR : TMemoryBlock.create, Unsupported data type ')
  end;

end;

constructor TMemoryBlock.Create(const aShape : TArray<Int64>; const aName:string; const dType:TQNNDataType; const aData:pointer);
begin
  self := create(product(aShape), aName, dType, aData);
  shape := aShape
end;

procedure TMemoryBlock.reSize(const aSize:NativeInt);
begin
  assert(isAllocated(), 'TMemoryBlock.resize : cannot resize a pointer assigned memory!');
  case DATATYPE_BITS[DataType] of
    //1 : setlength(Data1, aSize);
    4  : begin
      {$ifdef USE_CALLOC}
      Data4 := realloc(Data4, (aSize*DATATYPE_BITS[Datatype]) div 8);
      {$else}
      setLength(Data4 , aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('resize', (size*DATATYPE_BITS[DataType]+4) div 8, (aSize*DATATYPE_BITS[DataType]+4) div 8, self);
      size := aSize;
      assert(assigned(data4), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    8  : begin
      {$ifdef USE_CALLOC}
      Data8 := realloc(Data8, (aSize*DATATYPE_BITS[Datatype]) div 8);
      {$else}
      setLength(Data8 , aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('resize', size*DATATYPE_BITS[DataType] div 8, (aSize*DATATYPE_BITS[DataType]) div 8, self);
      size := aSize;
      assert(assigned(data8), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    16 : begin
      {$ifdef USE_CALLOC}
      Data16 := realloc(Data16, (aSize*DATATYPE_BITS[Datatype]) div 8);
      {$else}
      setLength(Data16, aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('resize', size*DATATYPE_BITS[DataType] div 8, (aSize*DATATYPE_BITS[DataType]) div 8, self);
      size := aSize;
      assert(assigned(data16), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    32 : begin
      {$ifdef USE_CALLOC}
      Data32 := realloc(Data32, (aSize*DATATYPE_BITS[Datatype]) div 8);
      {$else}
      setLength(Data32, aSize);
      {$endif}
      if assigned(onMemoryUpdate) then
        onMemoryUpdate('resize', size*DATATYPE_BITS[DataType] div 8, (aSize*DATATYPE_BITS[DataType]) div 8, self);
      size := aSize;
      assert(assigned(data32), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;
  else
    assert(false, 'ERROR : TMemoryBlock.create, Unsupported data type ')
  end;
end;

procedure TMemoryBlock.reSize(const aShape:TArray<int64>);
begin
  resize(product(aShape));
  shape := aShape;
end;

procedure TMemoryBlock.free();
var m:pointer;
begin
  if offset>0 then exit;
  if isAllocated() then begin

    if assigned(onMemoryUpdate) and isAllocated() then
      if DATATYPE_BITS[DataType]=4 then
        onMemoryUpdate('free', (Size*DATATYPE_BITS[DataType]+4) div 8, 0, self)
      else
        onMemoryUpdate('free', (Size*DATATYPE_BITS[DataType]) div 8, 0, self);
    {$ifdef USE_CALLOC}
    case DATATYPE_BITS[DataType] of
      4 :if assigned(Data4 ) and (size>0) then begin m := Data4 ; Data4  :=nil; quicknn_common.free(m) end;
      8 :if assigned(Data8 ) and (size>0) then begin m := Data8 ; Data8  :=nil; quicknn_common.free(m) end;
      16:if assigned(Data16) and (size>0) then begin m := Data16; Data16 :=nil; quicknn_common.free(m) end;
      32:if assigned(Data32) and (size>0) then begin m := Data32; Data32 :=nil; quicknn_common.free(m) end;
    end;
    {$else}
  //  case DATATYPE_BITS[DataType] of
  //    4 :setLength(Data4 , 0);
  //    8 :setLength(Data8 , 0);
  //    16:setLength(Data16, 0);
  //    32:setLength(Data32, 0);
  //  end;
    if length(Data32)>0 then
      setLength(Data32, 0);
    if length(Data16)>0 then
      setLength(Data16, 0);
    if length(Data8)>0  then
      setLength(Data8 , 0);
    if length(Data4)>0  then
      setLength(Data4 , 0);
    {$endif}
  end;
  self.DataType := dtUndef;
  self.offset := 0;
  self.size := 0;
  self.shape := nil;
  self.DataPtr := nil;
  self.name := '';
 // fillchar(self, sizeof(self), #0);
end;

function TMemoryBlock.count:NativeInt;
begin
  {$ifdef USE_CALLOC}
  result := size;
  {$else}
  case DATATYPE_BITS[DataType] of
    4  : exit(length(Data4));
    8  : exit(length(Data8));
    16 : exit(length(Data16));
    32 : exit(length(Data32))
  end;
  result := 0
  {$endif}
end;

function TMemoryblock.channels():longint;
begin
  assert(length(shape)>2 ,'ERROR : Memory has no channels dimension');
  result := shape[high(shape)-2]
end;

function TMemoryblock.height():longint;
begin
  assert(length(shape)>1,'ERROR : Memory has no height dimension');
  result := shape[high(shape)-1]
end;

function TMemoryblock.width():longint;
begin
  result := shape[high(shape)]
end;

function TMemoryBlock.isAssigned():boolean;
begin
  result := false;
  case DATATYPE_BITS[DataType] of
{$ifdef USE_CALLOC}
    4  : result := assigned(Data4 );
    8  : result := assigned(Data8 );
    16 : result := assigned(Data16);
    32 : result := assigned(Data32)
{$else}
    4  : result := length(Data4 )>0;
    8  : result := length(Data8 )>0;
    16 : result := length(Data16)>0;
    32 : result := length(Data32)>0
{$endif}

  end;
  if not result then
    result := assigned(DataPtr)
end;

function TMemoryBlock.isAllocated():boolean;
begin
  {$ifdef USE_CALLOC}
  result := assigned(Data4) or assigned(Data8) or assigned(Data16) or assigned(Data32);
  {$else}
  result := (length(Data4)>0) or (length(Data8)>0) or (length(Data16)>0) or (length(Data32)>0);
  {$endif}
  if result then
    assert(not assigned(DataPtr), 'ERROR [TMemoryBlock.isAllocated()] : Overlapping assigned memory and allocated memory were found! while one should exist!');
end;
procedure TMemoryBlock.printStat;
begin
  case DataType of
    dtF32:
      quicknncpu.printStat(self);
  else
    assert(false, 'ERROR : Datatype '+ GetEnumName(TypeInfo(TQNNDataType), ord(DataType))+' not implemented!')
  end;
end;

procedure TMemoryBlock.assignPtr(const ptr: PSingle; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dtF32;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: PBF16; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dtbf16;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: PFP16; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dtF16;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: PLongint; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dts32;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: PSmallInt; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dts16;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: PShortInt; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dtS8;
  DataPtr := ptr;
end;

procedure TMemoryBlock.assignPtr(const ptr: Pint4; const aShape:TArray<int64>);
begin
  assert(not isAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  shape := aShape;
  size := product(aShape);
  DataType := dtS4;
  DataPtr := ptr;
end;

//class operator TMemoryBlock.implicit(const val:Pointer ):TMemoryBlock;
//begin
//  assert(false, 'ERROR : Untyped tensors not allowed.')
//end;

{$ifndef USE_CALLOC}
class operator TMemoryBlock.implicit(const val:TArray<longint> ):TMemoryBlock;
begin
  result.DataType := dts32;
  result.Data32 := TArray<longword>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TArray<single>  ):TMemoryBlock;
begin
  result.DataType := dtF32;
  result.Data32 := TArray<longword>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TArray<BF16>    ):TMemoryBlock;
begin
  result.DataType := dtBF16;
  result.Data16 := TArray<word>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TArray<FP16>    ):TMemoryBlock;
begin
  result.DataType := dtF16;
  result.Data16 := TArray<word>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TArray<smallint>):TMemoryBlock;
begin
  result.DataType := dtS16;
  result.Data16 := TArray<word>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TArray<shortint>):TMemoryBlock;
begin
  result.DataType := dtS8;
  result.Data8 := TArray<Byte>(val);
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.implicit(const val: TArray<INT4>):TMemoryBlock;
begin
  result.DataType := dtS4;
  result.Data4 := val;
  result.size := length(val);
  //if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, (result.Size*DATATYPE_BITS[result.DataType]+4) div 8, result);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<longint>  ;
begin
  assert(val.DataType=dts32, ERRSTR_CAST_TYPE+'INT32');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<longint>(val.Data32);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<single>  ;
begin
  assert(val.DataType=dtf32, ERRSTR_CAST_TYPE+'FP32');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<single>(val.Data32);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<BF16>    ;
begin
  assert(val.DataType=dtBf16, ERRSTR_CAST_TYPE+'BF16');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<BF16>(val.Data16);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<FP16>    ;
begin
  assert(val.DataType=dtF16, ERRSTR_CAST_TYPE+'FP16');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<FP16>(val.Data16);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<smallint>;
begin
  assert(val.DataType=dts16, ERRSTR_CAST_TYPE+'INT16');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<SmallInt>(val.Data16);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):TArray<shortint>;
begin
  assert(val.DataType=dtS8, ERRSTR_CAST_TYPE+'INT8');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<shortint>(val.Data8);
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock): TArray<INT4>;
begin
  assert(val.DataType=dtS4, ERRSTR_CAST_TYPE+'INT4');
  assert(val.offset=0, ERRSTR_CAST_Array);
  result := TArray<INT4>(val.Data4);
end;

{$endif}

class operator TMemoryBlock.implicit(const val:TMemoryBlock):boolean ;
begin
  result := val.isAssigned();
end;

(*
class operator TMemoryBlock.implicit(const val:PLongint ):TMemoryBlock;
begin
  result.DataType := dts32;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PSingle  ):TMemoryBlock;
begin
  result.DataType := dtf32;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PBF16    ):TMemoryBlock;
begin
  result.DataType := dtBf16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PFP16    ):TMemoryBlock;
begin
  result.DataType := dtF16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:Psmallint):TMemoryBlock;
begin
  result.DataType := dtS16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PShortint):TMemoryBlock;
begin
  result.DataType := dtS8;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PINT4    ):TMemoryBlock;
begin
  result.DataType := dtS4;
  result.DataPtr := val;
end;
*)

class operator TMemoryBlock.implicit(const val:TMemoryBlock):Plongint  ;
begin
  assert(val.DataType=dtS32, 'ERROR : Data is not of '+'INT32');
  if assigned(val.Data32) then
    result := Pointer(val.Data32)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PSingle  ;
begin
  assert(val.DataType=dtF32, 'ERROR : Data is not of '+'FP32');
  if assigned(val.Data32) then
    result := Pointer(val.Data32)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PBF16    ;
begin
  assert(val.DataType=dtBf16, 'ERROR : Data is not of '+'BF16');
  if assigned(val.Data16) then
    result := Pointer(val.Data16)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PFP16    ;
begin
  assert(val.DataType=dtF16, 'ERROR : Data is not of '+'FP16');
  if assigned(val.Data16) then
    result := Pointer(val.Data16)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PSmallint;
begin
  assert(val.DataType=dts16, 'ERROR : Data is not of '+'INT16');
  if assigned(val.Data16) then
    result := Pointer(val.Data16)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PShortint;
begin
  assert(val.DataType=dtS8, 'ERROR : Data is not of '+'s8');
  if assigned(val.Data8) then
    result := Pointer(val.Data8)
  else
    result := val.DataPtr;
  inc(result, val.offset)
end;

class operator TMemoryBlock.implicit(const val:TMemoryBlock):PINT4    ;
begin
  assert(val.DataType=dtS4, 'ERROR : Data is not of '+'INT4');
  if assigned(val.Data4) then
    result := Pointer(val.Data4)
  else
    result := val.DataPtr;
  inc(result, val.offset)  // todo Implicit to PInt4 : check validity of int4 pointer offseting
end;

class operator TMemoryBlock.add(const src:TMemoryBlock; const aOffset:longint):TMemoryBlock  ;
begin
  result := src;
  result.offset:= src.offset + aOffset;
end;

class operator TMemoryBlock.add(const src:TMemoryBlock; const aOffset:int64):TMemoryBlock  ;
begin
  result := src;
  result.offset:= src.offset + aOffset;
end;

procedure TMemoryBlock.printCompare(const src:TMemoryBlock; const isSumSqrDiff:boolean =false);
var md,src1,src2: single;
begin
  assert((count = src.count) and( datatype=src.DataType), 'Tensor sizes do not match! '+IntToStr(count())+'<>'+IntToStr(src.Count()));
  printStat;
  src.printStat;
  case DataType of
    dtF32: begin
      if isSumSqrDiff then begin
        md := TQNNSingleOPS.QNNSqrDistance(count, self, src);
        writeln('SqrDistance :', md:1:5);
      end else begin
        md := TQNNSingleOPS.QNNMaxAbsDiff2(count, self, src, src1, src2);
        if md<>0 then begin
          src.print(psSIXELDithered, 3);
          readln;
        end;
        writeln('MaxAbsDiff :', md:1:5, ' max src1 :', src1:1:6, ' max src2:', src2:1:6);
      end;
    end
    else
      assert(false, 'printCompare : datatype not implemented!')
  end;
end;

function TMemoryBlock.TypeName:string;
begin
  result := GetEnumName(TypeInfo(TQNNDataType), ord(DataType));
end;

function toStr(const f:single):string; overload;
begin
  Str(f:1:4, result)
end;

function toStr(const f:double):string; overload;
begin
  Str(f:1:4, result)
end;

function toStr(const f:Integer):string; overload;
begin
  Str(f, result)
end;

function toStr(const f:Int64):string; overload;
begin
  Str(f, result)
end;

function toStr(const f:shortint):string; overload;
begin
  Str(f, result)
end;

function toStr(const f:SmallInt):string; overload;
begin
  Str(f, result)
end;

function TMemoryBlock.print(const consolePixel: TTensorPrintStyle; tile: Integer; minVal: double; maxVal: double): TArray<NativeInt>;
const
  csi = #$1B'[';
  up = #$1B'[A';
  dw = #$1B'[B';
  fw = #$1B'[C';
  bw = #$1B'[D';
  sc = #$1B'[s';
  rc = #$1B'[u';
  er = #$1B'[0K';
  cpos = #$1B'[6n';
var
  amin, amax: QNNFloat;
  i, j, k, t, _w, _h, _c, _area, index, outArgMin, outArgMax, ow, oh, _size: Integer;
  l: longword;
  range, r, g, b, r2, g2, b2: double;
  S: widestring;
  pixels : RawByteString;
  isHalfChar, isColor, isSIXEL : boolean;
  _Data : PQNNFloat;
const
  __shade: array[0..4] of string = (' ', '░', '▒', '▓', '█');
  // delphi will complain if no string type defined, MacOS will type rubbish if string type is defined!! :(
  {$if defined(MSWINDOWS) or defined(POSIX)}
  halfChar :widechar = #$2580;
  {$else}
  halfChar :ansistring= '▀';   // this will fail to compile on delphi - android
  {$endif}
begin
  assert(DataType = dtF32 , 'print : Tensor visualization is not impelemted for this type!');

  if not isAssigned() then exit;
  _size:=count();
  S := TypeName() + ' Tensor (';
  for i := 0 to High(Shape) do
    if i = 0 then
      S := S + ToStr(Shape[i])
    else
    begin
      S := S + ' X ';
      S := S + ToStr(Shape[i]);
    end;
  S := S + ')';
  _Data := Self;
  ow := length(S);
  oh := 1;
  Write(S, csi, length(S), 'D', dw);
  //writeln(S);
  ow := length(S);
  oh := 1;
  isHalfChar := not (consolePixel in [psValues, psGray5, psSIXELGray, psSIXEL, psSIXELDithered]);
  isColor := consolePixel in [psColor8, psColor24, psSIXEL, psSIXELDithered];
  isSIXEL := consolePixel in [psSIXEL, psSIXELGray, psSIXELDithered];
  if consolePixel <> psValues then
  begin
    if minVal = maxVal then
    begin
      TQNNSingleOPS.QNNMinMax(Count, PQNNFloat(Self), amin, amax, @outArgMin, @outArgMax);
      minVal := aMin; maxVal := aMax;
      S := '[min : ' + toStr(amin) + '@' + toStr(outArgMin) +
        ', max : ' + toStr(amax) + '@' + toStr(outArgMax) + ']';
      Write(S, csi, length(S), 'D', dw);
      //writeln(S);
      ow := Math.max(ow, length(S));
      Inc(oh);
    end;

    if isNan(minVal) or isNan(maxVal) then
    begin
      result := [ow, oh];
      exit();
    end;
    _w := Shape[high(Shape)];
    if length(Shape) > 1 then
    begin
      _h := Shape[high(Shape)-1];
      _area := _h * _w;
    end
    else
    begin
      _h := 1;
      _area := _w;
    end;
    _c := (1 + 2 * Ord(isColor));
    range := maxVal - minVal;
    Result := [ow, oh];
    if (range < EPSILON) or (tile < 1) then exit;
    S := '';
    pixels :='';
    l := 0;
    for i := 0 to _size div (_c * _area * tile) - 1 do
    begin
      for j := 0 to ceil(_h / (1 + Ord(isHalfChar))) - 1 do
      begin
        for t := 0 to tile - 1 do
        begin
          for k := 0 to _w - 1 do
          begin
            index := i * _c * tile * _area + t * _c * _h * _w + j *
              (1 + Ord(isHalfChar)) * _w + k;
            if index < _size then
            begin
              r := _Data[index];
              case consolePixel of
                psGray5: S := S + __shade[round(4 * (r - minVal) / range)];
                psGray24:
                begin
                  Inc(index, _w);
                  if index<_size then
                    r2 := _Data[index]
                  else
                    r2:=minVal;
                  r := 232 + 23 * (r - minVal) / range;
                  r2 := 232 + 23 * (r2 - minVal) / range;
                  S := S + #$1B'[38;5;' + ToStr(round(r)) +
                    'm' + #$1B'[48;5;' + ToStr(round(r2)) + 'm' + halfChar;
                end;
                psGray:
                begin
                  Inc(index, _w);
                  if index<_size then
                    r2 := _Data[index]
                  else
                    r2:=minVal;
                  r := $ff * (r - minVal) / range;
                  r2 := $ff * (r2 - minVal) / range;
                  S := S + #$1B'[38;2;' + ToStr(round(r)) + ';' +
                    ToStr(round(r)) + ';' + ToStr(round(r)) +
                    'm' + #$1B'[48;2;' + ToStr(round(r2)) + ';' +
                    ToStr(round(r2)) + ';' + ToStr(round(r2)) + 'm' + halfChar;
                end;
                psColor8:
                begin
                  g  := _Data[index + _area]       ;
                  b  := _Data[index + _area * 2]   ;
                  // next line
                  Inc(index, _w);
                  if index<_size then begin
                    r2 := _Data[index]            ;
                    g2 := _Data[index + _area]    ;
                    b2 := _Data[index + _area * 2]
                  end else begin
                    r2 := minVal;
                    g2 := minVal;
                    b2 := minVal
                  end;

                  r := 5 * (r - minVal) / range;
                  g := 5 * (g - minVal) / range;
                  b := 5 * (b - minVal) / range;

                  r2 := 5 * (r2 - minVal) / range;
                  g2 := 5 * (g2 - minVal) / range;
                  b2 := 5 * (b2 - minVal) / range;

                  S := S + #$1B'[38;5;' +
                    ToStr(16 +
                         round(b) +
                    6 *  round(g) +
                    36 * round(r)) +
                    'm' + #$1B'[48;5;' +
                    ToStr(16 +
                         round(b2) +
                     6 * round(g2) +
                    36 * round(r2)) +
                    'm' + halfChar;
                end;
                psColor24:
                begin
                  g := _Data[index + _area]    ;
                  b := _Data[index + _area * 2];
                  // nex line
                  Inc(index, _w);
                  if index<_size then begin
                    r2 := _Data[index];
                    g2 := _Data[index + _area];
                    b2 := _Data[index + _area * 2];
                  end else begin
                    r2 := minVal;
                    g2 := minVal;
                    b2 := minVal
                  end;

                  r := $ff * (r - minVal) / range;
                  g := $ff * (g - minVal) / range;
                  b := $ff * (b - minVal) / range;

                  r2 := $ff * (r2 - minVal) / range;
                  g2 := $ff * (g2 - minVal) / range;
                  b2 := $ff * (b2 - minVal) / range;

                  S := S + #$1B'[38;2;' +
                    ToStr(round(r)) + ';' +
                    ToStr(round(g)) + ';' +
                    ToStr(round(b)) +
                    'm' + #$1B'[48;2;' +
                    ToStr(round(r2)) + ';' +
                    ToStr(round(g2)) + ';' +
                    ToStr(round(b2)) +
                    'm' + halfChar;
                end;
                psSIXEL, psSIXELDithered: begin
                  g := _Data[index + _area]    ;
                  b := _Data[index + _area * 2];

                  r := $ff * (r - minVal) / range;
                  g := $ff * (g - minVal) / range;
                  b := $ff * (b - minVal) / range;
                  pixels:= pixels + chr(round(r)) + chr(round(g)) + chr(round(b));
                end;
                psSIXELGray: begin
                  r := $ff * (r - minVal) / range;
                  pixels:= pixels + ansichar(round(r));
                end;
              end;
            end;
          end;
          //if consolePixel<>psGray5 then
          //  S := S + #$1B'[0m '
        end;
        if not isSIXEL then begin
          Write(S, csi, _w * tile, 'D', dw);
          S := '';
        end;
        //writeln(S);
        ow := Math.max(ow, _w * tile);
        Inc(oh);
        Inc(l);
      end;
      if isSIXEL then begin
        printSixel(@pixels[1], _w*tile, _h, consolePixel=psSIXELDithered, consolePixel=psSIXELGray);
        pixels := ''
      end
    end;
    if consolePixel <> psGray5 then
      S := resetAllModes;//#$1B'[0m';
    if not isSIXEL then
      Write(S);
    //writeln(S);
    Result := [ow, oh];
    exit;
  end;

  //writeln(toString());
end;


class operator TMemoryBlock.Initialize({$ifdef FPC}var{$else}out{$endif} val:TMemoryBlock);
begin
  fillchar(val, sizeof(TMemoryBlock), #0);
end;



function UTF8CharSize(const str:ansichar):NativeInt;
var c : byte absolute str;
begin
  if (c and $80) = 0   then exit(1);
  if (c and $E0) = $C0 then exit(2);
  if (c and $F0) = $E0 then exit(3);
  if (c and $F8) = $F0 then exit(4);
  result := 1  (* if invalid *)
end;

function UTF8Length(const str:RawByteString):NativeInt;
var
  s : PAnsiChar absolute str;
  i:NativeInt;
begin
  result :=0;
  i :=0;
  while i < length(str) do begin
    inc(i, UTF8CharSize(s[i]));
    inc(result)
  end;
end;

type
  PConversionBits = ^ConversionBits;
  ConversionBits = record
    case boolean of
      false : (i32: longword);
      true  : (f32: single);
  end;

procedure BF16ToSingle(const N: NativeInt; const src: PBF16; dst: PSingle);
var
  i: NativeInt;
  cdst:PConversionBits absolute dst;
begin
  for i:=0 to N-1 do
    cdst[i].i32 := src[i] shl 16;
end;

procedure SingleToBF16(const N: NativeInt; const src: PSingle; dst: PBF16);
var
  i: NativeInt;
  csrc:PConversionBits absolute src;
begin
  for i:=0 to N-1 do
    dst[i] := csrc[i].i32 shr 16;
end;

procedure FP16ToSingle(const N: NativeInt; const src: PFP16; dst: PSingle);
const mantissa_table:array[0..2047] of longword = (
    $00000000, $33800000, $34000000, $34400000, $34800000, $34A00000, $34C00000, $34E00000, $35000000, $35100000, $35200000, $35300000, $35400000, $35500000, $35600000, $35700000,
    $35800000, $35880000, $35900000, $35980000, $35A00000, $35A80000, $35B00000, $35B80000, $35C00000, $35C80000, $35D00000, $35D80000, $35E00000, $35E80000, $35F00000, $35F80000,
    $36000000, $36040000, $36080000, $360C0000, $36100000, $36140000, $36180000, $361C0000, $36200000, $36240000, $36280000, $362C0000, $36300000, $36340000, $36380000, $363C0000,
    $36400000, $36440000, $36480000, $364C0000, $36500000, $36540000, $36580000, $365C0000, $36600000, $36640000, $36680000, $366C0000, $36700000, $36740000, $36780000, $367C0000,
    $36800000, $36820000, $36840000, $36860000, $36880000, $368A0000, $368C0000, $368E0000, $36900000, $36920000, $36940000, $36960000, $36980000, $369A0000, $369C0000, $369E0000,
    $36A00000, $36A20000, $36A40000, $36A60000, $36A80000, $36AA0000, $36AC0000, $36AE0000, $36B00000, $36B20000, $36B40000, $36B60000, $36B80000, $36BA0000, $36BC0000, $36BE0000,
    $36C00000, $36C20000, $36C40000, $36C60000, $36C80000, $36CA0000, $36CC0000, $36CE0000, $36D00000, $36D20000, $36D40000, $36D60000, $36D80000, $36DA0000, $36DC0000, $36DE0000,
    $36E00000, $36E20000, $36E40000, $36E60000, $36E80000, $36EA0000, $36EC0000, $36EE0000, $36F00000, $36F20000, $36F40000, $36F60000, $36F80000, $36FA0000, $36FC0000, $36FE0000,
    $37000000, $37010000, $37020000, $37030000, $37040000, $37050000, $37060000, $37070000, $37080000, $37090000, $370A0000, $370B0000, $370C0000, $370D0000, $370E0000, $370F0000,
    $37100000, $37110000, $37120000, $37130000, $37140000, $37150000, $37160000, $37170000, $37180000, $37190000, $371A0000, $371B0000, $371C0000, $371D0000, $371E0000, $371F0000,
    $37200000, $37210000, $37220000, $37230000, $37240000, $37250000, $37260000, $37270000, $37280000, $37290000, $372A0000, $372B0000, $372C0000, $372D0000, $372E0000, $372F0000,
    $37300000, $37310000, $37320000, $37330000, $37340000, $37350000, $37360000, $37370000, $37380000, $37390000, $373A0000, $373B0000, $373C0000, $373D0000, $373E0000, $373F0000,
    $37400000, $37410000, $37420000, $37430000, $37440000, $37450000, $37460000, $37470000, $37480000, $37490000, $374A0000, $374B0000, $374C0000, $374D0000, $374E0000, $374F0000,
    $37500000, $37510000, $37520000, $37530000, $37540000, $37550000, $37560000, $37570000, $37580000, $37590000, $375A0000, $375B0000, $375C0000, $375D0000, $375E0000, $375F0000,
    $37600000, $37610000, $37620000, $37630000, $37640000, $37650000, $37660000, $37670000, $37680000, $37690000, $376A0000, $376B0000, $376C0000, $376D0000, $376E0000, $376F0000,
    $37700000, $37710000, $37720000, $37730000, $37740000, $37750000, $37760000, $37770000, $37780000, $37790000, $377A0000, $377B0000, $377C0000, $377D0000, $377E0000, $377F0000,
    $37800000, $37808000, $37810000, $37818000, $37820000, $37828000, $37830000, $37838000, $37840000, $37848000, $37850000, $37858000, $37860000, $37868000, $37870000, $37878000,
    $37880000, $37888000, $37890000, $37898000, $378A0000, $378A8000, $378B0000, $378B8000, $378C0000, $378C8000, $378D0000, $378D8000, $378E0000, $378E8000, $378F0000, $378F8000,
    $37900000, $37908000, $37910000, $37918000, $37920000, $37928000, $37930000, $37938000, $37940000, $37948000, $37950000, $37958000, $37960000, $37968000, $37970000, $37978000,
    $37980000, $37988000, $37990000, $37998000, $379A0000, $379A8000, $379B0000, $379B8000, $379C0000, $379C8000, $379D0000, $379D8000, $379E0000, $379E8000, $379F0000, $379F8000,
    $37A00000, $37A08000, $37A10000, $37A18000, $37A20000, $37A28000, $37A30000, $37A38000, $37A40000, $37A48000, $37A50000, $37A58000, $37A60000, $37A68000, $37A70000, $37A78000,
    $37A80000, $37A88000, $37A90000, $37A98000, $37AA0000, $37AA8000, $37AB0000, $37AB8000, $37AC0000, $37AC8000, $37AD0000, $37AD8000, $37AE0000, $37AE8000, $37AF0000, $37AF8000,
    $37B00000, $37B08000, $37B10000, $37B18000, $37B20000, $37B28000, $37B30000, $37B38000, $37B40000, $37B48000, $37B50000, $37B58000, $37B60000, $37B68000, $37B70000, $37B78000,
    $37B80000, $37B88000, $37B90000, $37B98000, $37BA0000, $37BA8000, $37BB0000, $37BB8000, $37BC0000, $37BC8000, $37BD0000, $37BD8000, $37BE0000, $37BE8000, $37BF0000, $37BF8000,
    $37C00000, $37C08000, $37C10000, $37C18000, $37C20000, $37C28000, $37C30000, $37C38000, $37C40000, $37C48000, $37C50000, $37C58000, $37C60000, $37C68000, $37C70000, $37C78000,
    $37C80000, $37C88000, $37C90000, $37C98000, $37CA0000, $37CA8000, $37CB0000, $37CB8000, $37CC0000, $37CC8000, $37CD0000, $37CD8000, $37CE0000, $37CE8000, $37CF0000, $37CF8000,
    $37D00000, $37D08000, $37D10000, $37D18000, $37D20000, $37D28000, $37D30000, $37D38000, $37D40000, $37D48000, $37D50000, $37D58000, $37D60000, $37D68000, $37D70000, $37D78000,
    $37D80000, $37D88000, $37D90000, $37D98000, $37DA0000, $37DA8000, $37DB0000, $37DB8000, $37DC0000, $37DC8000, $37DD0000, $37DD8000, $37DE0000, $37DE8000, $37DF0000, $37DF8000,
    $37E00000, $37E08000, $37E10000, $37E18000, $37E20000, $37E28000, $37E30000, $37E38000, $37E40000, $37E48000, $37E50000, $37E58000, $37E60000, $37E68000, $37E70000, $37E78000,
    $37E80000, $37E88000, $37E90000, $37E98000, $37EA0000, $37EA8000, $37EB0000, $37EB8000, $37EC0000, $37EC8000, $37ED0000, $37ED8000, $37EE0000, $37EE8000, $37EF0000, $37EF8000,
    $37F00000, $37F08000, $37F10000, $37F18000, $37F20000, $37F28000, $37F30000, $37F38000, $37F40000, $37F48000, $37F50000, $37F58000, $37F60000, $37F68000, $37F70000, $37F78000,
    $37F80000, $37F88000, $37F90000, $37F98000, $37FA0000, $37FA8000, $37FB0000, $37FB8000, $37FC0000, $37FC8000, $37FD0000, $37FD8000, $37FE0000, $37FE8000, $37FF0000, $37FF8000,
    $38000000, $38004000, $38008000, $3800C000, $38010000, $38014000, $38018000, $3801C000, $38020000, $38024000, $38028000, $3802C000, $38030000, $38034000, $38038000, $3803C000,
    $38040000, $38044000, $38048000, $3804C000, $38050000, $38054000, $38058000, $3805C000, $38060000, $38064000, $38068000, $3806C000, $38070000, $38074000, $38078000, $3807C000,
    $38080000, $38084000, $38088000, $3808C000, $38090000, $38094000, $38098000, $3809C000, $380A0000, $380A4000, $380A8000, $380AC000, $380B0000, $380B4000, $380B8000, $380BC000,
    $380C0000, $380C4000, $380C8000, $380CC000, $380D0000, $380D4000, $380D8000, $380DC000, $380E0000, $380E4000, $380E8000, $380EC000, $380F0000, $380F4000, $380F8000, $380FC000,
    $38100000, $38104000, $38108000, $3810C000, $38110000, $38114000, $38118000, $3811C000, $38120000, $38124000, $38128000, $3812C000, $38130000, $38134000, $38138000, $3813C000,
    $38140000, $38144000, $38148000, $3814C000, $38150000, $38154000, $38158000, $3815C000, $38160000, $38164000, $38168000, $3816C000, $38170000, $38174000, $38178000, $3817C000,
    $38180000, $38184000, $38188000, $3818C000, $38190000, $38194000, $38198000, $3819C000, $381A0000, $381A4000, $381A8000, $381AC000, $381B0000, $381B4000, $381B8000, $381BC000,
    $381C0000, $381C4000, $381C8000, $381CC000, $381D0000, $381D4000, $381D8000, $381DC000, $381E0000, $381E4000, $381E8000, $381EC000, $381F0000, $381F4000, $381F8000, $381FC000,
    $38200000, $38204000, $38208000, $3820C000, $38210000, $38214000, $38218000, $3821C000, $38220000, $38224000, $38228000, $3822C000, $38230000, $38234000, $38238000, $3823C000,
    $38240000, $38244000, $38248000, $3824C000, $38250000, $38254000, $38258000, $3825C000, $38260000, $38264000, $38268000, $3826C000, $38270000, $38274000, $38278000, $3827C000,
    $38280000, $38284000, $38288000, $3828C000, $38290000, $38294000, $38298000, $3829C000, $382A0000, $382A4000, $382A8000, $382AC000, $382B0000, $382B4000, $382B8000, $382BC000,
    $382C0000, $382C4000, $382C8000, $382CC000, $382D0000, $382D4000, $382D8000, $382DC000, $382E0000, $382E4000, $382E8000, $382EC000, $382F0000, $382F4000, $382F8000, $382FC000,
    $38300000, $38304000, $38308000, $3830C000, $38310000, $38314000, $38318000, $3831C000, $38320000, $38324000, $38328000, $3832C000, $38330000, $38334000, $38338000, $3833C000,
    $38340000, $38344000, $38348000, $3834C000, $38350000, $38354000, $38358000, $3835C000, $38360000, $38364000, $38368000, $3836C000, $38370000, $38374000, $38378000, $3837C000,
    $38380000, $38384000, $38388000, $3838C000, $38390000, $38394000, $38398000, $3839C000, $383A0000, $383A4000, $383A8000, $383AC000, $383B0000, $383B4000, $383B8000, $383BC000,
    $383C0000, $383C4000, $383C8000, $383CC000, $383D0000, $383D4000, $383D8000, $383DC000, $383E0000, $383E4000, $383E8000, $383EC000, $383F0000, $383F4000, $383F8000, $383FC000,
    $38400000, $38404000, $38408000, $3840C000, $38410000, $38414000, $38418000, $3841C000, $38420000, $38424000, $38428000, $3842C000, $38430000, $38434000, $38438000, $3843C000,
    $38440000, $38444000, $38448000, $3844C000, $38450000, $38454000, $38458000, $3845C000, $38460000, $38464000, $38468000, $3846C000, $38470000, $38474000, $38478000, $3847C000,
    $38480000, $38484000, $38488000, $3848C000, $38490000, $38494000, $38498000, $3849C000, $384A0000, $384A4000, $384A8000, $384AC000, $384B0000, $384B4000, $384B8000, $384BC000,
    $384C0000, $384C4000, $384C8000, $384CC000, $384D0000, $384D4000, $384D8000, $384DC000, $384E0000, $384E4000, $384E8000, $384EC000, $384F0000, $384F4000, $384F8000, $384FC000,
    $38500000, $38504000, $38508000, $3850C000, $38510000, $38514000, $38518000, $3851C000, $38520000, $38524000, $38528000, $3852C000, $38530000, $38534000, $38538000, $3853C000,
    $38540000, $38544000, $38548000, $3854C000, $38550000, $38554000, $38558000, $3855C000, $38560000, $38564000, $38568000, $3856C000, $38570000, $38574000, $38578000, $3857C000,
    $38580000, $38584000, $38588000, $3858C000, $38590000, $38594000, $38598000, $3859C000, $385A0000, $385A4000, $385A8000, $385AC000, $385B0000, $385B4000, $385B8000, $385BC000,
    $385C0000, $385C4000, $385C8000, $385CC000, $385D0000, $385D4000, $385D8000, $385DC000, $385E0000, $385E4000, $385E8000, $385EC000, $385F0000, $385F4000, $385F8000, $385FC000,
    $38600000, $38604000, $38608000, $3860C000, $38610000, $38614000, $38618000, $3861C000, $38620000, $38624000, $38628000, $3862C000, $38630000, $38634000, $38638000, $3863C000,
    $38640000, $38644000, $38648000, $3864C000, $38650000, $38654000, $38658000, $3865C000, $38660000, $38664000, $38668000, $3866C000, $38670000, $38674000, $38678000, $3867C000,
    $38680000, $38684000, $38688000, $3868C000, $38690000, $38694000, $38698000, $3869C000, $386A0000, $386A4000, $386A8000, $386AC000, $386B0000, $386B4000, $386B8000, $386BC000,
    $386C0000, $386C4000, $386C8000, $386CC000, $386D0000, $386D4000, $386D8000, $386DC000, $386E0000, $386E4000, $386E8000, $386EC000, $386F0000, $386F4000, $386F8000, $386FC000,
    $38700000, $38704000, $38708000, $3870C000, $38710000, $38714000, $38718000, $3871C000, $38720000, $38724000, $38728000, $3872C000, $38730000, $38734000, $38738000, $3873C000,
    $38740000, $38744000, $38748000, $3874C000, $38750000, $38754000, $38758000, $3875C000, $38760000, $38764000, $38768000, $3876C000, $38770000, $38774000, $38778000, $3877C000,
    $38780000, $38784000, $38788000, $3878C000, $38790000, $38794000, $38798000, $3879C000, $387A0000, $387A4000, $387A8000, $387AC000, $387B0000, $387B4000, $387B8000, $387BC000,
    $387C0000, $387C4000, $387C8000, $387CC000, $387D0000, $387D4000, $387D8000, $387DC000, $387E0000, $387E4000, $387E8000, $387EC000, $387F0000, $387F4000, $387F8000, $387FC000,
    $38000000, $38002000, $38004000, $38006000, $38008000, $3800A000, $3800C000, $3800E000, $38010000, $38012000, $38014000, $38016000, $38018000, $3801A000, $3801C000, $3801E000,
    $38020000, $38022000, $38024000, $38026000, $38028000, $3802A000, $3802C000, $3802E000, $38030000, $38032000, $38034000, $38036000, $38038000, $3803A000, $3803C000, $3803E000,
    $38040000, $38042000, $38044000, $38046000, $38048000, $3804A000, $3804C000, $3804E000, $38050000, $38052000, $38054000, $38056000, $38058000, $3805A000, $3805C000, $3805E000,
    $38060000, $38062000, $38064000, $38066000, $38068000, $3806A000, $3806C000, $3806E000, $38070000, $38072000, $38074000, $38076000, $38078000, $3807A000, $3807C000, $3807E000,
    $38080000, $38082000, $38084000, $38086000, $38088000, $3808A000, $3808C000, $3808E000, $38090000, $38092000, $38094000, $38096000, $38098000, $3809A000, $3809C000, $3809E000,
    $380A0000, $380A2000, $380A4000, $380A6000, $380A8000, $380AA000, $380AC000, $380AE000, $380B0000, $380B2000, $380B4000, $380B6000, $380B8000, $380BA000, $380BC000, $380BE000,
    $380C0000, $380C2000, $380C4000, $380C6000, $380C8000, $380CA000, $380CC000, $380CE000, $380D0000, $380D2000, $380D4000, $380D6000, $380D8000, $380DA000, $380DC000, $380DE000,
    $380E0000, $380E2000, $380E4000, $380E6000, $380E8000, $380EA000, $380EC000, $380EE000, $380F0000, $380F2000, $380F4000, $380F6000, $380F8000, $380FA000, $380FC000, $380FE000,
    $38100000, $38102000, $38104000, $38106000, $38108000, $3810A000, $3810C000, $3810E000, $38110000, $38112000, $38114000, $38116000, $38118000, $3811A000, $3811C000, $3811E000,
    $38120000, $38122000, $38124000, $38126000, $38128000, $3812A000, $3812C000, $3812E000, $38130000, $38132000, $38134000, $38136000, $38138000, $3813A000, $3813C000, $3813E000,
    $38140000, $38142000, $38144000, $38146000, $38148000, $3814A000, $3814C000, $3814E000, $38150000, $38152000, $38154000, $38156000, $38158000, $3815A000, $3815C000, $3815E000,
    $38160000, $38162000, $38164000, $38166000, $38168000, $3816A000, $3816C000, $3816E000, $38170000, $38172000, $38174000, $38176000, $38178000, $3817A000, $3817C000, $3817E000,
    $38180000, $38182000, $38184000, $38186000, $38188000, $3818A000, $3818C000, $3818E000, $38190000, $38192000, $38194000, $38196000, $38198000, $3819A000, $3819C000, $3819E000,
    $381A0000, $381A2000, $381A4000, $381A6000, $381A8000, $381AA000, $381AC000, $381AE000, $381B0000, $381B2000, $381B4000, $381B6000, $381B8000, $381BA000, $381BC000, $381BE000,
    $381C0000, $381C2000, $381C4000, $381C6000, $381C8000, $381CA000, $381CC000, $381CE000, $381D0000, $381D2000, $381D4000, $381D6000, $381D8000, $381DA000, $381DC000, $381DE000,
    $381E0000, $381E2000, $381E4000, $381E6000, $381E8000, $381EA000, $381EC000, $381EE000, $381F0000, $381F2000, $381F4000, $381F6000, $381F8000, $381FA000, $381FC000, $381FE000,
    $38200000, $38202000, $38204000, $38206000, $38208000, $3820A000, $3820C000, $3820E000, $38210000, $38212000, $38214000, $38216000, $38218000, $3821A000, $3821C000, $3821E000,
    $38220000, $38222000, $38224000, $38226000, $38228000, $3822A000, $3822C000, $3822E000, $38230000, $38232000, $38234000, $38236000, $38238000, $3823A000, $3823C000, $3823E000,
    $38240000, $38242000, $38244000, $38246000, $38248000, $3824A000, $3824C000, $3824E000, $38250000, $38252000, $38254000, $38256000, $38258000, $3825A000, $3825C000, $3825E000,
    $38260000, $38262000, $38264000, $38266000, $38268000, $3826A000, $3826C000, $3826E000, $38270000, $38272000, $38274000, $38276000, $38278000, $3827A000, $3827C000, $3827E000,
    $38280000, $38282000, $38284000, $38286000, $38288000, $3828A000, $3828C000, $3828E000, $38290000, $38292000, $38294000, $38296000, $38298000, $3829A000, $3829C000, $3829E000,
    $382A0000, $382A2000, $382A4000, $382A6000, $382A8000, $382AA000, $382AC000, $382AE000, $382B0000, $382B2000, $382B4000, $382B6000, $382B8000, $382BA000, $382BC000, $382BE000,
    $382C0000, $382C2000, $382C4000, $382C6000, $382C8000, $382CA000, $382CC000, $382CE000, $382D0000, $382D2000, $382D4000, $382D6000, $382D8000, $382DA000, $382DC000, $382DE000,
    $382E0000, $382E2000, $382E4000, $382E6000, $382E8000, $382EA000, $382EC000, $382EE000, $382F0000, $382F2000, $382F4000, $382F6000, $382F8000, $382FA000, $382FC000, $382FE000,
    $38300000, $38302000, $38304000, $38306000, $38308000, $3830A000, $3830C000, $3830E000, $38310000, $38312000, $38314000, $38316000, $38318000, $3831A000, $3831C000, $3831E000,
    $38320000, $38322000, $38324000, $38326000, $38328000, $3832A000, $3832C000, $3832E000, $38330000, $38332000, $38334000, $38336000, $38338000, $3833A000, $3833C000, $3833E000,
    $38340000, $38342000, $38344000, $38346000, $38348000, $3834A000, $3834C000, $3834E000, $38350000, $38352000, $38354000, $38356000, $38358000, $3835A000, $3835C000, $3835E000,
    $38360000, $38362000, $38364000, $38366000, $38368000, $3836A000, $3836C000, $3836E000, $38370000, $38372000, $38374000, $38376000, $38378000, $3837A000, $3837C000, $3837E000,
    $38380000, $38382000, $38384000, $38386000, $38388000, $3838A000, $3838C000, $3838E000, $38390000, $38392000, $38394000, $38396000, $38398000, $3839A000, $3839C000, $3839E000,
    $383A0000, $383A2000, $383A4000, $383A6000, $383A8000, $383AA000, $383AC000, $383AE000, $383B0000, $383B2000, $383B4000, $383B6000, $383B8000, $383BA000, $383BC000, $383BE000,
    $383C0000, $383C2000, $383C4000, $383C6000, $383C8000, $383CA000, $383CC000, $383CE000, $383D0000, $383D2000, $383D4000, $383D6000, $383D8000, $383DA000, $383DC000, $383DE000,
    $383E0000, $383E2000, $383E4000, $383E6000, $383E8000, $383EA000, $383EC000, $383EE000, $383F0000, $383F2000, $383F4000, $383F6000, $383F8000, $383FA000, $383FC000, $383FE000,
    $38400000, $38402000, $38404000, $38406000, $38408000, $3840A000, $3840C000, $3840E000, $38410000, $38412000, $38414000, $38416000, $38418000, $3841A000, $3841C000, $3841E000,
    $38420000, $38422000, $38424000, $38426000, $38428000, $3842A000, $3842C000, $3842E000, $38430000, $38432000, $38434000, $38436000, $38438000, $3843A000, $3843C000, $3843E000,
    $38440000, $38442000, $38444000, $38446000, $38448000, $3844A000, $3844C000, $3844E000, $38450000, $38452000, $38454000, $38456000, $38458000, $3845A000, $3845C000, $3845E000,
    $38460000, $38462000, $38464000, $38466000, $38468000, $3846A000, $3846C000, $3846E000, $38470000, $38472000, $38474000, $38476000, $38478000, $3847A000, $3847C000, $3847E000,
    $38480000, $38482000, $38484000, $38486000, $38488000, $3848A000, $3848C000, $3848E000, $38490000, $38492000, $38494000, $38496000, $38498000, $3849A000, $3849C000, $3849E000,
    $384A0000, $384A2000, $384A4000, $384A6000, $384A8000, $384AA000, $384AC000, $384AE000, $384B0000, $384B2000, $384B4000, $384B6000, $384B8000, $384BA000, $384BC000, $384BE000,
    $384C0000, $384C2000, $384C4000, $384C6000, $384C8000, $384CA000, $384CC000, $384CE000, $384D0000, $384D2000, $384D4000, $384D6000, $384D8000, $384DA000, $384DC000, $384DE000,
    $384E0000, $384E2000, $384E4000, $384E6000, $384E8000, $384EA000, $384EC000, $384EE000, $384F0000, $384F2000, $384F4000, $384F6000, $384F8000, $384FA000, $384FC000, $384FE000,
    $38500000, $38502000, $38504000, $38506000, $38508000, $3850A000, $3850C000, $3850E000, $38510000, $38512000, $38514000, $38516000, $38518000, $3851A000, $3851C000, $3851E000,
    $38520000, $38522000, $38524000, $38526000, $38528000, $3852A000, $3852C000, $3852E000, $38530000, $38532000, $38534000, $38536000, $38538000, $3853A000, $3853C000, $3853E000,
    $38540000, $38542000, $38544000, $38546000, $38548000, $3854A000, $3854C000, $3854E000, $38550000, $38552000, $38554000, $38556000, $38558000, $3855A000, $3855C000, $3855E000,
    $38560000, $38562000, $38564000, $38566000, $38568000, $3856A000, $3856C000, $3856E000, $38570000, $38572000, $38574000, $38576000, $38578000, $3857A000, $3857C000, $3857E000,
    $38580000, $38582000, $38584000, $38586000, $38588000, $3858A000, $3858C000, $3858E000, $38590000, $38592000, $38594000, $38596000, $38598000, $3859A000, $3859C000, $3859E000,
    $385A0000, $385A2000, $385A4000, $385A6000, $385A8000, $385AA000, $385AC000, $385AE000, $385B0000, $385B2000, $385B4000, $385B6000, $385B8000, $385BA000, $385BC000, $385BE000,
    $385C0000, $385C2000, $385C4000, $385C6000, $385C8000, $385CA000, $385CC000, $385CE000, $385D0000, $385D2000, $385D4000, $385D6000, $385D8000, $385DA000, $385DC000, $385DE000,
    $385E0000, $385E2000, $385E4000, $385E6000, $385E8000, $385EA000, $385EC000, $385EE000, $385F0000, $385F2000, $385F4000, $385F6000, $385F8000, $385FA000, $385FC000, $385FE000,
    $38600000, $38602000, $38604000, $38606000, $38608000, $3860A000, $3860C000, $3860E000, $38610000, $38612000, $38614000, $38616000, $38618000, $3861A000, $3861C000, $3861E000,
    $38620000, $38622000, $38624000, $38626000, $38628000, $3862A000, $3862C000, $3862E000, $38630000, $38632000, $38634000, $38636000, $38638000, $3863A000, $3863C000, $3863E000,
    $38640000, $38642000, $38644000, $38646000, $38648000, $3864A000, $3864C000, $3864E000, $38650000, $38652000, $38654000, $38656000, $38658000, $3865A000, $3865C000, $3865E000,
    $38660000, $38662000, $38664000, $38666000, $38668000, $3866A000, $3866C000, $3866E000, $38670000, $38672000, $38674000, $38676000, $38678000, $3867A000, $3867C000, $3867E000,
    $38680000, $38682000, $38684000, $38686000, $38688000, $3868A000, $3868C000, $3868E000, $38690000, $38692000, $38694000, $38696000, $38698000, $3869A000, $3869C000, $3869E000,
    $386A0000, $386A2000, $386A4000, $386A6000, $386A8000, $386AA000, $386AC000, $386AE000, $386B0000, $386B2000, $386B4000, $386B6000, $386B8000, $386BA000, $386BC000, $386BE000,
    $386C0000, $386C2000, $386C4000, $386C6000, $386C8000, $386CA000, $386CC000, $386CE000, $386D0000, $386D2000, $386D4000, $386D6000, $386D8000, $386DA000, $386DC000, $386DE000,
    $386E0000, $386E2000, $386E4000, $386E6000, $386E8000, $386EA000, $386EC000, $386EE000, $386F0000, $386F2000, $386F4000, $386F6000, $386F8000, $386FA000, $386FC000, $386FE000,
    $38700000, $38702000, $38704000, $38706000, $38708000, $3870A000, $3870C000, $3870E000, $38710000, $38712000, $38714000, $38716000, $38718000, $3871A000, $3871C000, $3871E000,
    $38720000, $38722000, $38724000, $38726000, $38728000, $3872A000, $3872C000, $3872E000, $38730000, $38732000, $38734000, $38736000, $38738000, $3873A000, $3873C000, $3873E000,
    $38740000, $38742000, $38744000, $38746000, $38748000, $3874A000, $3874C000, $3874E000, $38750000, $38752000, $38754000, $38756000, $38758000, $3875A000, $3875C000, $3875E000,
    $38760000, $38762000, $38764000, $38766000, $38768000, $3876A000, $3876C000, $3876E000, $38770000, $38772000, $38774000, $38776000, $38778000, $3877A000, $3877C000, $3877E000,
    $38780000, $38782000, $38784000, $38786000, $38788000, $3878A000, $3878C000, $3878E000, $38790000, $38792000, $38794000, $38796000, $38798000, $3879A000, $3879C000, $3879E000,
    $387A0000, $387A2000, $387A4000, $387A6000, $387A8000, $387AA000, $387AC000, $387AE000, $387B0000, $387B2000, $387B4000, $387B6000, $387B8000, $387BA000, $387BC000, $387BE000,
    $387C0000, $387C2000, $387C4000, $387C6000, $387C8000, $387CA000, $387CC000, $387CE000, $387D0000, $387D2000, $387D4000, $387D6000, $387D8000, $387DA000, $387DC000, $387DE000,
    $387E0000, $387E2000, $387E4000, $387E6000, $387E8000, $387EA000, $387EC000, $387EE000, $387F0000, $387F2000, $387F4000, $387F6000, $387F8000, $387FA000, $387FC000, $387FE000
  );
const exponent_table : array[0..63] of longword = (
    $00000000, $00800000, $01000000, $01800000, $02000000, $02800000, $03000000, $03800000, $04000000, $04800000, $05000000, $05800000, $06000000, $06800000, $07000000, $07800000,
    $08000000, $08800000, $09000000, $09800000, $0A000000, $0A800000, $0B000000, $0B800000, $0C000000, $0C800000, $0D000000, $0D800000, $0E000000, $0E800000, $0F000000, $47800000,
    $80000000, $80800000, $81000000, $81800000, $82000000, $82800000, $83000000, $83800000, $84000000, $84800000, $85000000, $85800000, $86000000, $86800000, $87000000, $87800000,
    $88000000, $88800000, $89000000, $89800000, $8A000000, $8A800000, $8B000000, $8B800000, $8C000000, $8C800000, $8D000000, $8D800000, $8E000000, $8E800000, $8F000000, $C7800000
  );
const offset_table : array[0..63] of word = (
    0, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400,
    0, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400, $400
  );
var  bits : ConversionBits; i:NativeInt; value:word;
begin
  for i:=0 to N-1 do begin
    value := word(src[i]);
    bits.i32 := mantissa_table[offset_table[value shr 10] + (value and $3FF)] + exponent_table[value shr 10];
    dst[i] := bits.f32;
  end;
end;

procedure SingleToFP16(const N: NativeInt; const src: PSingle; dst: PFP16);
    const base_table : array[0..511] of word = (
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000,
      $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0001, $0002, $0004, $0008, $0010, $0020, $0040, $0080, $0100,
      $0200, $0400, $0800, $0C00, $1000, $1400, $1800, $1C00, $2000, $2400, $2800, $2C00, $3000, $3400, $3800, $3C00,
      $4000, $4400, $4800, $4C00, $5000, $5400, $5800, $5C00, $6000, $6400, $6800, $6C00, $7000, $7400, $7800, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00, $7C00,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8000,
      $8000, $8000, $8000, $8000, $8000, $8000, $8000, $8001, $8002, $8004, $8008, $8010, $8020, $8040, $8080, $8100,
      $8200, $8400, $8800, $8C00, $9000, $9400, $9800, $9C00, $A000, $A400, $A800, $AC00, $B000, $B400, $B800, $BC00,
      $C000, $C400, $C800, $CC00, $D000, $D400, $D800, $DC00, $E000, $E400, $E800, $EC00, $F000, $F400, $F800, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00,
      $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00, $FC00
    );
    const shift_table : array[0..511] of byte= (
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
      24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13
    );
var
  bits: ConversionBits;
  i:NativeInt;
begin
  for i:=0 to N-1 do begin
    bits.f32 := src[i];
    PWord(dst)[i] := base_table[bits.i32 shr 23] + word((bits.i32 and $7FFFFF) shr shift_table[bits.i32 shr 23]);
  end
end;

function ifthen(const cond:boolean; const ifTrue, ifFalse:longint):longint;
begin
  if cond then result:= ifTrue else result := iffalse
end;

function ifthen(const cond: boolean; const ifTrue, ifFalse: int64): int64;
begin
  if cond then result:= ifTrue else result := iffalse
end;

function ifthen(const cond: boolean; const ifTrue, ifFalse: single): single;
begin
  if cond then result:= ifTrue else result := iffalse
end;

function ifthen(const cond: boolean; const ifTrue, ifFalse: double): double;
begin
  if cond then result:= ifTrue else result := iffalse
end;

function ifthen(const cond: boolean; const ifTrue, ifFalse: string): string;
begin
  if cond then result:= ifTrue else result := iffalse
end;

{ TMemoryArena }

constructor TMemoryArena.Create(const aPool: NativeUInt);
begin
  FData := AllocMem(aPool);
  assert(assigned(FData), 'ERROR Creating Memory Arena'+sLineBreak+'Not enough memory!');
  {$ifdef DEBUG}
  writeln('Create : RefCount ', refcount);
  readln;
  {$endif}
  FMemPos:=0;
end;

function TMemoryArena.ArenaSize: NativeUInt;
begin
   result := FSize
end;

procedure TMemoryArena.ReSize(const aSize: NativeUInt);
begin
  ReAllocMem(FData, aSize);
  assert(assigned(FData), 'ERROR Resizing Memory Arena'+sLineBreak+'Not enough memory!');
  FSize := aSize
end;

function TMemoryArena.Allocate(const aSize: NativeUInt): pointer;
begin
  assert(FMemPos+aSize<=ArenaSize, 'ERROR not enough memory size in the memory arena');
  inc(FMemPos, aSize);
  result := PByte(FData) + FMemPos
end;

destructor TMemoryArena.Destroy;
begin
  Freemem(FData);
  {$ifdef DEBUG}
  writeln('Destroy RefCount ', RefCount);
  readln;
  {$endif}
  inherited Destroy;
end;

var arena : IMemoryArena;
  {$ifdef MSWINDOWS}
    hConsole : THandle;
    cMode : longword;
  {$endif}
initialization
  {$ifdef MSWINDOWS}
  if IsConsole then
  begin
    hConsole := GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleMode(hConsole, cMode);
    SetConsoleMode(hConsole, (cmode or ENABLE_VIRTUAL_TERMINAL_PROCESSING or
      ENABLE_PROCESSED_OUTPUT){ and not ENABLE_WRAP_AT_EOL_OUTPUT});
  end;
  //write(#$1B'[?1049h'); // set Console Alternative Buffer
  {$endif}
  TQNNImage.computePngCRCTable();
  //arena := TMemoryArena.create($ff);

end.

