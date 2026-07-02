unit quicknn_kernels;

{$ifdef FPC}
  {$ifdef USE_CPP_GEMM}
    {$linklib msvcrt.dll}
    {$linklib gomp.dll.a}
  {$endif}
  {$PackRecords C}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$modeswitch nestedprocvars}
  {$ifdef CPUX64}
    {$asmmode intel}
    {$FPUType AVX2}
  {$endif}
  {$if defined(darwin)}
    {$LinkFramework accelerate}
  {$endif}
{$endif}
{$ifdef CPUX64}
//{$EXCESSPRECISION OFF}
{$endif}
{$C+} // enable assertions
{$H+} // longstrings
{$M+} // include typeinfo
{$pointermath on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds
{$Z4} // aligh enums to 4 bytes (compatibility with external routines)

//{$define USE_MULTITHREADING}

interface
uses
  SysUtils, classes, Math, typinfo
  {$if defined(MSWINDOWS)}
  , windows
  {$elseif defined(POSIX)}
  , unixbase
  {$endif}
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  , quicknn_common
  , sixel
  , nVulkanHelper
  ;

const EPSILON = 1e-6;
{$if not declared(blasint)}
type
  blasint = longint;
{$endif}
//{$if not declared(CBLAS_LAYOUT)}
type
  CBLAS_Layout = (CblasRowMajor = 101, CblasColMajor = 102);
//{$else}
//const
//  CblasRowMajor = CBLAS_Layout.CblasRowMajor;
//  CblasColMajor = CBLAS_Layout.CblasColMajor;
//{$endif}

//{$if not declared(CBLAS_ORDER)}
type
  CBLAS_ORDER = CBLAS_Layout;
//{$endif}
//{$if not declared(CBLAS_TRANSPOSE)}
type
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans =
    113, CblasConjNoTrans = 114);
//{$else}
//const
  //CblasNoTrans = CBLAS_TRANSPOSE.CblasNoTrans;
  //CblasTrans = CBLAS_TRANSPOSE.CblasTrans;
  //CblasConjTrans = CBLAS_TRANSPOSE.CblasConjTrans;
  //CblasConjNoTrans = CBLAS_TRANSPOSE.CblasConjNoTrans ;
//{$endif}

type
  TSingleArray = TArray<Single>;
  PFloatarray = ^TFloatArray;
  TFloatArray = array[0..MaxLongint div 32] of single;
  TInterpolation = ( iNearest, iLinear, iCubic, iLanczos);

  {$if defined(MACOS) or defined(DARWIN) or defined(USE_STATIC_LINK)}
  procedure cblas_saxpy(n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ; external;
  function  cblas_sdot (n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):single; winapi ; external;
  function  cblas_sasum(n:blasint; x:Psingle; incx:blasint):single; winapi ; external;
  procedure cblas_sscal(N:blasint; alpha:single; X:Psingle; incX:blasint); winapi ; external;
  procedure cblas_sgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ; external;

  procedure cblas_daxpy(n:blasint; alpha:double; x:PDouble; incx:blasint; y:PDouble; incy:blasint); winapi ; external;
  function  cblas_ddot (n:blasint; x:PDouble; incx:blasint; y:PDouble; incy:blasint):double; winapi ; external;
  function  cblas_dasum(n:blasint; x:PDouble; incx:blasint):double; winapi ; external;
  procedure cblas_dscal(N:blasint; alpha:double; X:PDouble; incX:blasint); winapi ; external;
  procedure cblas_dgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:double; A:PDouble; lda:blasint; B:PDouble; ldb:blasint; beta:double; C:PDouble; ldc:blasint); winapi ; external;
  {$endif}
var
  hBLASLib : HMODULE;
  cblas_gemm : procedure (Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:PQNNFloat; ldc:blasint); winapi ;
  cblas_axpy : procedure (n:blasint; alpha:single; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint); winapi ;
  cblas_dot : function (n:blasint; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint):single; winapi ;
  cblas_asum : function (n:blasint; x:PQNNFloat; incx:blasint):QNNFloat; winapi ;
  cblas_scal : procedure (N:blasint; alpha:QNNFloat; X:PQNNFloat; incX:blasint = 1); winapi ;
  cblas_sbgemm : procedure(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PBF16; lda:blasint; B:PBF16; ldb:blasint; beta:single; C:PSingle; ldc:blasint); winapi ;


  workspace : TArray<QNNFloat>;
(* Global callback pointers - set by caller before inference *)
// Add and Mul terms used for vector vector element wise operations
// Scale and Bias terms used for vector Scalar element wise operations

procedure QNNAdd(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNAddInplace(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNMul(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNMulInplace(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNScale(const dst, src:PQNNFloat; const aScale:QNNFloat; const N:longint);   overload;
procedure QNNScaleInplace(const dst: PQNNFloat; const aScale: QNNFloat; const N: longint);overload;
procedure QNNBias(const dst, src:PQNNFloat; const aBias:QNNFloat; const N:longint);overload;
procedure QNNBiasInplace(const dst:PQNNFloat; const aBias:QNNFloat; const N:longint);overload;


procedure QNNFusedMulAdd(const dst, src, srcM, srcA:PQNNFloat; const N:longint);overload;
procedure QNNFusedAddMul(const dst, src, srcA, srcM:PQNNFloat; const N:longint);overload;
procedure QNNFusedScaleBias(const dst, src:PQNNFloat; const aScale, aBias:QNNFloat; const N:longint);overload;
procedure QNNFusedBiasScale(const dst, src:PQNNFloat; const aBias, aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedMulScale(const dst, src, srcM:PQNNFloat; const aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedBiasMulScale(const dst, src, srcM:PQNNFloat; const aBias, aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedScaleAdd(const dst, src, srcAdd:PQNNFloat; const aScale:QNNFloat; const N:longint); overload;
procedure QNNFusedBiasAdd(const dst, src, srcA:PQNNFloat; const aBias:QNNFloat; const N:longint); overload;
function QNNDot(const a, b:PQNNFloat; const N:longint):QNNFloat;

procedure QNNAccAdd(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNAccMul(const dst, a, b: PQNNFloat; const N:integer);   overload;

procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N:integer);   overload;
procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N:integer);   overload;
procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N:integer);   overload;

procedure QNNMatTranspose(const dst, src: PQNNFloat; const srcRows, srcCols: longint);
procedure QNNLinear(const dst, x, W, B:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias(const dst, x, W:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias_BF16(const dst, x:PQNNFloat; const W: PBF16; const seqLen, inDIM, outDIM: integer);
procedure QNNIm2Col(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth
                        , padHeight, padWidth
                        , strideY, strideX
                        , dilationY, dilationX: integer
                        ; const im: PQNNFloat; const col: PQNNFloat
                        ; const imOffset: integer =0 ; const colOffset: integer =0
                        ; const MultiThread: boolean = False);

procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth, padHeight, padWidth
                        , strideY, strideX, dilationY, dilationX: integer
                        ; const im: PQNNFloat; const imStride, imOffset: integer
                        ; const col: PQNNFloat; const colStride, colOffset: integer
                        ; const batch:integer);

procedure QNNConv2d( const dst, src, weights, bias:PQNNFloat;
                     const in_ch, out_ch, H, W, kH, kW, stride, padding:longint; const batch:longint =1);

procedure QNNNorm(const N:longint; const dst, src:PQNNFloat; const weights:PQNNFloat = nil; const stride:longint = 1);      overload;

procedure QNNRMSNorm(const N:longint; const dst, src:PQNNFloat; const weights:PQNNFloat = nil; const stride:longint = 1);   overload;
procedure QNNRMSNormRows(const dst, src, weight:PQNNFloat; const rows, dim: integer); overload;
procedure QNNTanh(const dst:PQNNFloat; const N:longint);

procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights:PQNNFloat; const seq, heads, headDim:longint); overload;
procedure QNNQKRMSNorm(const dst, normWeights:PQNNFloat; const seq, heads, headDim:longint); overload;

procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat; const batch, channels, H, W, num_groups:integer);
procedure QNNBatchNorm(const dst, src, running_mean, running_var, gamma, beta : PQNNFloat; const batch, channels, H, W: integer);

function QNNMinMax(const N:integer; const src:PQNNFloat; out outMin, outMax: QNNFloat; const argMin: PInteger; const argMax:PInteger = nil; const stride:integer=1):QNNFloat;
function QNNMax(const N:integer; const src:PQNNFloat; const arg:PInteger = nil; const stride:integer=1):QNNFloat;
function QNNMin(const N:integer; const src:PQNNFloat; const arg:PInteger = nil; const stride:integer=1):QNNFloat;
function QNNMaxAbs(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNMinAbs(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNArgMax(const N:integer; const src:PQNNFloat; const stride:integer=1):integer;
function QNNArgMin(const N:integer; const src:PQNNFloat; const stride:integer=1):integer;
function QNNSum(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNSumSqr(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNSumSqrDiff(const N: integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer=1):QNNFloat;overload;
function QNNSqrDistance(const N: integer; const src1, src2:PQNNFloat; const stride:integer=1):QNNFloat;overload;
function QNNMaxAbsDiff(const N: integer; const src1, src2: PQNNFloat; out outSrc1, outSrc2:QNNFloat; const stride: integer=1): QNNFloat; overload;
function QNNMaxAbsDiff(const N: integer; const src1, src2: PQNNFloat; const stride: integer=1): QNNFloat; overload;
function QNNSumAbsDiff(const N: integer; const src1, src2: PQNNFloat; const stride: integer=1): QNNFloat;  overload;
function QNNSumAbsDiff(const N: integer; const src: PQNNFloat; const aMean:QNNFloat; const stride: integer=1): QNNFloat;  overload;
function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNVariance(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer=1; const isPopulation:boolean=true):QNNFloat;
procedure QNNSigmoid(const x: PQNNFloat; const N:integer); overload;
procedure QNNSigmoid(const dst, src: PQNNFloat; const N:integer); overload;
procedure QNNSiluInplace(const x:PQNNFloat; const N:integer);     overload;
procedure QNNSilu(const dst, src:PQNNFloat; const N:integer);     overload;
procedure QNNSiluMul(const gate, up:PQNNFloat; const N:integer);
procedure QNNSoftmax(const x: PQNNFloat; const N: integer); overload;
procedure QNNSoftmaxRows(const x: PQNNFloat; const rows, cols: integer); overload;
procedure QNNAttention(const dst, Q, K, V: PQNNFloat; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHead(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat; const tile_scores: PQNNFloat; const q_tile_size, k_tile_size: longint);
procedure QNNFlashAttention(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
procedure QNNUpSampleNearest(const dst, src: PQNNFloat; const batch, channels, H, W, scale_h, scale_w: longint);
procedure QNNUpSample(const dst, src: PQNNFloat; const batch, channels, H, W:longint; const scale_h, scale_w: QNNFloat; const interpolation:TInterpolation = iNearest);
procedure QNNPatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNUnpatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNCopy(const dst, src:PQNNFloat; const N:integer); overload;
procedure QNNCopyStrided(const dst:PQNNFloat; const dstStride:integer; const src:PQNNFloat; const srcStride: integer; const N:integer); overload;
procedure QNNFill(const dst:PQNNFloat; const val:QNNFloat; const N:integer; const stride:integer=1);
procedure QNNBroadcast(const dst:pointer; const src; const srcSize, N:integer; const stride:integer=1);
procedure QNNGatedAdd(const dst, gate, proj:psingle; const seq, hidden:integer);
procedure QNNComputeRoPE(const dst:PQNNFloat; const maxSeq, dim:longint; const theta:QNNFloat);           overload;
procedure QNNComputeRoPE(const cosOut,sinOut:PQNNFloat; const maxSeqLen, headDim:longint; const theta:QNNFloat); overload;
procedure QNNComputeRoPE2D(const cosDst,sinDst:PQNNFloat; const patch_h, patch_w, dim:longint; const theta:QNNFloat);
procedure QNNComputeRoPE2DOffset(const cosDst,sinDst:PQNNFloat; const patch_h, patch_w, dim:longint; const theta:QNNFloat; const offset_t:longint);
procedure QNNComputeRoPEText(const cosDst, sinDst: PQNNFloat; const txt_seq, axis_dim: longint; const theta: QNNFloat);
procedure QNNApplyRoPE(const dst, freqs: PQNNFloat; const batch, seq, heads, head_dim: longint);        overload;
procedure QNNApplyRoPE(const dst, cosIn, sinIn: PQNNFloat; const numHeads, headDim: longint); overload;
procedure QNNApplyRoPE(const dst, cosIn, sinIn: PQNNFloat; const seqLen, numHeads, headDim: longint); overload;
procedure QNNApplyRoPE(const q, k, cos_cache, sin_cache:PQNNFloat; const seq_len, num_q_heads, num_kv_heads, head_dim:longint); overload;

procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq : PQNNFloat; const seq, heads, head_dim, dim: longint);
//Positional Embedding
function QNNTimeStepEmbedding(const t : QNNFloat;const dim:integer; const max_period:QNNFloat):TMemoryBlock;

// Adaptive Layer Normalization
procedure QNNAdaLN(const dst, src, bias, scale:PQNNFloat; const seq, hidden:longint; const eps:QNNFloat=EPSILON);

procedure QNNLINEAR_BF16_OR_F32(const dst, src:PSingle; const weight:TMemoryBlock; const weight_bf16:TMemoryBlock; const seq, srcDim, dstDim:integer);inline;

{$if not declared(FillWord)}
   {$define FILLWORD}
procedure FillWord(var dst; const count:IntPtr; const val:Word);
{$endif}
{$if not declared(FillDWord)}
   {$define FILLDWORD}
procedure FillDWord(var dst; const count:IntPtr; const val:longword);
{$endif}
{$if not declared(FillQWord)}
   {$define FILLQWORD}
procedure FillQWord(var dst; const count:IntPtr; const val:UInt64);
{$endif}

procedure printStat(const src:PQNNFloat; const N:longint);  overload;
procedure printStat(const src:TMemoryBlock);                overload;
procedure printCompare(const N:longint; const src1, src2:PQNNFloat; const isSumSqrDiff:boolean =false);
procedure writeTensor(const buf:TMemoryBlock);
function readTensor():TMemoryBlock;
function readArray():TArray<integer>;
procedure compareArray(const a, b:TArray<Integer>);             overload;
procedure compareArray(const a, b: PInteger; const N:longint);  overload;
procedure compareArray(const a, b: PSingle; const N:longint);  overload;

procedure qgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:PQNNFloat; ldc:blasint); WINAPI;


implementation

{$if defined(FILLWORD)}
procedure FillWord(var dst ; const count:IntPtr; const val:Word);
var i:IntPtr;  d:PWord;
begin
  d := @dst;
  for i:=0 to count-1 do
    d[i] := val
end;
{$endif}

{$if defined(FILLDWORD)}
procedure FillDWord(var dst ; const count:IntPtr; const val:longword);
var i:IntPtr;  d:PLongword;
begin
  d := @dst;
  for i:=0 to count-1 do
    d[i] := val
end;
{$endif}

{$if defined(FILLQWORD)}
procedure FillQWord(var dst ; const count:IntPtr; const val:UInt64);
var i:IntPtr; d:PUInt64;
begin
  d := @dst;
  for i:=0 to count-1 do
    d[i] := val
end;
{$endif}

function sqr(const x:QNNFloat):QNNFloat;inline;
begin
  exit(x*x)
end;

//{$macro on}
//{$define fast_exp:=exp}

function fast_exp(const x:single):single; inline;
var
  n, r, p:single;
  v : record
    case boolean of
      false :(i:longint);
      true  :(f:single)
  end;
begin
  //result := exp(x)
    if (x < -87.3) then  exit( 0.0);
    if (x > 88.7)  then exit( 1e38);
    n := floor(x * 1.4426950408889634 + 0.5);
    r := x - n * 0.6931471805599453;
    p := 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 +
              r * (0.04166667 + r * 0.00833333))));
    v.f := p;
    inc(v.i, trunc(n) shl 23);
    result := v.f;
end;

function QNNMinMax(const N: integer; const src: PQNNFloat; out outMin,
  outMax: QNNFloat; const argMin: PInteger; const argMax: PInteger;
  const stride: integer): QNNFloat;
var i,m:integer;
begin
  assert(assigned(src), 'ERROR QNNMax: [src] is empty!');
  outMin := src[0];
  outMax := src[0];
  if assigned(argMax) then argMax^ := 0;
  if assigned(argMin) then argMin^ := 0;
  if stride=1 then begin
    for i:=1 to N-1 do begin
      if src[i]>outMax then begin
        outMax := src[i];
        if assigned(argMax) then argMax^ :=i
      end;
      if src[i]<outMin then begin
        outMin := src[i];
        if assigned(argMin) then argMin^ :=i
      end;
    end
  end else
    for i:=1 to N-1 do begin
      if src[i*stride]<outMax then begin
        outMax := src[i*stride];
        if assigned(argMax) then argMax^ :=i*stride
      end;
      if src[i*stride]<outMin then begin
        outMin := src[i*stride];
        if assigned(argMin) then argMin^ :=i*stride
      end
    end
end;

function QNNMax(const N: integer; const src: PQNNFloat; const arg: PInteger;
  const stride: integer): QNNFloat;
var i,m:integer;
begin
  assert(assigned(src), 'ERROR QNNMax: [src] is empty!');
  result := src[0];
  if assigned(arg) then arg^ := 0;
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]>result then begin
        result := src[i];
        if assigned(arg) then arg^ :=i
      end
  end else
    for i:=1 to N-1 do
      if src[i*stride]>result then begin
        result := src[i*stride];
        if assigned(arg) then arg^ :=i*stride
      end

end;

function QNNMin(const N: integer; const src: PQNNFloat; const arg:PInteger; const stride: integer): QNNFloat;
var i,m:integer;
begin
  assert(assigned(src), 'ERROR QNNMin: [src] is empty!');
  result := src[0];
  if assigned(arg) then arg^ := 0;
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]<result then begin
        result := src[i];
        if assigned(arg) then arg^ := i
      end;
  end else
    for i:=1 to N-1 do
      if src[i*stride]<result then begin
        result := src[i*stride];
        if assigned(arg) then arg^ := i*stride
      end
end;

function QNNMaxAbs(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i:integer;
  ab:QNNFloat;
begin
  assert(assigned(src), 'ERROR QNNMin: [src] is empty!');
  result := abs(src[0]);
  if stride=1 then begin
    for i:=1 to N-1 do begin
      ab := abs(src[i]);
      if ab>result then result := ab;
    end;
  end else begin
    for i:=1 to N-1 do
      ab := abs(src[i*stride]);
      if ab>result then result := ab;
  end;
end;

function QNNMinAbs(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i:integer;
  ab:QNNFloat;
begin
  assert(assigned(src), 'ERROR QNNMin: [src] is empty!');
  result := abs(src[0]);
  if stride=1 then begin
    for i:=1 to N-1 do begin
      ab := abs(src[i]);
      if ab<result then result := ab;
    end;
  end else begin
    for i:=1 to N-1 do
      ab := abs(src[i*stride]);
      if ab<result then result := ab;
  end;
end;

function QNNArgMax(const N: integer; const src: PQNNFloat; const stride: integer): integer;
var
  i:integer;
  ma : QNNFloat;
begin
  assert(assigned(src), 'ERROR QNNArgMax: [src] is empty!');
  ma := src[0];
  result := 0;
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]>ma then begin ma := src[i]; result := i end;
  end else
    for i:=1 to N-1 do
      if src[i*stride]>ma then begin ma := src[i*stride]; result := i end;

end;

function QNNArgMin(const N: integer; const src: PQNNFloat; const stride: integer): integer;
var i:integer;
    mi : QNNFloat;
begin
  assert(assigned(src), 'ERROR QNNArgMin: [src] is empty!');
  mi := src[0];
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]<mi then begin mi := src[i]; result := i end;
  end else
    for i:=1 to N-1 do
      if src[i*stride]<mi then begin mi := src[i*stride]; result := i end;
end;

function QNNSum(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify mean
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + src[i]
  else
    for i:=0 to N-1 do
      result := result + src[i*stride];
end;

function QNNSumSqr(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify SumSqr
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + sqr(src[i])
  else
    for i:=0 to N-1 do
      result := result + sqr(src[i*stride]);
end;

function QNNSumSqrDiff(const N: integer; const src: PQNNFloat; const aMean: QNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify mean
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + sqr(src[i]-aMean)
  else
    for i:=0 to N-1 do
      result := result + sqr(src[i*stride]-aMean);
end;

function QNNSqrDistance(const N: integer; const src1, src2: PQNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify mean
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + sqr(src1[i]-src2[i])
  else
    for i:=0 to N-1 do
      result := result + sqr(src1[i*stride]-src2[i*stride]);
end;

function QNNMaxAbsDiff(const N: integer; const src1, src2: PQNNFloat; out outSrc1, outSrc2:QNNFloat; const stride: integer): QNNFloat;
var i: integer;
    ab2
    , ab
    :double;
begin
  // todo simdify mean
  outSrc1 := 0;
  outSrc2 := 0;

  result := 0;
  if stride=1 then
    for i:=0 to N-1 do begin
      ab2 := abs(src1[i]-src2[i]);
      //if src1[i] <> 0 then
      //  ab := abs(ab2/src1[i])
      //else
      //  ab:=ab2;
      //assert((ab<0.01) or (ab2< 0.01), 'at pos '+ intToStr(i));
      if ab2>result then begin
        outSrc1 := src1[i];
        outSrc2 := src2[i];
        result := ab2;
      end
    end
  else
    for i:=0 to N-1 do
      result := math.max(result, abs(src1[i*stride]-src2[i*stride]));
end;

function QNNMaxAbsDiff(const N: integer; const src1, src2: PQNNFloat;
  const stride: integer): QNNFloat;
var i: integer;
    ab2
    , ab
    :double;
begin
  // todo simdify mean
  assert(assigned(src1) and assigned(src2), 'ERROR : src1 and src2must be assigned');
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do begin
      ab2 := abs(src1[i]-src2[i]);
      //  ab := abs(ab2/src1[i])
      //else
      //  ab:=ab2;
      //assert((ab<0.01) or (ab2< 0.01), 'at pos '+ intToStr(i));
      if ab2>result then begin
        result := ab2;
      end
    end
  else
    for i:=0 to N-1 do
      result := math.max(result, abs(src1[i*stride]-src2[i*stride]));
end;

function QNNSumAbsDiff(const N: integer; const src1, src2: PQNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify mean
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + abs(src1[i]-src2[i])
  else
    for i:=0 to N-1 do
      result := result + abs(src1[i*stride]-src2[i*stride]);
end;

function QNNSumAbsDiff(const N: integer; const src: PQNNFloat;
  const aMean: QNNFloat; const stride: integer): QNNFloat;
var i: integer;
begin
  // todo simdify mean
  result := 0;
  if stride=1 then
    for i:=0 to N-1 do
      result := result + abs(src[i]-aMean)
  else
    for i:=0 to N-1 do
      result := result + abs(src[i*stride]-aMean);
end;

function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer):QNNFloat;
begin
  // todo simdify mean
  if n=0 then exit(0);
  result := QNNSum(N, src, stride);
  result := result / N
end;

function QNNVariance(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer; const isPopulation:boolean):QNNFloat;
begin
  if ((N=1) and not isPopulation) or (N=0) then exit(0);
  result := QNNSumSqrDiff(N, src, aMean, stride);
  if isPopulation then
    result := result / N
  else
    result := result / (N-1)
end;

procedure QNNAdd(const dst, a, b: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo simdify add
  for i:=0 to N-1 do dst[i] := a[i]+b[i]
end;

procedure QNNAddInplace(const dst, a: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo SIMDIFY ADD_inplace
  for i:=0 to N-1 do dst[i] := dst[i] + a[i]
end;

procedure QNNMul(const dst, a, b: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo simdify add
  for i:=0 to N-1 do dst[i] := a[i]*b[i]
end;

procedure QNNMulInplace(const dst, a: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo SIMDIFY ADD_inplace
  for i:=0 to N-1 do dst[i] := a[i]*dst[i]
end;

procedure QNNScale(const dst, src: PQNNFloat; const aScale: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNScale
  if dst=src then
    cblas_scal(N, aScale, dst, 1)
  else for i:=0 to N-1 do
      dst[i] := src[i]*aScale
end;

procedure QNNScaleInplace(const dst: PQNNFloat; const aScale: QNNFloat;
  const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNScale
  for i:=0 to N-1 do
      dst[i] := dst[i]*aScale
end;

procedure QNNBias(const dst, src: PQNNFloat; const aBias: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNBias
    for i:=0 to N-1 do
      dst[i] := src[i]+aBias;
end;

procedure QNNBiasInplace(const dst: PQNNFloat; const aBias: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNBias
    for i:=0 to N-1 do
      dst[i] := dst[i]+aBias;
end;

procedure QNNFusedMulAdd(const dst, src, srcM, srcA: PQNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedMulAdd
    for i:=0 to N-1 do
      dst[i] := src[i]*srcM[i]+srcA[i]
end;

procedure QNNFusedScaleBias(const dst, src: PQNNFloat; const aScale, aBias: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedScaleBias (Scalars)
    for i:=0 to N-1 do
      dst[i] := src[i]*aScale + aBias

end;

procedure QNNFusedAddMul(const dst, src, srcA, srcM: PQNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedAddMul
    for i:=0 to N-1 do
      dst[i] := (src[i]+srcA[i])*srcM[i]
end;

procedure QNNFusedBiasScale(const dst, src: PQNNFloat; const aBias, aScale: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedScaleBias (Scalars)
  case (2*longint(aBias=0.0)) or longint(aScale=1.0) of
    1 : QNNBias(dst, src, aBias, N);
    2 : QNNScale(dst, src, aScale, N);
    3 : QNNCopy(dst, src, N);
    else
      for i:=0 to N-1 do
        dst[i] := (src[i] + aBias)*aScale
  end;
end;

procedure QNNFusedMulScale(const dst, src, srcM: PQNNFloat; const aScale: QNNFloat; const N: longint);
var i:integer;
begin
    // todo priority SIMDIFY QNNFusedMulScale (Scalars)
  if aScale=1.0 then
    QNNMul(dst, src, srcM, N)
  else
    for i:=0 to N-1 do
      dst[i] := src[i]*srcM[i]*aScale
end;

procedure QNNFusedBiasMulScale(const dst, src, srcM: PQNNFloat; const aBias, aScale: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedBiasMulScale (Scalars)
    for i:=0 to N-1 do
      dst[i] := (src[i] + aBias)*srcM[i]*aScale
end;

procedure QNNFusedScaleAdd(const dst, src, srcAdd: PQNNFloat; const aScale: QNNFloat; const N: longint);
var i:longint;
begin
  if srcAdd=dst then
    cblas_axpy(N, aScale, src, 1, dst, 1)
  else
    for i:=0 to N-1 do
      dst[i] := aScale*src[i] + srcAdd[i]
end;

procedure QNNFusedBiasAdd(const dst, src, srcA: PQNNFloat;
  const aBias: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedBiasAdd (Scalars)
    for i:=0 to N-1 do
      dst[i] := aBias + src[i] + srcA[i]
end;

function QNNDot(const a, b: PQNNFloat; const N: longint): QNNFloat;
begin
  result := cblas_dot(N, a, 1, b, 1);
end;


procedure QNNGatedAdd(const dst, gate, proj:psingle; const seq, hidden:integer);
var i:integer;
begin
  // todo gated add [priority SIMDIFY]
  for i:=0 to seq-1 do
    QNNAccMul(dst + i*hidden, gate, proj + i*hidden, hidden);
end;

//RoPE (Rotary Position Embeddings)
procedure QNNComputeRoPE(const dst: PQNNFloat; const maxSeq, dim: longint; const theta: QNNFloat);
var
  pos, i, halfdim, d:longint;
  freq, angle:QNNFloat;
begin
  halfdim := dim div 2;
  for pos := 0 to maxSeq-1 do begin
      d := pos*halfdim*2;
      for i :=0 to halfdim-1 do begin
          freq  := 1.0 / power(theta, (2*i)/dim);
          angle := pos*freq;
          inc(d, 2);
          dst[d]     := cos(angle);
          dst[d + 1] := Sin(angle);
      end;
  end;
end;

procedure QNNComputeRoPE(const cosOut, sinOut: PQNNFloat; const maxSeqLen,
  headDim: longint; const theta: QNNFloat);
var
  half_dim, pos, i:longint;
  freq, angle : QNNFloat;
begin
  half_dim := headDim div 2;

  for pos := 0 to maxSeqLen-1 do begin
      for i := 0 to half_dim-1 do begin
          freq  := 1.0 / power(theta, (2*i) / headDim);
          angle := pos * freq;
          cosOut[pos * half_dim + i] := cos(angle);
          sinOut[pos * half_dim + i] := sin(angle);
      end
  end;
end;

procedure QNNComputeRoPE2D(const cosDst, sinDst: PQNNFloat; const patch_h,
  patch_w, dim: longint; const theta: QNNFloat);
var
  d, hy, wx, pos, halfdim:longint;
  cos_p, sin_p : PQNNFloat;
  angle_h, angle_w, cos_h, cos_w, sin_h, sin_w: QNNFloat;
  freqs: array[0..15] of QNNFloat;
begin
  halfdim := dim div 2;  (* 16 dims per half-axis *)
  //int seq = patch_h * patch_w;
  //(void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for d:= 0 to halfdim-1 do
    freqs[d] := 1 / power(theta, (2*d) / dim);

  for hy := 0 to patch_h-1 do begin
    for wx := 0 to patch_w-1 do begin
      pos := hy * patch_w + wx;
      cos_p := cosDst + pos*dim*4;  (* 4 axes * 32 dims each = 128 *)
      sin_p := sinDst + pos*dim*4;

      (* Axis 0 (dims 0-31): T position = 0, so cos=1, sin=0 *)
      for d := 0 to dim-1 do begin
        cos_p[d] := 1.0;
        sin_p[d] := 0.0;
      end;

      (* Axis 1 (dims 32-63): H position (y/height)
       * Python RoPE stacks [cos, -sin, sin, cos] as 2x2 matrix per freq.
       * For apply_rope: out = [[cos, -sin], [sin, cos]] @ [x0, x1]
       * We store cos/sin per pair and apply_rope_2d handles the rotation.
       *)
      for d := 0 to halfdim-1 do begin
          angle_h := hy * freqs[d];
          cos_h := cos(angle_h);
          sin_h := sin(angle_h);
          (* Each frequency contributes to a pair of dimensions *)
          pos := dim + d * 2;
          cos_p[pos] := cos_h;
          cos_p[pos + 1] := cos_h;
          sin_p[pos] := sin_h;
          sin_p[pos + 1] := sin_h;
      end;

      (* Axis 2 (dims 64-95): W position (x/width) *)
      for d := 0 to halfdim-1 do begin
          angle_w := wx * freqs[d];
          cos_w := cos(angle_w);
          sin_w := sin(angle_w);
          pos := dim*2 + d*2;
          cos_p[pos] := cos_w;
          cos_p[pos + 1] := cos_w;
          sin_p[pos] := sin_w;
          sin_p[pos + 1] := sin_w;
      end;

      (* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 *)
      for d := 0 to dim -1  do begin
          pos := dim*3 + d;
          cos_p[pos] := 1.0;
          sin_p[pos] := 0.0;
      end
    end
  end

end;

procedure QNNComputeRoPE2DOffset(const cosDst, sinDst: PQNNFloat;
  const patch_h, patch_w, dim: longint; const theta: QNNFloat;
  const offset_t: longint);
var
  d, hy, wx,pos, halfdim:longint;
  cos_p, sin_p : PQNNFloat;
  angle_h, angle_w, angle_t, cos_h, cos_w, cos_t, sin_h, sin_w, sin_t: QNNFloat;
  freqs: array[0..15] of QNNFloat;
begin
  halfdim := dim div 2;  (* 16 dims per half-axis *)
  //int seq = patch_h * patch_w;
  //(void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for d:= 0 to halfdim-1 do
    freqs[d] := 1 / power(theta, (2 * d) / dim);

  for hy := 0 to patch_h-1 do begin
    for wx := 0 to patch_w-1 do begin
      pos := hy * patch_w + wx;
      cos_p := cosDst + pos*dim*4;  (* 4 axes * 32 dims each = 128 *)
      sin_p := sinDst + pos*dim*4;


      (* Axis 0 (dims 0-31): T position = t_offset (non-zero for refs) *)
      for d := 0 to halfdim-1 do begin
          angle_t := offset_t * freqs[d];
          cos_t := cos(angle_t);
          sin_t := sin(angle_t);
          cos_p[d * 2] := cos_t;
          cos_p[d * 2 + 1] := cos_t;
          sin_p[d * 2] := sin_t;
          sin_p[d * 2 + 1] := sin_t;
      end;

      for d := 0 to halfdim-1 do begin
          angle_h := hy * freqs[d];
          cos_h := cos(angle_h);
          sin_h := sin(angle_h);
          (* Each frequency contributes to a pair of dimensions *)
          cos_p[dim + d * 2] := cos_h;
          cos_p[dim + d * 2 + 1] := cos_h;
          sin_p[dim + d * 2] := sin_h;
          sin_p[dim + d * 2 + 1] := sin_h;
      end;

      (* Axis 2 (dims 64-95): W position (x/width) *)
      for d := 0 to halfdim-1 do begin
          angle_w := wx * freqs[d];
          cos_w := cos(angle_w);
          sin_w := sin(angle_w);
          cos_p[dim * 2 + d * 2] := cos_w;
          cos_p[dim * 2 + d * 2 + 1] := cos_w;
          sin_p[dim * 2 + d * 2] := sin_w;
          sin_p[dim * 2 + d * 2 + 1] := sin_w;
      end;

      (* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 *)
      for d := 0 to dim -1  do begin
          cos_p[dim * 3 + d] := 1.0;
          sin_p[dim * 3 + d] := 0.0;
      end
    end
  end
end;

procedure QNNComputeRoPEText(const cosDst, sinDst: PQNNFloat; const txt_seq, axis_dim: longint; const theta: QNNFloat);
var
  half_axis, head_dim, d, s, pos: longint;
  cos_p: PQNNFloat;
  sin_p: PQNNFloat;
  angle, cos_l, sin_l: QNNFloat;
  freqs : array[0..15]of QNNFloat;
begin
  half_axis := axis_dim div 2;
  head_dim := axis_dim * 4;
  for d := 0 to half_axis -1 do
    freqs[d] := 1.0 / power(theta, (2 * d) / axis_dim);
  for s := 0 to txt_seq -1 do
    begin
      cos_p := cosDst+s * head_dim;
      sin_p := sinDst+s * head_dim;
      for d := 0 to axis_dim * 3 -1 do
        begin
            cos_p[d] := 1.0;
            sin_p[d] := 0.0
        end;
      for d := 0 to half_axis -1 do
        begin
          angle := s * freqs[d];
          cos_l := cos(angle);
          sin_l := sin(angle);
          pos := axis_dim*3 + d*2;
          cos_p[pos]     := cos_l;
          cos_p[pos + 1] := cos_l;
          sin_p[pos]     := sin_l;
          sin_p[pos + 1] := sin_l
        end
    end
end;

procedure qscale(N:blasint; alpha:single; X:PQNNFloat; incX:blasint);  WINAPI;
var i:blasint;
begin
  // todo qscale Simdify
  if incX=1 then
    for i:=0 to N-1 do
      X[i]:= X[i]*alpha
  else
    for i:=0 to N-1 do
      X[i*incX] := X[i*incX]* alpha
end;

{$if defined(CPUX64)}
const
  SIMD_REGS = 8;
  {$ifdef FPC}
  SIMD_SHFT  = BsfQWord(SIMD_REGS);
  {$else}
  {$if SIMD_REGS = 8}SIMD_SHFT = 3{$else} SIMD_SHFT = 2{$endif};
  {$endif}
  SIMD_OFF = SIMD_REGS * sizeof(single);
  ymmd : array[0..7] of int32 = (0, 1, 2, 3, 4, 5, 6, 7);

function sdot_avx2(const N:longint; const A,B:PSingle):single;assembler;{$ifdef FPC}nostackframe;{$endif}
asm
{$ifndef FPC}
  .NOFRAME
{$endif}
   mov              r11d     ,    N
   vpxor            ymm0    ,    ymm0   ,   ymm0
   shr              r11d     ,    SIMD_SHFT
   jz               @rem
@while:
   vmovups          ymm1    ,    yword [A]
   vfmadd231ps      ymm0    ,    ymm1   , yword [B]
   add              A       ,    SIMD_OFF
   add              B       ,    SIMD_OFF
   dec              r11d
   jnz              @while

@rem:
   mov              r11d     ,    N
   and              r11d     ,    SIMD_REGS -1
   jz               @done
   vmovd            xmm3    ,    r11d
   vpxor            ymm1    ,    ymm1    , ymm1
   vpxor            ymm2    ,    ymm2    , ymm2
   vpbroadcastd     ymm3    ,    xmm3
   vpcmpgtd         ymm3    ,    ymm3    , [rip+ymmd]
   vmaskmovps       ymm1    ,    ymm3    , [A]
   vmaskmovps       ymm2    ,    ymm3    , [B]
   vfmadd231ps      ymm0    ,    ymm1    , ymm2

@done:
   vextractf128     xmm1    ,    ymm0   ,   $1
   vzeroupper
   vaddps           xmm0    ,    xmm0   ,   xmm1
   vhaddps          xmm0    ,    xmm0   ,   xmm0
   vhaddps          xmm0    ,    xmm0   ,   xmm0

end;

procedure saxpy_avx2(const N:longint; const a:single; const x,y:PSingle);assembler;{$ifdef FPC}nostackframe;{$endif}
asm
  //push         r11
  //push         N
  {$ifndef FPC}
  .NOFRAME
  {$endif}

//  movss         xmm2   , a
  vbroadcastss ymm1   , a
  mov          r11d    , N
  shr          r11d    , (SIMD_SHFT + 2)    // div by 16 (4*4) = turns * SIMD_REGS
  jz           @rem1

@while:
  vmovups      ymm0   , yword [y]
  vmovups      ymm2   , yword [y+SIMD_OFF]
  vmovups      ymm3   , yword [y+SIMD_OFF*2]
  vmovups      ymm4   , yword [y+SIMD_OFF*3]

  vfmadd231ps  ymm0   , ymm1       , yword [x]                 //xmm0
  vfmadd231ps  ymm2   , ymm1       , yword [x+SIMD_OFF]  //xmm2
  vfmadd231ps  ymm3   , ymm1       , yword [x+SIMD_OFF*2]//xmm8
  vfmadd231ps  ymm4   , ymm1       , yword [x+SIMD_OFF*3]//xmm3

  vmovups      yword [y]             , ymm0
  vmovups      yword [y+SIMD_OFF]    , ymm2
  vmovups      yword [y+SIMD_OFF*2]  , ymm3
  vmovups      yword [y+SIMD_OFF*3]  , ymm4

  add          x      , 4 * SIMD_OFF   // turns * offset
  add          y      , 4 * SIMD_OFF
  dec          r11d
  jnz          @while

@rem1:
  mov          r11d    , N
  and          r11d    , (4*SIMD_REGS-1)       // mod 32  ( turns * SIMD_REGS)
  shr          r11d    , SIMD_SHFT             // div SIMD_REGS
  jz           @rem

@while1:
  vmovups      ymm0   , [y]

  vfmadd231ps  ymm0   , ymm1       , [x]
  vmovups      [y]    , ymm0
  add          x      , SIMD_OFF
  add          y      , SIMD_OFF
  dec          r11d
  jnz          @while1

@rem:
  mov          r11d    , N
  and          r11d    , (SIMD_REGS -1)       // mod SIMD_REGS
  jz           @done

@while2:
  vmovss       xmm0   , dword [y]
  vfmadd231ss  xmm0   , xmm1, [x]
  vmovss       dword [y]    , xmm0
  add          x      , 4
  add          y      , 4
  dec          r11d
  jnz          @while2

@done:
  //pop          r11
  //vzeroupper
end;
{$endif}

procedure qaxpy(n:blasint; alpha:single; x:PQNNFloat; y:PQNNFloat); overload;
var
  i:blasint;
begin
  // todo qaxpy Simdify
    {$if defined(CPUX64)}
    saxpy_avx2(N, alpha, x, y)
    {$else}
    for i:=0 to N-1 do
      y[i]:= alpha*x[i] + y[i]
    {$endif}
end;

procedure qaxpy(n:blasint; alpha:single; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint); winapi ; overload;
var
  i:blasint;
begin
  // todo qaxpy Simdify
  if (incx=1) and (incy=1) then
    {$if defined(CPUX64)}
    saxpy_avx2(N, alpha, x, y)
    {$else}
    for i:=0 to N-1 do
      y[i]:= alpha*x[i] + y[i]
    {$endif}
  else
    for i:=0 to N-1 do
      y[i*incy]:= alpha*x[i*incx] + y[i*incy]
end;

function qdot(N:blasint; x:PQNNFloat; y:PQNNFloat):single; overload;
var
  i:blasint;
begin
  // todo qdot Simdify

    {$if defined(CPUX64)}
    result := sdot_avx2(N, x, y)
    {$else}
    result := 0;
    for i:=0 to N-1 do
      result := result + x[i]*y[i]
    {$endif}
end;

function qdot(n:blasint; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint):single; winapi;overload;
var
  i:blasint;
begin
  // todo qdot Simdify
  if (incx=1) and (incy=1) then begin
    {$if defined(CPUX64)}
    result := sdot_avx2(N, x, y)
    {$else}
    result := 0;
    for i:=0 to N-1 do
      result := result + x[i]*y[i]
    {$endif}
  end
  else begin
    result := 0;
    for i:=0 to N-1 do
      result := result + x[i*incx]*y[i*incy]
  end;
end;


type prn_t = procedure(const str:pansichar{const m, n, k:longint; Alpha:single; lda, ldb:longint; Beta:single; ldc:longint});cdecl varargs;


{$ifdef USE_CPP_GEMM}
{$link sgemm2.o}
procedure set_prn(const p:prn_t);winapi;external;
procedure cpp_sgemm(TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; const M, N, K:blasint; const alpha:single; const A:PQNNFloat; const lda:blasint; const B:PQNNFloat; const ldb:blasint; const beta:single; C:PQNNFloat; const ldc:blasint; const ithread:longint = 0; const nthreads:longint = 0); winapi; external;
procedure cblas_sgemm(layout: CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; const M, N, K:blasint; const alpha:single; const A:PQNNFloat; const lda:blasint; const B:PQNNFloat; const ldb:blasint; const beta:single; C:PQNNFloat; const ldc:blasint); winapi; external;

procedure printf(const str:pansichar);cdecl varargs; external 'msvcrt.dll';

//procedure printf(const str:pansichar; const args:array of const);winapi; external 'msvcrt.dll'; overload;

//procedure prn(str:pansichar; const a:array of const{const m, n, k:longint; Alpha:single; lda, ldb:longint; Beta:single; ldc:longint});winapi;
//begin
//  printf(str,  @a[0] {m, n, k, Alpha, lda, ldb, Beta, ldc});
//end;
{$endif}



procedure qgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:PQNNFloat; ldc:blasint); WINAPI;

{$ifdef FPC}
procedure gemm_thread(const start, finish: IntPtr; const data:pointer);
{$else}
var gemm_thread:TGroupProcNested;
begin
  gemm_thread := procedure(const start, finish: IntPtr; const data:pointer)
{$endif}
var
  //start, finish,
    mm, nn ,kk : blasint;
  a_part : single;
  AA, BB, CC : PQNNFloat;
  idx : IntPtr;
begin
  {$ifdef USE_CPP_GEMM}
  cpp_sgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, start, finish-start+1);
  {$else}
  //writeln('gemm m:',start,' to:', finish);
  if (transA=CblasNoTrans) and (transB=CblasNoTrans)then begin
    for idx :=start to finish do begin
      CC := C + idx*ldc;
      AA := A + idx*lda;
      for kk:= 0 to K-1 do begin
        qaxpy(N, alpha*AA[kk], B+kk*ldb, CC);
        //for nn:=0 to N-1 do
        //  CC[nn] := CC[nn] + a_part*B[kk*ldb + nn];
      end;
    end;
    exit;
  end;
  if (transA=CblasNoTrans) and (transB=CblasTrans)then begin
    for idx:=start to finish do begin
      CC := C + idx*ldc;
      AA := A + idx*lda;
      for nn := 0 to N-1 do begin
          BB := B + nn*ldb;
          a_part := qdot(K, AA, BB);
          //a_part := 0.0;
          //for kk:=0 to K-1 do begin
          //    a_part := a_part + AA[kk]*BB[kk];
          //end;
          CC[nn] := CC[nn] + ALPHA*a_part;
      end;
    end;
    exit
  end;
  if (transA=CblasTrans) and (transB=CblasNoTrans)then begin
    for idx :=start to finish do begin
      CC := C + idx*ldc;
      for kk:=0 to K-1 do begin
        a_part := ALPHA*A[kk*lda + idx];
        BB := B + kk*ldb;
        qaxpy(N, a_part, BB, CC);
        //for nn:=0 to N-1 do begin
        //  CC[nn] := CC[nn] + a_part*BB[nn]
        //end;
      end;
    end;
    exit;
  end;
  {$endif}
end;
var i: blasint;
{$ifdef FPC}
begin
{$endif}
  // todo qgemm Simdify
  assert((order=CblasRowMajor) and not((TransA=CblasTrans) and (TransB=CblasTrans)),'ERROR : Operation is not supported using the provided arguments!');
  {$ifdef USE_CPP_GEMM}
  cblas_sgemm(order, transA, transB, M, N,K, alpha, A, lda, B, ldb, beta, C, ldc);
  exit;
  {$endif}

  //start :=0 ;
  //finish := M-1;
    if BETA=0 then
      FillChar(C^, (M*N*sizeOf(QNNFloat)), 0)
    else if BETA<>1 then
      cblas_scal(M*N, BETA, C, 1);
  {$if defined(USE_MULTITHREADING)}
    mp.&For(gemm_thread, 0, M-1);
  {$else}
  gemm_thread(0, M-1, nil);
  //for i:=0 to M-1 do gemm_thread(i, nil)
  {$endif}
end;

function qasum(n:blasint; x:PQNNFloat; incx:blasint):QNNFloat; winapi ;
var
  i:longint;
begin
  result := 0;
  if incx=1 then
    for i:=0 to n-1 do
      result := result + abs(x[i])
  else
    for i:=0 to n-1 do
      result := result + abs(x[i*incx])
end;

procedure QNNAccAdd(const dst, a, b: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo simdify accadd
  for i:=0 to N-1 do dst[i] := dst[i] + a[i] + b[i]
end;

procedure QNNAccMul(const dst, a, b: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo simdify accmul
  for i:=0 to N-1 do
    dst[i] := dst[i] + a[i]*b[i]
end;

procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  assert((A.count>=M*K) or (assigned(A.DataPtr) and (A.count=0)), 'ERROR : QNNMatMul out of bound [A]');
  assert((B.count>=K*N) or (assigned(B.dataPtr) and (B.count=0)), 'ERROR : QNNMatMul out of bound [B]');
  assert((C.count>=M*N) or (assigned(C.dataPtr) and (C.count=0)), 'ERROR : QNNMatMul out of bound [C]');
    cblas_gemm(CBLASRowMajor, CBLASNoTrans, CBLASNoTrans
    , M, N, K, 1
    , A, K
    , B, N
    , 0
    , C, N);
end;

procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  assert((A.count>=M*K) or (assigned(A.DataPtr) and (A.count=0)), 'ERROR : QNNMatMul out of bound [A]');
  assert((B.count>=N*K) or (assigned(B.dataPtr) and (B.count=0)), 'ERROR : QNNMatMul out of bound [B]');
  assert((C.count>=M*N) or (assigned(C.dataPtr) and (C.count=0)), 'ERROR : QNNMatMul out of bound [C]');
    cblas_gemm(CBLASRowMajor, CblasNoTrans, CBLASTrans
    , M, N, K, 1
    , A, K
    , B, K
    , 0
    , C, N);
end;

procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  assert((A.count>=K*M) or (assigned(A.DataPtr) and (A.count=0)), 'ERROR : QNNMatMul out of bound [A]');
  assert((B.count>=N*K) or (assigned(B.dataPtr) and (B.count=0)), 'ERROR : QNNMatMul out of bound [B]');
  assert((C.count>=M*N) or (assigned(C.dataPtr) and (C.count=0)), 'ERROR : QNNMatMul out of bound [C]');
    cblas_gemm(CBLASRowMajor, CblasTrans, CBLASNoTrans
    , M, N, K, 1
    , A, K
    , B, K
    , 0
    , C, N);
end;

procedure QNNMatTranspose(const dst, src: PQNNFloat; const srcRows, srcCols: longint);
var c, r:longint;
begin
  for c:=0 to srcCols-1 do
    for r:=0 to srcRows-1 do
      dst[c*srcRows + r] := src[r*srcCols + c]
end;

procedure QNNLinear(const dst, x, W, b: PQNNFloat; const seqLen, inDIM, outDIM: integer);
var i: integer;
begin
  cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              seqLen, outDIM, inDIM,
              1.0, x, inDIM, W, inDIM,
              0.0, dst, outDIM);

  (* Add bias if present *)
  if assigned(b) then
    for i := 0 to seqLen-1 do
       QNNAddInplace(dst + i*outDIM, B, outDIM)
end;

procedure QNNLinearNoBias(const dst, x, W: PQNNFloat; const seqLen, inDIM, outDIM: integer);
begin
  cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              seqLen, outDIM, inDIM,
              1.0, x, inDIM, W, inDIM,
              0.0, dst, outDIM);
end;

procedure QNNLinearNoBias_BF16(const dst, x: PQNNFloat; const W: PBF16; const seqLen, inDIM, outDIM: integer);
begin
  //if assigned(cblas_sbgemm) then
  //  cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
  //            seqLen, outDIM, inDIM,
  //            1.0,
  //            x, inDIM,
  //            W, inDIM,
  //            0.0,
  //            dst, outDIM)
  //else
  begin
    if length(workspace) < inDIM*outDIM then
      setLength(workspace, inDIM*outDIM);
    BF16ToSingle(inDIM*outDIM, W, pointer(workspace));
    QNNLinearNoBias(dst, x, pointer(workspace), seqLen, inDIM, outDIM);
  end
end;

procedure QNNIm2Col(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: integer; const im: PQNNFloat; const col: PQNNFloat;
  const imOffset: integer; const colOffset: integer; const MultiThread: boolean
  );
var
  channel, output_h, output_w, channel_size, out_channel_size, kernel_size: integer;
  {$ifdef FPC}
  procedure i2c_ext(idx:IntPtr; ptr:Pointer);
  {$else}
  i2c_ext: TThreadProcNested;
begin
  //{$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2colExt);{$endif}
  i2c_ext := procedure(idx: IntPtr; ptr: Pointer)
  {$endif}
  var
    kernel_row, kernel_col, output_col, output_row, input_row, input_col, i: NativeInt;
    d_im, d_col: PQNNFloat;
  begin
    d_im := im + imOffset + channel_size * idx;
    d_col := col + colOffset + kernel_size * out_channel_size * idx;
    for kernel_row := 0 to kernelHeight - 1 do
      for kernel_col := 0 to kernelWidth - 1 do
        begin
          //fillChar(d_col^, output_w*output_h*sizeOf(single), #0);
          input_row := -padHeight + kernel_row * dilationY;
          for output_row := 0 to output_h - 1 do  begin
            if (NativeUInt(input_row) < NativeUInt(aHeight)) then begin
              input_col := -padWidth + kernel_col * dilationX;
              if (strideX=1) then begin
                //fillchar(d_col[0], output_w*SizeOf(d_col^), #0);
                move(d_im[input_row*aWidth + ord(input_col>=0)*input_col], d_col[ord(input_col<0)*abs(input_col)], (output_w-abs(input_col))*sizeof(d_col^));
              end
              else
                for output_col:=0 to output_w-1 do  begin
                  i := output_col*strideX + input_col;
                  d_col[output_col] := ord(NativeUInt(i) < NativeUInt(aWidth))*d_im[input_row*aWidth + i];
                end; // or use the following (UnOptimized)
            end
            else
              //fillchar(d_col^, output_w*SizeOf(d_col^), #0)
              ;
            Inc(d_col, output_w);
            Inc(input_row, strideY);
          end;
        end;
  end;

  {$ifdef FPC}
begin
  {$endif}
  output_w := (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) div strideX + 1;
  output_h := (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) div strideY + 1;
  channel_size := aHeight * aWidth;
  out_channel_size := output_w * output_h;
  kernel_size := kernelWidth * kernelHeight;
  {$ifdef USE_MULTITHREADING}
  if MultiThread then
    mp.&for(i2c_ext, 0, aChannels{, @p})
  else
  for channel:=0 to aChannels-1 do
      i2c_ext(channel,{@p}nil);
  {$else}
  for channel := 0 to aChannels - 1 do
    i2c_ext(channel,{@p}nil);
  {$endif}
  //{$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opIm2colExt);{$endif}
end;

procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth,
  kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: integer; const im: PQNNFloat; const imStride, imOffset: integer;
  const col: PQNNFloat; const colStride, colOffset: integer;
  const batch: integer);
var b : NativeInt; mt:boolean;

{$ifdef FPC}
  procedure i2c(idx:IntPtr; ptr:Pointer);
{$else}
  i2c : TThreadProcNested;
begin
  i2c := procedure (idx:IntPtr; ptr:Pointer)
{$endif}
  begin
    QNNIm2Col(
      aChannels, aHeight, aWidth,
      kernelHeight, kernelWidth, padHeight, padWidth,
      strideY, strideX, dilationY, dilationX, im+idx*imStride, col+idx*colStride, imOffset, colOffset, mt);
  end;
{$ifdef FPC}
begin
{$endif}
  mt := batch=1;
  {$ifdef USE_MULTITHREADING}
  if not mt then
    mp.&For(i2c, 0, batch, nil)
  else
  {$endif}
  for b:=0 to batch-1 do
    QNNIm2Col(
      aChannels, aHeight, aWidth,
      kernelHeight, kernelWidth, padHeight, padWidth,
      strideY, strideX, dilationY, dilationX,
      im + b*imStride,
      col + b*colStride, imOffset, colOffset,
      mt);
end;

{$define USE_IM2Col}
procedure QNNConv2d(const dst, src, weights, bias: PQNNFloat; const in_ch,
  out_ch, H, W, kH, kW, stride, padding: longint; const batch: longint);
const MAX_COL_SIZE = 256 * 1024 * 1024;
var
  outW, outH, kSize, imColSize, filters, outImgSize
  , k, inVolume, b, c, strideA, strideB, strideC: integer;
  workspacePtr : PQNNFloat;
{$if defined(USE_NAIVE)}
procedure kernel(idx:IntPtr; data:pointer);
var
  oc, oh, ow, ic, kernel_row, kernel_col, ih, iw, in_idx, w_idx, out_idx:longint;
  sum : QNNFloat;
  SS, WW, DD : PQNNFloat;
begin
  SS := src; WW := weights; DD := dst;
  oc := idx div out_ch;
  oh := idx mod outH;
  for ow := 0 to outW-1 do begin
      if assigned(bias) then sum := bias[oc] else sum := 0.0;
      for ic := 0 to in_ch-1 do
          for kernel_row := 0 to kH-1 do
              for kernel_col := 0 to KW-1 do begin
                  ih := oh * stride - padding + kernel_row;
                  iw := ow * stride - padding + kernel_col;

                  if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W) then begin
                      in_idx := b * in_ch * H * W + ic * H * W + ih * W + iw;
                      w_idx  := oc * in_ch * kH * kW + ic * kH * kW + kernel_row * kW + kernel_col;
                      sum := sum + SS[in_idx] * WW[w_idx];
                  end;
              end;

      out_idx := b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
      DD[out_idx] := sum;
  end;
end;

var
  i:longint;
{$else}
    col_size, {max_col_size, }tile_col_size, row_size: IntPtr;
    col, col2: TMemoryBlock;
    col_ptr, in_b, out_b, out_tile, out_ch_ptr: PQNNFloat;
    tile_rows, tile_start, tile_end, tile_h, tile_pixels,
      col_row, ic, kernel_row, kernel_col, oh, ow, ih, iw, {col_idx, }oc, i, outSpatial,
      input_col: longint;
    b_val: single;
{$endif}
begin
  assert(assigned(dst) and assigned(src) and assigned(weights), 'ERROR : dst, src and weights must be assigned');

{$if defined(USE_NAIVE)}
  outH := (H + 2 * padding - kH) div stride + 1;
  outW := (W + 2 * padding - kW) div stride + 1;
  // Naive implementation with threadings (fallback)
  for b := 0 to batch-1 do
    {$ifdef USE_MULTITHREADING}
    mp.&For(kernel, 0,  out_ch*outH);
    {$else}
    for i:= 0 to out_ch*outH-1 do
      kernel(i, nil);
    {$endif}
      //for oc := 0 to out_ch-1 do
      //    for oh := 0 to outH-1 do
              //for ow := 0 to outW-1 do begin
              //    if assigned(bias) then sum := bias[oc] else sum := 0.0;
              //    for ic := 0 to in_ch-1 do
              //        for kernel_row := 0 to kH-1 do
              //            for kernel_col := 0 to KW-1 do begin
              //                ih := oh * stride - padding + kernel_row;
              //                iw := ow * stride - padding + kernel_col;
              //
              //                if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W) then begin
              //                    in_idx := b * in_ch * H * W + ic * H * W + ih * W + iw;
              //                    w_idx  := oc * in_ch * kH * kW + ic * kH * kW + kernel_row * kW + kernel_col;
              //                    sum := sum + src[in_idx] * weights[w_idx];
              //                end;
              //            end;
              //
              //    out_idx := b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
              //    dst[out_idx] := sum;
              //end;
  exit;
{$else}
    outH := (H + 2 * padding - kH) div stride + 1;
    outW := (W + 2 * padding - kW) div stride + 1;
    outSpatial := outH*outW;
    col_size := in_ch * kH * kW * outH * outW;

    tile_rows := outH;
    {$if not defined(USE_IM2COL)}
    if col_size > MAX_COL_SIZE then
        begin
            row_size := in_ch * kH * kW * outW;
            tile_rows := MAX_COL_SIZE div row_size;
            if tile_rows < 1 then
                tile_rows := 1
        end;
    tile_col_size := in_ch * kH * kW * tile_rows * outW;
    {$else}
    {$endif}
    if KH*KW<>1 then
       col := TMemoryBlock.Create([in_ch, kH, kW, tile_rows, outW]);
      //col2 := TMemoryBlock.Create([in_ch, kH, kW, tile_rows, outW]);

    for b := 0 to batch -1 do  begin
        in_b := src + b * in_ch * H * W;
        out_b := dst + b * out_ch * outH * outW;
        tile_start := 0;
        while tile_start < outH do begin
          tile_end := tile_start+tile_rows;
          if tile_end > outH then
              tile_end := outH;
          tile_h := tile_end-tile_start;
          tile_pixels := tile_h * outW;
          col_row := 0;
          if KH*KW=1 then // if pointwise conv
            col.assignPtr(in_b, [in_ch, kH, kW, tile_rows, outW])
          else
          {$if defined(USE_IM2COL)}
            QNNIm2Col(in_ch, H, W, KH, KW, padding, padding, stride, stride, 1, 1, in_b, col);
          {$else}
            for ic := 0 to in_ch -1 do
              for kernel_row := 0 to kH -1 do
                for kernel_col := 0 to kW -1 do begin
                  for oh := tile_start to tile_end -1 do begin
                    ih := oh * stride + kernel_row - padding;
                    //col_idx := col_row * tile_pixels+(oh-tile_start) * outW;
                    col_ptr := col + (col_row * tile_pixels+(oh-tile_start) * outW);
                    input_col := kernel_col - padding;
                    if (longword(ih) < longword(H)) then
                      if stride=1 then begin
                        FillChar(col_ptr[0], outW*sizeof(QNNFloat), #0);
                        move(in_b[ic*H*W + ih*W + ord(input_col>=0)*input_col], col_ptr[ord(input_col<0)*abs(input_col)], (outW-abs(input_col))*sizeof(QNNFloat))
                        //for ow := 0 to outW -1 do begin
                        //  iw := ow -padding + kernel_col;
                        //  if (longword(iw) < longword(W)) then
                        //    col_ptr[ow] := in_b[ic * H * W+ih * W+iw]
                        //  //else
                        //  //  col_ptr[ow] := 0.0
                        //end
                      end else
                        for ow := 0 to outW -1 do begin
                          iw := ow * stride  - padding+kernel_col;
                          if (longword(iw) < longword(W)) then
                            col_ptr[ow] := in_b[ic * H * W+ih * W+iw]
                          else
                            col_ptr[ow] := 0.0
                        end
                    else
                      FillChar(col_ptr[0], outW*sizeof(QNNFloat), #0);
                  end;
                  inc(col_row)
                end;
          {$endif}
          //QNNIm2Col(in_ch, H, W, KH, KW, padding, padding, stride, stride, 1, 1, in_b, col2);
          //col.printCompare(col2);
          K := in_ch * kH * kW;
          out_tile := out_b+tile_start * outW;
          cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_ch, tile_pixels, K, 1.0, weights, K, col, tile_pixels, 0.0, out_tile, outH * outW);
          inc(tile_start, tile_rows)
        end;
        if assigned(bias) then
          for oc := 0 to out_ch -1 do begin
            b_val := bias[oc];
            QNNBiasInplace(out_b + oc*outSpatial, b_val, outSpatial);
            //out_ch_ptr := out_b + oc*outH*outW;
            //for i := 0 to outSpatial -1 do
            //    out_ch_ptr[i] := out_ch_ptr[i] + b_val
          end
    end;
    col.free;
    col2.free;
{$endif}
end;

procedure QNNNorm(const N: longint; const dst, src: PQNNFloat; const weights: PQNNFloat; const stride: longint);
var
  mean, invStdDev: QNNFloat;
  i:longint;
begin
  mean := QNNMean(N, src, stride);
  invStdDev := 1/sqrt(QNNVariance(N, src, mean, stride) + EPSILON);
  if assigned(weights) then
    if stride=1 then
      QNNFusedBiasMulScale(dst, src, weights, -mean, invStdDev, N)
      //for i:=0 to N-1 do
      //  dst[i] := (src[i]-mean)*invStdDev*weights[i]
    else
      for i:=0 to N-1 do
        dst[i*stride] := (src[i*stride]-mean)*invStdDev*weights[i*stride]
  else
    if stride=1 then
      QNNFusedBiasScale(dst, src, -mean, invStdDev, N)
      //for i:=0 to N-1 do
      //  dst[i] := (src[i]-mean)*invStdDev
    else
      for i:=0 to N-1 do
        dst[i*stride] := (src[i*stride]-mean)*invStdDev
end;

procedure QNNRMSNorm(const N: longint; const dst, src: PQNNFloat; const weights: PQNNFloat; const stride: longint);
var
  invRMS, sum_sq, rms: single;
  i:longint;
begin
  //sum_sq := 0.0;
  //for i := 0 to N-1 do
  //  sum_sq := sum_sq + src[i]*src[i];
  //rms := sqrt(sum_sq / N + EPSILON);
  //invRMS := 1.0 / rms;
  //for i := 0 to N-1 do
  //    dst[i] := src[i]*invRMS*weights[i];
  invRMS := 1.0/sqrt(QNNSumSqr(N, src, stride)/N + EPSILON);
  if assigned(weights) then
    if stride =1 then
      QNNFusedMulScale(dst, src, weights, invRMS, N)
    else
      for i:= 0 to N-1 do
        dst[i*stride] := src[i*stride]*weights[i]*invRMS
  else
    if stride=1 then
      QNNScale(dst, src, invRMS, N)
    else
      for i:= 0 to N-1 do
        dst[i*stride] := src[i*stride]*invRMS

end;

procedure QNNRMSNormRows(const dst, src, weight: PQNNFloat; const rows, dim: integer);
var
   r
   //, i
   : integer;
   //sq_sum, rms, rms_inv:QNNFloat;
   //x_row, out_row : PQNNFloat;
begin
  // todo RMS_NORM Simdify
  for r :=0 to rows-1 do
      QNNRMSNorm(dim, dst + r*dim, src + r*dim, weight, 1);

end;

procedure QNNTanh(const dst: PQNNFloat; const N: longint);
var
  i: Integer;
begin
  for i:=0 to N-1 do
    dst[i] := tanh(dst[i])
end;

procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights: PQNNFloat; const seq, heads, headDim: longint);
var
  qh, kh : PQNNFloat;
  rmsInv:QNNFloat;
  i, j, d:longint;
begin
  for i := 0 to seq-1 do begin
      for j := 0 to heads-1 do begin
          (* Q normalization *)
          qh := Q + i*heads*headDim + j*headDim;

          //float sum_sq = 0.0f;
          //for (int d = 0; d < head_dim; d++) begin
          //    sum_sq += qh[d] * qh[d];
          //end

          //rmsInv := 1.0 / sqrtf(sum_sq / head_dim + eps);
          //for (int d = 0; d < head_dim; d++) begin
          //    qh[d] = qh[d] * rms_inv * q_weight[d];
          //end
          QNNRMSNorm(headDim, qh, qh, QWeights);

          (* K normalization *)
          kh := K + i*heads*headDim + j*headDim;
          //sum_sq = 0.0f;
          //for (int d = 0; d < head_dim; d++) begin
          //    sum_sq += kh[d] * kh[d];
          //end
          //rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
          //for (int d = 0; d < head_dim; d++) begin
          //    kh[d] = kh[d] * rms_inv * k_weight[d];
          //end
          QNNRMSNorm(headDim, kh, kh, KWeights);

      end
  end

end;

procedure QNNQKRMSNorm(const dst, normWeights: PQNNFloat; const seq, heads, headDim: longint);
var
  s:longint;
  head : PQNNFloat;
begin
  for s:= 0 to seq-1 do begin
    head := dst + s*heads*headDim;
    QNNRMSNormRows(head, head, normWeights, heads, headDim);
  end;
end;

procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat; const batch, channels, H, W, num_groups:integer);
var
  channels_per_group, spatial, b, g
  , count, c_start, c_end, c, idx, i: integer;
  mean, diff, variance, std_inv, norm : QNNFloat;
  src1, dst1 : PQNNFloat;
begin
    channels_per_group := channels div num_groups;
    spatial := H * W;
    // todo group norm simdify
    for b := 0 to batch-1 do begin
        for g := 0 to num_groups-1 do begin
            c_start := g * channels_per_group;
            c_end := c_start + channels_per_group;

            //mean := 0.0;
            //count := 0;
            //for c := c_start to c_end-1 do begin
            //    for i := 0 to spatial-1 do begin
            //        idx := b*channels*spatial + c*spatial + i;
            //        mean := mean + src[idx];
            //        inc(count);
            //    end
            //end;
            //mean := mean/count;

            src1 := src + b*channels*spatial + c_start*spatial;
            dst1 := dst + b*channels*spatial + c_start*spatial;
            mean :=QNNMean(spatial*channels_per_group, src1, 1);

            //variance := 0.0;
            //for c := c_start to c_end-1 do begin
            //    for i := 0 to spatial-1 do begin
            //        idx := b*channels*spatial + c*spatial + i;
            //        diff := src[idx] - mean;
            //        variance := variance + diff*diff;
            //    end
            //end;
            //variance := variance / count;
            variance := QNNVariance(spatial*channels_per_group, src1, mean);

            std_inv := 1.0 / sqrt(variance + EPSILON);

            for c:= 0 to channels_per_group-1 do
              QNNFusedScaleBias(dst1 + c*spatial, src1 + c*spatial, gamma[c+c_start]*std_inv, beta[c+c_start] - mean*gamma[c+c_start]*std_inv, spatial);
            //for c := c_start to c_end-1 do begin
            //  for i := 0 to spatial-1 do begin
            //    idx := b*channels*spatial + c * spatial + i;
            //    dst[idx] := src[idx]*gamma[c]*std_inv - mean*gamma[c]*std_inv + beta[c];
            //  end
            //end
        end
    end
end;

procedure QNNBatchNorm(const dst, src,
                     running_mean, running_var, gamma, beta : PQNNFloat;
                     const batch, channels, H, W: integer);
var
  spatial, c, n, i, idx : integer;
  mean, variance, std_inv, g, b_val:QNNFloat;
  src1, dst1 : PQNNFloat;
begin
  // todo BatchNorm simdify
    spatial := H * W;
    for c := 0 to channels-1 do begin
        mean     := running_mean[c];
        variance := running_var[c];
        std_inv  := 1.0 / sqrt(variance + EPSILON);
        if assigned(gamma) then g :=  gamma[c] else g := 1.0;
        if assigned(beta) then b_val := beta[c] else b_val := 0.0;

        for n := 0 to batch-1 do begin
            src1:= src + n*channels*spatial + c*spatial;
            dst1:= dst + n*channels*spatial + c*spatial;
            QNNFusedScaleBias(dst1, src1, g*std_inv, b_val-mean*g*std_inv, spatial);
        end

        //for n := 0 to batch-1 do begin
        //    for i := 0 to spatial-1 do begin
        //        idx := n * channels * spatial + c * spatial + i;
        //        dst[idx] := g * (src[idx] - mean) * std_inv + b_val;
        //    end
        //end

    end
end;

procedure QNNSigmoid(const x: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo sigmoid inplace simdify
  for i := 0 to N-1 do
    x[i] := 1.0 / (1.0 + fast_exp(-x[i]));
end;

procedure QNNSigmoid(const dst, src: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo sigmoid simdify
  for i := 0 to N-1 do
    dst[i] := 1.0 / (1.0 + fast_exp(-src[i]));
end;

procedure QNNSiluInplace(const x: PQNNFloat; const N: integer);
var
  i:integer;
  val :QNNFloat;
begin
  // todo silu inplace simdify
    for i := 0 to N-1 do begin
        val := x[i];
        x[i] := val / (1.0 + fast_exp(-val));
    end
end;

procedure QNNSilu(const dst, src: PQNNFloat; const N: integer);
var
  i:integer;
  val :QNNFloat;
begin
  // todo silu priority simdify
    for i := 0 to N-1 do begin
        val := src[i];
        dst[i] := val / (1.0 + fast_exp(-val));
    end
end;

procedure QNNSiluMul(const gate, up:PQNNFloat; const N:integer);
var
  i: integer;
  val : QNNFloat;
begin

  // todo siluMul priority simdify
  for i := 0 to N-1 do begin
    val := gate[i];
    gate[i] := (val / (1.0 + fast_exp(-val))) * up[i];
  end
end;

procedure QNNSoftmax(const x: PQNNFloat; const N: integer);
var
  i: longint;
  max_val, sum, inv_sum: QNNFloat;
begin
  // todo softmax simdify
  max_val := x[0];
  for i := 1 to N -1 do
      if x[i] > max_val then
          max_val := x[i];
  //max_val := QNNMax(N, x);
  sum := 0.0;
  for i := 0 to N-1 do begin
      x[i] := fast_exp(x[i]-max_val);
      sum := sum + x[i]
  end;
  inv_sum := 1.0 / sum;
  for i := 0 to N-1 do
      x[i] := x[i] * inv_sum
end;

procedure QNNSoftmaxRows(const x: PQNNFloat; const rows, cols: integer);
var
    i, c: longint;
    row: PQNNFloat;
    max_val, sum, inv_sum: QNNFloat;
begin
  // todo softmax simdify
    for i := 0 to rows -1 do
      QNNSoftmax(x + i*cols, cols);
      //begin
      //  row := x+ i*cols;
      //  max_val := row[0];
      //  for c := 1 to cols -1 do
      //      if row[c] > max_val then
      //          max_val := row[c];
      //  sum := 0.0;
      //  for c := 0 to cols -1 do
      //      begin
      //          row[c] := fast_exp(row[c]-max_val);
      //          sum := sum + row[c]
      //      end;
      //  inv_sum := 1.0 / sum;
      //  for c := 0 to cols -1 do
      //      row[c] := row[c] * inv_sum
      //end
end;

(* ========================================================================
 * Attention Operations
 * ======================================================================== *)

(* SDPA (Scaled dot-product attention): softmax(Q @ K^T / sqrt(d)) @ V.
 * This is the naive implementation that materializes the full seq_q x seq_k
 * attention matrix. Used only for small sequences; the transformer's main
 * attention path uses iris_flash_attention() or the GPU kernel instead. *)
procedure QNNAttention(const dst, Q, K, V: PQNNFloat; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
var
    scores, qq, kk, vv, o: PQNNFloat;
    b, h, i, j, d: longint;
    dot, sum: QNNFloat;
begin
  //todo QNNAttention SIMDIFY
    if length(workspace) < seq_q*seq_k then
      setlength(workspace, seq_q*seq_k);
    scores := pointer(workspace);
    for b := 0 to batch -1 do
        for h := 0 to heads -1 do
            begin
                qq := Q+(b * heads+h) * seq_q * head_dim;
                kk := K+(b * heads+h) * seq_k * head_dim;
                vv := V+(b * heads+h) * seq_k * head_dim;
                o := dst+(b * heads+h) * seq_q * head_dim;
                {$if 1=1}
                cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_q, seq_k, head_dim, scale,  qq, head_dim, kk, head_dim, 0, scores, seq_k);
                //if scale<>1 then QNNScale(scores, scores, scale, seq_q*seq_k);
                {$else}
                for i := 0 to seq_q -1 do
                    for j := 0 to seq_k -1 do
                        begin
                            dot := 0.0;
                            for d := 0 to head_dim -1 do
                                dot := dot + (qq[i * head_dim+d] * kk[j * head_dim+d]);
                            scores[i * seq_k+j] := dot * scale
                        end;
                {$endif}
                QNNSoftmaxRows(scores, seq_q, seq_k);
                {$if 1=1}
                cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_q, head_dim, seq_k, 1, scores, seq_k, vv, head_dim, 0, o, head_dim);
                {$else}
                for i := 0 to seq_q -1 do
                    for d := 0 to head_dim -1 do
                        begin
                            sum := 0.0;
                            for j := 0 to seq_k -1 do
                                sum := sum + (scores[i*seq_k + j] * vv[j*head_dim + d]);
                            o[i * head_dim+d] := sum
                        end
                {$endif}
            end;
end;

(* ========================================================================
 * Flash Attention - Memory-Efficient Tiled Attention
 *
 * Uses online softmax algorithm to compute attention without materializing
 * the full [seq_q, seq_k] attention matrix. Reduces memory from O(n²) to O(n).
 *
 * Algorithm (for each query position):
 * 1. Initialize: max_score = -inf, sum = 0, output = 0
 * 2. For each key/value block:
 *    - Compute local scores = Q @ K^T * scale
 *    - Update running max and sum with correction factors
 *    - Accumulate weighted values into output
 * 3. Normalize: output /= sum
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
 * ======================================================================== *)

(*
 * Flash attention for a single head.
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * Uses O(head_dim) working memory per query instead of O(seq_k).
 *)
procedure QNNFlashAttentionHead(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
var
    i, d, j: longint;
    q_row, o_row, k_row, v_row: PQNNFloat;
    max_score, sum_exp, score, correction, weight, inv_sum: QNNFloat;
begin
    // todo FlashAttentionHead simdify
    for i := 0 to seq_q -1 do
        begin
            q_row := Q   + i*head_dim;
            o_row := dst + i*head_dim;
            max_score := -1;
            sum_exp := 0.0;
            QNNFill(o_row, 0, head_dim);
            //for d := 0 to head_dim -1 do
            //    o_row[d] := 0.0;
            for j := 0 to seq_k -1 do
                begin
                    k_row := K+j * head_dim;
                    v_row := V+j * head_dim;
                    //score := 0.0;
                    //for d := 0 to head_dim -1 do
                    //    score := score + (q_row[d] * k_row[d]);
                    //score := score * scale;
                    score := scale*QNNDot(q_row, k_row, head_dim);
                    if score > max_score then
                        begin
                            correction := fast_exp(max_score-score);
                            sum_exp := sum_exp*correction + 1.0;
                            QNNFusedScaleAdd(o_row, o_row, v_row, correction, head_dim);
                            //for d := 0 to head_dim -1 do
                            //    o_row[d] := o_row[d]*correction + v_row[d];
                            max_score := score
                        end
                    else
                        begin
                            weight := fast_exp(score-max_score);
                            sum_exp := sum_exp + weight;
                            cblas_axpy(head_dim, weight, v_row, 1, o_row, 1);
                            //for d := 0 to head_dim -1 do
                            //    o_row[d] := o_row[d] + (weight * v_row[d])
                        end
                end;
            inv_sum := 1.0 / sum_exp;
            cblas_scal(head_dim, inv_sum, o_row, 1);
            //for d := 0 to head_dim -1 do
            //    o_row[d] := o_row[d] * inv_sum
        end
end;

(*
 * Flash attention with BLAS-optimized tiling.
 * Processes queries in tiles for better cache utilization.
 * Uses BLAS for tile-level matrix operations when available.
 *
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * tile_scores: scratch buffer of size [q_tile_size, k_tile_size]
 *)
procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat; const tile_scores: PQNNFloat; const q_tile_size, k_tile_size: longint);
var
    max_scores, sum_exps, Q_tile, K_tile, V_tile, score_row, o_row, out_tile, v_row: PQNNFloat;
    i, k_start, k_end, k_len, q_start, q_end, q_len, qi, ki, d: longint;
    tile_max, old_max, new_max, correction, weight, inv_sum: QNNFloat;
begin
  // todo FlashAttentionHeadTiled simdify
  // cache frendly flash attention head
    if length(workspace)< 2*seq_q then setLength(workspace, 2*seq_q);
    max_scores := pointer(workspace);
    sum_exps := @workspace[seq_q];

    //for i := 0 to seq_q -1 do
    //    begin
    //        max_scores[i] := -1;
    //        sum_exps[i] := 0.0
    //    end;

    QNNFill(max_scores, -1, seq_q);
    QNNFill(sum_exps,  0.0, seq_q);


    //fillchar(dst^, seq_q*head_dim*sizeof(QNNFloat), 0);
    QNNFill(dst, 0, seq_q*head_dim);
    k_start := 0;
    while k_start < seq_k do begin
        if (k_start+k_tile_size < seq_k) then
            k_end := k_start+k_tile_size
        else
            k_end := seq_k;
        k_len := k_end-k_start;
        q_start := 0;
        while q_start < seq_q do begin
            if (q_start+q_tile_size < seq_q) then
                q_end := q_start+q_tile_size
            else
                q_end := seq_q;
            q_len := q_end-q_start;
            Q_tile := Q+q_start * head_dim;
            K_tile := K+k_start * head_dim;
            V_tile := V+k_start * head_dim;
            out_tile := dst+q_start * head_dim;
            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_len, k_len, head_dim, scale, Q_tile, head_dim, K_tile, head_dim, 0.0, tile_scores, k_tile_size);
            for qi := 0 to q_len -1 do begin
                i := q_start+qi;
                score_row := tile_scores+qi * k_tile_size;
                o_row := out_tile+qi * head_dim;
                tile_max := QNNMax(k_len, score_row);
                //tile_max := score_row[0];
                //for ki := 1 to k_len -1 do
                //    if score_row[ki] > tile_max then
                //        tile_max := score_row[ki];
                old_max := max_scores[i];
                if (tile_max > old_max) then
                    new_max := tile_max
                else
                    new_max := old_max;
                if old_max > -1 then begin
                    correction := fast_exp(old_max-new_max);
                    sum_exps[i] := sum_exps[i] * correction;
                    cblas_scal(head_dim, correction, o_row, 1);
                    //for d := 0 to head_dim -1 do
                    //    o_row[d] := o_row[d] * correction
                end;
                for ki := 0 to k_len -1 do  begin
                    weight := fast_exp(score_row[ki]-new_max);
                    sum_exps[i] := sum_exps[i] + weight;
                    v_row := V_tile+ki * head_dim;
                    cblas_axpy(head_dim, weight, v_row, 1, o_row, 1);
                    //for d := 0 to head_dim -1 do
                    //    o_row[d] := o_row[d] + (weight * v_row[d])
                end;
                max_scores[i] := new_max
            end;
            inc(q_start, q_tile_size)
        end;
        k_start := k_start + k_tile_size
    end;
    for i := 0 to seq_q -1 do  begin
        cblas_scal(head_dim, 1.0/sum_exps[i], dst+i*head_dim, 1);
        //inv_sum := 1.0 / sum_exps[i];
        //o_row := dst+i * head_dim;
        //for d := 0 to head_dim -1 do
        //    o_row[d] := o_row[d] * inv_sum
    end;
end;

(*
 * Flash attention for multi-head attention.
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 *
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 *
 * Memory usage: O(seq_q + tile_size²) instead of O(seq_q * seq_k)
 *)
procedure QNNFlashAttention(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
const
    q_tile_size = 32;
    k_tile_size = 64;
var
    h, i, d, j, hidden, off: longint;
    Q_head, K_head, V_head, out_head, Q_contig, K_contig, V_contig, out_contig: PQNNFloat;
    tile_scores : array[0..q_tile_size * k_tile_size-1] of QNNFloat;

    //scores : PQNNFloat;
begin
  //tile_scores := PQNNFloat(malloc(q_tile_size * k_tile_size * sizeof(float)));
  hidden := heads * head_dim;
  if length(workspace) < 2*head_dim*(seq_k+seq_q) then
    setLength(workspace, 2*head_dim*(seq_k+seq_q));

(*
  scores := pointer(workspace);
  for i := 0 to heads-1 do begin
      //qh := Q      + i*head_dim;
      //kh := K      + i*head_dim;
      //vh := V      + i*head_dim;
      //oh := dst    + i*head_dim;
      //sh := scores + i*seq_q*seq_k;

      cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  seq_q, seq_k, head_dim,
                  scale,
                  Q      + i*head_dim, hidden,
                  K      + i*head_dim, hidden,
                  0.0,
                  scores + i*seq_q*seq_k, seq_k);
      QNNSoftmax(scores + i*seq_q*seq_k, seq_q, seq_k);

      cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  seq_k, head_dim, seq_q,
                  1.0,
                  scores + i*seq_q*seq_k, seq_k,
                  V      + i*head_dim, hidden,
                  0.0,
                  dst    + i*head_dim, hidden);
  end;
  exit;
*)

  off :=0;
  Q_contig   := @workspace[off]; inc(off, seq_q*head_dim);
  K_contig   := @workspace[off]; inc(off, seq_k*head_dim);
  V_contig   := @workspace[off]; inc(off, seq_k*head_dim);
  out_contig := @workspace[off];

  for h := 0 to heads -1 do begin
    Q_head := Q+h * head_dim;
    K_head := K+h * head_dim;
    V_head := V+h * head_dim;
    out_head := dst + h*head_dim;
    if (seq_q <= 64) and (seq_k <= 128) then begin
      for i := 0 to seq_q -1 do
          QNNCopy(Q_contig+i*head_dim, Q_head+i*hidden, head_dim);
          //for d := 0 to head_dim -1 do
          //    Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
      for j := 0 to seq_k -1 do begin
          QNNCopy(K_contig+j*head_dim, K_head+j*hidden{heads*head_dim}, head_dim);
          QNNCopy(V_contig+j*head_dim, V_head+j*hidden{heads*head_dim}, head_dim);
          //for d := 0 to head_dim -1 do
          //    begin
          //        K_contig[j * head_dim+d] := K_head[j * hidden+d];
          //        V_contig[j * head_dim+d] := V_head[j * hidden+d]
          //    end;
      end;
      QNNFlashAttentionHead(out_contig, Q_contig, K_contig, V_contig, seq_q, seq_k, head_dim, scale);
      for i := 0 to seq_q -1 do
          QNNCopy(out_head+i*hidden, out_contig+i*head_dim, head_dim);
          //for d := 0 to head_dim -1 do
          //    out_head[i * hidden+d] := out_contig[i * head_dim+d];
    end else begin
      for i := 0 to seq_q -1 do
          QNNCopy(Q_contig+i*head_dim, Q_head+i*hidden, head_dim);
          //for d := 0 to head_dim -1 do
          //    Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
      for j := 0 to seq_k -1 do begin
          QNNCopy(K_contig+j*head_dim, K_head+j*hidden{heads*head_dim}, head_dim);
          QNNCopy(V_contig+j*head_dim, V_head+j*hidden{heads*head_dim}, head_dim);
          //for d := 0 to head_dim -1 do
          //    begin
          //        K_contig[j * head_dim+d] := K_head[j * hidden+d];
          //        V_contig[j * head_dim+d] := V_head[j * hidden+d]
          //    end;
      end;
      QNNFlashAttentionHeadTiled(out_contig, Q_contig, K_contig, V_contig, seq_q, seq_k, head_dim, scale, @tile_scores[0], q_tile_size, k_tile_size);

      for i := 0 to seq_q -1 do
          QNNCopy(out_head+i*hidden, out_contig+i*head_dim, head_dim);
          //for d := 0 to head_dim -1 do
          //    out_head[i * hidden+d] := out_contig[i * head_dim+d];
    end
  end;
end;

(* Apply precomputed RoPE (Rotary Position Embedding) in-place using the
 * split-half convention: dim d pairs with dim d+half for rotation. This is
 * the Flux convention (4-axis, split-half); Z-Image uses consecutive pairs
 * via a separate kernel. RoPE lets the transformer learn relative position
 * from the dot-product structure of Q and K. *)
procedure QNNApplyRoPE(const dst, freqs: PQNNFloat; const batch, seq, heads,
  head_dim: longint);
var
    head: PQNNFloat;
    half_dim, b, s, h, i: longint;
    cos_val, sin_val: QNNFloat;
    x0, x1: QNNFloat;
begin
    //todo QNNApply RoPE SIMDIFY
    half_dim := head_dim div 2;
    for b := 0 to batch -1 do
      for s := 0 to seq -1 do
        for h := 0 to heads -1 do begin
            head := dst+((b * seq+s) * heads+h) * head_dim;
            for i := 0 to half_dim -1 do begin
                cos_val := freqs[s*half_dim*2 + i*2];
                sin_val := freqs[s*half_dim*2 + i*2+1];
                x0 := head[i];
                x1 := head[i+half_dim];
                head[i]          := x0*cos_val - x1*sin_val;
                head[i+half_dim] := x0*sin_val + x1*cos_val
            end
        end
end;

procedure QNNApplyRoPE(const dst, cosIn, sinIn: PQNNFloat; const numHeads, headDim: longint);
var
  halfDim, h, i :longint;
  head : PQNNFloat;
  x0, x1 : QNNFloat;
begin
  halfDim := headDim div 2;           // todo Priority SIMD
  for h:=0 to numHeads-1 do begin
    head := dst + h*headDim;
    for i:=0 to halfDim-1 do begin
      x0 := head[i];
      x1 := head[i+halfDim];
      head[i]           := x0*cosIn[i] - x1*sinIn[i];
      head[i + halfDim] := x0*sinIn[i] + x1*cosIn[i];
    end;
  end;
end;

procedure QNNApplyRoPE(const dst, cosIn, sinIn: PQNNFloat; const seqLen, numHeads, headDim: longint);
var
  halfDim, s:longint;
begin
  halfDim := headDim div 2;      // todo Priority SIMD
  for s := 0 to seqLen-1 do
    QNNApplyRope(dst + s*numHeads*headDim, cosIn + s*halfDim, sinIn + s*halfDim, numHeads, headDim)
end;

procedure QNNApplyRoPE(const q, k, cos_cache, sin_cache: PQNNFloat; const seq_len, num_q_heads, num_kv_heads, head_dim: longint);
var
  half_dim, s, i, h : longint;
  x0, x1, cos_val, sin_val : double;
  q_head, k_head, cos_row, sin_row : PQNNFloat;
begin
  half_dim := head_dim div 2;

  (* Apply RoPE to Q *)
  for s := 0 to seq_len-1 do begin
      cos_row := cos_cache + s * half_dim;
      sin_row := sin_cache + s * half_dim;

      for h := 0 to num_q_heads-1 do begin
          q_head := q + s * num_q_heads * head_dim + h * head_dim;

          for i := 0 to half_dim-1 do begin
              x0 := q_head[i];
              x1 := q_head[i + half_dim];
              cos_val := cos_row[i];
              sin_val := sin_row[i];
              q_head[i]            := x0*cos_val - x1*sin_val;
              q_head[i + half_dim] := x0*sin_val + x1*cos_val;
          end;
      end;
  end;

  (* Apply RoPE to K *)
  for s := 0 to seq_len-1 do begin
      cos_row := cos_cache + s * half_dim;
      sin_row := sin_cache + s * half_dim;

      for h := 0 to num_kv_heads-1 do begin
          k_head := k + s*num_kv_heads*head_dim + h*head_dim;

          for i := 0 to half_dim-1 do begin
              x0 := k_head[i];
              x1 := k_head[i + half_dim];
              cos_val := cos_row[i];
              sin_val := sin_row[i];

              k_head[i]            := x0*cos_val - x1*sin_val;
              k_head[i + half_dim] := x0*sin_val + x1*cos_val;
          end;
      end;
  end;
end;


procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq: PQNNFloat; const seq, heads, head_dim, dim: longint);
var s, d, h:longint;
  cos_s, sin_s, vec :PQNNFloat;
  cos_val,sin_val, x0, x1 : QNNFloat;
begin
  //axis_dim;  (* head_dim = 128 = 4 * axis_dim (axis_dim = 32) *)
  for s := 0 to seq-1 do begin         // todo priority SIMD
    cos_s := cos_freq + s*head_dim;  (* [128] *)
    sin_s := sin_freq + s*head_dim;

    for h := 0 to heads-1 do begin
      vec := dst + (s*heads + h) * head_dim;

      (* Apply rotation to all 128 dims in pairs (0,1), (2,3), ... (126,127) *)
      d := 0;
      while d < head_dim do begin
          cos_val := cos_s[d];  (* cos[d] == cos[d+1] due to repeat_interleave *)
          sin_val := sin_s[d];
          x0 := vec[d];
          x1 := vec[d + 1];
          (* Complex rotation: (x0 + i*x1) * (cos + i*sin) *)
          vec[d] := x0 * cos_val - x1 * sin_val;
          vec[d + 1] := x1 * cos_val + x0 * sin_val;
          inc(d, 2)
      end
    end
  end
end;

function QNNTimeStepEmbedding(const t: QNNFloat; const dim: integer; const max_period: QNNFloat): TMemoryBlock;
var
  half_dim, i : longint;
  log_max, freq, angle : QNNFloat;
  dst_ptr : PQNNFloat;
begin
  half_dim := dim div 2;
  log_max := ln(max_period);
  result := TMemoryBlock.Create([2, half_dim]);
  dst_ptr := result;
  for i := 0 to half_dim-1 do begin
      (* freq = exp(-log(max_period) * i / half_dim) *)
      freq := exp(-log_max * (i / half_dim));
      angle := t * freq;
      dst_ptr[i] := cos(angle);           (* cos part first (flip_sin_to_cos=True) *)
      dst_ptr[i + half_dim] := sin(angle);    (* sin part second *)
  end
end;

// Adaptive Layer Normalization
procedure QNNAdaLN(const dst, src, bias, scale: PQNNFloat; const seq, hidden: longint; const eps: QNNFloat);
var
  i, j:longint;
  inRow, outRow:PQNNFloat;
  Mean, variance, stdInv, norm:QNNFloat;
begin
  for i:=0 to seq-1 do begin      // todo priority SIMD
    inRow  := src + i*hidden;
    outRow := dst + i*hidden;
    mean     := QNNMean(hidden, inRow);
    variance := QNNVariance(hidden, inRow, mean);
    stdInv := 1.0/sqrt(variance + EPSILON);
    for j:=0 to hidden-1 do begin
      norm := (inRow[j] - mean)*stdInv;
      outRow[j] := (1 + scale[j])*norm + bias[j]
    end;
  end;

end;

procedure QNNLINEAR_BF16_OR_F32(const dst, src: PSingle;
  const weight: TMemoryBlock; const weight_bf16: TMemoryBlock; const seq,
  srcDim, dstDim: integer);
begin
  assert(weight.isAssigned() or weight_bf16.isAssigned(),'ERROR QNNLINEAR_BF16_OR_F32 : no weight or weight_16 assigned!');
  if weight_bf16.isAssigned() then
    QNNLinearNoBias_BF16(dst, src, weight_bf16, seq, srcDim, dstDim)
  else
    QNNLinearNoBias     (dst, src, weight     , seq, srcDim, dstDim);
end;


procedure QNNUpSampleNearest(const dst, src: PQNNFloat; const batch, channels, H, W, scale_h, scale_w: longint);
var
    outH, outW, b, c, oh, ow, ih, iw, in_idx, out_idx: longint;
begin
    outH := H * scale_h;
    outW := W * scale_w;
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for oh := 0 to outH -1 do
          for ow := 0 to outW -1 do
            begin
              ih := oh div scale_h;
              iw := ow div scale_w;
              in_idx := b * channels * H * W+c * H * W+ih * W+iw;
              out_idx := b * channels * outH * outW+c * outH * outW+oh * outW+ow;
              dst[out_idx] := src[in_idx]
            end
end;

function linearInterpolation(const a, b, t:QNNFloat):QNNFloat;
begin
  result := a + t*(b-a)
end;

function cubicInterpolation(const a, b, c, d, t:QNNFloat):QNNFloat; overload;
begin
  //result := b + t*(0.5*c - 0.5*a + t*(a -2.5*b + 2*c -0.5*d + t*(1.5*b - 0.5*a - 1.5*c + 0.5*d))) ;

  result := b + 0.5 * t*(c - a + t*(2.0*a - 5.0*b + 4.0*c - d + t*(3.0*(b - c) + d - a)))
end;

function cubicInterpolation(const b, c, t:QNNFloat):QNNFloat;  overload;
begin
  //result := b + t*(0.5*c - 0.5*a + t*(a -2.5*b + 2*c -0.5*d + t*(1.5*b - 0.5*a - 1.5*c + 0.5*d))) ;

  //result := b + 0.5 * t*(c - a + t*(2.0*a - 5.0*b + 4.0*c - d + t*(3.0*(b - c) + d - a)))
  result := b + 0.5 * t*(c + t*(-5.0*b + 4.0*c + t*(3.0*(b - c))))

end;


procedure QNNUpSample(const dst, src: PQNNFloat; const batch, channels, H, W: longint; const scale_h, scale_w: QNNFloat; const interpolation: TInterpolation);
var
    outH, outW, b, c, oh, ow, ih, iw, in_idx, out_idx: longint;
    fx, fy, p1, p2, p3, p4: QNNFloat;
begin
  // [QNNUpSample] simdify
  assert((scale_w>=1) and (scale_h>=1),'ERROR QNNUpSample: Scales must be larger or equal than 1.0') ;
  assert(interpolation in [iNearest, iLinear, iCubic],'ERROR QNNUpSample: Selected interpolation is not implemented!') ;
  outH := ceil(H * scale_h);
  outW := ceil(W * scale_w);
  if interpolation = iNearest then begin
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for oh := 0 to outH -ceil(scale_h) do begin
          ih := trunc(oh / scale_h);
          for ow := 0 to outW -ceil(scale_w) do
            begin
              iw := trunc(ow / scale_w);
              out_idx := b * channels * outH * outW+c * outH * outW+oh * outW+ow;
              in_idx := b * channels * H * W+c * H * W+ih * W+iw;
              dst[out_idx] := src[in_idx];
            end;
        end;
    exit
  end;
  if interpolation = iLinear then begin
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for oh := 0 to outH - ceil(scale_h)-1 do begin
          fy := oh/scale_h;
          ih := trunc(fy);
          fy := frac(fy);
          for ow := 0 to outW - ceil(scale_w)-1 do
            begin
              fx := ow/scale_w;
              iw := trunc(fx);
              fx := frac(fx);
              out_idx := b * channels * outH * outW + c * outH * outW+oh * outW+ow;
              in_idx := b * channels * H * W+c * H * W+ih * W+iw;
              p1 := linearInterpolation(src[in_idx], src[in_idx+1], fx);
              p2 := linearInterpolation(src[in_idx+W], src[in_idx+W+1], fx);
              dst[out_idx] := linearInterpolation(p1, p2, fy);
            end;
        end;
    exit
  end;
  if interpolation = iCubic then begin
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for oh := 0 to outH - ceil(scale_h)-1 do begin
          fy := oh/scale_h;
          ih := trunc(fy);
          fy := frac(fy);
          for ow := 0 to outW - ceil(scale_w)-1 do
            begin
              fx := ow/scale_w;
              iw := trunc(fx);
              fx := frac(fx);
              out_idx := b * channels * outH * outW + c * outH * outW+oh * outW+ow;
              in_idx := b * channels * H * W+c * H * W+ih * W+iw;
              p1 := cubicInterpolation(src[in_idx], src[in_idx+1], fx);
              p2 := cubicInterpolation(src[in_idx+W], src[in_idx+W+1], fx);
              dst[out_idx] := cubicInterpolation(p1, p2, fy);
            end;
          //fy := oh/scale_h;
          //ih := trunc(fy);
          //fy := frac(fy);
          //if ih<1 then begin
          //  ih:=1 ; fy := -1+fy
          //end else
          //if ih> H-2 then begin
          //  ih := H-2 ; fy := 1
          //end else
          //if ih> H-3 then begin
          //  ih := H-3 ; fy := 1+fy
          //end;
          //for ow := 0 to outW - ceil(scale_w)-1 do
          //  begin
          //    fx := ow/scale_w;
          //    iw := trunc(fx);
          //    fx := frac(fx);
          //    out_idx := b * channels * outH * outW + c * outH * outW+oh * outW+ow;
          //    if iw<1 then begin
          //      iw:=1 ; fx := -1+fx
          //    end else
          //    if iw> W-2 then begin
          //      iw := W-2 ; fx := 1
          //    end else
          //    if iw> W-3 then begin
          //      iw := W-3 ; fx := 1+fx
          //    end;
          //    in_idx := b * channels * H * W+c * H * W+ih * W+iw;
          //    p1 := cubicInterpolation(src[in_idx-W-1], src[in_idx-W], src[in_idx-W+1], src[in_idx-W+2], fx);
          //    p2 := cubicInterpolation(src[in_idx-1], src[in_idx], src[in_idx+1], src[in_idx+2], fx);
          //    p3 := cubicInterpolation(src[in_idx+W-1], src[in_idx+W], src[in_idx+W+1], src[in_idx+W+2], fx);
          //    p4 := cubicInterpolation(src[in_idx+2*W-1], src[in_idx+2*W], src[in_idx+2*W+1], src[in_idx+2*W+2], fx);
          //    dst[out_idx] := cubicInterpolation(p1, p2, p3, p4, fy);
          //  end;
        end;
    exit
  end;

end;


(* Convert spatial latent to patch tokens for the diffusion transformer.
 * Groups each ps x ps spatial block into a single token vector:
 * [batch, channels, H, W] -> [batch, channels*ps*ps, H/ps, W/ps].
 * The transformer operates on these patch tokens, not individual spatial
 * positions, reducing sequence length by ps*ps (4x for ps=2). *)
procedure QNNPatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
var outH, outW, out_ch, b, c, ph, pw, pi, pj, ih, iw, in_idx, out_c, out_idx: longint;
begin
    outH := H div patch_size;
    outW := W div patch_size;
    out_ch := channels * patch_size * patch_size;
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for ph := 0 to outH -1 do
          for pw := 0 to outW -1 do
            for pi := 0 to patch_size -1 do
              for pj := 0 to patch_size -1 do
                begin
                  ih := ph * patch_size+pi;
                  iw := pw * patch_size+pj;
                  in_idx := b * channels * H * W + c * H * W + ih * W + iw;
                  out_c := c * patch_size * patch_size + pi * patch_size + pj;
                  out_idx := b * out_ch * outH * outW+out_c * outH * outW+ph * outW+pw;
                  dst[out_idx] := src[in_idx]
                end
end;

procedure QNNUnpatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
var in_ch, outH, outW, b, c, ph, pw, pi, pj, in_c, in_idx, oh, ow, out_idx: longint;
begin
    in_ch := channels * patch_size * patch_size;
    outH := H * patch_size;
    outW := W * patch_size;
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for ph := 0 to H -1 do
          for pw := 0 to W -1 do
            for pi := 0 to patch_size -1 do
              for pj := 0 to patch_size -1 do
                begin
                  in_c := c*patch_size*patch_size + pi*patch_size + pj;
                  in_idx := b*in_ch*H*W + in_c*H*W + ph*W + pw;
                  oh := ph*patch_size + pi;
                  ow := pw*patch_size + pj;
                  out_idx := b*channels*outH*outW + c*outH*outW + oh*outW + ow;
                  dst[out_idx] := src[in_idx]
                end
end;

procedure QNNCopy(const dst, src:PQNNFloat; const N:integer);
begin
  move(src^, dst^, N*sizeOf(QNNFloat))
end;

procedure QNNCopyStrided(const dst: PQNNFloat; const dstStride: integer; const src: PQNNFloat; const srcStride: integer; const N: integer);
var i: integer;
begin
  if dstStride*srcStride=1 then
    move(src^, dst^, N*sizeOf(QNNFloat))
  else
    for i:=0 to N-1 do dst[i*dstStride]:=src[i*srcStride]
end;

procedure QNNFill(const dst: PQNNFloat; const val: QNNFloat; const N: integer; const stride: integer);
var i:integer;
begin
  if (val=0) and (stride=1) then
    FillChar(dst^, N*SizeOf(QNNFloat), 0)
  else
    for i:=0 to N-1 do dst[i*stride]:= val
end;

procedure QNNBroadcast(const dst: pointer; const src; const srcSize,
  N: integer; const stride: integer);
var i:integer;
begin
   for i:= 0 to N-1 do
       move(src, (PByte(dst)+ i*stride*srcSize)^, srcSize)
end;



const
  {$if defined(MSWINDOWS)}
  blaslib = 'openblas.dll';
  {$elseif defined(LINUX)}
  blaslib = 'libopenblas.so';
  {$elseif defined(MACOS) or defined(DARWIN)}
  {$endif}

(*
procedure s2b(const a:TArray<single>; const result : TArray<byte>);
var
  i: Integer;
  v : single;
begin
  assert(length(a)=length(result));
  //result := nil;
  //setLength(result, length(a));
  for i:= 0 to high(a) do begin
    if a[i]>1 then v:=1 else
    if a[i]<0 then v:=0 else v := a[i];
    result[i] := round(v*255)
  end;
end;

procedure b2s(const a:TArray<byte>; const result : TArray<single>);
var
  i: Integer;
begin
  assert(length(a)=length(result));
  //result := nil;
  //setLength(result, length(a));
  for i:= 0 to high(a) do
    result[i] := a[i]/255
end;

const scale = 40;
  WW = 11;
  HH = 10;
var
  im1, om1:  TArray<byte>;
  im2, om2 : TArray<single>;
  i : longint;
*)

procedure printStat(const src:PQNNFloat; const N:longint);
const DECIMALS =3;
var mi, ma, ami, ama, mean, stddev:QNNFLoat;
  argma, argmi : longint;
begin
  mi := QNNMin(N, src, @argmi);
  ma := QNNMax(N, src, @argma);
  ami := QNNMinAbs(N, src);
  ama := QNNMaxAbs(N, src);
  mean := QNNMean(N, src);
  stdDev := sqrt(QNNVariance(N, src, mean));
  writeln('[', N,']', 'Min :', mi:1:DECIMALS,'@',argmi, ', Max:', ma:1:DECIMALS,'@',argma,', minAbs:', ami:1:DECIMALS,', maxAbs:', ama:1:DECIMALS,', mean:', mean:1:DECIMALS, ', stddev:', stddev:1:DECIMALS);
end;

procedure printStat(const src: TMemoryBlock);
begin
  printStat(src, src.count)
end;

procedure printCompare(const N:longint; const src1, src2:PQNNFloat; const isSumSqrDiff:boolean =false);
var md,out1,out2: single;
begin
  assert(assigned(src1) and assigned(src2), 'ERROR : src1 and src2 must be assigned');
  writeln('');
  printStat(src1, N);
  printStat(src2, N);
  if isSumSqrDiff then begin
    md := QNNSqrDistance(N, src1, src2);
    writeln('SqrDistance :', md:1:5);
  end else begin
    md := QNNMaxAbsDiff(N, src1, src2, out1, out2);
    writeln('MaxAbsDiff :', md:1:5, ' max src1 :', out1:1:6, ' max src2:', out2:1:6);
  end;
end;

procedure writeTensor(const buf:TMemoryBlock);
var f: file; rt:integer;
begin
  assignFile(f, 'c:\development\flux2.c\tensor.1');
  Rewrite(f, sizeof(single));
  BlockWrite(f, PSingle(buf)^, buf.count(), rt);
  closefile(f);
  assert(rt = buf.Count(), 'Unmatch blocksize write');
end;

function readTensor(): TMemoryBlock;
var f:File; rd:integer;
begin
  assignFile(f, 'c:\development\flux2.c\tensor');
  reset(f, sizeof(single));
  result := TMemoryBlock.Create(fileSize(f));
  BlockRead(f, PSingle(result)^, result.count, rd);
  CloseFile(f);
  assert(rd = result.Count(), 'Unmatch blcoksize read');
end;

function readArray: TArray<integer>;
var f:File; rd:integer;
begin
  assignFile(f, 'c:\development\flux2.c\tensor');
  reset(f, sizeof(integer));
  setLength(result, fileSize(f));
  BlockRead(f, PInteger(result)^, length(result), rd);
  CloseFile(f);
  assert(rd = length(result), 'Unmatch blcoksize read');
end;

procedure compareArray(const a, b: TArray<Integer>);
var i:longint;
begin
  assert(high(a)=high(b),'unmatched array length');
  for i:=0 to high(a)-1 do
    assert(a[i]=b[i],' Unmatched element at '+intToStr(i))
end;

procedure compareArray(const a, b: PInteger; const N:longint);
var i:longint;
begin
  for i:=0 to N-1 do
    assert(a[i]=b[i],' Unmatched element at '+intToStr(i))
end;

procedure compareArray(const a, b: PSingle; const N: longint);
var i:longint;
begin
  for i:=0 to N-1 do
    assert(a[i]=b[i],' Unmatched element at '+intToStr(i))
end;



//procedure __imp_prn(const str: pchar);
//begin
//  printf(str)
//end;

procedure printArray(const src:PQNNFloat; const N:longint);
var i:longint;
begin
  write('[');
  if N>0 then begin
    write(src[0]:1:4);
    for i:=1 to N-1 do
      write(', ', src[i]:1:4)

  end;
  writeln(']')
end;


type
  PF = ^TF;
  TF = array[0..7] of QNNFloat;
const
  batch = 1; heads = 64; seq = 64; dim = 64;


  M  = 1;
  N  = 511*13;
  KK = 519*9;

var
  A, B, O, O2 : TMemoryBlock;
  QPTR,KPTR,VPTR, OPTR, OPTR2 : PQNNFloat;
  i : longint;
  t, t2:Int64;

  d1, d2 : QNNFloat;
  dptr : PQNNFloat;

initialization


  //SetPrecisionMode(TFPUPrecisionMode.pmDouble);
(*
  setLength(im1, 3*WW*HH);
  setLength(im2, length(im1));
  setlength(om1, 3*ceil(WW*scale)*ceil(HH*scale));
  setlength(om2, length(om1));
  for i := 1 to high(im1) do
    im1[i] := not(im1[i-1]);

  b2s(im1, im2);
  QNNUpSample(pointer(om2), pointer(im2), 1, 3, HH, WW, scale, scale, iNearest);
  s2b(om2, om1);
  //printSixel(pointer(im1), WW, HH, true, false, poCHW);
  printSixel(pointer(om1), ceil(WW*scale), ceil(HH*scale), true, false, poCHW);

  fillchar(om2[0], length(om1)*sizeof(om2[0]), #0);
  b2s(im1, im2);
  QNNUpSample(pointer(om2), pointer(im2), 1, 3, HH, WW, scale, scale, iLinear);
  s2b(om2, om1);
  //printSixel(pointer(im1), WW, HH, true, false, poCHW);
  printSixel(pointer(om1), ceil(WW*scale), ceil(HH*scale), true, false, poCHW);

  fillchar(om2[0], length(om1)*sizeof(om2[0]), #0);
  b2s(im1, im2);
  QNNUpSample(pointer(om2), pointer(im2), 1, 3, HH, WW, scale, scale, iCubic);
  s2b(om2, om1);
  //printSixel(pointer(im1), WW, HH, true, false, poCHW);
  printSixel(pointer(om1), ceil(WW*scale), ceil(HH*scale), true, false, poCHW);
//writeln(cubicInterpolation(-1, 3.3, 2, 0, 0):1:4);
  //writeln(cubicInterpolation(-1, 3.3, 2, 0, 1):1:4);
  //writeln(cubicInterpolation(-1, 3.3, 2, 0, 2):1:4);
  //writeln(cubicInterpolation(-1, 3.3, 2, 0, 3):1:4);
  readln;
*)

  cblas_axpy := qaxpy;
  cblas_dot  := qdot;
  cblas_scal := qscale;
  cblas_gemm := qgemm;
  cblas_asum := qasum;

  hBLASLib :=0;

  {$if defined(MACOS) or defined(darwin) or defined(USE_STATIC_LINK)}
  case GetTypeData(TypeInfo(QNNFloat)).FloatType of
    ftSingle :
      begin
        cblas_gemm := cblas_sgemm  ;
        cblas_axpy := cblas_saxpy  ;
        cblas_dot  := cblas_sdot   ;
        cblas_asum := cblas_sasum  ;
        cblas_scal := cblas_sscal  ;
      end;
    //ftDouble :
    //  begin
    //    cblas_gemm := cblas_dgemm  ;
    //    cblas_axpy := cblas_daxpy  ;
    //    cblas_dot  := cblas_ddot   ;
    //    cblas_asum := cblas_dasum  ;
    //    cblas_scal := cblas_dscal  ;
    //  end;
  end;
  {$else}
  hBLASLib := LoadLibrary(blaslib);
  {$ifdef MSWINDOWS}
  if hBLASLib=0 then
    hBLASLib:=LoadLibrary('lib'+blaslib);
  {$endif}
  if (hBLASLib>0) then
    case GetTypeData(TypeInfo(QNNFloat)).FloatType of
      ftSingle :
        begin
          cblas_gemm   := getProcAddress(hBLASLib, 'cblas_sgemm');
          cblas_axpy   := getProcAddress(hBLASLib, 'cblas_saxpy');
          cblas_dot    := getProcAddress(hBLASLib, 'cblas_sdot' );
          cblas_asum   := getProcAddress(hBLASLib, 'cblas_sasum');
          cblas_scal   := getProcAddress(hBLASLib, 'cblas_sscal');
          cblas_sbgemm := getProcAddress(hBLASLib, 'cblas_sbgemm');
        end;
      //ftDouble :
      //  begin
      //    cblas_gemm := getProcAddress(hBLASLib, 'cblas_dgemm');
      //    cblas_axpy := getProcAddress(hBLASLib, 'cblas_daxpy');
      //    cblas_dot  := getProcAddress(hBLASLib, 'cblas_ddot' );
      //    cblas_asum := getProcAddress(hBLASLib, 'cblas_dasum');
      //    cblas_scal := getProcAddress(hBLASLib, 'cblas_dscal');
      //  end;
    end;
  {$endif}


(*
  randomize;
  A := TMemoryBlock.create(M*KK);
  dptr := A;
  for i:=0 to A.Count-1 do
    dptr[i] := 10*(random()*2 -1);
  B := TMemoryBlock.create(KK*N);
  dptr := B;
  for i:=0 to B.Count-1 do
    dptr[i] := 10*(random()*2 -1);

  O := TMemoryBlock.create([M, N]);
  dptr := O;
  for i:=0 to O.Count-1 do
    dptr[i] := 10*(random()*2 -1);

  O2 := TMemoryBlock.create([M, N]);
  QNNCopy(O2, O, O.count);

  //set_prn(printf);

  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, KK, 1.5, A, KK, B, N, 0, O, N) ;
  //      qgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, KK, 1.5, A, KK, B, N, 0, O2, N) ;
   //cpp_sgemm(               CblasNoTrans, CblasNoTrans, M, N, KK, 1.5, Q, KK, K, N, 0, O2, N, 0 , M) ;

   for i:=0 to 10 do
       cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, KK, 1, A, KK, B, KK, 1, O , N) ;
   for i:=0 to 10 do
       cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, KK, 1, A, KK, B, KK, 1, O2, N) ;
  //cpp_sgemm(                CblasNoTrans, CblasTrans, M, N, KK, 1.5, Q, KK, K, KK, 0, O2, N, 0, M) ;

  //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, KK, 1.5, Q, M, K, N, 0, O, N) ;
  //     qgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, KK, 1.5, Q, M, K, N, 0, O2, N) ;

  //O .print();
  //O2.print();           sgemm2

  O.printCompare(O2, false);

  readln;

*)



finalization

  if hBLASLib>0 then begin
    FreeLibrary(hBLASLib);
    cblas_gemm := nil;
    cblas_axpy := nil;
    cblas_dot  := nil;
    cblas_asum := nil;
    cblas_scal := nil;
    cblas_sbgemm := nil
  end;

end.

