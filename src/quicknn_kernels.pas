unit quicknn_kernels;
{$ifdef FPC}
  {$PackRecords C}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$modeswitch nestedprocvars}
  {$ifdef CPUX64}
    {$asmmode intel}
  {$endif}
  {$if defined(darwin)}
    {$LinkFramework accelerate}
  {$endif}
{$endif}
{$C+} // enable assertions
{$H+} // longstrings
{$pointermath on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds

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
  ;

const EPSILON = 0.000001;
{$if not declared(blasint)}
type
  blasint = longint;
{$endif}
{$if not declared(CBLAS_LAYOUT)}
type
  CBLAS_Layout = (CblasRowMajor = 101, CblasColMajor = 102);
{$else}
const
  CblasRowMajor = CBLAS_Layout.CblasRowMajor;
  CblasColMajor = CBLAS_Layout.CblasColMajor;
{$endif}

{$if not declared(CBLAS_ORDER)}
type
  CBLAS_ORDER = CBLAS_Layout;
{$endif}
{$if not declared(CBLAS_TRANSPOSE)}
type
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans =
    113, CblasConjNoTrans = 114);
{$else}
const
  CblasNoTrans = CBLAS_TRANSPOSE.CblasNoTrans;
  CblasTrans = CBLAS_TRANSPOSE.CblasTrans;
  CblasConjTrans = CBLAS_TRANSPOSE.CblasConjTrans;
  CblasConjNoTrans = CBLAS_TRANSPOSE.CblasConjNoTrans ;
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
  TSubstepCallback = procedure( &type :TSubstepType; index :longint; total: longint);

  (*
   * Step callback - called at sampling step boundaries.
   * step: current step (1-based), or 0 to indicate sampling is starting
   * total: total number of steps
   *)
  TStepCallback = procedure (step:longint; total:longint);


  (*
   * Phase callback - called at major phase boundaries.
   * phase: descriptive name ("encoding text", "decoding image", etc.)
   * done: 0 when starting, 1 when finished
   *)
  TPhaseCallback = procedure (const phase :pchar (* todo maybe a string*); done:longint);

  (*
   * Step image callback - called after each denoising step with decoded image.
   * step: current step (1-based)
   * total: total number of steps
   * img: decoded image at this step (caller must NOT free)
   *
   * To use: set both iris_step_image_callback and iris_step_image_vae before
   * calling the sampling function. The callback is only invoked when both are set.
   *)
  TStepImageCallback = procedure (step: longint; total: longint; const img : PQNNimage(* todo maybe its an array? *));

  (*
   * Text encoder progress callback - called once per Qwen3 layer.
   * layer: current layer (0-based)
   * total: total number of layers (36)
   *)
  TTextProgressCallback = procedure (layer : longint; total : longint);

  (*
   * VAE progress callback - called once per resblock/attention block.
   * block: current block (0-based)
   * total: total number of blocks (11 for encoder, 15 for decoder)
   *)
  TVAEProgressCallback = procedure ( block: longint; total: longint );


  TInterpolation = ( iNearest, iLinear, iCubic, iLanczos);

var
  hBLASLib : HMODULE;
  {$if defined(MACOS) or defined(DARWIN)}
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
  cblas_gemm : procedure (Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ;
  cblas_axpy : procedure (n:blasint; alpha:single; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint); winapi ;
  cblas_dot : function (n:blasint; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint):single; winapi ;
  cblas_asum : function (n:blasint; x:PQNNFloat; incx:blasint):QNNFloat; winapi ;
  cblas_scal : procedure (N:blasint; alpha:QNNFloat; X:PQNNFloat; incX:blasint = 1); winapi ;
  cblas_sbgemm : procedure(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PBF16; lda:blasint; B:PBF16; ldb:blasint; beta:single; C:PSingle; ldc:blasint); winapi ;


  workspace : TArray<QNNFloat>;
(* Global callback pointers - set by caller before inference *)
  substep_callback : TSubstepCallback;
  step_callback    : TStepCallback;
  phase_callback   : TPhaseCallBack;
  step_image_callback : TStepImageCallback;
  step_image_vae : pointer;  (* Set to iris_vae_t* for step image decoding *)
  text_progress_callback_t : TTextProgressCallback;
  vae_progress_callback_t  : TVAEProgressCallback;
  verbose : longint;

// Add and Mul terms used for vector vector element wise operations
// Scale and Bias terms used for vector Scalar element wise operations

procedure QNNAdd(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNAdd(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNMul(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNMul(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNScale(const dst, src:PQNNFloat; const aScale:QNNFloat; const N:longint);
procedure QNNBias(const dst, src:PQNNFloat; const aBias:QNNFloat; const N:longint);

procedure QNNFusedMulAdd(const dst, src, srcM, srcA:PQNNFloat; const N:longint);overload;
procedure QNNFusedAddMul(const dst, src, srcA, srcM:PQNNFloat; const N:longint);overload;
procedure QNNFusedScaleBias(const dst, src:PQNNFloat; const aScale, aBias:QNNFloat; const N:longint);overload;
procedure QNNFusedBiasScale(const dst, src:PQNNFloat; const aBias, aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedMulScale(const dst, src, srcM:PQNNFloat; const aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedBiasMulScale(const dst, src, srcM:PQNNFloat; const aBias, aScale:QNNFloat; const N:longint);overload;
procedure QNNFusedScaleAdd(const dst, src, srcA:PQNNFloat; const aScale:QNNFloat; const N:longint); overload;
function QNNDot(const a, b:PQNNFloat; const N:longint):QNNFloat;

procedure QNNAccAdd(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNAccMul(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNMatMulNN(const A, B, C: PQNNFloat; const M,N,K:integer);   overload;
procedure QNNMatMulNT(const A, B, C: PQNNFloat; const M,N,K:integer);   overload;
procedure QNNLinear(const dst, x, W, b:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias(const dst, x, W:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias_BF16(const dst, x:PQNNFloat; const W: PBF16; const seqLen, inDIM, outDIM: integer);
procedure QNNIm2Col(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth
                        , padHeight, padWidth
                        , strideY, strideX
                        , dilationY, dilationX: blasint
                        ; const im: PQNNFloat; const col: PQNNFloat
                        ; const imOffset: blasint =0 ; const colOffset: blasint =0
                        ; const MultiThread: boolean = False);

procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth, padHeight, padWidth
                        , strideY, strideX, dilationY, dilationX: blasint
                        ; const im: PQNNFloat; const imStride, imOffset: blasint
                        ; const col: PQNNFloat; const colStride, colOffset: blasint
                        ; const batch:blasint);

procedure QNNConv2d( const dst, src, weights, bias:PQNNFloat;
                     const in_ch, out_ch, H, W, kH, kW, stride, padding:blasint; const batch:blasint =1);

procedure QNNNorm(const N:longint; const dst, src:PQNNFloat; const weights:PQNNFloat = nil; const stride:longint = 1);      overload;

procedure QNNRMSNorm(const N:longint; const dst, src:PQNNFloat; const weights:PQNNFloat = nil; const stride:longint = 1);   overload;
procedure QNNRMSNorm(const dst, src, weight:PQNNFloat; const seqLen, hidden: integer); overload;
procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights:PQNNFloat; const seq, heads, headDim:longint);
procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat; const batch, channels, H, W, num_groups:integer);
procedure QNNBatchNorm(const dst, src,
                      running_mean, running_var, gamma, beta : PQNNFloat;
                      const batch, channels, H, W: integer);
function QNNMax(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNMin(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNArgMax(const N:integer; const src:PQNNFloat; const stride:integer=1):integer;
function QNNArgMin(const N:integer; const src:PQNNFloat; const stride:integer=1):integer;
function QNNSum(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNSumSqr(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNSumSqrDiff(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer=1):QNNFloat;
function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNVariance(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer=1; const isPopulation:boolean=true):QNNFloat;
procedure QNNSilu(const x:PQNNFloat; const n:integer);
procedure QNNSiluMul(const gate, up:PQNNFloat; const N:integer);
procedure QNNSoftmax(const x: PQNNFloat; const rows, cols: integer);
procedure QNNAttention(const dst, Q, K, V: PQNNFloat; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHead(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat; const tile_scores: PQNNFloat; const q_tile_size, k_tile_size: longint);
procedure QNNFlashAttention(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
procedure QNNUpSampleNearest(const dst, src: PQNNFloat; const batch, channels, H, W, scale_h, scale_w: longint);
procedure QNNUpSample(const dst, src: PQNNFloat; const batch, channels, H, W:longint; const scale_h, scale_w: QNNFloat; const interpolation:TInterpolation = iNearest);
procedure QNNPatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNUnpatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNCopy(const dst, src:PQNNFloat; const N:integer); overload;
procedure QNNCopy(const dst:PQNNFloat; const dstStride:integer; const src:PQNNFloat; const srcStride: integer; const N:integer); overload;
procedure QNNFill(const dst:PQNNFloat; const val:QNNFloat; const N:integer; const stride:integer=1);
procedure QNNBroadcast(const dst:pointer; const src; const srcSize, N:integer; const stride:integer=1);
procedure QNNGatedAdd(const dst, gate, proj:psingle; const seq, hidden:integer);
procedure QNNComputeRoPE(const dst:PQNNFloat; const maxSeq, dim:longint; const theta:QNNFloat);
procedure QNNComputeRoPE2D(const cosDst,sinDst:PQNNFloat; const patch_h, patch_w, dim:longint; const theta:QNNFloat);
procedure QNNComputeRoPE2DOffset(const cosDst,sinDst:PQNNFloat; const patch_h, patch_w, dim:longint; const theta:QNNFloat; const offset_t:longint);
procedure QNNComputeRoPEText(const cosDst, sinDst: PQNNFloat; const txt_seq, axis_dim: longint; const theta: QNNFloat);
procedure QNNApplyRoPE(const dst, freqs: PQNNFloat; const batch, seq, heads, head_dim: longint);
procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq : PQNNFloat; const seq, heads, head_dim, dim: longint);

procedure QNNTimestepEmbedding(const dst :PQNNFloat; const t : QNNFloat;const dim:integer; const max_period:QNNFloat);

// Adaptive Layer Normalization
procedure QNNAdaLN(const dst, src, bias, scale:PQNNFloat; const seq, hidden:longint; const eps:QNNFloat=EPSILON);

procedure QNNLINEAR_BF16_OR_F32(const dst, src:PSingle; const weight:PSingle; const weight_bf16:PBF16; const seq, srcDim, dstDim:integer);inline;

implementation

function sqr(const x:QNNFloat):QNNFloat;inline;
begin
  exit(x*x)
end;

function fast_expf(const x:single):single; inline;
var
  n, r, p:single;
  v : record
    case boolean of
      false :(i:longint);
      true  :(f:single)
  end;

begin
    if (x < -87.3) then  exit( 0.0);
    if (x > 88.7)  then exit( 1e38);
    n := floor(x * 1.4426950408889634 + 0.5);
    r := x - n * 0.6931471805599453;
    p := 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 +
              r * (0.04166667 + r * 0.00833333))));
    v.f := p;
    v.i := v.i + PLongword(@n)^ shl 23;
    result := v.f;
end;

function QNNMax(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i:integer;
begin
  assert(assigned(src), 'ERROR QNNMax: [src] is empty!');
  result := src[0];
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]>result then result := src[i]
  end else
    for i:=1 to N-1 do
      if src[i*stride]>result then result := src[i*stride]

end;

function QNNMin(const N: integer; const src: PQNNFloat; const stride: integer): QNNFloat;
var i:integer;
begin
  assert(assigned(src), 'ERROR QNNMin: [src] is empty!');
  result := src[0];
  if stride=1 then begin
    for i:=1 to N-1 do
      if src[i]<result then result := src[i]
  end else
    for i:=1 to N-1 do
      if src[i*stride]<result then result := src[i*stride]
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
  // todo simdify mean
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

function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer):QNNFloat;
begin
  // todo simdify mean
  result := QNNSum(N, src, stride);
  result := result / N
end;

function QNNVariance(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer; const isPopulation:boolean):QNNFloat;
begin
  if (N=1) and not isPopulation then exit(0);
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

procedure QNNAdd(const dst, a: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo SIMDIFY ADD_inplace
  for i:=0 to N-1 do dst[i] := a[i]+dst[i]
end;

procedure QNNMul(const dst, a, b: PQNNFloat; const N: integer);
var i:integer;
begin
  // todo simdify add
  for i:=0 to N-1 do dst[i] := a[i]*b[i]
end;

procedure QNNMul(const dst, a: PQNNFloat; const N: integer);
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

procedure QNNBias(const dst, src: PQNNFloat; const aBias: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNBias
    for i:=0 to N-1 do
      dst[i] := src[i]+aBias;
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
    for i:=0 to N-1 do
      dst[i] := (src[i] + aBias)*aScale
end;

procedure QNNFusedMulScale(const dst, src, srcM: PQNNFloat; const aScale: QNNFloat; const N: longint);
var i:integer;
begin
    // todo SIMDIFY QNNFusedMulScale (Scalars)
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

procedure QNNFusedScaleAdd(const dst, src, srcA: PQNNFloat; const aScale: QNNFloat; const N: longint);
var i:longint;
begin
  if srcA=dst then
    cblas_axpy(N, aScale, src, 1, dst, 1)
  else
    for i:=0 to N-1 do
      dst[i] := aScale*src[i] + srcA[i]
end;

function QNNDot(const a, b: PQNNFloat; const N: longint): QNNFloat;
begin
  result := cblas_dot(N, a, 1, b, 1);
end;


procedure QNNGatedAdd(const dst, gate, proj:psingle; const seq, hidden:integer);
var i:integer;
begin
  // todo gated add [SIMDIFY]
  for i:=0 to seq-1 do
    QNNAccMul(dst + i*hidden, gate, proj + i*hidden, hidden);
end;

//RoPE (Rotary Position Embeddings)
procedure QNNComputeRoPE(const dst: PQNNFloat; const maxSeq, dim: longint; const theta: QNNFloat);
var
  i, d, halfdim:longint;
  freq, angle:QNNFloat;
begin
  halfdim := dim div 2;
  for i := 0 to maxSeq-1 do
    for d :=0 to halfdim-1 do begin
      freq  :=  1/power(theta, (2*d)/dim);
      angle := i*freq;
      dst[i*halfdim*2 + d*2]     := cos(angle);
      dst[i*halfdim*2 + d*2 + 1] := Sin(angle);
    end;
end;

procedure QNNComputeRoPE2D(const cosDst, sinDst: PQNNFloat; const patch_h,
  patch_w, dim: longint; const theta: QNNFloat);
var
  i, d, hy, wx,pos, halfdim:longint;
  cos_p, sin_p : PQNNFloat;
  angle_h, angle_w, cos_h, cos_w, sin_h, sin_w: QNNFloat;
  freqs: array[0..15] of QNNFloat;
begin
  halfdim := dim div 2;  (* 16 dims per half-axis *)
  //int seq = patch_h * patch_w;
  //(void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for i:= 0 to halfdim-1 do
    freqs[d] := 1 / power(theta, (2 * d) / halfdim);

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

procedure QNNComputeRoPE2DOffset(const cosDst, sinDst: PQNNFloat;
  const patch_h, patch_w, dim: longint; const theta: QNNFloat;
  const offset_t: longint);
var
  i, d, hy, wx,pos, halfdim:longint;
  cos_p, sin_p : PQNNFloat;
  angle_h, angle_w, angle_t, cos_h, cos_w, cos_t, sin_h, sin_w, sin_t: QNNFloat;
  freqs: array[0..15] of QNNFloat;
begin
  halfdim := dim div 2;  (* 16 dims per half-axis *)
  //int seq = patch_h * patch_w;
  //(void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for i:= 0 to halfdim-1 do
    freqs[d] := 1 / power(theta, (2 * d) / halfdim);

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
  half_axis, head_dim, d, s: longint;
  cos_p: PQNNFloat;
  sin_p: PQNNFloat;
  angle, cos_l, sin_l: QNNFloat;
  freqs : array[0..15]of QNNFloat;
begin
  half_axis := axis_dim div 2;
  head_dim := axis_dim * 4;
  for d := 0 to half_axis -1 do
    freqs[d] := 1.0 / power(theta, (2 * d) div axis_dim);
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
          cos_p[axis_dim*3 + d*2]     := cos_l;
          cos_p[axis_dim*3 + d*2 + 1] := cos_l;
          sin_p[axis_dim*3 + d*2]     := sin_l;
          sin_p[axis_dim*3 + d*2 + 1] := sin_l
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
      X[i*incX] := X[incX]* alpha
end;

procedure qaxpy(n:blasint; alpha:single; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint); winapi ;
var
  i:blasint;
begin
  // todo qaxpy Simdify
  if (incx=1) and (incy=1) then
    for i:=0 to N-1 do
      y[i]:= alpha*x[i] + y[i]
  else
    for i:=0 to N-1 do
      y[i*incy]:= alpha*x[i*incx] + y[i*incy]
end;

function qdot(n:blasint; x:PQNNFloat; incx:blasint; y:PQNNFloat; incy:blasint):single; winapi ;
var
  i:blasint;
begin
  result := 0;
  // todo qdot Simdify
  if (incx=1) and (incy=1) then
    for i:=0 to N-1 do
      result := result + x[i]*y[i]
  else
    for i:=0 to N-1 do
      result := result + x[i*incx]*y[i*incy]
end;

procedure qgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:PSingle; ldc:blasint); WINAPI;
var
  mm ,nn ,kk : blasint;
  a_part : QNNFloat;
  AA, BB, CC : PQNNFloat;
begin
  // todo qgemm Simdify
  assert((order=CblasRowMajor) and not((TransA=CblasTrans) and (TransB=CblasTrans)),'ERROR : Operation is not supported using the provided arguments!');
  if BETA=0 then
    FillChar(C^, M*N*sizeOf(QNNFloat), 0)
  else if BETA<>1 then
    cblas_scal(M*N, BETA, C, 1);

  if (transA=CblasNoTrans) and (transB=CblasNoTrans)then begin
    for mm :=0 to M-1 do begin
      CC := C + mm*ldc;
      for kk := 0 to K-1 do begin
        a_part := ALPHA*A[mm*lda + kk];
        BB := B + kk*ldb;
        qaxpy(N, a_part, BB, 1, CC, 1);
        //for nn := 0 to N-1 do begin
        //    CC[nn] := CC[nn] + a_part*BB[nn];
        //end;
      end
    end;
    exit;
  end;
  if (transA=CblasNoTrans) and (transB=CblasTrans)then begin
    for mm:=0 to M-1 do begin
      CC := C + mm*ldc;
      AA := A + mm*lda;
      for nn := 0 to N-1 do begin
          BB := B + nn*ldb;
          a_part := qdot(N, AA, 1, BB, 1);
          CC[nn] := CC[nn] + ALPHA*a_part;
          //a_part := 0.0;
          //for kk:=0 to K-1 do begin
          //    a_part := a_part + AA[kk]*BB[kk];
          //end;
      end;
    end;
    exit
  end;
  if (transA=CblasTrans) and (transB=CblasNoTrans)then begin
    for mm :=0 to M-1 do begin
      CC := C + mm*ldc;
      for kk:=0 to K-1 do begin
        a_part := ALPHA*A[kk*lda + mm];
        BB := B + kk*ldb;
        qaxpy(N, a_part, BB, 1, CC, 1);
        //for nn:=0 to N-1 do begin
        //  CC[nn] := CC[nn] + a_part*BB[nn]
        //end;
      end;
    end;
    exit;
  end;
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
  for i:=0 to N-1 do dst[i] := dst[i] + a[i]*b[i]
end;

procedure QNNMatMulNN(const A, B, C: PQNNFloat; const M, N, K: integer);
begin
    cblas_gemm(CBLASRowMajor, CBLASNoTrans, CBLASNoTrans
    , M, N, K, 1
    , A, K
    , B, N
    , 0
    , C, N);
end;

procedure QNNMatMulNT(const A, B, C: PQNNFloat; const M, N, K: integer);
begin
    cblas_gemm(CBLASRowMajor, CblasNoTrans, CBLASTrans
    , M, N, K, 1
    , A, K
    , B, K
    , 0
    , C, N);
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
       QNNAdd(dst + i*outDIM, b, outDIM)
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
  dilationX: blasint; const im: PQNNFloat; const col: PQNNFloat;
  const imOffset: blasint; const colOffset: blasint; const MultiThread: boolean
  );
var
  channel, output_h, output_w, channel_size, out_channel_size, kernel_size: NativeInt;
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
    d_im, d_col: PSingle;
  begin
    d_im := im + imOffset + channel_size * idx;
    d_col := col + colOffset + kernel_size * out_channel_size * idx;
    for kernel_row := 0 to kernelHeight - 1 do
      for kernel_col := 0 to kernelWidth - 1 do
        begin
          //fillChar(d_col^, output_w*output_h*sizeOf(single), #0);
          input_row := -padHeight + kernel_row * dilationY;
          for output_row := 0 to output_h - 1 do  begin
            if {(input_row>=0) and} (NativeUInt(input_row) < NativeUInt(aHeight)) then begin
              input_col := -padWidth + kernel_col * dilationX;
              if (strideX=1) then begin
                fillchar(d_col[0], output_w*SizeOf(d_col^), #0);
                move(d_im[input_row*aWidth + ord(input_col>=0)*input_col], d_col[ord(input_col<0)*abs(input_col)], (output_w-abs(input_col))*sizeof(d_col^));
              end
              else
                for output_col:=0 to output_w-1 do  begin
                  i := output_col*strideX + input_col;
                  d_col[output_col] := ord(NativeUInt(i) < NativeUInt(aWidth))*d_im[input_row*aWidth + i];
                end; // or use the following (UnOptimized)
                //while output_col < output_w do  begin
                //  if {(input_col>=0) and} (SizeUInt(input_col) < SizeUInt(aWidth)) then
                //    d_col[output_col] := d_im[input_row * aWidth + input_col]
                //  else
                //    d_col[output_col] := 0;
                //  ;
                //  Inc(output_col);
                //  Inc(input_col, strideX);
                //end;
            end
            else
              fillchar(d_col^, output_w*SizeOf(d_col^), #0);
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
  dilationX: blasint; const im: PQNNFloat; const imStride, imOffset: blasint;
  const col: PQNNFloat; const colStride, colOffset: blasint;
  const batch: blasint);
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

procedure QNNConv2d(const dst, src, weights, bias: PQNNFloat; const in_ch,
  out_ch, H, W, kH, kW, stride, padding: blasint; const batch: blasint);
var
  outW, outH, kSize, imColSize, filters, outImgSize
  , k, inVolume, b, strideA, strideB, strideC: integer;
  workspacePtr : PQNNFloat;
begin
  outW := W div Stride + Padding * 2 - kW + 1 ;
  outH := H div Stride + Padding * 2 - kH + 1 ;
  kSize := kW*kH;
  outImgSize := outW*outH;
  filters :=out_ch;// {or dst.c() both are the same}
  k := in_ch * kSize;
  imColSize := in_ch * kSize * outImgSize;
  strideA :=0 ; // placeholdeAKernelsr
  strideC := outImgSize * filters;
  inVolume := in_ch*H*W;
  if (kSize <> 1) or
     //(xDilation * yDilation <> 1) or
     (Stride <> 1) then
  begin
      strideB := imColSize;
      if length(workspace) < batch * imColSize then
          setLength(workspace, batch * imColSize);
      workspacePtr := pointer(workspace);
      QNNIm2ColStridedBatched(
        in_ch {or AKernels.c()}, h, w, kH, kW,
        Padding, Padding, Stride, Stride, 1 {xDilation}, 1{yDilation},
        src, inVolume, 0,
        workspacePtr, strideB, 0, batch);
  end else begin
      strideB := inVolume;
      workspacePtr := src;
  end;
  if assigned(bias) then
    QNNCopy(dst, bias, batch*strideC)
  {else
    QNNCopy(dst, bias, batch*strideC)};
  for b := 0 to batch - 1 do
  begin
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     filters, outImgSize, k, 1,
                     weights + b*strideA, k,
                     workspacePtr + b * strideB, outImgSize,
                     ord(assigned(bias)), dst + b * strideC, outImgSize);
  end;

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
  invRMS: single;
  i:longint;
begin
  invRMS := 1/sqrt(QNNSumSqr(N, src, stride)/N + EPSILON);
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

procedure QNNRMSNorm(const dst, src, weight: PQNNFloat; const seqLen, hidden: integer);
var
   s
   //, i
   : integer;
   //sq_sum, rms, rms_inv:QNNFloat;
   //x_row, out_row : PQNNFloat;
begin
  // todo RMS_NORM Simdify
  for s :=0 to seqLen-1 do begin
      QNNRMSNorm(hidden, dst + s*hidden, src + s*hidden, weight, 1);

      //(* Compute RMS *)
      //sq_sum := 0.0;
      //for i := 0 to hidden-1 do
      //    sq_sum := sq_sum + sqr(x_row[i]);
      //
      //rms := sqrt(sq_sum / hidden + EPSILON);
      //rms_inv := 1.0 / rms;
      //
      //(* Normalize and scale *)
      //for i := 0 to hidden-1 do
      //    out_row[i] := x_row[i] * rms_inv * weight[i];
  end
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

procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat;
                     const batch, channels, H, W, num_groups:integer);
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
            //for c_start to c_end-1 do begin
                //for i := 0 to spatial-1 do begin
                //    idx := b * channels * spatial + c * spatial + i;
                //    mean := mean + src[idx];
                //    inc(count);
                //end
            //end;
            //mean := mean/count;

            src1 := src + b*channels*spatial + c_start*spatial;
            dst1 := src + b*channels*spatial + c_start*spatial;
            mean :=QNNMean(spatial*channels_per_group, src1, 1);

            //variance := 0.0;
            //for c := c_start to c_end-1 do begin
            //    for i := 0 to spatial-1 do begin
            //        idx := b * channels * spatial + c * spatial + i;
            //        diff := src[idx] - mean;
            //        variance := variance + diff * diff;
            //    end
            //end;
            //variance := variance / count;
            variance := QNNVariance(spatial*channels_per_group, src1, mean);

            std_inv := 1.0 / sqrt(variance + EPSILON);
            for c:= 0 to channels_per_group-1 do
              QNNFusedScaleBias(dst1+c*spatial, src1+c*spatial, gamma[c]*std_inv, -mean*gamma[c]*std_inv+beta[c], spatial);
            //for c := c_start to c_end-1 do begin
            //    for i := 0 to spatial-1 do begin
            //        idx := b * channels * spatial + c * spatial + i;
            //        dst[idx] := (src[idx] - mean)*gamma[c]*std_inv + beta[c];
            //    end
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
        mean := running_mean[c];
        variance := running_var[c];
        std_inv := 1.0 / sqrt(variance + EPSILON);
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

procedure QNNSilu(const x: PQNNFloat; const n: integer);
var
  i:integer;
  val :QNNFloat;
begin
  // todo silu simdify
    for i := 0 to n-1 do begin
        val := x[i];
        x[i] := val / (1.0 + fast_expf(-val));
    end
end;

procedure QNNSiluMul(const gate, up:PQNNFloat; const N:integer);
var
  i: integer;
  val : QNNFloat;
begin

  // todo siluMul simdify
  for i := 0 to N-1 do begin
    val := gate[i];
    gate[i] := (val / (1.0 + fast_expf(-val))) * up[i];
  end
end;

procedure QNNSoftmax(const x: PQNNFloat; const rows, cols: integer);
var
    r, c: longint;
    row: PQNNFloat;
    max_val, sum, inv_sum: QNNFloat;
begin
  // todo softmax simdify
    for r := 0 to rows -1 do
        begin
            row := x+r * cols;
            max_val := row[0];
            for c := 1 to cols -1 do
                if row[c] > max_val then
                    max_val := row[c];
            sum := 0.0;
            for c := 0 to cols -1 do
                begin
                    row[c] := fast_expf(row[c]-max_val);
                    sum := sum + row[c]
                end;
            inv_sum := 1.0 / sum;
            for c := 0 to cols -1 do
                row[c] := row[c] * inv_sum
        end
end;

(* ========================================================================
 * Attention Operations
 * ======================================================================== *)

(* Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V.
 * This is the naive implementation that materializes the full seq_q x seq_k
 * attention matrix. Used only for small sequences; the transformer's main
 * attention path uses iris_flash_attention() or the GPU kernel instead. *)
procedure QNNAttention(const dst, Q, K, V: PQNNFloat; const batch, heads,
  seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
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
                for i := 0 to seq_q -1 do
                    for j := 0 to seq_k -1 do
                        begin
                            dot := 0.0;
                            for d := 0 to head_dim -1 do
                                dot := dot + (qq[i * head_dim+d] * kk[j * head_dim+d]);
                            scores[i * seq_k+j] := dot * scale
                        end;
                QNNSoftmax(scores, seq_q, seq_k);
                for i := 0 to seq_q -1 do
                    for d := 0 to head_dim -1 do
                        begin
                            sum := 0.0;
                            for j := 0 to seq_k -1 do
                                sum := sum + (scores[i * seq_k+j] * vv[j * head_dim+d]);
                            o[i * head_dim+d] := sum
                        end
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
                            correction := fast_expf(max_score-score);
                            sum_exp := sum_exp*correction + 1.0;
                            QNNFusedScaleAdd(o_row, o_row, v_row, correction, head_dim);
                            //for d := 0 to head_dim -1 do
                            //    o_row[d] := o_row[d]*correction + v_row[d];
                            max_score := score
                        end
                    else
                        begin
                            weight := fast_expf(score-max_score);
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

    for i := 0 to seq_q -1 do
        begin
            max_scores[i] := -1;
            sum_exps[i] := 0.0
        end;

    fillchar(dst^, seq_q*head_dim*sizeof(QNNFloat), 0);
    //QNNFill(dst, 0, seq_q*head_dim)
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
                    correction := fast_expf(old_max-new_max);
                    sum_exps[i] := sum_exps[i] * correction;
                    cblas_scal(head_dim, correction, o_row, 1);
                    //for d := 0 to head_dim -1 do
                    //    o_row[d] := o_row[d] * correction
                end;
                for ki := 0 to k_len -1 do  begin
                    weight := fast_expf(score_row[ki]-new_max);
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
      //qh := Q      + h*head_dim;
      //kh := K      + h*head_dim;
      //vh := V      + h*head_dim;
      //oh := dst    + h*head_dim;
      //sh := scores + h*seq_q*seq_k;

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
  Q_contig   := @workspace[off]; inc(off, seq_q * head_dim);
  K_contig   := @workspace[off]; inc(off, seq_k * head_dim);
  V_contig   := @workspace[off]; inc(off, seq_k * head_dim);
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
    vec: PQNNFloat;
    half_dim, b, s, h, d: longint;
    cos_val, sin_val: QNNFloat;
    x0, x1: QNNFloat;
begin
    //todo QNNApply RoPE SIMDIFY
    half_dim := head_dim div 2;
    for b := 0 to batch -1 do
      for s := 0 to seq -1 do
        for h := 0 to heads -1 do
          begin
            vec := dst+((b * seq+s) * heads+h) * head_dim;
            for d := 0 to half_dim -1 do
              begin
                  cos_val := freqs[s * half_dim * 2+d * 2];
                  sin_val := freqs[s * half_dim * 2+d * 2+1];
                  x0 := vec[d];
                  x1 := vec[d+half_dim];
                  vec[d] := x0 * cos_val-x1 * sin_val;
                  vec[d+half_dim] := x0 * sin_val+x1 * cos_val
              end
            end
end;

procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq: PQNNFloat; const seq, heads, head_dim, dim: longint);
var s, d, h:longint;
  cos_s, sin_s, vec :PQNNFloat;
  cos_val,sin_val, x0, x1 : QNNFloat;
begin
  //axis_dim;  (* head_dim = 128 = 4 * axis_dim (axis_dim = 32) *)
  for s := 0 to seq-1 do begin
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

procedure QNNTimestepEmbedding(const dst: PQNNFloat; const t: QNNFloat; const dim: integer; const max_period: QNNFloat);
var
  half, i : longint;
  log_max, freq, angle : QNNFloat;
begin
  half := dim div 2;
  log_max := ln(max_period);

  for i := 0 to half-1 do begin
      (* freq = exp(-log(max_period) * i / half_dim) *)
      freq := exp(-log_max * (i / half));
      angle := t * freq;
      dst[i] := cos(angle);           (* cos part first (flip_sin_to_cos=True) *)
      dst[i + half] := sin(angle);    (* sin part second *)
  end
end;

// Adaptive Layer Normalization
procedure QNNAdaLN(const dst, src, bias, scale: PQNNFloat; const seq, hidden: longint; const eps: QNNFloat);
var
  i, j:longint;
  inRow, outRow:PQNNFloat;
  Mean, variance, stdInv, norm:QNNFloat;
begin
  for i:=0 to seq-1 do begin
    inRow  := src + i*hidden;
    outRow := dst + i*hidden;
    mean     := QNNMean(hidden, inRow);
    variance := QNNVariance(hidden, inRow, mean);
    stdInv := 1.0/sqrt(variance + EPSILON);
    for j:=0 to hidden-1 do begin
      norm := (inRow[i] - mean)*stdInv;
      outRow[j] := (1 + scale[j])*norm + bias[j]
    end;
  end;

end;

procedure QNNLINEAR_BF16_OR_F32(const dst, src: PSingle; const weight: PSingle;
  const weight_bf16: PBF16; const seq, srcDim, dstDim: integer);
begin
  assert(assigned(weight) or assigned(weight_bf16),'ERROR QNNLINEAR_BF16_OR_F32 : no weight or weight_16 assigned!');
  if assigned(weight_bf16) then
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

function cubicInterpolation(const a, b, c, d, t:QNNFloat):QNNFloat;
begin
  //result := b + t*(0.5*c - 0.5*a + t*(a -2.5*b + 2*c -0.5*d + t*(1.5*b - 0.5*a - 1.5*c + 0.5*d))) ;

  result := b + 0.5 * t*(c - a + t*(2.0*a - 5.0*b + 4.0*c - d + t*(3.0*(b - c) + d - a)))
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
              iw := round(ow / scale_w);
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
        for oh := 0 to outH - trunc(scale_h)-1 do begin
          fy := oh/scale_h;
          ih := trunc(fy);
          fy := frac(fy);
          for ow := 0 to outW - trunc(scale_w)-1 do
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
        for oh := trunc(scale_h) to outH - trunc(2*scale_h)-1 do begin
          fy := oh/scale_h;
          ih := trunc(fy);
          fy := frac(fy);
          for ow := trunc(scale_w) to outW - trunc(2*scale_w)-1 do
            begin
              fx := ow/scale_w;
              iw := trunc(fx);
              fx := frac(fx);
              out_idx := b * channels * outH * outW + c * outH * outW+oh * outW+ow;
              in_idx := b * channels * H * W+c * H * W+ih * W+iw;
              p1 := cubicInterpolation(src[in_idx-W-1], src[in_idx-W], src[in_idx-W+1], src[in_idx-W+2], fx);
              p2 := cubicInterpolation(src[in_idx-1], src[in_idx], src[in_idx+1], src[in_idx+2], fx);
              p3 := cubicInterpolation(src[in_idx+W-1], src[in_idx+W], src[in_idx+W+1], src[in_idx+W+2], fx);
              p4 := cubicInterpolation(src[in_idx+2*W-1], src[in_idx+2*W], src[in_idx+2*W+1], src[in_idx+2*W+2], fx);
              dst[out_idx] := cubicInterpolation(p1, p2, p3, p4, fy);
            end;
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
var p, outH, outW, out_ch, b, c, ph, pw, pi, pj, ih, iw, in_idx, out_c, out_idx: longint;
begin
    p := patch_size;
    outH := H div p;
    outW := W div p;
    out_ch := channels * p * p;
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for ph := 0 to outH -1 do
          for pw := 0 to outW -1 do
            for pi := 0 to p -1 do
              for pj := 0 to p -1 do
                begin
                  ih := ph * p+pi;
                  iw := pw * p+pj;
                  in_idx := b * channels * H * W+c * H * W+ih * W+iw;
                  out_c := c * p * p+pi * p+pj;
                  out_idx := b * out_ch * outH * outW+out_c * outH * outW+ph * outW+pw;
                  dst[out_idx] := src[in_idx]
                end
end;

procedure QNNUnpatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
var p, in_ch, outH, outW, b, c, ph, pw, pi, pj, in_c, in_idx, oh, ow, out_idx: longint;
begin
    p := patch_size;
    in_ch := channels * p * p;
    outH := H * p;
    outW := W * p;
    for b := 0 to batch -1 do
      for c := 0 to channels -1 do
        for ph := 0 to H -1 do
          for pw := 0 to W -1 do
            for pi := 0 to p -1 do
              for pj := 0 to p -1 do
                begin
                  in_c := c*p*p + pi*p + pj;
                  in_idx := b*in_ch*H*W + in_c*H*W + ph*W + pw;
                  oh := ph*p + pi;
                  ow := pw*p + pj;
                  out_idx := b*channels*outH*outW + c*outH*outW + oh*outW + ow;
                  dst[out_idx] := src[in_idx]
                end
end;

procedure QNNCopy(const dst, src:PQNNFloat; const N:integer);
begin
  move(src^, dst^, N*sizeOf(QNNFloat))
end;

procedure QNNCopy(const dst: PQNNFloat; const dstStride: integer; const src: PQNNFloat; const srcStride: integer; const N: integer);
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

initialization

  setLength(im1, 3*WW*HH);
  setLength(im2, length(im1));
  setlength(om1, 3*ceil(WW*scale)*ceil(HH*scale));
  setlength(om2, length(om1));
  for i := 1 to high(im1) do
    im1[i] := not(im1[i-1]);

  b2s(im1, im2);

  QNNUpSample(pointer(om2), pointer(im2), 1, 3, HH, WW, scale, scale, iCubic);
  s2b(om2, om1);
  printSixel(pointer(im1), WW, HH, true, false, poCHW);
  printSixel(pointer(om1), ceil(WW*scale), ceil(HH*scale), true, false, poCHW);
  readln;


  cblas_axpy := qaxpy;
  cblas_dot  := qdot;
  cblas_scal := qscale;
  cblas_gemm := qgemm;
  cblas_asum := qasum;

  hBLASLib := LoadLibrary(blaslib);
  {$ifdef MSWINDOWS}
  if hBLASLib=0 then
    hBLASLib:=LoadLibrary('lib'+blaslib);
  {$endif}

  {$if defined(MACOS) or defined(darwin)}
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


finalization

  if hBLASLib>0 then begin
    FreeLibrary(hBLASLib);
    cblas_gemm := nil;
    cblas_axpy := nil;
    cblas_dot  := nil;
    cblas_asum := nil;
    cblas_scal := nil;
    cblas_sbgemm := nil
  end

end.

