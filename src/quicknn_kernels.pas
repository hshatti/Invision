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
{$H+}
{$pointermath on}

{$define USE_MULTITHREADING}

interface
uses
  SysUtils, classes, Math
  {$if defined(MSWINDOWS)}
  , windows
  {$elseif defined(POSIX)}
  , unixbase
  {$endif}
  , steroids
  , quicknn_common;

const EPS = 0.000001;
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




var
  hBLASLib : HANDLE;
  {$if defined(MACOS) or defined(DARWIN)}
  procedure cblas_saxpy(n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ; external;
  function  cblas_sdot (n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):single; winapi ; external;
  function  cblas_sasum(n:blasint; x:Psingle; incx:blasint):single; winapi ; external;
  procedure cblas_sscal(N:blasint; alpha:single; X:Psingle; incX:blasint); winapi ; external;
  procedure cblas_sgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ; external;
  {$else}
  cblas_sgemm : procedure (Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ;
  cblas_saxpy : procedure (n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ;
  cblas_sdot : function (n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):single; winapi ;
  cblas_sasum : function (n:blasint; x:Psingle; incx:blasint):single; winapi ;
  cblas_sscal : procedure (N:blasint; alpha:single; X:Psingle; incX:blasint); winapi ;
  {$endif}

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

procedure QNNAdd(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNAdd(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNMul(const dst, a, b: PQNNFloat; const N:integer);   overload;
procedure QNNMul(const dst, a: PQNNFloat; const N:integer);   overload;
procedure QNNMatMulNN(const A, B, C: PQNNFloat; const M,N,K:integer);   overload;
procedure QNNMatMulNT(const A, B, C: PQNNFloat; const M,N,K:integer);   overload;
procedure QNNLinear(const dst, x, W, b:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias(const dst, x, W:PQNNFloat; const seqLen, inDIM, outDIM: integer);
procedure QNNLinearNoBias_BF16(const dst, x:PQNNFloat; const W: PBF16; const seqLen, inDIM, outDIM: integer);
procedure QNNConv2d(const dst, src, weights, bias:PQNNFloat;
                 const batch, in_ch, out_ch, H, W, kH, kW, stride, padding:integer);

procedure QNNRMSNorm(const dst, src, weight:PQNNFloat; const seqLen, hidden: integer);
procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat; const batch, channels, H, W, num_groups:integer);
procedure QNNBatchNorm(const dst, src,
                     running_mean, running_var, gamma, beta : PQNNFloat;
                     const batch, channels, H, W: integer);
function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer=1):QNNFloat;
function QNNVarianc(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer=1; const isPopulation:boolean=true):QNNFloat;
procedure QNNSilu(const x:PQNNFloat; const n:integer);
procedure QNNSiluMul(const gate, up:PQNNFloat; const N:integer);
procedure QNNSoftmax(const x: PQNNFloat; const rows, cols: integer);
procedure QNNAttention(const dst, Q, K, V: PQNNFloat; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHead(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat; const tile_scores: PQNNFloat; const q_tile_size, k_tile_size: longint);
procedure QNNFlashAttention(const dst, Q, K, V: PQNNFloat; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
procedure QNNApplyROPE(const dst, freqs: PQNNFloat; const batch, seq, heads, head_dim: longint);
procedure QNNComputeROPEFreqs(freqs: PQNNFloat; const pos: Plongint; seq: longint; dim: longint; theta: QNNFloat);
procedure QNNUpSampleNearest(const dst, src: PQNNFloat; const batch, channels, H, W, scale_h, scale_w: longint);
procedure QNNPatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNUnpatchify(const dst, src: PQNNFloat; const batch, channels, H, W, patch_size: longint);
procedure QNNCopy(const dst, src:PQNNFloat; const N:integer);

implementation

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
    v.i := v.i + longint(n) shl 23;
    result := v.f;
end;


function QNNMean(const N:integer; const src:PQNNFloat; const stride:integer):QNNFloat;
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
  result := result / N
end;

function QNNVarianc(const N:integer; const src:PQNNFloat; const aMean:QNNFloat; const stride:integer; const isPopulation:boolean):QNNFloat;
var i: integer;
begin
  //todo simdify variance
  if (N=1) and not isPopulation then exit(0);
  if stride=1 then
    for i:=0 to N-1 do
      result := result + sqr(src[i]-aMean)
  else
    for i:=0 to N-1 do
      result := result + sqr(src[i*stride]-aMean);
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

procedure qscale(N:blasint; alpha:single; X:Psingle; incX:blasint);  WINAPI;
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

procedure qaxpy(n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ;
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

function qdot(n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):single; winapi ;
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

procedure qgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; beta:single; C:Psingle; ldc:blasint); WINAPI;
var
  mm ,nn ,kk : blasint;
  a_part : QNNFloat;
  AA, BB, CC : PQNNFloat;
begin
  // todo qgemm Simdify
  assert((order=CblasRowMajor) and (TransA=CblasNoTrans),'ERROR : Operation not supported using provided arguments!');
  if BETA=0 then
    FillChar(C^, M*N*sizeOf(QNNFloat), 0)
  else if BETA<>1 then
    cblas_sscal(M*N, BETA, C, 1);

  if transB=CblasNoTrans then
    for mm :=0 to M-1 do begin
      CC := C + mm*ldc;
      for kk := 0 to K-1 do begin
        a_part := ALPHA*A[mm*lda + kk];
        for nn := 0 to N-1 do begin
            CC[nn] := CC[nn] + a_part*B[kk*ldb + nn];
        end;
      end
    end
  else
    for mm:=0 to M-1 do begin
      CC := C + mm*ldc;
      AA := A + mm*lda;
      for nn := 0 to N-1 do begin
          BB := B + nn*ldb;
          a_part := 0.0;
          for kk:=0 to K-1 do begin
              a_part := a_part + AA[kk]*BB[kk];
          end;
          CC[nn] := ALPHA*a_part;
      end;
    end
end;

procedure QNNMatMulNN(const A, B, C: PQNNFloat; const M, N, K: integer);
begin
    cblas_sgemm(CBLASRowMajor, CBLASNoTrans, CBLASNoTrans
    , M, N, K, 1
    , A, K
    , B, N
    , 0
    , C, N);
end;

procedure QNNMatMulNT(const A, B, C: PQNNFloat; const M, N, K: integer);
begin
    cblas_sgemm(CBLASRowMajor, CblasNoTrans, CBLASTrans
    , M, N, K, 1
    , A, K
    , B, K
    , 0
    , C, N);
end;

procedure QNNLinear(const dst, x, W, b: PQNNFloat; const seqLen, inDIM, outDIM: integer);
var i: integer;
begin
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
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
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              seqLen, outDIM, inDIM,
              1.0, x, inDIM, W, inDIM,
              0.0, dst, outDIM);
end;

procedure QNNLinearNoBias_BF16(const dst, x: PQNNFloat; const W: PBF16; const seqLen, inDIM, outDIM: integer);
begin
  if length(workspace) < inDIM*outDIM then
    setLength(workspace, inDIM*outDIM);
  BF16ToSingle(inDIM*outDIM, W, pointer(workspace));
  QNNLinearNoBias(dst, x, pointer(workspace), seqLen, inDIM, outDIM);
end;

procedure qim2Col(const aChannels, aHeight, aWidth: blasint;
  const kernelHeight, kernelWidth, padHeight, padWidth, strideY,
  strideX, dilationY, dilationX: blasint; const im: PQNNFloat;
  const imOffset: blasint; const col: PQNNFloat; const colOffset: blasint;
  const MultiThread: boolean = False);
var
  channel, output_h, output_w, channel_size, out_channel_size, kernel_size: SizeInt;
  {$ifdef FPC}
  procedure i2c_ext(idx:IntPtr; ptr:Pointer);
  {$else}
  i2c_ext: TThreadProcNested;
begin
  //{$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2colExt);{$endif}
  i2c_ext := procedure(idx: IntPtr; ptr: Pointer)
  {$endif}
  var
    kernel_row, kernel_col, output_col, output_row, input_row, input_col, i: SizeInt;
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
            if {(input_row>=0) and} (SizeUInt(input_row) < SizeUInt(aHeight)) then begin
              input_col := -padWidth + kernel_col * dilationX;
              if (strideX=1) then begin
                fillchar(d_col[0], output_w*SizeOf(d_col^), #0);
                move(d_im[input_row*aWidth + ord(input_col>=0)*input_col], d_col[ord(input_col<0)*abs(input_col)], (output_w-abs(input_col))*sizeof(d_col^));
              end
              else
                for output_col:=0 to output_w-1 do  begin
                  i := output_col*strideX + input_col;
                  d_col[output_col] := ord(SizeUInt(i) < SizeUInt(aWidth))*d_im[input_row*aWidth + i];
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

procedure qim2colStridedBatched(
  const aChannels, aHeight, aWidth: Sizeint;
  const kernelHeight, kernelWidth, padHeight, padWidth,
  strideY, strideX, dilationY, dilationX: SizeInt;
  const im: PSingle; const imStride, imOffset: SizeInt;
  const col: PSingle; const colStride, colOffset: SizeInt;
  const batchCount:SizeInt);
var b : SizeInt; mt:boolean;

{$ifdef FPC}
  procedure i2c(idx:IntPtr; ptr:Pointer);
{$else}
  i2c : TThreadProcNested;
begin
  i2c := procedure (idx:IntPtr; ptr:Pointer)
{$endif}
  begin
    qim2Col(
      aChannels, aHeight, aWidth,
      kernelHeight, kernelWidth, padHeight, padWidth,
      strideY, strideX, dilationY, dilationX, im+idx*imStride, imOffset, col+idx*colStride, colOffset, mt);
  end;
{$ifdef FPC}
begin
{$endif}
  mt := batchCount=1;
  {$ifdef USE_MULTITHREADING}
  if not mt then
    mp.&For(i2c, 0, batchCount, nil)
  else
  {$endif}
  for b:=0 to batchCount-1 do
    qim2Col(
      aChannels, aHeight, aWidth,
      kernelHeight, kernelWidth, padHeight, padWidth,
      strideY, strideX, dilationY, dilationX,
      im + b*imStride, imOffset,
      col + b*colStride, colOffset,
      mt);
end;

procedure QNNConv2d(const dst, src, weights, bias: PQNNFloat; const batch,
  in_ch, out_ch, H, W, kH, kW, stride, padding: integer);
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
      qIm2colStridedBatched(
        in_ch {or AKernels.c()}, h, w, kH, kW,
        Padding, Padding, Stride, Stride, 1 {xDilation}, 1{yDilation},
        src, inVolume, 0,
        workspacePtr, strideB, 0, batch);
  end else begin
      strideB := inVolume;
      workspacePtr := src;
  end;

  for b := 0 to batch - 1 do
  begin
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     filters, outImgSize, k, 1,
                     weights + b*strideA, k,
                     workspacePtr + b * strideB, outImgSize,
                     0, dst + b * strideC, outImgSize);
  end;

end;

procedure QNNRMSNorm(const dst, src, weight: PQNNFloat; const seqLen, hidden: integer);
var s, i: integer;
   sq_sum, rms, rms_inv:QNNFloat;
   x_row, out_row : PQNNFloat;
begin
  // todo RMS_NORM Simdify
  for s :=0 to seqLen-1 do begin
      x_row := src + s * hidden;
      out_row := dst + s * hidden;

      (* Compute RMS *)
      sq_sum := 0.0;
      for i := 0 to hidden-1 do
          sq_sum := sq_sum + sqr(x_row[i]);

      rms := sqrt(sq_sum / hidden + eps);
      rms_inv := 1.0 / rms;

      (* Normalize and scale *)
      for i := 0 to hidden-1 do
          out_row[i] := x_row[i] * rms_inv * weight[i];
  end
end;

procedure QNNGroupNorm(const dst, src, gamma, beta : PQNNFloat;
                     const batch, channels, H, W, num_groups:integer);
var
  channels_per_group, spatial, b, g
  , count, c_start, c_end, c, idx, i: integer;
  mean, diff, variance, std_inv, norm : QNNFloat;
begin
    channels_per_group := channels div num_groups;
    spatial := H * W;
    // todo group norm simdify
    for b := 0 to batch-1 do begin
        for g := 0 to num_groups-1 do begin
            c_start := g * channels_per_group;
            c_end := c_start + channels_per_group;

            mean := 0.0;
            count := 0;
            for c := c_start to c_end-1 do begin
                for i := 0 to spatial-1 do begin
                    idx := b * channels * spatial + c * spatial + i;
                    mean := mean + src[idx];
                    inc(count);
                end
            end;
            mean := mean/count;

            variance := 0.0;
            for c := c_start to c_end-1 do begin
                for i := 0 to spatial-1 do begin
                    idx := b * channels * spatial + c * spatial + i;
                    diff := src[idx] - mean;
                    variance := variance + diff * diff;
                end
            end;
            variance := variance / count;

            std_inv := 1.0 / sqrt(variance + eps);

            for c := c_start to c_end-1 do begin
                for i := 0 to spatial-1 do begin
                    idx := b * channels * spatial + c * spatial + i;
                    norm := (src[idx] - mean) * std_inv;
                    dst[idx] := gamma[c] * norm + beta[c];
                end
            end
        end
    end
end;

procedure QNNBatchNorm(const dst, src,
                     running_mean, running_var, gamma, beta : PQNNFloat;
                     const batch, channels, H, W: integer);
var
  spatial, c, n, i, idx : integer;
  mean, variance, std_inv, g, b_val:QNNFloat;
begin
  // todo BatchNorm simdify
    spatial := H * W;
    for c := 0 to channels-1 do begin
        mean := running_mean[c];
        variance := running_var[c];
        std_inv := 1.0 / sqrt(variance + eps);
        if assigned(gamma) then g :=  gamma[c] else g := 1.0;
        if assigned(beta) then b_val := beta[c] else b_val := 0.0;

        for n := 0 to batch-1 do begin
            for i := 0 to spatial-1 do begin
                idx := n * channels * spatial + c * spatial + i;
                dst[idx] := g * (src[idx] - mean) * std_inv + b_val;
            end
        end
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
            q_row := Q+i * head_dim;
            o_row := dst+i * head_dim;
            max_score := -1;
            sum_exp := 0.0;
            for d := 0 to head_dim -1 do
                o_row[d] := 0.0;
            for j := 0 to seq_k -1 do
                begin
                    k_row := K+j * head_dim;
                    v_row := V+j * head_dim;
                    score := 0.0;
                    for d := 0 to head_dim -1 do
                        score := score + (q_row[d] * k_row[d]);
                    score := score * scale;
                    if score > max_score then
                        begin
                            correction := fast_expf(max_score-score);
                            sum_exp := sum_exp * correction+1.0;
                            for d := 0 to head_dim -1 do
                                o_row[d] := o_row[d] * correction+v_row[d];
                            max_score := score
                        end
                    else
                        begin
                            weight := fast_expf(score-max_score);
                            sum_exp := sum_exp + weight;
                            for d := 0 to head_dim -1 do
                                o_row[d] := o_row[d] + (weight * v_row[d])
                        end
                end;
            inv_sum := 1.0 / sum_exp;
            for d := 0 to head_dim -1 do
                o_row[d] := o_row[d] * inv_sum
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

    fillchar(dst^, seq_q * head_dim * sizeof(QNNFloat), 0);

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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_len, k_len, head_dim, scale, Q_tile, head_dim, K_tile, head_dim, 0.0, tile_scores, k_tile_size);
            for qi := 0 to q_len -1 do
                begin
                    i := q_start+qi;
                    score_row := tile_scores+qi * k_tile_size;
                    o_row := out_tile+qi * head_dim;
                    tile_max := score_row[0];
                    for ki := 1 to k_len -1 do
                        if score_row[ki] > tile_max then
                            tile_max := score_row[ki];
                    old_max := max_scores[i];
                    if (tile_max > old_max) then
                        new_max := tile_max
                    else
                        new_max := old_max;
                    if old_max > -1 then
                        begin
                            correction := fast_expf(old_max-new_max);
                            sum_exps[i] := sum_exps[i] * correction;
                            for d := 0 to head_dim -1 do
                                o_row[d] := o_row[d] * correction
                        end;
                    for ki := 0 to k_len -1 do
                        begin
                            weight := fast_expf(score_row[ki]-new_max);
                            sum_exps[i] := sum_exps[i] + weight;
                            v_row := V_tile+ki * head_dim;
                            for d := 0 to head_dim -1 do
                                o_row[d] := o_row[d] + (weight * v_row[d])
                        end;
                    max_scores[i] := new_max
                end;
            q_start := q_start + q_tile_size
        end;
        k_start := k_start + k_tile_size
    end;
    for i := 0 to seq_q -1 do
        begin
            inv_sum := 1.0 / sum_exps[i];
            o_row := dst+i * head_dim;
            for d := 0 to head_dim -1 do
                o_row[d] := o_row[d] * inv_sum
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
begin
    //tile_scores := PQNNFloat(malloc(q_tile_size * k_tile_size * sizeof(float)));

    if length(workspace) < 2*head_dim*(seq_k+seq_q) then
      setLength(workspace, 2*head_dim*(seq_k+seq_q));

    off :=0;
    Q_contig   := @workspace[off]; inc(off, seq_q * head_dim);
    K_contig   := @workspace[off]; inc(off, seq_k * head_dim);
    V_contig   := @workspace[off]; inc(off, seq_k * head_dim);
    out_contig := @workspace[off];

    for h := 0 to heads -1 do
        begin
            Q_head := Q+h * head_dim;
            K_head := K+h * head_dim;
            V_head := V+h * head_dim;
            out_head := dst+h * head_dim;
            hidden := heads * head_dim;
            if (seq_q <= 64) and (seq_k <= 128) then
                begin
                    for i := 0 to seq_q -1 do
                        for d := 0 to head_dim -1 do
                            Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
                    for j := 0 to seq_k -1 do
                        for d := 0 to head_dim -1 do
                            begin
                                K_contig[j * head_dim+d] := K_head[j * hidden+d];
                                V_contig[j * head_dim+d] := V_head[j * hidden+d]
                            end;
                    QNNFlashAttentionHead(out_contig, Q_contig, K_contig, V_contig, seq_q, seq_k, head_dim, scale);
                    for i := 0 to seq_q -1 do
                        for d := 0 to head_dim -1 do
                            out_head[i * hidden+d] := out_contig[i * head_dim+d];
                end
            else
                begin
                    for i := 0 to seq_q -1 do
                        for d := 0 to head_dim -1 do
                            Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
                    for j := 0 to seq_k -1 do
                        for d := 0 to head_dim -1 do
                            begin
                                K_contig[j * head_dim+d] := K_head[j * hidden+d];
                                V_contig[j * head_dim+d] := V_head[j * hidden+d]
                            end;
                    QNNFlashAttentionHeadTiled(out_contig, Q_contig, K_contig, V_contig, seq_q, seq_k, head_dim, scale, tile_scores, q_tile_size, k_tile_size);
                    for i := 0 to seq_q -1 do
                        for d := 0 to head_dim -1 do
                            out_head[i * hidden+d] := out_contig[i * head_dim+d];
                end
        end;
end;

(* Apply precomputed RoPE (Rotary Position Embedding) in-place using the
 * split-half convention: dim d pairs with dim d+half for rotation. This is
 * the Flux convention (4-axis, split-half); Z-Image uses consecutive pairs
 * via a separate kernel. RoPE lets the transformer learn relative position
 * from the dot-product structure of Q and K. *)
procedure QNNApplyROPE(const dst, freqs: PQNNFloat; const batch, seq, heads, head_dim: longint);
var
    vec: PQNNFloat;
    half_dim, b, s, h, d: longint;
    cos_val, sin_val: QNNFloat;
    x0, x1: QNNFloat;
begin
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

procedure QNNComputeROPEFreqs(freqs: PQNNFloat; const pos: Plongint; seq: longint; dim: longint; theta: QNNFloat);
var
    half_dim, s, d: longint;
    p, freq, angle: QNNFloat;
begin
    half_dim := dim div 2;
    for s := 0 to seq -1 do
        begin
            p := single(pos[s]);
            for d := 0 to half_dim -1 do
                begin
                    freq := 1.0 / power(theta, (2 * d) / dim);
                    angle := p * freq;
                    freqs[s * half_dim * 2+d * 2] := cos(angle);
                    freqs[s * half_dim * 2+d * 2+1] := sin(angle)
                end
        end
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
                                in_c := c * p * p+pi * p+pj;
                                in_idx := b * in_ch * H * W+in_c * H * W+ph * W+pw;
                                oh := ph * p+pi;
                                ow := pw * p+pj;
                                out_idx := b * channels * outH * outW+c * outH * outW+oh * outW+ow;
                                dst[out_idx] := src[in_idx]
                            end
end;

procedure QNNCopy(const dst, src:PQNNFloat; const N:integer);
begin
  move(src^, dst^, N*sizeOf(QNNFloat))
end;

const
  {$if defined(MSWINDOWS)}
  blaslib = 'openblas.dll';
  {$elseif defined(LINUX)}
  blaslib = 'libopenblas.so';
  {$elseif defined(MACOS) or defined(DARWIN)}
  {$endif}

initialization
  cblas_saxpy := qaxpy;
  cblas_sdot := qdot;
  cblas_sscal := qscale;
  cblas_sgemm := qgemm;

  hBLASLib := LoadLibrary(blaslib);
  {$ifdef MSWINDOWS}
  if hBLASLib=0 then
    hBLASLib:=LoadLibrary('lib'+blaslib);
  {$endif}

  {$if defined(MACOS) or defined(darwin)}
  {$else}
  if hBLASLib>0 then begin
    cblas_sgemm := getProcAddress(hBLASLib, 'cblas_sgemm');
    cblas_saxpy := getProcAddress(hBLASLib, 'cblas_saxpy');
    cblas_sdot  := getProcAddress(hBLASLib, 'cblas_sdot' );
    cblas_sasum := getProcAddress(hBLASLib, 'cblas_sasum');
    cblas_sscal := getProcAddress(hBLASLib, 'cblas_sscal');
  end;
  {$endif}

finalization

if hBLASLib>0 then begin
    UnloadLibrary(hBLASLib);
    cblas_sgemm := nil;
    cblas_saxpy := nil;
    cblas_sdot  := nil;
    cblas_sasum := nil;
    cblas_sscal := nil;
  end

end.

