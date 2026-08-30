unit quicknn_kernels;

{$ifdef FPC}
  {$PackRecords C}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$modeswitch nestedprocvars}
  {$ifdef CPUX64}
    {$asmmode intel}
    //{$FPUType AVX2}
  {$endif}
  //{$if defined(darwin)}
  //  {$LinkFramework accelerate}
  //{$endif}
{$endif}
{$ifdef CPUX64}

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
  {$elseif defined(DARWIN)}
  {$elseif defined(POSIX)}
  , unixbase
  {$endif}
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  , quicknn_common
  , sixel
  {$ifdef USE_PROFILING}
  , nOpMetrics
  {$endif}
  ;

{
  TQNNOps = class
    class procedure cblas_gemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint;
          alpha:single; A:TMemoryBlock; lda:blasint; B:TMemoryBlock; ldb:blasint; beta:single; C:TMemoryBlock; ldc:blasint);virtual;abstract;
    class procedure cblas_axpy(n:blasint; alpha:single; x:TMemoryBlock; incx:blasint; y:TMemoryBlock; incy:blasint);virtual;abstract;
    class function cblas_dot(n:blasint; x:TMemoryBlock; incx:blasint; y:TMemoryBlock; incy:blasint):single;virtual;abstract;
    class function cblas_asum(n:blasint; x:TMemoryBlock; incx:blasint):QNNFloat;virtual;abstract;
    class procedure cblas_scal(N:blasint; alpha:QNNFloat; X:TMemoryBlock; incX:blasint = 1);virtual;abstract;

  (* Global callback pointers - set by caller before inference *)
  // Add and Mul terms used for vector vector element wise operations
  // Scale and Bias terms used for vector Scalar element wise operations

    class procedure QNNAdd(const dst, a, b: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNAddInplace(const dst, a: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNMul(const dst, a, b: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNMulInplace(const dst, a: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNScale(const dst, src:TMemoryBlock; const aScale:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNScaleInplace(const dst: TMemoryBlock; const aScale: QNNFloat; const N: longint);virtual;abstract;
    class procedure QNNBias(const dst, src:TMemoryBlock; const aBias:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNBiasInplace(const dst:TMemoryBlock; const aBias:QNNFloat; const N:longint);virtual;abstract;


    class procedure QNNFusedMulAdd(const dst, src, srcM, srcA:TMemoryBlock; const N:longint);virtual;abstract;
    class procedure QNNFusedAddMul(const dst, src, srcA, srcM:TMemoryBlock; const N:longint);virtual;abstract;
    class procedure QNNFusedScaleBias(const dst, src:TMemoryBlock; const aScale, aBias:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNFusedBiasScale(const dst, src:TMemoryBlock; const aBias, aScale:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNFusedMulScale(const dst, src, srcM:TMemoryBlock; const aScale:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNFusedBiasMulScale(const dst, src, srcM:TMemoryBlock; const aBias, aScale:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNFusedScaleAdd(const dst, src, srcAdd:TMemoryBlock; const aScale:QNNFloat; const N:longint);virtual;abstract;
    class procedure QNNFusedBiasAdd(const dst, src, srcA:TMemoryBlock; const aBias:QNNFloat; const N:longint);virtual;abstract;
    class function QNNDot(const a, b:TMemoryBlock; const N:longint):QNNFloat;virtual;abstract;

    class procedure QNNAccAdd(const dst, a, b: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNAccMul(const dst, a, b: TMemoryBlock; const N:integer);virtual;abstract;

    class procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N:integer);virtual;abstract;
    class procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N:integer);virtual;abstract;
    class procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N:integer);virtual;abstract;

    class procedure QNNMatTranspose(const dst, src: TMemoryBlock; const srcRows, srcCols: longint);virtual;abstract;
    class procedure QNNLinear(const dst, x, W, B:TMemoryBlock; const seqLen, inDIM, outDIM: integer);virtual;abstract;
    class procedure QNNLinearNoBias(const dst, x, W:TMemoryBlock; const seqLen, inDIM, outDIM: integer);virtual;abstract;
    class procedure QNNLinearNoBias_BF16(const dst, x:TMemoryBlock; const W: PBF16; const seqLen, inDIM, outDIM: integer);virtual;abstract;
    class procedure QNNIm2Col(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: integer;
          const im: TMemoryBlock; const col: TMemoryBlock; const imOffset: integer =0 ; const colOffset: integer =0; const MultiThread: boolean = False);virtual;abstract;

    class procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: integer;
          const im: TMemoryBlock; const imStride, imOffset: integer; const col: TMemoryBlock; const colStride, colOffset: integer; const batch:integer);virtual;abstract;

    class procedure QNNConv2d( const dst, src, weights, bias:TMemoryBlock; const in_ch, out_ch, H, W, kH, kW, stride, padding:longint; const batch:longint =1);virtual;abstract;

    class procedure QNNNorm(const N:longint; const dst, src:TMemoryBlock; const weights:TMemoryBlock ; const stride:longint = 1);virtual;abstract;

    class procedure QNNRMSNorm(const N:longint; const dst, src:TMemoryBlock; const weights:TMemoryBlock ; const stride:longint = 1);virtual;abstract;
    class procedure QNNRMSNormRows(const dst, src, weight:TMemoryBlock; const rows, dim: integer);virtual;abstract;
    class procedure QNNTanh(const dst:TMemoryBlock; const N:longint);virtual;abstract;

    class procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights:TMemoryBlock; const seq, heads, headDim:longint);virtual;abstract;
    class procedure QNNRMSNormSeq(const dst, normWeights:TMemoryBlock; const seq, heads, headDim:longint);virtual;abstract;

    class procedure QNNGroupNorm(const dst, src, gamma, beta : TMemoryBlock; const batch, channels, H, W, num_groups:integer);virtual;abstract;
    class procedure QNNBatchNormGammaBeta(const dst, src, running_mean, running_var, gamma, beta : TMemoryBlock; const batch, channels, H, W: integer);virtual;abstract;
    class procedure QNNBatchNorm(const dst, src, running_mean, running_var : TMemoryBlock; const batch, channels, H, W: integer);virtual;abstract;

    class function QNNMinMax(const N:integer; const src:TMemoryBlock; out outMin, outMax: QNNFloat;
          const argMin: PInteger; const argMax:PInteger = nil; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNMax(const N:integer; const src:TMemoryBlock; const arg:PInteger = nil; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNMin(const N:integer; const src:TMemoryBlock; const arg:PInteger = nil; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNMaxAbs(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNMinAbs(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNArgMax(const N:integer; const src:TMemoryBlock; const stride:integer=1):integer;virtual;abstract;
    class function QNNArgMin(const N:integer; const src:TMemoryBlock; const stride:integer=1):integer;virtual;abstract;
    class function QNNSum(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNSumSqr(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNSumSqrDiff(const N: integer; const src:TMemoryBlock; const aMean:QNNFloat; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNSqrDistance(const N: integer; const src1, src2:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNMaxAbsDiff2(const N: integer; const src1, src2: TMemoryBlock; out outSrc1, outSrc2:QNNFloat; const stride: integer=1): QNNFloat;virtual;abstract;
    class function QNNMaxAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer=1): QNNFloat;virtual;abstract;
    class function QNNSumAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer=1): QNNFloat;virtual;abstract;
    class function QNNSumAbsDiffScalar(const N: integer; const src: TMemoryBlock; const aMean:QNNFloat; const stride: integer=1): QNNFloat;virtual;abstract;
    class function QNNMean(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;virtual;abstract;
    class function QNNVariance(const N:integer; const src:TMemoryBlock; const aMean:QNNFloat; const stride:integer=1; const isPopulation:boolean=true):QNNFloat;virtual;abstract;
    class procedure QNNSigmoidInplace(const x: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNSigmoid(const dst, src: TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNSiluInplace(const x:TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNSilu(const dst, src:TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNSiluMul(const gate, up:TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNSoftmax(const x: TMemoryBlock; const N: integer);virtual;abstract;
    class procedure QNNSoftmaxRows(const x: TMemoryBlock; const rows, cols: integer);virtual;abstract;
    class procedure QNNAttention(const dst, Q, K, V: TMemoryBlock; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);virtual;abstract;
    class procedure QNNFlashAttentionHead(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);virtual;abstract;
    class procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint;
          const scale: QNNFloat; const tile_scores: TMemoryBlock; const q_tile_size, k_tile_size: longint);virtual;abstract;
    class procedure QNNFlashAttention(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);virtual;abstract;
    class procedure QNNUpSampleNearest(const dst, src: TMemoryBlock; const batch, channels, H, W, scale_h, scale_w: longint);virtual;abstract;
    class procedure QNNUpSample(const dst, src: TMemoryBlock; const batch, channels, H, W:longint; const scale_h, scale_w: QNNFloat;
          const interpolation:TInterpolation = iNearest);virtual;abstract;
    class procedure QNNPatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);virtual;abstract;
    class procedure QNNUnpatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);virtual;abstract;
    class procedure QNNCopy(const dst, src:TMemoryBlock; const N:integer);virtual;abstract;
    class procedure QNNCopyStrided(const dst:TMemoryBlock; const dstStride:integer; const src:TMemoryBlock; const srcStride: integer; const N:integer);virtual;abstract;
    class procedure QNNFill(const dst:TMemoryBlock; const val:QNNFloat; const N:integer; const stride:integer=1);virtual;abstract;
    class procedure QNNBroadcast(const dst:pointer; const src; const srcSize, N:integer; const stride:integer=1);virtual;abstract;
    class procedure QNNGatedAdd(const dst, gate, proj:TMemoryBlock; const seq, hidden:integer);virtual;abstract;
    class procedure QNNComputeRoPE(const dst:TMemoryBlock; const maxSeq, dim:longint; const theta:QNNFloat);virtual;abstract;
    class procedure QNNComputeRoPE2(const cosOut,sinOut:TMemoryBlock; const maxSeqLen, headDim:longint; const theta:QNNFloat);virtual;abstract;
    class procedure QNNComputeRoPE2D(const cosOut,sinDst:TMemoryBlock; const patch_h, patch_w, dim:longint; const theta:QNNFloat);virtual;abstract;
    class procedure QNNComputeRoPE2DOffset(const cosOut, sinOut:TMemoryBlock; const patch_h, patch_w, dim:longint; const theta:QNNFloat; const offset_t:longint);virtual;abstract;
    class procedure QNNComputeRoPEText(const cosOut, sinOut: TMemoryBlock; const txt_seq, axis_dim: longint; const theta: QNNFloat);virtual;abstract;
    class procedure QNNApplyRoPE(const dst, freqs: TMemoryBlock; const batch, seq, heads, head_dim: longint);virtual;abstract;
    class procedure QNNApplyRoPE2(const dst, cosIn, sinIn: TMemoryBlock; const numHeads, headDim: longint);virtual;abstract;
    class procedure QNNApplyRoPE3(const dst, cosIn, sinIn: TMemoryBlock; const seqLen, numHeads, headDim: longint);virtual;abstract;
    class procedure QNNApplyRoPEQK(const q, k, cos_cache, sin_cache:TMemoryBlock; const seq_len, num_q_heads, num_kv_heads, head_dim:longint);virtual;abstract;

    class procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq : TMemoryBlock; const seq, heads, head_dim, dim: longint);virtual;abstract;
  //Positional Embedding
    class function QNNTimeStepEmbedding(const t : QNNFloat;const dim:integer; const max_period:QNNFloat):TMemoryBlock;virtual;abstract;
    class procedure QNNMatTriangularFill(const dim:integer; const dst:TMemoryBlock; const val:QNNFloat; const mask:PLongint=nil);virtual;abstract;
    class procedure QNNMaskFill(const dst: TMemoryBlock; const mask:PLongint; const val:QNNFloat; const rows, cols:longint);virtual;abstract;

  // Adaptive Layer Normalization
    class procedure QNNAdaLN(const dst, src, bias, scale:TMemoryBlock; const seq, hidden:longint; const eps:QNNFloat=EPSILON);virtual;abstract;
  end;
}
var
  hBLASLib : HMODULE;
  workspace : TArray<QNNFloat>;

  {$if defined(USE_PROFILING)}
  metrics : TTensorMetrics;
  {$endif}

  procedure cblas_gemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint;
            alpha:single; A:TMemoryBlock; lda:blasint; B:TMemoryBlock; ldb:blasint; beta:single; C:TMemoryBlock; ldc:blasint); winapi ;
  procedure cblas_axpy(n:blasint; alpha:single; x:TMemoryBlock; incx:blasint; y:TMemoryBlock; incy:blasint); winapi ;
  function cblas_dot(n:blasint; x:TMemoryBlock; incx:blasint; y:TMemoryBlock; incy:blasint):single; winapi ;
  function cblas_asum(n:blasint; x:TMemoryBlock; incx:blasint):QNNFloat; winapi ;
  procedure cblas_scal(N:blasint; alpha:QNNFloat; X:TMemoryBlock; incX:blasint = 1); winapi ;

(* Global callback pointers - set by caller before inference *)
// Add and Mul terms used for vector vector element wise operations
// Scale and Bias terms used for vector Scalar element wise operations

  procedure QNNAdd(const dst, a, b: TMemoryBlock; const N:integer);
  procedure QNNAddInplace(const dst, a: TMemoryBlock; const N:integer);
  procedure QNNMul(const dst, a, b: TMemoryBlock; const N:integer);
  procedure QNNMulInplace(const dst, a: TMemoryBlock; const N:integer);
  procedure QNNScale(const dst, src:TMemoryBlock; const aScale:QNNFloat; const N:longint);
  procedure QNNScaleInplace(const dst: TMemoryBlock; const aScale: QNNFloat; const N: longint);
  procedure QNNBias(const dst, src:TMemoryBlock; const aBias:QNNFloat; const N:longint);
  procedure QNNBiasInplace(const dst:TMemoryBlock; const aBias:QNNFloat; const N:longint);


  procedure QNNFusedMulAdd(const dst, src, srcM, srcA:TMemoryBlock; const N:longint);
  procedure QNNFusedAddMul(const dst, src, srcA, srcM:TMemoryBlock; const N:longint);
  procedure QNNFusedScaleBias(const dst, src:TMemoryBlock; const aScale, aBias:QNNFloat; const N:longint);
  procedure QNNFusedBiasScale(const dst, src:TMemoryBlock; const aBias, aScale:QNNFloat; const N:longint);
  procedure QNNFusedMulScale(const dst, src, srcM:TMemoryBlock; const aScale:QNNFloat; const N:longint);
  procedure QNNFusedBiasMulScale(const dst, src, srcM:TMemoryBlock; const aBias, aScale:QNNFloat; const N:longint);
  procedure QNNFusedScaleAdd(const dst, src, srcAdd:TMemoryBlock; const aScale:QNNFloat; const N:longint);
  procedure QNNFusedBiasAdd(const dst, src, srcA:TMemoryBlock; const aBias:QNNFloat; const N:longint);
  function QNNDot(const a, b:TMemoryBlock; const N:longint):QNNFloat;

  procedure QNNAccAdd(const dst, a, b: TMemoryBlock; const N:integer);
  procedure QNNAccMul(const dst, a, b: TMemoryBlock; const N:integer);

  procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N:integer);
  procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N:integer);
  procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N:integer);

  procedure QNNMatTranspose(const dst, src: TMemoryBlock; const srcRows, srcCols: longint);
  procedure QNNLinear(const dst, x, W, B:TMemoryBlock; const seqLen, inDIM, outDIM: integer);
  procedure QNNLinearNoBias(const dst, x, W:TMemoryBlock; const seqLen, inDIM, outDIM: integer);
  procedure QNNLinearNoBias_BF16(const dst, x:TMemoryBlock; const W: PBF16; const seqLen, inDIM, outDIM: integer);
  procedure QNNIm2Col(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth
                        , padHeight, padWidth
                        , strideY, strideX
                        , dilationY, dilationX: integer
                        ; const im: TMemoryBlock; const col: TMemoryBlock
                        ; const imOffset: integer =0 ; const colOffset: integer =0
                        ; const MultiThread: boolean = False);

  procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth
                        , kernelHeight, kernelWidth, padHeight, padWidth
                        , strideY, strideX, dilationY, dilationX: integer
                        ; const im: TMemoryBlock; const imStride, imOffset: integer
                        ; const col: TMemoryBlock; const colStride, colOffset: integer
                        ; const batch:integer);

  function QNNConv2DShape(const batch, filters, srcH, srcW, kH, kW, padding, stride:int64):TArray<int64>;

  procedure QNNConv2d(var dst:TMemoryBlock; const src, weights, bias:TMemoryBlock;
                     const in_ch, out_ch, H, W, kH, kW, stride, padding:longint; const batch:longint =1);

  procedure QNNNorm(const N:longint; const dst, src:TMemoryBlock; const weights:TMemoryBlock ; const stride:longint = 1);

  procedure QNNRMSNorm(const N:longint; const dst, src:TMemoryBlock; const weights:TMemoryBlock ; const stride:longint = 1);
  procedure QNNRMSNormRows(const dst, src, weight:TMemoryBlock; const rows, dim: integer);
  procedure QNNTanh(const dst:TMemoryBlock; const N:longint);

  procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights:TMemoryBlock; const seq, heads, headDim:longint);
  procedure QNNRMSNormSeq(const dst, normWeights:TMemoryBlock; const seq, heads, headDim:longint);

  procedure QNNGroupNorm(const dst, src, gamma, beta : TMemoryBlock; const batch, channels, H, W, num_groups:integer);
  procedure QNNBatchNormGammaBeta(const dst, src, running_mean, running_var, gamma, beta : TMemoryBlock; const batch, channels, H, W: integer);
  procedure QNNBatchNorm(const dst, src, running_mean, running_var : TMemoryBlock; const batch, channels, H, W: integer);

  procedure QNNMinMax(const N:integer; const src:TMemoryBlock; out outMin, outMax: QNNFloat; const argMin: PInteger; const argMax:PInteger = nil; const stride:integer=1);
  function QNNMax(const N:integer; const src:TMemoryBlock; const arg:PInteger = nil; const stride:integer=1):QNNFloat;
  function QNNMin(const N:integer; const src:TMemoryBlock; const arg:PInteger = nil; const stride:integer=1):QNNFloat;
  function QNNMaxAbs(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNMinAbs(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNArgMax(const N:integer; const src:TMemoryBlock; const stride:integer=1):integer;
  function QNNArgMin(const N:integer; const src:TMemoryBlock; const stride:integer=1):integer;
  function QNNSum(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNSumSqr(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNSumSqrDiff(const N: integer; const src:TMemoryBlock; const aMean:QNNFloat; const stride:integer=1):QNNFloat;
  function QNNSqrDistance(const N: integer; const src1, src2:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNMaxAbsDiff2(const N: integer; const src1, src2: TMemoryBlock; out outSrc1, outSrc2:QNNFloat; const stride: integer=1): QNNFloat;
  function QNNMaxAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer=1): QNNFloat;
  function QNNSumAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer=1): QNNFloat;
  function QNNSumAbsDiffScalar(const N: integer; const src: TMemoryBlock; const aMean:QNNFloat; const stride: integer=1): QNNFloat;
  function QNNMean(const N:integer; const src:TMemoryBlock; const stride:integer=1):QNNFloat;
  function QNNVariance(const N:integer; const src:TMemoryBlock; const aMean:QNNFloat; const stride:integer=1; const isPopulation:boolean=true):QNNFloat;
  procedure QNNSigmoidInplace(const x: TMemoryBlock; const N:integer);
  procedure QNNSigmoid(const dst, src: TMemoryBlock; const N:integer);
  procedure QNNSiluInplace(const x:TMemoryBlock; const N:integer);
  procedure QNNSilu(const dst, src:TMemoryBlock; const N:integer);
  procedure QNNSiluMul(const gate, up:TMemoryBlock; const N:integer);
  procedure QNNSoftmax(const x: TMemoryBlock; const N: integer);
  procedure QNNSoftmaxRows(const x: TMemoryBlock; const rows, cols: integer);
  procedure QNNAttention(const dst, Q, K, V: TMemoryBlock; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
  procedure QNNFlashAttentionHead(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
  procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint;
                     const scale: QNNFloat; const tile_scores: TMemoryBlock; const q_tile_size, k_tile_size: longint);
  procedure QNNFlashAttention(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
  procedure QNNUpSampleNearest(const dst, src: TMemoryBlock; const batch, channels, H, W, scale_h, scale_w: longint);
  procedure QNNUpSample(const dst, src: TMemoryBlock; const batch, channels, H, W:longint; const scale_h, scale_w: QNNFloat; const interpolation:TInterpolation = iNearest);
  procedure QNNPatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);
  procedure QNNUnpatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);
  procedure QNNPermut(var dst:TMemoryBlock; const src:TMemoryBlock; const axis:TArray<byte>);   overload;
  function QNNPermut(const src:TMemoryBlock; const axis:TArray<byte>):TMemoryBlock;             overload;
  procedure QNNCopy(const dst, src:TMemoryBlock; const N:integer);
  procedure QNNCopyStrided(const dst:TMemoryBlock; const dstStride:integer; const src:TMemoryBlock; const srcStride: integer; const N:integer);
  procedure QNNFill(const dst:TMemoryBlock; const val:QNNFloat; const N:integer; const stride:integer=1);
  procedure QNNBroadcast(const dst:pointer; const src; const srcSize, N:integer; const stride:integer=1);
  procedure QNNGatedAdd(const dst, gate, proj:TMemoryBlock; const seq, hidden:integer);
  procedure QNNComputeRoPE(const dst:TMemoryBlock; const maxSeq, dim:longint; const theta:QNNFloat);
  procedure QNNComputeRoPE2(const cosOut,sinOut:TMemoryBlock; const maxSeqLen, headDim:longint; const theta:QNNFloat);
  procedure QNNComputeRoPE2D(const cosOut,sinDst:TMemoryBlock; const patch_h, patch_w, dim:longint; const theta:QNNFloat);
  procedure QNNComputeRoPE2DOffset(const cosOut, sinOut:TMemoryBlock; const patch_h, patch_w, dim:longint; const theta:QNNFloat; const offset_t:longint);
  procedure QNNComputeRoPEText(const cosOut, sinOut: TMemoryBlock; const txt_seq, axis_dim: longint; const theta: QNNFloat);
  procedure QNNApplyRoPE(const dst, freqs: TMemoryBlock; const batch, seq, heads, head_dim: longint);
  procedure QNNApplyRoPE2(const dst, cosIn, sinIn: TMemoryBlock; const numHeads, headDim: longint);
  procedure QNNApplyRoPE3(const dst, cosIn, sinIn: TMemoryBlock; const seqLen, numHeads, headDim: longint);
  procedure QNNApplyRoPEQK(const q, k, cos_cache, sin_cache:TMemoryBlock; const seq_len, num_q_heads, num_kv_heads, head_dim:longint);

  procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq : TMemoryBlock; const seq, heads, head_dim, dim: longint);
//Positional Embedding
  function QNNTimeStepEmbedding(const t : QNNFloat;const dim:integer; const max_period:QNNFloat):TMemoryBlock;
  procedure QNNMatTriangularFill(const dim:integer; const dst:TMemoryBlock; const val:QNNFloat; const mask:PLongint=nil);
  procedure QNNMaskFill(const dst: TMemoryBlock; const mask:PLongint; const val:QNNFloat; const rows, cols:longint);

// Adaptive Layer Normalization
  procedure QNNAdaLN(const dst, src, bias, scale:TMemoryBlock; const seq, hidden:longint; const eps:QNNFloat=EPSILON);

  procedure QNNLINEAR_BF16_OR_F32(const dst, src:PSingle; const weight:TMemoryBlock; const weight_bf16:TMemoryBlock; const seq, srcDim, dstDim:integer);


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

function isUsingBlas():boolean;
(*
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
*)

implementation
uses quicknncpu;

const OP_NOT_IMPL= 'Operation is not implemented for this datatype!';

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


function isUsingBlas():boolean;
begin
  result := TQNNSingleOPS.isUsingBlas
end;

procedure OP_IMPL_FAIL();inline;
begin
  assert(false, OP_NOT_IMPL)
end;

procedure cblas_gemm(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
  TransB: CBLAS_TRANSPOSE; M: blasint; N: blasint; K: blasint; alpha: single;
  A: TMemoryBlock; lda: blasint; B: TMemoryBlock; ldb: blasint; beta: single;
  C: TMemoryBlock; ldc: blasint); winapi;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case C.DataType of
    dtF32 :
      TQNNSingleOPS.cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure cblas_axpy(n: blasint; alpha: single; x: TMemoryBlock; incx: blasint;
  y: TMemoryBlock; incy: blasint); winapi;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAxpy);
  {$endif}
  case x.DataType of
    dtF32 :
      TQNNSingleOPS.cblas_axpy(n, alpha, x, incx, y, incy);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAxpy);
  {$endif}
end;

function cblas_dot(n: blasint; x: TMemoryBlock; incx: blasint; y: TMemoryBlock; incy: blasint): single; winapi;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opDot);
  {$endif}
  case x.DataType of
    dtF32 :
      result := TQNNSingleOPS.cblas_dot(n, x, incx, y, incy);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opDot);
  {$endif}
end;

function cblas_asum(n: blasint; x: TMemoryBlock; incx: blasint): QNNFloat;
  winapi;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAbsSum);
  {$endif}
  case x.DataType of
    dtF32 :
      result := TQNNSingleOPS.cblas_asum(n, x, incx);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAbsSum);
  {$endif}
end;

procedure cblas_scal(N: blasint; alpha: QNNFloat; X: TMemoryBlock; incX: blasint); winapi;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMulvs);
  {$endif}
  case x.DataType of
    dtF32 :
      TQNNSingleOPS.cblas_scal(N, alpha, X, incX);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMulvs);
  {$endif}
end;

procedure QNNAdd(const dst, a, b: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAddvv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNAdd] : N is out of source tesnor bounds');
  assert(b.offset+N <= b.size, 'ERROR [QNNAdd] : N is out of destination tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNAdd] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAdd(dst, a, b, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAddvv);
  {$endif}
end;

procedure QNNAddInplace(const dst, a: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAddvv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNAddInplace] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNAddInplace] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAddInplace(dst, a, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAddvv);
  {$endif}
end;

procedure QNNMul(const dst, a, b: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMulvv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNMul] : N is out of source tesnor bounds');
  assert(b.offset+N <= b.size, 'ERROR [QNNMul] : N is out of destination tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNMul] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMul(dst, a, b, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMulvv);
  {$endif}
end;

procedure QNNMulInplace(const dst, a: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMulvv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNMulInplace] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNMulInplace] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMulInplace(dst, a, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMulvv);
  {$endif}
end;

procedure QNNScale(const dst, src: TMemoryBlock; const aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMulvs);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNScale] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNScale] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNScale(dst, src, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMulvs);
  {$endif}
end;

procedure QNNScaleInplace(const dst: TMemoryBlock; const aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMulvs);
  {$endif}
  assert(dst.offset+N <= dst.size, 'ERROR [QNNScaleInplace] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNScaleInplace(dst, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMulvs);
  {$endif}
end;

procedure QNNBias(const dst, src: TMemoryBlock; const aBias: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAddvs);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNBias] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNBias] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNBias(dst, src, aBias, N)
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAddvs);
  {$endif}
end;

procedure QNNBiasInplace(const dst: TMemoryBlock; const aBias: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAddvs);
  {$endif}
  assert(dst.offset+N <= dst.size, 'ERROR [QNNBiasInplace] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNBiasInplace(dst, aBias, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAddvs);
  {$endif}
end;

procedure QNNFusedMulAdd(const dst, src, srcM, srcA: TMemoryBlock; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavv);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNFusedMulAdd] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNFusedMulAdd] : N is out of destination tesnor bounds');
  assert(srcA.offset+N <= srcA.Size, 'ERROR [QNNFusedMulAdd] : N is out of sourceA tesnor bounds');
  assert(srcM.offset+N <= srcM.size, 'ERROR [QNNFusedMulAdd] : N is out of sourceM tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedMulAdd(dst, src, srcM, srcA, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavv);
  {$endif}
end;

procedure QNNFusedAddMul(const dst, src, srcA, srcM: TMemoryBlock; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavv);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNFusedAddMul] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNFusedAddMul] : N is out of destination tesnor bounds');
  assert(srcA.offset+N <= srcA.Size, 'ERROR [QNNFusedAddMul] : N is out of sourceA tesnor bounds');
  assert(srcM.offset+N <= srcM.size, 'ERROR [QNNFusedAddMul] : N is out of sourceM tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedAddMul(dst, src, srcA, srcM, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavv);
  {$endif}
end;

procedure QNNFusedScaleBias(const dst, src: TMemoryBlock; const aScale, aBias: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavss);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNFusedScaleBias] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNFusedScaleBias] : N is out of destination tesnor bounds');  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedScaleBias(dst, src, aScale, aBias, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavss);
  {$endif}
end;

procedure QNNFusedBiasScale(const dst, src: TMemoryBlock; const aBias, aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavss);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNFusedBiasScale] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNFusedBiasScale] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedBiasScale(dst, src, aBias, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavss);
  {$endif}
end;

procedure QNNFusedMulScale(const dst, src, srcM: TMemoryBlock; const aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavvs);
  {$endif}
  assert(src.offset+N  <= src.Size, 'ERROR [QNNFusedMulScale] : N is out of source tesnor bounds');
  assert(srcM.offset+N <= srcM.size, 'ERROR [QNNFusedMulScale] : N is out of sourceM tesnor bounds');
  assert(dst.offset+N  <= dst.size, 'ERROR [QNNFusedMulScale] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedMulScale(dst, src, srcM, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavvs);
  {$endif}
end;

procedure QNNFusedBiasMulScale(const dst, src, srcM: TMemoryBlock; const aBias, aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavss);
  {$endif}
  assert(src.offset+N  <= src.Size, 'ERROR [QNNFusedBiasMulScale] : N is out of source tesnor bounds');
  assert(srcM.offset+N <= srcM.size, 'ERROR [QNNFusedBiasMulScale] : N is out of sourceM tesnor bounds');
  assert(dst.offset+N  <= dst.size, 'ERROR [QNNFusedBiasMulScale] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedBiasMulScale(dst, src, srcM, aBias, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavss);
  {$endif}
end;

procedure QNNFusedScaleAdd(const dst, src, srcAdd: TMemoryBlock; const aScale: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavvs);
  {$endif}
  assert(src.offset+N  <= src.Size, 'ERROR [QNNFusedScaleAdd] : N is out of source tesnor bounds');
  assert(srcAdd.offset+N <= srcAdd.size, 'ERROR [QNNFusedScaleAdd] : N is out of dstAdd tesnor bounds');
  assert(dst.offset+N  <= dst.size, 'ERROR [QNNFusedScaleAdd] : N is out of destination tesnor bounds');

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedScaleAdd(dst, src, srcAdd, aScale, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavvs);
  {$endif}
end;

procedure QNNFusedBiasAdd(const dst, src, srcA: TMemoryBlock; const aBias: QNNFloat; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavvs);
  {$endif}
  assert(src.offset+N  <= src.Size, 'ERROR [QNNFusedBiasAdd] : N is out of source tesnor bounds');
  assert(srcA.offset+N <= srcA.size, 'ERROR [QNNFusedBiasAdd] : N is out of sourceAdd tesnor bounds');
  assert(dst.offset+N  <= dst.size, 'ERROR [QNNFusedBiasAdd] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFusedBiasAdd(dst, src, srcA, aBias, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavvs);
  {$endif}
end;

function QNNDot(const a, b: TMemoryBlock; const N: longint): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opDot);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNDot] : N is out of source tesnor bounds');
  assert(b.offset+N <= b.size, 'ERROR [QNNDot] : N is out of destination tesnor bounds');
  case a.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNDot(a, b, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opDot);
  {$endif}
end;

procedure QNNAccAdd(const dst, a, b: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNAccAdd] : N is out of source tesnor bounds');
  assert(b.offset+N <= b.size, 'ERROR [QNNAccAdd] : N is out of destination tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNAccAdd] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAccAdd(dst, a, b, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavv);
  {$endif}
end;

procedure QNNAccMul(const dst, a, b: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavv);
  {$endif}
  assert(a.offset+N <= a.Size, 'ERROR [QNNAccMul] : N is out of source tesnor bounds');
  assert(b.offset+N <= b.size, 'ERROR [QNNAccMul] : N is out of destination tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNAccMul] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAccMul(dst, a, b, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavv);
  {$endif}
end;

procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case C.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMatMulNN(C, A, B, M, K, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case C.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMatMulNT(C, A, B, M, K, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case C.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMatMulTN(C, A, B, M, K, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNMatTranspose(const dst, src: TMemoryBlock; const srcRows, srcCols: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opOther);
  {$endif}
  assert(src.offset + srcRows*srcCols<= src.size, 'QNNTranspose : src dimensions out of bounds.');
  assert(dst.offset + srcRows*srcCols<= dst.size, 'QNNTranspose : dst dimensions out of bounds.');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMatTranspose(dst, src, srcRows, srcCols);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opOther);
  {$endif}
end;

procedure QNNLinear(const dst, x, W, B: TMemoryBlock; const seqLen, inDIM, outDIM: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNLinear(dst, x, W, B, seqLen, inDIM, outDIM);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNLinearNoBias(const dst, x, W: TMemoryBlock; const seqLen, inDIM, outDIM: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNLinearNoBias(dst, x, W, seqLen, inDIM, outDIM);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNLinearNoBias_BF16(const dst, x: TMemoryBlock; const W: PBF16; const seqLen, inDIM, outDIM: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNLinearNoBias_BF16(dst, x, W, seqLen, inDIM, outDIM);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;

procedure QNNIm2Col(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: integer;
  const im: TMemoryBlock; const col: TMemoryBlock; const imOffset: integer; const colOffset: integer; const MultiThread: boolean);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opIm2col);
  {$endif}

  case col.DataType of
    dtF32 :
      TQNNSingleOPS.QNNIm2Col(aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth,
                                         strideY, strideX, dilationY, dilationX, im, col, imOffset, colOffset, MultiThread);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opIm2col);
  {$endif}
end;

procedure QNNIm2ColStridedBatched(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: integer;
  const im: TMemoryBlock; const imStride, imOffset: integer; const col: TMemoryBlock; const colStride, colOffset: integer; const batch: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opIm2ColExt);
  {$endif}

  case col.DataType of
    dtF32 :
      TQNNSingleOPS.QNNIm2ColStridedBatched(aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth,
         strideY, strideX, dilationY, dilationX, im, imStride, imOffset, col, colStride, colOffset, batch);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opIm2ColExt);
  {$endif}
end;

// calculate Conv2D target shape dimension from an input dimension
function QNNConv2DShape(const batch, filters, srcH, srcW, kH, kW, padding, stride:int64):TArray<int64>;
begin
  result := [batch, filters, (srcH + 2 * padding - kH) div stride + 1, (srcW + 2 * padding - kW) div stride + 1];
end;

const sz : int64 = 0;
procedure QNNConv2d(var dst: TMemoryBlock; const src, weights,
  bias: TMemoryBlock; const in_ch, out_ch, H, W, kH, kW, stride,
  padding: longint; const batch: longint);
var  dims:TArray<int64>;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opConv2D);
  {$endif}
  dims := QNNConv2DShape(batch, out_ch, H, W, kH, kW, padding, stride);
  assert(dst.offset + product(dims)<=dst.Size, 'ERROR [Conv2D] : Incorrect destination size');
  dst.reshape(dims);
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNConv2d(dst, src, weights, bias, in_ch, out_ch, H, W, kH, kW, stride, padding, batch);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opConv2D);
  {$endif}
end;


procedure QNNNorm(const N: longint; const dst, src: TMemoryBlock; const weights: TMemoryBlock; const stride: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opNormalize);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNNorm] : N is out of source tesnor bounds');
  assert(weights.offset+N <= weights.size, 'ERROR [QNNNorm] : N is out of weights tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNNorm] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNNorm(N, dst, src, weights, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opNormalize);
  {$endif}
end;

procedure QNNRMSNorm(const N: longint; const dst, src: TMemoryBlock; const weights: TMemoryBlock; const stride: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opRMSNorm);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNRMSNorm] : N is out of source tesnor bounds');
  assert(weights.offset+N <= weights.size, 'ERROR [QNNRMSNorm] : N is out of weights tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNRMSNorm] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNRMSNorm(N, dst, src, weights, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opRMSNorm);
  {$endif}
end;

procedure QNNRMSNormRows(const dst, src, weight: TMemoryBlock; const rows, dim: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opRMSNorm);
  {$endif}
  assert(src.offset+dim*rows <= src.Size, 'ERROR [QNNRMSNormRows] : out of source tesnor bounds');
  assert(weight.offset+dim <= weight.size, 'ERROR [QNNRMSNormRows] : dim is out of weight tesnor bounds');
  assert(dst.offset+dim*rows <= dst.size, 'ERROR [QNNRMSNormRows] : out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNRMSNormRows(dst, src, weight, rows, dim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opRMSNorm);
  {$endif}
end;

procedure QNNTanh(const dst: TMemoryBlock; const N: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opTanh);
  {$endif}
  assert(dst.offset+N <= dst.size, 'ERROR [QNNTanh] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNTanh(dst, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opTanh);
  {$endif}
end;

procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights: TMemoryBlock; const seq, heads, headDim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opRMSNorm);
  {$endif}
  assert(Q.offset+seq*heads*headDim <= Q.Size, 'ERROR [QNNQKRMSNorm] : out of Q tesnor bounds');
  assert(Qweights.offset+headDim <= Qweights.size, 'ERROR [QNNQKRMSNorm] : headDim is out of Qweights tesnor bounds');
  assert(K.offset+seq*heads*headDim <= K.Size, 'ERROR [QNNQKRMSNorm] : out of K tesnor bounds');
  assert(Kweights.offset+headDim <= Kweights.size, 'ERROR [QNNQKRMSNorm] : headDim is out of Kweights tesnor bounds');
  case Q.DataType of
    dtF32 :
      TQNNSingleOPS.QNNQKRMSNorm(Q, K, QWeights, KWeights, seq, heads, headDim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opRMSNorm);
  {$endif}
end;

procedure QNNRMSNormSeq(const dst, normWeights: TMemoryBlock; const seq, heads, headDim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opRMSNorm);
  {$endif}
  assert(normWeights.offset+headDim <= normWeights.size, 'ERROR [QNNRMSNormSeq] : headDim is out of weights tesnor bounds');
  assert(dst.offset+seq*heads*headDim <= dst.size, 'ERROR [QNNRMSNormSeq] : out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNRMSNormSeq(dst, normWeights, seq, heads, headDim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opRMSNorm);
  {$endif}
end;

procedure QNNGroupNorm(const dst, src, gamma, beta: TMemoryBlock; const batch, channels, H, W, num_groups: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGroupNorm);
  {$endif}
  assert(gamma.offset+channels <= gamma.size, 'ERROR [QNNGroupNorm] : out of gamma tesnor bounds');
  assert(beta.offset+channels <= beta.size, 'ERROR [QNNGroupNorm] : out of beta tesnor bounds');
  assert(src.offset+batch*channels*H*W <= src.size, 'ERROR [QNNGroupNorm] : out of source tesnor bounds');
  assert(dst.offset+batch*channels*H*W <= dst.size, 'ERROR [QNNGroupNorm] : out of destination tesnor bounds');

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNGroupNorm(dst, src, gamma, beta, batch, channels, H, W, num_groups)
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opGroupNorm);
  {$endif}
end;

procedure QNNBatchNormGammaBeta(const dst, src, running_mean, running_var, gamma, beta: TMemoryBlock; const batch, channels, H, W: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opBatchNorm);
  {$endif}
  assert(running_mean.offset+channels <= running_mean.size, 'ERROR [QNNBatchNormGammaBeta] : out of running_mean tesnor bounds');
  assert(running_var.offset+channels <=running_var.size, 'ERROR [QNNBatchNormGammaBeta] : out of running_var tesnor bounds');
  assert(gamma.offset+channels <= gamma.size, 'ERROR [QNNBatchNormGammaBeta] : out of gamma tesnor bounds');
  assert(beta.offset+channels <= beta.size, 'ERROR [QNNBatchNormGammaBeta] : out of beta tesnor bounds');
  assert(src.offset+batch*channels*H*W <= src.size, 'ERROR [QNNBatchNormGammaBeta] : out of source tesnor bounds');
  assert(dst.offset+batch*channels*H*W <= dst.size, 'ERROR [QNNBatchNormGammaBeta] : out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNBatchNorm(dst, src, running_mean, running_var, gamma, beta, batch, channels, H, W);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opBatchNorm);
  {$endif}
end;

procedure QNNBatchNorm(const dst, src, running_mean, running_var : TMemoryBlock; const batch, channels, H, W: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opBatchNorm);
  {$endif}
  assert(running_mean.offset+channels <= running_mean.size, 'ERROR [QNNBatchNorm] : out of running_mean tesnor bounds');
  assert(running_var.offset+channels <=running_var.size, 'ERROR [QNNBatchNorm] : out of running_var tesnor bounds');
  assert(src.offset+batch*channels*H*W <= src.size, 'ERROR [QNNBatchNorm] : out of source tesnor bounds');
  assert(dst.offset+batch*channels*H*W <= dst.size, 'ERROR [QNNBatchNorm] : out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNBatchNorm(dst, src, running_mean, running_var, nil, nil, batch, channels, H, W);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opBatchNorm);
  {$endif}
end;

procedure QNNMinMax(const N: integer; const src: TMemoryBlock; out outMin, outMax: QNNFloat; const argMin: PInteger; const argMax: PInteger; const stride: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMinAndMax);
  {$endif}

  case src.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMinMax(N, src, outMin, outMax, argMin, argMax, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMinAndMax);
  {$endif}
end;

function QNNMax(const N: integer; const src: TMemoryBlock; const arg: PInteger; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMax);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMax(N, src, arg, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMax);
  {$endif}
end;

function QNNMin(const N: integer; const src: TMemoryBlock; const arg: PInteger; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMin);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMin(N, src, arg, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMin);
  {$endif}
end;

function QNNMaxAbs(const N: integer; const src: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMaxAbs);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMaxAbs(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMaxAbs);
  {$endif}
end;

function QNNMinAbs(const N: integer; const src: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMinAbs);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMinAbs(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMinAbs);
  {$endif}
end;

function QNNArgMax(const N: integer; const src: TMemoryBlock; const stride: integer): integer;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opArgMax);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNArgMax(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opArgMax);
  {$endif}
end;

function QNNArgMin(const N: integer; const src: TMemoryBlock; const stride: integer): integer;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opArgMin);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNArgMin(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opArgMin);
  {$endif}
end;

function QNNSum(const N: integer; const src: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSum);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSum(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSum);
  {$endif}
end;

function QNNSumSqr(const N: integer; const src: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSumSqr);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSumSqr(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSumSqr);
  {$endif}
end;

function QNNSumSqrDiff(const N: integer; const src: TMemoryBlock; const aMean: QNNFloat; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSumSqrDiffScaler);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSumSqrDiff(N, src, aMean, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSumSqrDiffScaler);
  {$endif}
end;

function QNNSqrDistance(const N: integer; const src1, src2: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opDistance);
  {$endif}

  case src1.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSqrDistance(N, src1, src2, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opDistance);
  {$endif}
end;

function QNNMaxAbsDiff2(const N: integer; const src1, src2: TMemoryBlock; out outSrc1, outSrc2: QNNFloat; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMaxAbsDiff);
  {$endif}

  case src1.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMaxAbsDiff2(N, src1, src2, outSrc1, outSrc2, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMaxAbsDiff);
  {$endif}
end;

function QNNMaxAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMaxAbsDiff);
  {$endif}

  case src1.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMaxAbsDiff(N, src1, src2, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMaxAbsDiff);
  {$endif}
end;

function QNNSumAbsDiff(const N: integer; const src1, src2: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSumAbsDiff);
  {$endif}

  case src1.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSumAbsDiff(N, src1, src2, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSumAbsDiff);
  {$endif}
end;

function QNNSumAbsDiffScalar(const N: integer; const src: TMemoryBlock; const aMean: QNNFloat; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAbsDiffScaler);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNSumAbsDiffScalar(N, src, aMean, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAbsDiffScaler);
  {$endif}
end;

function QNNMean(const N: integer; const src: TMemoryBlock; const stride: integer): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMeans);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNMean(N, src, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMeans);
  {$endif}
end;

function QNNVariance(const N: integer; const src: TMemoryBlock; const aMean: QNNFloat; const stride: integer; const isPopulation: boolean): QNNFloat;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opVariances);
  {$endif}

  case src.DataType of
    dtF32 :
      result := TQNNSingleOPS.QNNVariance(N, src, aMean, stride, isPopulation);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opVariances);
  {$endif}
end;

procedure QNNSigmoidInplace(const x: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSigmoidInplace);
  {$endif}

  case x.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSigmoidInplace(x, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSigmoidInplace);
  {$endif}
end;

procedure QNNSigmoid(const dst, src: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSigmoid);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSigmoid(dst, src, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSigmoid);
  {$endif}
end;

procedure QNNSiluInplace(const x: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSiLUInplace);
  {$endif}

  case x.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSiluInplace(x, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSiLUInplace);
  {$endif}
end;

procedure QNNSilu(const dst, src: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSiLU);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSilu(dst, src, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSiLU);
  {$endif}
end;

procedure QNNSiluMul(const gate, up: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSiLUMul);
  {$endif}

  case gate.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSiluMul(gate, up, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSiLUMul);
  {$endif}
end;

procedure QNNSoftmax(const x: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSoftmax);
  {$endif}

  case x.DataType of
    dtF32 :
      TQNNSingleOPS.QNNSoftmax(x, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSoftmax);
  {$endif}
end;

procedure QNNSoftmaxRows(const x: TMemoryBlock; const rows, cols: integer);
var i:longint;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opSoftmax);
  {$endif}

  case x.DataType of
    dtF32 :
      //if x.offset=0 then
        TQNNSingleOPS.QNNSoftmaxRows(x, rows, cols)
      //else begin
      //  assert(x.size - x.offset>= rows*cols, 'QNNSoftmax : out of bounds');
      //  for i:=0 to Rows-1 do
      //    TQNNSingleOPs.QNNSoftmax(x + i*cols, cols);
      //end
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opSoftmax);
  {$endif}
end;

procedure QNNAttention(const dst, Q, K, V: TMemoryBlock; const batch, heads, seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAttention);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAttention(dst, Q, K, V, batch, heads, seq_q, seq_k, head_dim, scale);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAttention);
  {$endif}
end;

procedure QNNFlashAttentionHead(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint; const scale: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAttentionHead);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFlashAttentionHead(dst, Q, K, V, seq_q, seq_k, head_dim, scale);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAttentionHead);
  {$endif}
end;

procedure QNNFlashAttentionHeadTiled(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, head_dim: longint;
  const scale: QNNFloat; const tile_scores: TMemoryBlock; const q_tile_size, k_tile_size: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAttentionHeadTiled);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFlashAttentionHeadTiled(dst, Q, K, V, seq_q, seq_k, head_dim, scale, tile_scores, q_tile_size, k_tile_size);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAttentionHeadTiled);
  {$endif}
end;

procedure QNNFlashAttention(const dst, Q, K, V: TMemoryBlock; const seq_q, seq_k, heads, head_dim: longint; const scale: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFlashAttention);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFlashAttention(dst, Q, K, V, seq_q, seq_k, heads, head_dim, scale);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFlashAttention);
  {$endif}
end;

procedure QNNUpSampleNearest(const dst, src: TMemoryBlock; const batch, channels, H, W, scale_h, scale_w: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opUpSampleNearest);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNUpSampleNearest(dst, src, batch, channels, H, W, scale_h, scale_w);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opUpSampleNearest);
  {$endif}
end;

procedure QNNUpSample(const dst, src: TMemoryBlock; const batch, channels, H, W: longint; const scale_h, scale_w: QNNFloat; const interpolation: TInterpolation);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opUpSample);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNUpSample(dst, src, batch, channels, H, W, scale_h, scale_w, interpolation);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opUpSample);
  {$endif}
end;

procedure QNNPatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opPatchify);
  {$endif}
  assert(dst.size = src.size, 'ERROR [QNNPatchify] : Source and Destination sizes mismatch.');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNPatchify(dst, src, batch, channels, H, W, patch_size);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opPatchify);
  {$endif}
end;

procedure QNNUnpatchify(const dst, src: TMemoryBlock; const batch, channels, H, W, patch_size: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opUnPatchify);
  {$endif}
  assert(dst.size = src.size, 'ERROR [QNNUnPatchify] : Source and Destination sizes mismatch.');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNUnpatchify(dst, src, batch, channels, H, W, patch_size);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opUnPatchify);
  {$endif}
end;

procedure QNNPermut(var dst: TMemoryBlock; const src: TMemoryBlock; const axis: TArray<byte>);
var
  i: int64;
  dims : set of byte = [];
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opTranspose);
  {$endif}
  assert(length(axis)=length(src.shape), 'ERROR [QNNPermute] : Axis and Source dimensions dimensions must be the same.');
  if not dst.isAssigned() then begin
    dst := TMemoryBlock.Create(copy(src.shape), '');
  end else
    assert(length(dst.shape)=length(src.shape), 'ERROR [QNNPermute] : Destination and Source dimensions must be the same.');

  for i := 0 to High(axis) do begin
    assert(not(axis[i] in dims) and (axis[i]<length(axis)), 'ERROR [QNNPermute] : axis cannot be out of dimensions boundries or duplicated!');
    include(dims, axis[i]);
  end;

  case src.DataType of
    dtF32 :
      //AltPermut(dst, src, src.shape, axis);
      TQNNSingleOPS.QNNPermut(dst, src, axis, src.shape);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opTranspose);
  {$endif}
end;

function QNNPermut(const src: TMemoryBlock; const axis: TArray<byte>): TMemoryBlock;
begin
  result := nil;
  QNNPermut(result, src, axis)
end;

procedure QNNCopy(const dst, src: TMemoryBlock; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opCopy);
  {$endif}
  assert(src.offset+N <= src.Size, 'ERROR [QNNCopy] : N is out of source tesnor bounds');
  assert(dst.offset+N <= dst.size, 'ERROR [QNNCopy] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNCopy(dst, src, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opCopy);
  {$endif}
end;

procedure QNNCopyStrided(const dst: TMemoryBlock; const dstStride: integer; const src: TMemoryBlock; const srcStride: integer; const N: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opCopyStrided);
  {$endif}
  assert(src.offset+N*srcStride <= src.Size, 'ERROR [QNNCopyStrided] : N is out of source tesnor bounds');
  assert(dst.offset+N*dstStride <= dst.size, 'ERROR [QNNCopyStrided] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNCopyStrided(dst, dstStride, src, srcStride, N);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opCopyStrided);
  {$endif}
end;

procedure QNNFill(const dst: TMemoryBlock; const val: QNNFloat; const N: integer; const stride: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFill);
  {$endif}
  assert(dst.offset+N*stride <= dst.size, 'ERROR [QNNFill] : N is out of destination tesnor bounds');
  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNFill(dst, val, N, stride);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFill);
  {$endif}
end;

procedure QNNBroadcast(const dst: pointer; const src; const srcSize, N: integer; const stride: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opOther);
  {$endif}


    OP_IMPL_FAIL() ;
  {$if defined(USE_PROFILING)}
  metrics.finish(opOther);
  {$endif}
end;

procedure QNNGatedAdd(const dst, gate, proj: TMemoryBlock; const seq, hidden: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opFmavv);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNGatedAdd(dst, gate, proj, seq, hidden);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opFmavv);
  {$endif}
end;

procedure QNNComputeRoPE(const dst: TMemoryBlock; const maxSeq, dim: longint; const theta: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opROPE);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNComputeRoPE(dst, maxSeq, dim, theta);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opROPE);
  {$endif}
end;

procedure QNNComputeRoPE2(const cosOut, sinOut: TMemoryBlock; const maxSeqLen, headDim: longint; const theta: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opROPE2);
  {$endif}

  case cosOut.DataType of
    dtF32 :
      TQNNSingleOPS.QNNComputeRoPE(cosOut, sinOut, maxSeqLen, headDim, theta);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opROPE2);
  {$endif}
end;

procedure QNNComputeRoPE2D(const cosOut, sinDst: TMemoryBlock; const patch_h, patch_w, dim: longint; const theta: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opROPE2D);
  {$endif}

  case cosOut.DataType of
    dtF32 :
      TQNNSingleOPS.QNNComputeRoPE2D(cosOut, sinDst, patch_h, patch_w, dim, theta);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opROPE2D);
  {$endif}
end;

procedure QNNComputeRoPE2DOffset(const cosOut, sinOut: TMemoryBlock; const patch_h, patch_w, dim: longint; const theta: QNNFloat; const offset_t: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opROPE2DOffset);
  {$endif}

  case cosOut.DataType of
    dtF32 :
      TQNNSingleOPS.QNNComputeRoPE2DOffset(cosOut, sinOut, patch_h, patch_w, dim, theta, offset_t);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opROPE2DOffset);
  {$endif}
end;

procedure QNNComputeRoPEText(const cosOut, sinOut: TMemoryBlock; const txt_seq, axis_dim: longint; const theta: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opROPEText);
  {$endif}

  case cosOut.DataType of
    dtF32 :
      TQNNSingleOPS.QNNComputeRoPEText(cosOut, sinOut, txt_seq, axis_dim, theta);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opROPEText);
  {$endif}
end;

procedure QNNApplyRoPE(const dst, freqs: TMemoryBlock; const batch, seq, heads, head_dim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opApplyROPE);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNApplyRoPE(dst, freqs, batch, seq, heads, head_dim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opApplyROPE);
  {$endif}
end;

procedure QNNApplyRoPE2(const dst, cosIn, sinIn: TMemoryBlock; const numHeads, headDim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opApplyROPE2);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNApplyRoPE2(dst, cosIn, sinIn, numHeads, headDim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opApplyROPE2);
  {$endif}
end;

procedure QNNApplyRoPE3(const dst, cosIn, sinIn: TMemoryBlock; const seqLen, numHeads, headDim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opApplyROPE3);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNApplyRoPE3(dst, cosIn, sinIn, seqLen, numHeads, headDim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opApplyROPE3);
  {$endif}
end;

procedure QNNApplyRoPEQK(const q, k, cos_cache, sin_cache: TMemoryBlock; const seq_len, num_q_heads, num_kv_heads, head_dim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opApplyROPEQK);
  {$endif}

  case q.DataType of
    dtF32 :
      TQNNSingleOPS.QNNApplyRoPEQK(q, k, cos_cache, sin_cache, seq_len, num_q_heads, num_kv_heads, head_dim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opApplyROPEQK);
  {$endif}
end;

procedure QNNApplyRoPE2D(const dst, cos_freq, sin_freq: TMemoryBlock; const seq, heads, head_dim, dim: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opApplyROPE2D);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNApplyRoPE2D(dst, cos_freq, sin_freq, seq, heads, head_dim, dim);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opApplyROPE2D);
  {$endif}
end;

function QNNTimeStepEmbedding(const t: QNNFloat; const dim: integer; const max_period: QNNFloat): TMemoryBlock;
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opTimeStepEmbedding);
  {$endif}
  result := TQNNSingleOPS.QNNTimeStepEmbedding(t, dim, max_period);
  //OP_IMPL_FAIL()
  {$if defined(USE_PROFILING)}
  metrics.finish(opTimeStepEmbedding);
  {$endif}
end;

procedure QNNMatTriangularFill(const dim: integer; const dst: TMemoryBlock; const val: QNNFloat; const mask: PLongint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opTriangularFill);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMatTriangularFill(dim, dst, val, mask);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opTriangularFill);
  {$endif}
end;

procedure QNNMaskFill(const dst: TMemoryBlock; const mask: PLongint; const val: QNNFloat; const rows, cols: longint);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opMaskFill);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNMaskFill(dst, mask, val, rows, cols);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opMaskFill);
  {$endif}
end;

procedure QNNAdaLN(const dst, src, bias, scale: TMemoryBlock; const seq, hidden: longint; const eps: QNNFloat);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opAdaLN);
  {$endif}

  case dst.DataType of
    dtF32 :
      TQNNSingleOPS.QNNAdaLN(dst, src, bias, scale, seq, hidden, eps);
  else
    OP_IMPL_FAIL()
  end;
  {$if defined(USE_PROFILING)}
  metrics.finish(opAdaLN);
  {$endif}
end;

procedure QNNLINEAR_BF16_OR_F32(const dst, src: PSingle; const weight: TMemoryBlock; const weight_bf16: TMemoryBlock; const seq, srcDim, dstDim: integer);
begin
  {$if defined(USE_PROFILING)}
  metrics.start(opGemm);
  {$endif}
  TQNNSingleOPS.QNNLINEAR_BF16_OR_F32(dst, src, weight, weight_bf16, seq, srcDim, dstDim);
    //OP_IMPL_FAIL()
  {$if defined(USE_PROFILING)}
  metrics.finish(opGemm);
  {$endif}
end;


var
  i:longint;
  m1, m2:TMemoryBlock;
  m1_ptr : PSingle;
initialization
  //printDigits:=0;
  //m1:=TMemoryBlock.create([2, 2, 2, 2, 2], '');
  //m2:=TMemoryBlock.create([2, 2, 2, 2, 2], '');
  //m1_ptr := m1;
  //for i:=0 to m1.size-1 do m1_ptr[i] := i div 2 ;
  //QNNPermute(m2, m1, m1.Shape,[1, 0, 2, 3, 4]);
  //m2.print(psValues);
  //readln

end.
