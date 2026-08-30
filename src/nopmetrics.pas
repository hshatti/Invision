unit nOpMetrics;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

type

{$if not declared(SizeInt)}
  SizeInt = IntPtr;
{$endif}
  TMeasureOps = (
    opAllocCPU, opIncFill, opFill, opCopy, opCopyStrided, opMaxNorm, opStdDevNorm, opRMSNorm, opBatchAddvs, opAddvs, opBatchMulvs, opGroupNorm, opBatchNorm,
    opMulvs, opBatchSubvs, opSubvs, opBatchDivvs, opDivvs, opPow, opAxpy, opMulvv, opAddvv, opSubvv, opDivvv, opROPE, opROPE2,opROPE2D, opROPEText, opROPEOffset, opROPE2DOffset,
    opApplyROPE, opApplyROPE2, opApplyROPE3, opApplyROPEQK, opApplyROPE2D, opFmavv, opDot, opBatchFmavss, opFmavss, opFmavvs, opGemm, opIm2col, opCol2im,
    opTimeStepEmbedding, opTriangularFill, opMaskFill, opAdaLN,
    opConv2D, opConcat, opAddConcat, opHostToDevice, opDeviceToHost, opMemAllocate, opMemRelease, opGPUAllocate, opArgMax, opArgMin, opConvert, opQuantize, opDequantize,
    opAttention, opAttentionHead, opAttentionHeadTiled, opFlashAttention, opGPURelease, opIm2ColExt, opCol2ImExt, opMeans, opVariances, opSoftmax, opSoftmaxInplace, opSiLU, opSiLUInplace, opSiLUMul, opCrossEntropySoftmax,
    opMeansVars, opNormalize,  opMeansVarsDelta, opNormalizeDelta, opAddDots, opForwardBias, opBackwardBias, opForwardScale, opForwardScaleAdd, opFwMaxpool, opBwMaxpool,
    opReduce , opFwDropout, opBwDropout, opL2, opClip, opInvSqrt, opMin, opMax, opSumAbsDiff, opSum, opSumSqrDiff, opSumSqrDiffScaler, opDistance, opSumSqr, opMaxAbs, opMinAbs, opMinAndMax,
    opInit, opDispose, opActivate, opDerive, opSigmoid, opSigmoidInplace, opPatchify, opUnPatchify, opTranspose, opUpSample, opUpSampleNearest, opUpSampleCubic, opMaxAbsDiff, opAbsDiff, opAbsDiffScaler, opAbsSum, opOther, opTanh
  );

  { TTensorOps }
  PTensorMetrics = ^TTensorMetrics;

  { TTensorMetrics }

  TTensorMetrics = record
  private
      m:array[0..999] of int64;
      stack: longint;
      function GetItem(i: TMeasureOps): int64;
  public
      elapsed: array[low(TMeasureOps)..high(TMeasureOps)] of int64;
      counts: array[low(TMeasureOps)..high(TMeasureOps)] of int64;
      //stackArray : TArray<TMeasureOps>;
      procedure start(const a:TMeasureOps);
      procedure finish(const a:TMeasureOps);
      procedure reset();
      procedure print();
      function total():int64;
      property Item[i:TMeasureOps]:int64 read GetItem ;default;
      class operator Initialize({$ifdef fpc}var {$else}out {$endif} dst:TTensorMetrics);
  end;

implementation
uses nChrono, sysutils;

{ TTensorMetric }

function TTensorMetrics.GetItem(i: TMeasureOps): int64;
begin
  result := elapsed[i]
end;

procedure TTensorMetrics.start(const a: TMeasureOps);
begin
  m[stack]:=clock;
  inc(counts[a]);
  inc(stack);
  //insert(a, stackArray, length(stackArray))
end;

procedure TTensorMetrics.finish(const a: TMeasureOps);
begin
  dec(stack);
  dec(counts[a]);
  elapsed[a] := elapsed[a] + clock()- m[stack];
  //delete(stackArray, high(stackArray), 1)
end;

procedure TTensorMetrics.reset();
begin
  fillchar(self, SizeOf(self), #0)
end;

procedure TTensorMetrics.print();
var t:int64; i:TMeasureOps; s:string;
begin
  t := 0;
  for i:= low(TMeasureOps) to high(TMeasureOps) do
    if elapsed[i]<>0 then begin
      inc(t, elapsed[i]);
      str(i, s);
      writeLn(format('%-20s', [copy(s, 3)]), elapsed[i]/CLOCKS_PER_MS:10:3, 'ms')
    end;
  writeLn('--------------------------------');
  writeLn(format('%-20s', ['total ']):20, t/CLOCKS_PER_MS:10:3, 'ms')

end;

function TTensorMetrics.total(): int64;
var
  i: TMeasureOps;
begin
  result := 0;
  for i:=low(TMeasureOps) to high(TMeasureOps) do
    inc(result, elapsed[i])
end;

class operator TTensorMetrics.Initialize({$ifdef fpc}var {$else}out {$endif} dst:TTensorMetrics);
begin
  fillchar(dst, SizeOf(dst), #0)
end;

end.

