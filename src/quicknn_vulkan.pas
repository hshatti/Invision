unit quicknn_vulkan;

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


interface
uses Sysutils, math, TypInfo, nVulkanHelper, quicknn_common;

type
  TQNNOperation = (
        opActivate
      , opActivate_swish
      , opAdd
      , opAxpy
      , opBackward_bias
      , opBackward_dropout
      , opBackward_maxpool
      , opBackward_scale
      , opClamp
      , opCol2im
      , opCopy
      , opCost_l2
      , opCross_entropy
      , opFill
      , opFma
      , opFma_scalar
      , opForward_bias
      , opForward_dropout
      , opForward_fma
      , opForward_maxpool
      , opForward_scale
      , opGemm_nn
      , opGemm_nt
      , opGemm_tn
      , opGemm_nn_WrapTiling
      , opGemm_nt_WrapTiling
      , opGemm_tn_WrapTiling
      , opGradient
      , opIm2col
      , opInverse_sqrt
      , opMean
      , opMean_var_gradient
      , opMul
      , opNorm
      , opNorm_gradient
      , opPower
      , opScale
      , opSoftmax
      , opSub
      , opUpsample
      , opVariance
    );



  { TVulkanNN }

  { TQNNVulkan }

  TQNNVulkan = class(TVulkanCompute)
  private
    vkA, vkB, vkC : TVulkanMemory;
  public

    procedure preparePipeline(const op:TQNNOperation);overload;
    procedure dispatchPipeline(const Op: TQNNOperation; const args: TVulkanArgs; const x:longword; const y:longword = 1; const z:longword =1); overload;
    procedure sgemm(const tranA, tranB:boolean; M, N, K :longint; const ALPHA :single; const A :PSingle; const lda :longint; const B:PSingle; const ldb :longint; const BETA :single; const C :PSingle; const ldc :longint);

  end;

implementation

{ TQNNVulkan }

procedure TQNNVulkan.preparePipeline(const op: TQNNOperation);
begin
  preparePipeline(longint(op), loadShader('../src/vulkan/spv/'+copy(getEnumName(TypeInfo(TQNNOperation), ord(op)), 3) + '.comp.float.spv'));
end;

procedure TQNNVulkan.dispatchPipeline(const Op: TQNNOperation;
  const args: TVulkanArgs; const x: longword; const y: longword;
  const z: longword);
begin
  dispatchPipeline(TKernelHandle(op), args, x, y, z);
end;

procedure TQNNVulkan.sgemm(const tranA, tranB: boolean; M, N, K: longint;
  const ALPHA: single; const A: PSingle; const lda: longint; const B: PSingle;
  const ldb: longint; const BETA: single; const C: PSingle; const ldc: longint);
const
  BN = 128;
  BM = 128;
  WORKGROUP_SIZE_X = 8 ; // 128;
  WORKGROUP_SIZE_Y = 8; //1;
  WORKGROUP_SIZE_Z = 1;
var aSize, bSize,cSize : int64;
begin
  assert(not tranA , 'TVulknCompte.sgemm : not implemrnted with the current arguments!');

  if tranA then begin
    aSize := K*lda*sizeof(single);
    //if vkA.size < aSize then
    begin
      if vkA.size > 0 then
        Self.freeMemory(vkA);
      vkA := createStorageMemory(aSize);
    end
  end else begin
    aSize := M*lda*sizeof(single);
    //if vkA.size < aSize then
    begin
      if vkA.size > 0 then
        Self.freeMemory(vkA);
      vkA := createStorageMemory(aSize);
    end;
  end;

  if tranB then begin
    bSize := N*ldb*sizeof(single);
    //if vkB.size <bSize then
    begin
      if vkB.size > 0 then
        Self.freeMemory(vkB);
      vkB := createStorageMemory(bSize);
    end
  end else begin
    bSize :=  K*ldb*sizeof(single);
    //if vkB.size < bSize then
    begin
      if vkB.size > 0 then
        Self.freeMemory(vkB);
      vkB := createStorageMemory(bSize);
    end;
  end;

  cSize := M*ldc*sizeof(single);
  //if vkC.size < cSize then
  begin
    if vkC.size > 0 then
      Self.freeMemory(vkC);
    vkC := createStorageMemory(cSize);
  end;


  if not(
    (high(vulkanPipelines) > max(longint(opGemm_nn_WrapTiling), longint(opGemm_nt))) and
    assigned(vulkanPipelines[longint(opGemm_nn_WrapTiling)].pipeline) and
    assigned(vulkanPipelines[longint(opGemm_nt_WrapTiling)].pipeline) and
    assigned(vulkanPipelines[longint(opGemm_nn)].pipeline)
    ) then begin
      preparePipeline(opGemm_nn_WrapTiling);
      preparePipeline(opGemm_nn);
      preparePipeline(opGemm_nt_WrapTiling);
  end;


  pushToDevice(A, vkA);
  pushToDevice(B, vkB);
  pushToDevice(C, vkC);
  beginCommadBuffer;

  if (not tranA) and (not tranB) then
    dispatchPipeline(opGemm_nn_WrapTiling, [M, N, K, ALPHA, vkA.buffer, 0, lda, vkB.buffer, 0, ldb, BETA, vkC.buffer, 0, ldc],
                                   CEIL_DIV(N, BN), CEIL_DIV(M, BM)
                        )
    //dispatchPipeline(opGemm_nn, [M, N, K, 1.0, vkA.buffer, 0, lda, vkB.buffer, 0, ldb, 0.0, vkC.buffer, 0, ldc],
    //                               (M + WORKGROUP_SIZE_X-1) div WORKGROUP_SIZE_X,
    //                               (N + WORKGROUP_SIZE_Y-1) div WORKGROUP_SIZE_Y,
    //                               1 //(K+WORKGROUP_SIZE_Z-1) div WORKGROUP_SIZE_Z
    //                    )
  else if (not tranA) and tranB then
    dispatchPipeline(opGemm_nt_WrapTiling, [M, N, K, ALPHA, vkA.buffer, 0, lda, vkB.buffer, 0, ldb, BETA, vkC.buffer, 0, ldc],
                                   CEIL_DIV(N, BN), CEIL_DIV(M, BM)
                        );
  if CommandBufferStarted then begin
    endCommandBuffer;
    finish
  end;
  pullFromDevice(C, vkC, cSize);

end;



const
  M : longint = $8;
  N : longint = $8;
  K : longint = $8;
  WORKGROUP_SIZE_X = 8 ; // 128;
  WORKGROUP_SIZE_Y = 32; //1;
  WORKGROUP_SIZE_Z = 1;
  //WORKGROUP_SIZE_X = 128;
  //WORKGROUP_SIZE_Y = 1;
  //WORKGROUP_SIZE_Z = 1;

  BN = 128;
  BM = 128;

var
  vk  : TVulkanCompute;
  v   : TVulkanArgs;
  i   : integer;
  A, B, C, res : TArray<nfloat>;
  AA,BB, CC: TVulkanMemory;
  //stagingMems : TArray<TVulkanMemory>;
  workgroups : LongWord;
  vkd : TArray<TVulkanDevice>;
  args : TVulkanArgs;
  i4 : int4;
  ms : uint64;
  hh : half;

initialization

(*
  vk := TVulkanCompute.Create(1);
  A := array_rand(M*K, -10, 10);
  B := array_rand(N*K, -10, 10);
  AA := vk.createStorageMemory(M*K*sizeof(nfloat));
  BB := vk.createStorageMemory(N*K*sizeof(nfloat));

  setLength(C  , M*N);//C := array_rand(M*N, -10, 10);
  setLength(res, M*N);

  CC := vk.createStorageMemory(M*N*sizeof(nfloat));  //vk.pushToDevice(C, CC);

  //array_stat(A);
  //array_stat(B);
  //array_add(a, b, c);
  //array_stat(C);

  vk.preparePipeline(ord(opGemm_nn_WrapTiling), vk.loadShader('../src/vulkan/spv/' + copy(getEnumName(TypeInfo(TQNNOperation), ord(opGemm_nn_WrapTiling)), 3)+'.comp.float.spv'));
  //vk.preparePipeline(ord(opGemm_nt_WrapTiling), vk.loadShader('../src/vulkan/spv/' + copy(getEnumName(TypeInfo(TQNNOperation), ord(opGemm_nt_WrapTiling)), 3)+'.comp.float.spv'));
  //vk.preparePipeline(ord(opGemm_tn_WrapTiling), vk.loadShader('../src/vulkan/spv/' + copy(getEnumName(TypeInfo(TQNNOperation), ord(opGemm_tn_WrapTiling)), 3)+'.comp.float.spv'));
  //vk.preparePipeline(ord(opGemm_nn), vk.loadShader('../src/vulkan/spv/' + copy(getEnumName(TypeInfo(TQNNOperation), ord(opGemm_nn)), 3)+'.comp.float.spv'));

  writeln('work groups : ',  CEIL_DIV(N, BN), ' X ', CEIL_DIV(M, BM));


  vk.pushToDevice(A, AA);
  vk.pushToDevice(B, BB);
  vk.beginCommadBuffer;
  ms := GetTickCount64;

  //workgroups := (N+WORKGROUP_SIZE-1) div WORKGROUP_SIZE;

  //vk.dispatchPipeline(ord(opGemm_nn), [M, N, K, 1.0, AA, 0, K, BB, 0, N, 0.0, CC, 0, N],
  //                               (M + WORKGROUP_SIZE_X-1) div WORKGROUP_SIZE_X,
  //                               (N + WORKGROUP_SIZE_Y-1) div WORKGROUP_SIZE_Y,
  //                               1 //(K+WORKGROUP_SIZE_Z-1) div WORKGROUP_SIZE_Z
  //                    );


  vk.dispatchPipeline(ord(opGemm_nn_WrapTiling), [M, N, K, 1.0, AA.buffer, 0, K, BB.buffer, 0, N, 0.0, CC.buffer, 0, N],
                                 CEIL_DIV(N, BN), CEIL_DIV(M, BM)
                      );
  //vk.dispatchPipeline(ord(opGemm_nt_WrapTiling), [M, N, K, 1.0, AA.buffer, 0, K, BB.buffer, 0, K, 0.0, CC.buffer, 0, N],
  //                               CEIL_DIV(N, BN), CEIL_DIV(M, BM)
  //                    );

  vk.endCommandBuffer;
  vk.finish;
  vk.pullFromDevice(C, CC);
  writeln('GPU : ',GetTickCount64-ms,'ms');


  ms := GetTickCount64;
  //array_gemm_nt(M, N, K, 1.0, A, K, B, K, 0.0, res, N);
  array_gemm_nn(M, N, K, 1.0, A, K, B, N, 0.0, res, N);
  //array_gemm_tn(M, N, K, 1.0, A, M, B, N, 0.0, res, N);
  writeln('CPU : ',GetTickCount64-ms,'ms');
  writeln('CPU/GPU Diff: ', single(array_sum_sqr_diff(C, res)):1:6);

  writeln('Vulkan Compute:');
  array_print(C, 1);
  array_stat(C, 2);

  writeln('CPU Compute:');
  array_print(res, 1);
  array_stat(res, 2);


  //vk.dispatchPipeline(NN_SGEMM1_NN, args, workgroups);
  //vk.pullFromDevice(mems[2], storageMems[2]);

  vk.freeMemory(AA);
  vk.freeMemory(BB);
  vk.freeMemory(CC);
  vk.free;
  //array_add(length(res), pointer(mems[0]), pointer(mems[1]), pointer(res));
  //array_fma(3, 4, mems[0], mems[2]);

  //array_print_diff(C, res, 0.05);

  readln
*)

finalization

end.

