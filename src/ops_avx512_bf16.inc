


{$if defined(CPUX64) and defined(VER3_3)}

procedure SingleToBF16_x8_avx512(const src: PSingle; var dst:PBF16);assembler; {$ifdef fpc} nostackframe;{$endif}
asm
  vcvtneps2bf16 xmm0        , yword [src]
  vmovups       oword [dst] , xmm0
end;

procedure SingleToBF16_x16_avx512(const src: PSingle; var dst:PBF16);assembler; {$ifdef fpc} nostackframe;{$endif}
asm
  vcvtneps2bf16 ymm0        , zword [src]
  vmovups       yword [dst] , ymm0
end;


procedure BF16ToSingle_x16_avx512(const src: PSingle; var dst:PBF16);assembler; {$ifdef fpc} nostackframe;{$endif}
asm
  vpmovzxwd     zmm0        , yword [src]
  vpslld        zmm0        , zmm0 , 16
  vmovups       zword [dst] , zmm0
end;

{$endif}
