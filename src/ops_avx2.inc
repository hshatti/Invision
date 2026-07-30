{$if defined(CPUX64)}
const
  SIMD_REGS = 8;
  //ONE : single =1.0;
  {$ifdef FPC}
  SIMD_SHFT  = BsfQWord(SIMD_REGS);
  {$else}
  {$if SIMD_REGS = 8}SIMD_SHFT = 3{$else} SIMD_SHFT = 2{$endif};
  {$endif}
  SIMD_OFF = SIMD_REGS * sizeof(single);
  ymmd : array[0..7] of int32 = (0, 1, 2, 3, 4, 5, 6, 7);

const
  L2E :single = 1.442695041;// log2(e);
  C0  :single = 1.00172476;
  C1  :single = 0.657636276;
  C2  :single = 0.3371894346;
  //MAX_EXP =  8.8722839052068352E+001;
  //MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP =  8.872283E+001;
  MIN_EXP = -8.733654E+001;

  ONE :array[0..7] of single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  ZERO:array[0..7] of single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  MX  :array[0..7] of single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  MN  :array[0..7] of single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP );

procedure QNNExp_avx2(const N:NativeInt; const src:PSingle; const dst:PSingle); assembler;
asm
  sub                  rsp      , $10*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + l2e]
  vpbroadcastd  ymm4  , [rip + c1]
  vpbroadcastd  ymm5  , [rip + c0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  //vxorps        ymm0  , ymm0        , ymm0              // zero
  vmovups       ymm1  , yword [src]
  vcmpgeps      ymm6  , ymm1        , [rip + mx]
  vcmpleps      ymm7  , ymm1        , [rip + mn]
  vblendvps     ymm1  , ymm1 , [rip + mx], ymm6
  vblendvps     ymm1  , ymm1 , [rip + mn], ymm7
  vmulps        ymm1  , ymm3        , ymm1
  vroundps      ymm2  , ymm1        , 1
  vsubps        ymm1  , ymm1        , ymm2
  vcvtps2dq     ymm0  , ymm2
  vpbroadcastd  ymm2  , [rip + c2]
  vfmadd213ps   ymm2  , ymm1        , ymm4
  vpslld        ymm0  , ymm0        , 23
  vfmadd213ps   ymm1  , ymm2        , ymm5
  vpaddd        ymm0  , ymm0        , ymm1
  vmovups       yword [dst] , ymm0

  add           src   , 32
  add           dst   , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done

@rem:

  vmovss        xmm1  , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + mx]
  vcmpleps      xmm7  , xmm1        , [rip + mn]
  vblendvps     xmm1  , xmm1 , [rip + mx], xmm6
  vblendvps     xmm1  , xmm1 , [rip + mn], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + c2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0       , xmm1
  vmovss        dword [dst] , xmm0

  add           src   , 4
  add           dst   , 4
  dec           N
  jnz           @rem

@done:
  vmovdqu              xmm6     , [rsp+$00]
  vmovdqu              xmm7     , [rsp+$10]
  add                  rsp      , $10*2                     // restoring stack
end;

// AKA SWISH
// sigmoid or dst can be nil

procedure QNNSiLU_avx2(const N:NativeInt; const src, up, outSigmoid, dst:PSingle);assembler;   // a function that does sigmoid, gated and SiLU
asm
  mov                  rax      , dst
  sub                  rsp      , $10*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + L2E]
  vpbroadcastd  ymm4  , [rip + C1]
  vpbroadcastd  ymm5  , [rip + C0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  vxorps        ymm0  , ymm0        , ymm0              // zero
  vsubps        ymm1  , ymm0        , yword [src]       // t1 = -src

  vcmpgeps      ymm6  , ymm1        , [rip + MX]        // clamp t1 to [MIN_EXP, MAX_EXP]
  vcmpleps      ymm7  , ymm1        , [rip + MN]
  vblendvps     ymm1  , ymm1 , [rip + MX], ymm6
  vblendvps     ymm1  , ymm1 , [rip + MN], ymm7

  vmulps        ymm1  , ymm3        , ymm1              // t1 = t1 * log2(e)
  vroundps      ymm2  , ymm1        , 1                 // t2 = round(t1 * log2(e))
  vsubps        ymm1  , ymm1        , ymm2              // t1 = t1 - t2
  vcvtps2dq     ymm0  , ymm2                            // t0 = cast to integer(t2)
  vpbroadcastd  ymm2  , [rip + C2]
  vfmadd213ps   ymm2  , ymm1        , ymm4              // t2 = C2 * t1 + C1
  vpslld        ymm0  , ymm0        , 23                // t0 = t0 shl 32
  vfmadd213ps   ymm1  , ymm2        , ymm5              // t1 = t1 * t2 + C0
  vpaddd        ymm0  , ymm0        , ymm1              // t0 = t0 + integer(t1) => exp(t1)
  vaddps        ymm1  , ymm0        , [rip + ONE]       // 1 +exp(-src)
  vrcpps        ymm1  , ymm1                            // 1/(1+exp(-src))

  cmp           outSigmoid   , 0
  je            @skip1
  vmovups       yword [outSigmoid] , ymm1
  add           outSigmoid   , 32
@skip1:
  cmp           rax       , 0
  je            @skipup1
  vmulps        ymm1            , ymm1 ,   yword [src]
  cmp           up        , 0
  je            @skipdst1
  vmulps        ymm1            , ymm1 ,   yword [up]
  add           up              , 32
@skipdst1:
  vmovups       yword [rax]     , ymm1
  add           rax             , 32

@skipup1:
  add           src       , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done
@rem:

  vpxor         xmm0  , xmm0        , xmm0
  vsubss        xmm1  , xmm0        , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + MX]
  vcmpleps      xmm7  , xmm1        , [rip + MN]
  vblendvps     xmm1  , xmm1 , [rip + MX], xmm6
  vblendvps     xmm1  , xmm1 , [rip + MN], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + C2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0        , xmm1
  vaddss        xmm1  , xmm0        , [rip + ONE]       // 1 +exp(-src)
  vrcpss        xmm1  , xmm1        , xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  cmp           outSigmoid   , 0   // sigmoid is NULL
  je            @skip2
  vmovss        dword [outSigmoid] , xmm1
  add           outSigmoid         , 4
@skip2:
  cmp           rax       , 0
  je            @skipup2
  vmulss        xmm1            , xmm1 ,   dword [src]
  cmp           up        , 0
  je            @skipdst2
  vmulss        xmm1            , xmm1 ,   dword [up]
  add           up              , 4
@skipdst2:
  vmovss        dword [rax]     , xmm1
  add           rax             , 4

@skipup2:
  add           src             , 4
  dec           N
  jnz           @rem
@done:
  vmovdqu       xmm6            , [rsp+$00]
  vmovdqu       xmm7            , [rsp+$10]
  add           rsp             , $10*2                     // restoring stack
end;

//const
//  YMMD : array [0..7] of longword =(0,1,2,3,4,5,6,7);//: array[0..7] of single = ()
function QNNMax_avx2(const N:NativeInt; const src:PSingle):Single;assembler;
asm
  vbroadcastss  ymm0 , dword [src]

  mov           r8   , N
  shr           r8   , 5
  jz            @rem1

  vmovaps       ymm1 , ymm0
  vmovaps       ymm2 , ymm0
  vmovaps       ymm3 , ymm0

@while1:
  vmaxps        ymm0 , ymm0, yword [src]
  vmaxps        ymm1 , ymm1, yword [src +32  ]
  vmaxps        ymm2 , ymm2, yword [src +32*2]
  vmaxps        ymm3 , ymm3, yword [src +32*3]
  add           src  , 32*4
  dec           r8
  jnz           @while1

  vmaxps        ymm0, ymm0, ymm1
  vmaxps        ymm2, ymm2, ymm3
  vmaxps        ymm0, ymm0, ymm2

@rem1:

  mov           r8   , N
  and           r8   , 31
  shr           r8   , 3
  jz            @rem

@while:
  vmaxps        ymm0 , ymm0, yword [src]
  add           src  , 32
  dec           r8
  jnz           @while

@rem:
  mov           r8   , N
  and           r8   , 7
  jz            @done

  vmovd         xmm1 , r8d
  vpbroadcastd  ymm1 , xmm1
  vpcmpgtd      ymm1 , ymm1, YMMD[rip]
  //      vmaskmovps   (%rsi,%rax,sz), %ymm1, %ymm2
  vblendvps     ymm1 , ymm0, yword[src], ymm1
  vmaxps        ymm0 , ymm0, ymm1

@done:
  vextractf128  xmm1 , ymm0, 1
  vzeroupper
  maxps         xmm0 , xmm1
  movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  maxps         xmm0 , xmm1
  movshdup      xmm1 , xmm0
  maxss         xmm0 , xmm1

end;

function QNNMin_avx2(const N:NativeInt; const src:PSingle):Single;assembler;
asm
  vbroadcastss  ymm0 , dword [src]

  mov           r8   , N
  shr           r8   , 5
  jz            @rem1

  vmovaps       ymm1 , ymm0
  vmovaps       ymm2 , ymm0
  vmovaps       ymm3 , ymm0

@while1:
  vminps        ymm0 , ymm0, yword [src]
  vminps        ymm1 , ymm1, yword [src +32  ]
  vminps        ymm2 , ymm2, yword [src +32*2]
  vminps        ymm3 , ymm3, yword [src +32*3]
  add           src  , 32*4
  dec           r8
  jnz           @while1

  vminps        ymm0, ymm0, ymm1
  vminps        ymm2, ymm2, ymm3
  vminps        ymm0, ymm0, ymm2

@rem1:

  mov           r8   , N
  and           r8   , 31
  shr           r8   , 3
  jz            @rem

@while:
  vminps        ymm0 , ymm0, yword [src]
  add           src  , 32
  dec           r8
  jnz           @while

@rem:
  mov           r8   , N
  and           r8   , 7
  jz            @done

  vmovd         xmm1 , r8d
  vpbroadcastd  ymm1 , xmm1
  vpcmpgtd      ymm1 , ymm1, YMMD[rip]
  //      vmaskmovps   (%rsi,%rax,sz), %ymm1, %ymm2
  vblendvps     ymm1 , ymm0, yword[src], ymm1
  vminps        ymm0 , ymm0, ymm1

@done:
  vextractf128  xmm1 , ymm0, 1
  vzeroupper
  minps         xmm0 , xmm1
  movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  minps         xmm0 , xmm1
  movshdup      xmm1 , xmm0
  minss         xmm0 , xmm1

end;

procedure QNNSoftmax_avx2(const N:NativeInt; const src:PSingle);assembler;
asm
  //max_val := QNNMax(N, x);
  //sum := 0.0;
  //for i := 0 to N-1 do begin
  //    x[i] := fast_exp(x[i]-max_val);
  //    //x[i] := exp(x[i]-max_val);
  //    sum := sum + x[i]
  //end;
  //inv_sum := single(1.0) / sum;
  //QNNScaleInplace(x, inv_sum, N);
  CALL QNNMax_avx2                              // max_src
  vbroadcastss ymm6, xmm0
  vxorps       ymm3, ymm3, ymm3                 // sum=0
  mov  rax , N
  shr  rax , 3
  jz   @rem

@while:
  vmovups       ymm1  , yword[src]
  vsubps        ymm1  , ymm1 , ymm6             // t1 = src - max_src

  vmulps        ymm1  , ymm3        , ymm1      // t1 = t1 * log2(e)
  vroundps      ymm2  , ymm1        , 1         // t2 = round(t1 * log2(e))
  vsubps        ymm1  , ymm1        , ymm2      // t1 = t1 - t2
  vcvtps2dq     ymm0  , ymm2                    // t0 = cast to integer(t2)
  vpbroadcastd  ymm2  , [rip + C2]
  vfmadd213ps   ymm2  , ymm1        , ymm4      // t2 = C2 * t1 + C1
  vpslld        ymm0  , ymm0        , 23        // t0 = t0 shl 32
  vfmadd213ps   ymm1  , ymm2        , ymm5      // t1 = t1 * t2 + C0
  vpaddd        ymm0  , ymm0        , ymm1      // t0 = t0 + integer(t1) => exp(t1)

  vaddps        ymm3  , ymm3        , ymm0

  add           src   , 32
  dec           rax
  jnz           @while


@rem:
  mov           rax , N
  and           rax , 7
  vmovd         xmm6 , eax
  vpbroadcastd  ymm6 , xmm6
  vpcmpgtd      ymm6 , ymm6, YMMD[rip]
  vmaskmovps    ymm1 , ymm6, yword[src]



end;

function sdot_avx2(const N:int64; const A,B:PSingle):single;assembler;{$ifdef FPC}nostackframe;{$endif}
asm
{$ifndef FPC}
  .NOFRAME
{$endif}
   mov              r11     ,    N
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
   mov              r11     ,    N
   and              r11     ,    SIMD_REGS -1
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

procedure saxpy_avx2(const N:int64; const a:single; const x,y:PSingle);assembler;{$ifdef FPC}nostackframe;{$endif}
asm
  //push         r11
  //push         N
  {$ifndef FPC}
  .NOFRAME
  {$endif}

//  movss         xmm2   , a
  vbroadcastss ymm1   , a
  mov          r11    , N
  shr          r11    , (SIMD_SHFT + 2)    // div by 16 (4*4) = turns * SIMD_REGS
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
  dec          r11
  jnz          @while

@rem1:
  mov          r11    , N
  and          r11    , (4*SIMD_REGS-1)       // mod 32  ( turns * SIMD_REGS)
  shr          r11    , SIMD_SHFT             // div SIMD_REGS
  jz           @rem

@while1:
  vmovups      ymm0   , yword[y]

  vfmadd231ps  ymm0   , ymm1       , [x]
  vmovups      yword [y]    , ymm0
  add          x      , SIMD_OFF
  add          y      , SIMD_OFF
  dec          r11
  jnz          @while1

@rem:
  mov          r11    , N
  and          r11    , (SIMD_REGS -1)       // mod SIMD_REGS
  jz           @done

@while2:
  vmovss       xmm0   , dword [y]
  vfmadd231ss  xmm0   , xmm1, [x]
  vmovss       dword [y]    , xmm0
  add          x      , 4
  add          y      , 4
  dec          r11
  jnz          @while2

@done:
  //pop          r11
  //vzeroupper
end;

function QNNSumSqr_avx2(const N:NativeInt; const src: PSingle):single;assembler;
asm

  mov           r8   , N
  shr           r8   , 4
  jz            @rem1

  vxorps        ymm0 , ymm0, ymm0
  vxorps        ymm1 , ymm1, ymm1

@while1:
  vmovups       ymm4,  yword[src]
  vmovups       ymm5,  yword[src +32]

  vfmadd231ps   ymm0 , ymm4, ymm4
  vfmadd231ps   ymm1 , ymm5, ymm5


  add           src  , 32*2
  dec           r8
  jnz           @while1

  vaddps        ymm0, ymm0, ymm1

@rem1:

  mov           r8   , N
  and           r8   , 15
  shr           r8   , 3
  jz            @rem

@while:
  vmovups       ymm1,  yword[src]
  vfmadd231ps   ymm0 , ymm1, ymm1
  add           src  , 32
  dec           r8
  jnz           @while

@rem:
  mov           r8   , N
  and           r8   , 7
  jz            @done

  vmovd         xmm1 , r8d
  vpbroadcastd  ymm1 , xmm1
  vpcmpgtd      ymm1 , ymm1, YMMD[rip]
  vmaskmovps    ymm2 , ymm1, yword[src]
  //vblendvps     ymm1 , ymm0, yword[src], ymm1
  vfmadd231ps   ymm0 , ymm2, ymm2

@done:
  //vbroadcastss  ymm1, [ONE+rip]
  vdpps         ymm0 , ymm0, [rip+ONE], $F1
  vextractf128  xmm1 , ymm0, 1
  vzeroupper
  addss         xmm0 , xmm1
  //vextractf128  xmm1 , ymm0, 1
  //vzeroupper
  //addps         xmm0 , xmm1
  //movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  //addps         xmm0 , xmm1
  //movshdup      xmm1 , xmm0
  //addss         xmm0 , xmm1

end;

function QNNSumSqrDiff_avx2(const N:NativeInt; const src, aMean: PSingle):single;assembler;
asm

  mov           r9   , N
  shr           r9   , 4
  jz            @rem1

  vbroadcastss  ymm2 , dword[aMean]
  vxorps        ymm0 , ymm0, ymm0
  vxorps        ymm1 , ymm1, ymm1

@while1:

  vsubps        ymm4,  ymm2, yword[src]
  vsubps        ymm5,  ymm2, yword[src + 32]

  vfmadd231ps   ymm0 , ymm4, ymm4
  vfmadd231ps   ymm1 , ymm5, ymm5


  add           src  , 32*2
  dec           r9
  jnz           @while1

  vaddps        ymm0, ymm0, ymm1

@rem1:

  mov           r9   , N
  and           r9   , 15
  shr           r9   , 3
  jz            @rem

@while:
  vsubps        ymm1 , ymm2, yword[src]
  vfmadd231ps   ymm0 , ymm1, ymm1
  add           src  , 32
  dec           r9
  jnz           @while

@rem:
  mov           r9   , N
  and           r9   , 7
  jz            @done

  vmovd         xmm1 , r9d
  vpbroadcastd  ymm1 , xmm1
  vpcmpgtd      ymm1 , ymm1, YMMD[rip]
  vmaskmovps    ymm3 , ymm1, yword[src]
  vsubps        ymm3 , ymm2, ymm3
  //vblendvps     ymm1 , ymm0, yword[src], ymm1
  vfmadd231ps   ymm0 , ymm3, ymm3

@done:
  //vbroadcastss  ymm1, [ONE+rip]
  vdpps         ymm0 , ymm0, [rip+ONE], $F1
  vextractf128  xmm1 , ymm0, 1
  vzeroupper
  addss         xmm0 , xmm1
  //vextractf128  xmm1 , ymm0, 1
  //vzeroupper
  //addps         xmm0 , xmm1
  //movhlps       xmm1 , xmm0 // = selecting offsets 2,3,0,0  = 0b1110
  //addps         xmm0 , xmm1
  //movshdup      xmm1 , xmm0
  //addss         xmm0 , xmm1

end;

procedure QNNFusedMulScale_avx2(const N:NativeInt; const dst, a, b: PSingle; const scale:Single);assembler;
asm
{$ifdef MSWINDOWS}
  movss         xmm4 , dword[scale]
{$else}
  movss         xmm4 , scale
{$endif}
  vbroadcastss  ymm4 , xmm4

  mov           rax  , N
  shr           rax  , 5
  jz            @rem1

@while:
  vmulps        ymm0 , ymm4, yword[a]
  vmulps        ymm1 , ymm4, yword[a+32*1]
  vmulps        ymm2 , ymm4, yword[a+32*2]
  vmulps        ymm3 , ymm4, yword[a+32*3]

  vmulps        ymm0 , ymm0, yword[b]
  vmulps        ymm1 , ymm1, yword[b+32*1]
  vmulps        ymm2 , ymm2, yword[b+32*2]
  vmulps        ymm3 , ymm3, yword[b+32*3]

  vmovups       yword[dst]      , ymm0
  vmovups       yword[dst+32*1] , ymm1
  vmovups       yword[dst+32*2] , ymm2
  vmovups       yword[dst+32*3] , ymm3

  add           a    , 32*4
  add           b    , 32*4
  add           dst  , 32*4
  dec           rax
  jnz           @while

@rem1:
  mov           rax  , N
  and           rax  , 31
  shr           rax  , 3
  jz            @rem

@while1:
  vmulps        ymm0 , ymm4, yword[a]
  vmulps        ymm0 , ymm0, yword[b]
  vmovups       yword[dst] , ymm0

  add           a    , 32
  add           b    , 32
  add           dst  , 32
  dec           rax
  jnz           @while1

@rem:
  mov           rax  , N
  and           rax  , 7
  jz            @done
  vmovd         xmm5 , eax
  vpbroadcastd  ymm5 , xmm5
  vpcmpgtd      ymm5 , ymm5, YMMD[rip]
  vmaskmovps    ymm0 , ymm5, yword[a]
  vmaskmovps    ymm1 , ymm5, yword[b]
  vmulps        ymm0,  ymm0, ymm4
  vmulps        ymm0,  ymm0, ymm1
  vmaskmovps    yword[dst] , ymm5, ymm0
@done:

end;

procedure QNNFusedScaleBias_avx2(const N:NativeInt; const dst, src:PSingle; const scale, bias:Single);assembler;
asm
{$ifdef MSWINDOWS}
  movss         xmm5 , dword[bias]
{$else}
  movss         xmm5 , bias
{$endif}
  vbroadcastss  ymm4 , scale
  vbroadcastss  ymm5 , xmm5

  mov           rax  , N
  shr           rax  , 5
  jz            @rem1

@while:

  vmovups       ymm0 , yword [src]
  vmovups       ymm1 , yword [src+32*1]
  vmovups       ymm2 , yword [src+32*2]
  vmovups       ymm3 , yword [src+32*3]

  vfmadd132ps   ymm0 , ymm5 , ymm4
  vfmadd132ps   ymm1 , ymm5 , ymm4
  vfmadd132ps   ymm2 , ymm5 , ymm4
  vfmadd132ps   ymm3 , ymm5 , ymm4

  vmovups       yword[dst]      , ymm0
  vmovups       yword[dst+32*1] , ymm1
  vmovups       yword[dst+32*2] , ymm2
  vmovups       yword[dst+32*3] , ymm3

  add           src  , 32*4
  add           dst  , 32*4
  dec           rax
  jnz           @while

@rem1:
  mov           rax  , N
  and           rax  , 31
  shr           rax  , 3
  jz            @rem

@while1:
  vmovups       ymm0 , yword [src]
  vfmadd132ps   ymm0 , ymm5 , ymm4
  vmovups       yword[dst]  , ymm0

  add           src  , 32
  add           dst  , 32
  dec           rax
  jnz           @while1

@rem:
  mov           rax  , N
  and           rax  , 7
  jz            @done
  vmovd         xmm1 , eax
  vpbroadcastd  ymm1 , xmm1
  vpcmpgtd      ymm1 , ymm1, YMMD[rip]
  vmaskmovps    ymm0 , ymm1, yword[src]
  vfmadd132ps   ymm0 , ymm5 , ymm4
  vmaskmovps    yword[dst] , ymm5, ymm0
@done:

end;

{$endif}

