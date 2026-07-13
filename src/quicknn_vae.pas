unit quicknn_vae;

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

interface

uses
  SysUtils, {$if defined(USE_CPU)} quicknn_cpu {$else} quicknn_kernels{$endif}, safetensor, quicknn_common;


type

  { TVAEResBlock }

  TVAEResBlock = record
    norm1_weight, norm1_bias,                (* [channels] *)
    conv1_weight, conv1_bias,                (* [out_ch, in_ch, 3, 3] *)
    norm2_weight, norm2_bias,                (* [channels] *)
    conv2_weight, conv2_bias,                (* [out_ch, out_ch, 3, 3] *)
    skip_weight , skip_bias : TMemoryBlock;  (* [out_ch, in_ch, 1, 1] if in_ch != out_ch *)
    in_channels, out_channels : longint;
    procedure forward(const dst, src, work: TMemoryBlock; const batch, H, W, num_groups: longint);
    //procedure load(var f: file);                                     overload;
    procedure load(const sf: TSafetensorFiles; const aPrefix:string; const in_ch, out_ch:longint; const useMMap:boolean = false);    overload;
    procedure free();
  end;

  { TVAEAttnBlock }

  TVAEAttnBlock = record
    norm_weight, norm_bias,                   (* [channels] *)
    q_weight, q_bias,                         (* [channels, channels, 1, 1] *)
    k_weight, k_bias,                         (* [channels, channels, 1, 1] *)
    v_weight, v_bias,                         (* [channels, channels, 1, 1] *)
    out_weight, out_bias : TMemoryBlock;      (* [channels, channels, 1, 1] *)
    channels : longint;
    procedure forward(const dst, src, work: TMemoryBlock; const batch, H, W, num_groups: longint);
    //procedure load(var f:file);                                    overload;
    procedure load(const sf:TSafetensorFiles; const aPrefix:string; const aChannels:longint; const useMMap:boolean = false);   overload;
    procedure free();
  end;

  { TVAEDownSample }

  TVAEDownSample = record
    conv_weight, conv_bias : TMemoryBlock;     (* [channels, channels, 3, 3] *)
    channels : longint;
    procedure free();
  end;

  TVAEUpSample = type TVAEDownSample;

  (* VAE context *)

  { TVAE }
  PVAE = ^TVAE;
  TVAE = record
      (* Configuration *)
      z_channels,         (* 32 (Flux) or 16 (Z-Image) *)
      latent_channels,    (* z_channels * 4: 128 (Flux) or 64 (Z-Image) *)
      base_channels,      (* 128 *)
      num_res_blocks,     (* 2 *)
      num_groups : longint;         (* 32 *)
      ch_mult : TCHMult;         (* {1, 2, 4, 4} *)
      eps, scaling_factor, shift_factor : single;     (* 0 = use batch norm (Flux), else Z-Image shift *)

      (* Encoder weights *)
      enc_conv_in_weight, enc_conv_in_bias : TMemoryBlock;   (* [128, 3, 3, 3] *)

      (* Encoder down blocks: 4 levels, each with num_res_blocks + optional downsample *)
      enc_down_blocks : TArray<TVAEResBlock>;    (* 4 * 2 = 8 resblocks *)
      enc_downsample : TArray<TVaeDownSample>;   (* 3 downsamples (not at last level) *)

      (* Encoder mid block *)
      enc_mid_block1 : TVAEResBlock  ;
      enc_mid_attn   : TVAEAttnBlock ;
      enc_mid_block2 : TVAEResBlock  ;

      (* Encoder output *)
      enc_norm_out_weight, enc_norm_out_bias, (*[512] *)
      enc_conv_out_weight, enc_conv_out_bias : TMemoryBlock; (* [64, 512, 3, 3] *)

      (* Decoder weights *)
      dec_conv_in_weight, dec_conv_in_bias : TMemoryBlock;   (* [512, z_channels, 3, 3] *)

      (* Decoder mid block *)
      dec_mid_block1: TVAEResBlock  ;
      dec_mid_attn  : TVAEAttnBlock ;
      dec_mid_block2: TVAEResBlock  ;

      (* Decoder up blocks: 4 levels, each with num_res_blocks+1 + optional upsample *)
      dec_up_blocks : TArray<TVAEResBlock>;      (* 4 * 3 = 12 resblocks *)
      dec_upsample  : TArray<TVAEUpSample>;       (* 3 upsamples *)

      (* Decoder output *)
      dec_norm_out_weight, dec_norm_out_bias, (* [128] *)
      dec_conv_out_weight, dec_conv_out_bias : TMemoryBlock; (* [3, 128, 3, 3] *)

      (* Normalization stats for latent space (Flux only, NULL for Z-Image) *)
      bn_mean: TMemoryBlock;         (* [latent_channels] *)
      bn_var : TMemoryBlock;          (* [latent_channels] *)

      (* Post-quantization conv (1x1) - Flux only, NULL for Z-Image *)
      quant_conv_weight,       (* [z_ch*2, z_ch*2, 1, 1] - encoder *)
      quant_conv_bias,         (* [z_ch*2] *)
      post_quant_conv_weight,  (* [z_ch, z_ch, 1, 1] - decoder *)
      post_quant_conv_bias : TMemoryBlock;    (* [z_ch] *)

      (* Working memory (allocated for max image size) *)
      max_h, max_w : longint;
      work1, work2, work3: TMemoryBlock; // PSingle;
      work_size : IntPtr;
      useMMap : boolean;
      function encode(const img: TMemoryBlock; const batch, H, W: longint; var out_h, out_w: longint): TMemoryBlock;
      function decode(const latent: TMemoryBlock; const batch, latent_h, latent_w: longint):TQNNImage;   overload;
      procedure decode(const dst, latent: TMemoryBlock; const batch, latent_h, latent_w: longint);       overload;
      //procedure load(var f: file);                                    overload;
      procedure load(const sf:TSafetensorFiles; const zChannels:longint; const scalingFactor, shiftFactor:QNNFloat);  overload;
      function isLoaded():boolean;
      class procedure padRightBottom(const dst, src: TMemoryBlock; batch, channels, H, W: longint); static;
      procedure free;
  end;


implementation

function readInt(var f:file):longint;
begin
    blockread(f, result, sizeof(longint))
end;

function readSingles(var f:file; const count:longint):TMemoryBlock;
var s : string;
begin
    s := TFileRec(f).name;
    result := TMemoryBlock.Create(count, s+'['+IntToStr(FilePos(f) div 128)+']');
    blockread(f, PSingle(result)^, count*sizeof(single))
end;

{ TVAEResBlock }

procedure TVAEResBlock.forward(const dst, src, work: TMemoryBlock; const batch,
  H, W, num_groups: longint);
var
    spatial: longint;
    conv1_out: TMemoryBlock;
begin
    spatial := H * W;
    if in_channels = out_channels then
        QNNCopy(dst, src, batch*in_channels*spatial)
    else
        QNNConv2d(dst, src, skip_weight, skip_bias, in_channels, out_channels, H, W, 1, 1, 1, 0, batch);
//printCompare(batch*in_channels*spatial, dst, readTensor());
    QNNGroupNorm(work, src, norm1_weight, norm1_bias, batch, in_channels, H, W, num_groups);
//printCompare(batch*in_channels*spatial, work, readTensor());
    QNNSiluInplace(work, batch*in_channels*spatial);
//printCompare(batch*in_channels*spatial, work, readTensor());
    conv1_out := work + batch*in_channels*spatial;
    QNNConv2d(conv1_out, work, conv1_weight, conv1_bias, in_channels, out_channels, H, W, 3, 3, 1, 1, batch);
    QNNGroupNorm(work, conv1_out, norm2_weight, norm2_bias, batch, out_channels, H, W, num_groups);
    QNNSiluInplace(work, batch*out_channels*spatial);
    QNNConv2d(conv1_out, work, conv2_weight, conv2_bias, out_channels, out_channels, H, W, 3, 3, 1, 1, batch);
    QNNAddInplace(dst, conv1_out, batch*out_channels*spatial);
//printCompare(batch*out_channels*spatial, dst, readTensor());
end;

//procedure TVAEResBlock.load(var f: file);
//begin
//  in_channels  := readInt(f);
//  out_channels := readInt(f);
//
//  norm1_weight := readSingles(f, in_channels);
//  norm1_bias   := readSingles(f, in_channels);
//  conv1_weight := readSingles(f, out_channels * in_channels * 3 * 3);
//  conv1_bias   := readSingles(f, out_channels);
//  norm2_weight := readSingles(f, out_channels);
//  norm2_bias   := readSingles(f, out_channels);
//  conv2_weight := readSingles(f, out_channels * out_channels * 3 * 3);
//  conv2_bias   := readSingles(f, out_channels);
//
//  if (in_channels <> out_channels) then begin
//      skip_weight := readSingles(f, out_channels * in_channels);
//      skip_bias   := readSingles(f, out_channels);
//  end else begin
//      //skip_weight := nil;
//      //skip_bias := nil;
//  end;
//
//end;

procedure TVAEResBlock.load(const sf: TSafetensorFiles; const aPrefix: string;
  const in_ch, out_ch: longint; const useMMap: boolean);
begin
    in_channels := in_ch;
    out_channels := out_ch;
    norm1_weight := sf.getTensorDataMemBlock(format('%s.norm1.weight', [aPrefix]), useMMap);
    norm1_bias   := sf.getTensorDataMemBlock(format('%s.norm1.bias'  , [aPrefix]), useMMap);
    conv1_weight := sf.getTensorDataMemBlock(format('%s.conv1.weight', [aPrefix]), useMMap);
    conv1_bias   := sf.getTensorDataMemBlock(format('%s.conv1.bias'  , [aPrefix]), useMMap);
    norm2_weight := sf.getTensorDataMemBlock(format('%s.norm2.weight', [aPrefix]), useMMap);
    norm2_bias   := sf.getTensorDataMemBlock(format('%s.norm2.bias'  , [aPrefix]), useMMap);
    conv2_weight := sf.getTensorDataMemBlock(format('%s.conv2.weight', [aPrefix]), useMMap);
    conv2_bias   := sf.getTensorDataMemBlock(format('%s.conv2.bias'  , [aPrefix]), useMMap);
    if in_ch <> out_ch then
      begin
        skip_weight := sf.getTensorDataMemBlock(format('%s.conv_shortcut.weight', [aPrefix]), useMMap);
        skip_bias   := sf.getTensorDataMemBlock(format('%s.conv_shortcut.bias', [aPrefix]), useMMap)
      end
    else
      begin
        //skip_weight := nil;
        //skip_bias   := nil
      end;

end;

procedure TVAEResBlock.free();
begin
  norm1_weight.free;
  norm1_bias.free;                (* [channels] *)
  conv1_weight.free;
  conv1_bias.free;                (* [out_ch, in_ch, 3, 3] *)
  norm2_weight.free;
  norm2_bias.free;                (* [channels] *)
  conv2_weight.free;
  conv2_bias.free;                (* [out_ch, out_ch, 3, 3] *)
  skip_weight.free;
  skip_bias.free;
end;

{ TVAEAttnBlock }

procedure TVAEAttnBlock.forward(const dst, src, work: TMemoryBlock;
  const batch, H, W, num_groups: longint);
var
    ch, spatial, c, i, b: longint;
    q, k, v, qb, kb, vb, ob, attn_out{, q_ptr, k_ptr, v_ptr, o_ptr}: TMemoryBlock;
    scale: QNNFloat;
    q_t, k_t, v_t, o_t, scores: TMemoryBlock;
begin
    ch := channels;
    spatial := H * W;
    QNNGroupNorm(work, src, norm_weight, norm_bias, batch, ch, H, W, num_groups);
    Q := work + batch*ch*spatial;
    K := Q + batch*ch*spatial;
    V := K + batch*ch*spatial;
    QNNConv2d(q, work, q_weight, q_bias, ch, ch, H, W, 1, 1, 1, 0, batch);
    QNNConv2d(k, work, k_weight, k_bias, ch, ch, H, W, 1, 1, 1, 0, batch);
    QNNConv2d(v, work, v_weight, v_bias, ch, ch, H, W, 1, 1, 1, 0, batch);
    scale := 1.0 / sqrt(ch);
    attn_out := v + batch*ch*spatial;
    q_t    := TMemoryBlock.Create(spatial * ch, 'VAE_FW_Q ' + TGUID.NewGuid.ToString()); //q_ptr := q_t;
    k_t    := TMemoryBlock.Create(spatial * ch, 'VAE_FW_K ' + TGUID.NewGuid.ToString()); //k_ptr := k_t;
    v_t    := TMemoryBlock.Create(spatial * ch, 'VAE_FW_V ' + TGUID.NewGuid.ToString()); //v_ptr := v_t;
    o_t    := TMemoryBlock.Create(spatial * ch, 'VAE_FW_O ' + TGUID.NewGuid.ToString()); //o_ptr := o_t;
    scores := TMemoryBlock.Create(spatial * spatial, 'VAE_FQ_SCORES ' + TGUID.NewGuid.ToString());
    for b := 0 to batch -1 do
        begin
            qb := Q + b * ch * spatial;
            kb := K + b * ch * spatial;
            vb := V + b * ch * spatial;
            ob := attn_out + b * ch * spatial;
            QNNMatTranspose(q_t, qb, ch, spatial);
            QNNScale(q_t, q_t, scale, spatial);
            QNNMatTranspose(k_t, kb, ch, spatial);
            QNNMatTranspose(v_t, vb, ch, spatial);

            //for c := 0 to ch -1 do
            //    for i := 0 to spatial -1 do
            //        begin
            //            q_ptr[i * ch+c] := qb[c * spatial+i] * scale;
            //            k_ptr[i * ch+c] := kb[c * spatial+i];
            //            v_ptr[i * ch+c] := vb[c * spatial+i]
            //        end;
            QNNMatMulNT(scores, q_t, k_t, spatial, ch, spatial);
            QNNSoftmaxRows(scores, spatial, spatial);
            QNNMatMulNN(o_t, scores, v_t, spatial, spatial, ch);
            QNNMatTranspose(ob, o_t, spatial, ch);
            //for c := 0 to ch -1 do
            //    for i := 0 to spatial -1 do
            //        ob[c * spatial+i] := o_ptr[i * ch+c]
        end;
    q_t   .free;
    k_t   .free;
    v_t   .free;
    o_t   .free;
    scores.free;
    QNNConv2d(work, attn_out, out_weight, out_bias, ch, ch, H, W, 1, 1, 1, 0, batch);
    QNNAdd(dst, src, work, batch * ch * spatial);
end;

//procedure TVAEAttnBlock.load(var f: file);
//begin
//  channels    := readInt(f);
//  norm_weight := readSingles(f, channels);
//  norm_bias   := readSingles(f, channels);
//  q_weight    := readSingles(f, channels * channels);
//  q_bias      := readSingles(f, channels);
//  k_weight    := readSingles(f, channels * channels);
//  k_bias      := readSingles(f, channels);
//  v_weight    := readSingles(f, channels * channels);
//  v_bias      := readSingles(f, channels);
//  out_weight  := readSingles(f, channels * channels);
//  out_bias    := readSingles(f, channels);
//end;

procedure TVAEAttnBlock.load(const sf: TSafetensorFiles; const aPrefix: string;
  const aChannels: longint; const useMMap: boolean);
begin
    channels := aChannels;

    norm_weight := sf.getTensorDataMemBlock(format('%s.group_norm.weight', [aPrefix]), useMMap);
    norm_bias   := sf.getTensorDataMemBlock(format('%s.group_norm.bias', [aPrefix]), useMMap);

    q_weight    := sf.getTensorDataMemBlock(format('%s.to_q.weight', [aPrefix]), useMMap);
    q_bias      := sf.getTensorDataMemBlock(format('%s.to_q.bias', [aPrefix]), useMMap);

    k_weight    := sf.getTensorDataMemBlock(format('%s.to_k.weight', [aPrefix]), useMMap);
    k_bias      := sf.getTensorDataMemBlock(format('%s.to_k.bias', [aPrefix]), useMMap);

    v_weight     := sf.getTensorDataMemBlock(format('%s.to_v.weight', [aPrefix]), useMMap);
    v_bias       := sf.getTensorDataMemBlock(format('%s.to_v.bias', [aPrefix]), useMMap);

    out_weight   := sf.getTensorDataMemBlock(format('%s.to_out.0.weight', [aPrefix]), useMMap);
    out_bias     := sf.getTensorDataMemBlock(format('%s.to_out.0.bias', [aPrefix]), useMMap);
end;

procedure TVAEAttnBlock.free();
begin
  norm_weight.free;
  norm_bias.free;                   (* [channels] *)
  q_weight.free;
  q_bias.free;                         (* [channels, channels, 1, 1] *)
  k_weight.free;
  k_bias.free;                         (* [channels, channels, 1, 1] *)
  v_weight.free;
  v_bias.free;                         (* [channels, channels, 1, 1] *)
  out_weight.free;
  out_bias.free;
end;

{ TVAEDownSample }

procedure TVAEDownSample.free();
begin
  conv_weight.free();
  conv_bias.free;
end;


{ TVAE }

function TVAE.isLoaded(): boolean;
begin
  result := enc_conv_in_weight.isAssigned() and dec_conv_in_weight.isAssigned();
end;

function TVAE.encode(const img: TMemoryBlock; const batch, H, W: longint; var out_h, out_w: longint): TMemoryBlock;
//const ch_mult : array[0..3] of longint = (1, 2, 4, 4);
var
   cur_h, cur_w, block_idx, down_idx
    , progress, total_blocks, level, ch_out, r, padded_h, padded_w, new_h, new_w
    , mid_ch, z_ch, latent_h, latent_w, z_spatial, b, patch_h, patch_w, lat_ch, n, i: longint;
    x, work, padded : TMemoryBlock;
    mean: TMemoryBlock;
begin
    if assigned(phase_callback) then
        phase_callback('VAE Encode', false);
    x := work1;
    work := work2;
    cur_h := H; cur_w := W;
    QNNConv2d(x, img, enc_conv_in_weight, enc_conv_in_bias, 3, base_channels, H, W, 3, 3, 1, 1, batch);
    block_idx := 0;
    down_idx := 0;
    progress := 0;
    total_blocks := 4 * num_res_blocks+3;
    for level := 0 to 4 -1 do begin
        ch_out := base_channels * ch_mult[level];
        for r := 0 to num_res_blocks -1 do begin
            enc_down_blocks[block_idx].forward(work, x, work3, batch, cur_h, cur_w, num_groups);
            inc(block_idx);
            QNNCopy(x, work, batch * ch_out * cur_h * cur_w);
            if assigned(vae_progress_callback) then begin
                vae_progress_callback(progress, total_blocks);
                inc(progress);
            end
        end;
        if level < 3 then begin
            padded := work3;
            padded_h := cur_h+1;
            padded_w := cur_w+1;
            new_h := (padded_h-3) div 2+1;
            new_w := (padded_w-3) div 2+1;
            padRightBottom(padded, x, batch, ch_out, cur_h, cur_w);
            QNNConv2d(work, padded, enc_downsample[down_idx].conv_weight, enc_downsample[down_idx].conv_bias, ch_out, ch_out, padded_h, padded_w, 3, 3, 2, 0, batch);
            inc(down_idx);
            cur_h := new_h;
            cur_w := new_w;
            QNNCopy(x, work, batch*ch_out*cur_h*cur_w);
        end
    end;
    mid_ch := base_channels * ch_mult[3];
    enc_mid_block1.forward(work, x, work3, batch, cur_h, cur_w, num_groups);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    enc_mid_attn.forward(x, work, work3, batch, cur_h, cur_w, num_groups);// < 0 then
        //exit(nil);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    enc_mid_block2.forward(work, x, work3, batch, cur_h, cur_w, num_groups);
    QNNCopy(x, work, batch * mid_ch * cur_h * cur_w);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    QNNGroupNorm(work, x, enc_norm_out_weight, enc_norm_out_bias, batch, mid_ch, cur_h, cur_w, num_groups);
    QNNSiluInplace(work, batch * mid_ch * cur_h * cur_w);
    z_ch := z_channels * 2;
    QNNConv2d(x, work, enc_conv_out_weight, enc_conv_out_bias, mid_ch, z_ch, cur_h, cur_w, 3, 3, 1, 1, batch);
    if boolean(quant_conv_weight) then
        begin
            QNNConv2d(work, x, quant_conv_weight, quant_conv_bias, z_ch, z_ch, cur_h, cur_w, 1, 1, 1, 0, batch);
            QNNCopy(x, work, batch * z_ch * cur_h * cur_w)
        end;
    latent_h := cur_h;
    latent_w := cur_w;
    z_spatial := latent_h * latent_w;
    mean := TMemoryBlock.Create([batch, z_channels, latent_h, latent_w], 'VAE_ENCODE_MEAN '+ TGUID.NewGuid.ToString());
    for b := 0 to batch -1 do
        QNNCopy(mean + b*z_channels*z_spatial, x + b*z_ch*z_spatial, z_channels*z_spatial);
    patch_h := latent_h div 2;
    patch_w := latent_w div 2;
    lat_ch := latent_channels;
    result := TMemoryBlock.Create([batch, lat_ch, patch_h, patch_w], 'VAE_ENCODE_RESULT '+ TGUID.NewGuid.ToString());
    QNNPatchify(result, mean, batch, z_channels, latent_h, latent_w, 2);
    mean.free;
    if scaling_factor <> 0.0 then begin
        n := batch * lat_ch * patch_h * patch_w;
        QNNFusedBiasScale(result, result, -shift_factor, scaling_factor, N);
        //for i := 0 to n -1 do
        //    latent[i] := (latent[i]-shift_factor) * scaling_factor
    end else begin
        QNNBatchNorm(work, result, bn_mean, bn_var, batch, lat_ch, patch_h, patch_w);
        QNNCopy(result, work, batch * lat_ch * patch_h * patch_w)
    end;
     out_h := patch_h;
     out_w := patch_w;
    if assigned(phase_callback) then
        phase_callback('VAE Encode', true);
end;

function TVAE.decode(const latent: TMemoryBlock; const batch, latent_h,
  latent_w: longint): TQNNImage;
//const ch_mult : array[0..3] of integer = (1, 2, 4, 4);
var
    x, work:TMemoryBlock; mean_ptr, var_ptr: PQNNFloat;
    lat_ch, z_spatial, n, i, b, c, idx, unpatch_h, unpatch_w, cur_w, cur_h, mid_ch, progress,
     total_blocks, block_idx, up_idx, level, ch_out, r, new_h, new_w, out_ch, H, W, y, ch: longint;
    mean, std, val: single;

begin
    if assigned(phase_callback) then
        phase_callback('VAE Decode', false);
    x := work1;
    work := work2;
    lat_ch := latent_channels;
    z_spatial := latent_h * latent_w;
    QNNCopy(x, latent, batch * lat_ch * z_spatial);
//printCompare( batch * lat_ch * z_spatial, x, readTensor());
    if scaling_factor <> 0.0 then
        QNNFusedScaleBias(x, x, 1/scaling_factor, shift_factor, batch*lat_ch*z_spatial)
        //begin
        //    n := batch * lat_ch * z_spatial;
        //    for i := 0 to n -1 do
        //        x[i] := x[i]/scaling_factor + shift_factor
        //end
    else begin
        //QNNBatchNorm(x, x, bn_mean, bn_var, nil, nil, batch, lat_ch, latent_h, latent_w);
        mean_ptr := bn_mean;
        var_ptr := bn_var;

        for c := 0 to lat_ch -1 do begin
            mean := mean_ptr[c];
            std := sqrt(var_ptr[c]+eps);
            for b := 0 to batch -1 do
                QNNFusedScaleBias(x+(b*lat_ch + c)*z_spatial, x+(b*lat_ch + c)*z_spatial, std, mean, z_spatial);
                //for i := 0 to z_spatial -1 do
                //    begin
                //        idx := b * lat_ch * z_spatial+c * z_spatial+i;
                //        x[idx] := x[idx] * std + mean
                //    end
        end;
    end;
//printCompare(batch * lat_ch * z_spatial, x, readTensor());
    unpatch_h := latent_h * 2;
    unpatch_w := latent_w * 2;
    QNNUnpatchify(work, x, batch, z_channels, latent_h, latent_w, 2);
    QNNCopy(x, work, batch * z_channels * unpatch_h * unpatch_w);
//printCompare(batch * z_channels * unpatch_h * unpatch_w, x, readTensor());
    cur_h := unpatch_h; cur_w := unpatch_w;
    if boolean(post_quant_conv_weight) then begin
      QNNConv2d(work, x, post_quant_conv_weight, post_quant_conv_bias, z_channels, z_channels, cur_h, cur_w, 1, 1, 1, 0, batch);
      QNNCopy(x, work, batch*z_channels*cur_h*cur_w)
    end;
    mid_ch := base_channels * ch_mult[3];
    QNNConv2d(work, x, dec_conv_in_weight, dec_conv_in_bias, z_channels, mid_ch, cur_h, cur_w, 3, 3, 1, 1, batch);
    QNNCopy(x, work, batch*mid_ch*cur_h*cur_w);
//printCompare(batch*mid_ch*cur_h*cur_w, x, readTensor());

    progress := 0;
    total_blocks := 3+4 * (num_res_blocks+1);
    dec_mid_block1.forward(work, x, work3, batch, cur_h, cur_w, num_groups);

    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    dec_mid_attn.forward(x, work, work3, batch, cur_h, cur_w, num_groups);// < 0 then

    //exit(nil);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    dec_mid_block2.forward(work, x, work3, batch, cur_h, cur_w, num_groups);
    QNNCopy(x, work, batch*mid_ch*cur_h*cur_w);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    block_idx := 0;
    up_idx := 0;
    level := 3;
    while level >= 0 do begin
        ch_out := base_channels * ch_mult[level];
        for r := 0 to num_res_blocks+1 -1 do
            begin
                dec_up_blocks[block_idx].forward(work, x, work3, batch, cur_h, cur_w, num_groups);
                inc(block_idx);
                QNNCopy(x, work, batch * ch_out * cur_h * cur_w);
                if assigned(vae_progress_callback) then begin
                    vae_progress_callback(progress, total_blocks);
                    inc(progress)
                end;
            end;
        if level > 0 then
            begin
                new_h := cur_h * 2;
                new_w := cur_w * 2;
                QNNUpSampleNearest(work, x, batch, ch_out, cur_h, cur_w, 2, 2);
                QNNConv2d(x, work, dec_upsample[up_idx].conv_weight, dec_upsample[up_idx].conv_bias, ch_out, ch_out, new_h, new_w, 3, 3, 1, 1, batch);
                inc(up_idx);
                cur_h := new_h;
                cur_w := new_w
            end;
        dec(level)
    end;
    out_ch := base_channels;
    QNNGroupNorm(work, x, dec_norm_out_weight, dec_norm_out_bias, batch, out_ch, cur_h, cur_w, num_groups);
    QNNSiluInplace(work, batch * out_ch * cur_h * cur_w);
    QNNConv2d(x, work, dec_conv_out_weight, dec_conv_out_bias, out_ch, 3, cur_h, cur_w, 3, 3, 1, 1, batch);
    H := cur_h;
    W := cur_w;

    result := TQNNImage.Create(W, H, 3, x);
    //if not assigned(result.data) then
    //    exit(nil);
    //for y := 0 to H -1 do
    //  for c := 0 to W -1 do
    //    for ch := 0 to 3 -1 do begin
    //      val := x[ch * H * W+y * W+c];
    //      val := (val+1.0) * 0.5;
    //      val := val * 255.0;
    //      if val < 0 then
    //          val := 0;
    //      if val > 255 then
    //          val := 255;
    //      result.data[(y*W + c)*3 + ch] := round(val)
    //    end;
    if assigned(phase_callback) then
        phase_callback('VAE Decode', true);

end;

procedure TVAE.decode(const dst, latent: TMemoryBlock; const batch, latent_h,
  latent_w: longint);
const ch_mult : array[0..3] of integer = (1, 2, 4, 4);
var
    x, work:TMemoryBlock; mean_ptr, var_ptr: PQNNFloat;
    lat_ch, z_spatial, b, c, unpatch_h, unpatch_w, cur_w, cur_h, mid_ch, progress,
     total_blocks, block_idx, up_idx, level, ch_out, r, new_h, new_w, out_ch: longint;
    mean, std: single;
begin
    if assigned(phase_callback) then
        phase_callback('VAE Decode', false);
    x := work1;
    work := work2;
    lat_ch := latent_channels;
    z_spatial := latent_h * latent_w;
    QNNCopy(x, latent, batch * lat_ch * z_spatial);
    if scaling_factor <> 0.0 then
        QNNFusedScaleBias(x, x, 1/scaling_factor, shift_factor, batch*lat_ch*z_spatial)
        //begin
        //    n := batch * lat_ch * z_spatial;
        //    for i := 0 to n -1 do
        //        x[i] := x[i]/scaling_factor + shift_factor
        //end
    else begin
        QNNBatchNorm(x, x, bn_mean, bn_var, batch, lat_ch, latent_h, latent_w);
        // x is [batch, latent_channels, latent_h, latent_w]
        //mean_ptr := bn_mean;
        //var_ptr := bn_var;
        //for c := 0 to lat_ch -1 do begin
        //    mean := mean_ptr[c];
        //    std := sqrt(var_ptr[c]+eps);
        //    for b := 0 to batch -1 do
        //        QNNFusedScaleBias(x+(b*lat_ch + c)*z_spatial, x+(b*lat_ch + c)*z_spatial, std, mean, z_spatial);// x is [batch, latent_channels, latent_h, latent_w]
        //        //for i := 0 to z_spatial -1 do
        //        //    begin
        //        //        idx := b * lat_ch * z_spatial+c * z_spatial+i;
        //        //        x[idx] := x[idx] * std + mean
        //        //    end
        //end;
    end;
    unpatch_h := latent_h * 2;
    unpatch_w := latent_w * 2;
    QNNUnpatchify(work, x, batch, z_channels, latent_h, latent_w, 2);
    QNNCopy(x, work, batch * z_channels * unpatch_h * unpatch_w);
    // x and work is [batch, z_channels, unpatch_h, unpatch_w]
    cur_h := unpatch_h; cur_w := unpatch_w;
    if boolean(post_quant_conv_weight) then begin
      QNNConv2d(work, x, post_quant_conv_weight, post_quant_conv_bias, z_channels, z_channels, cur_h, cur_w, 1, 1, 1, 0, batch);
      QNNCopy(x, work, batch*z_channels*cur_h*cur_w)
      // x and work is [batch, z_channel, latent_h*2, latent_w*2]
    end;
    mid_ch := base_channels * ch_mult[3];
    QNNConv2d(work, x, dec_conv_in_weight, dec_conv_in_bias, z_channels, mid_ch, cur_h, cur_w, 3, 3, 1, 1, batch);
    QNNCopy(x, work, batch*mid_ch*cur_h*cur_w);
    progress := 0;
    total_blocks := 3+4 * (num_res_blocks+1);
    dec_mid_block1.forward(work, x, work3, batch, cur_h, cur_w, num_groups);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    dec_mid_attn.forward(x, work, work3, batch, cur_h, cur_w, num_groups);// < 0 then
        //exit(nil);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    dec_mid_block2.forward(work, x, work3, batch, cur_h, cur_w, num_groups);
    QNNCopy(x, work, batch*mid_ch*cur_h*cur_w);
    if assigned(vae_progress_callback) then begin
        vae_progress_callback(progress, total_blocks);
        inc(progress)
    end;
    block_idx := 0;
    up_idx := 0;
    level := 3;
    while level >= 0 do begin
        ch_out := base_channels * ch_mult[level];
        for r := 0 to num_res_blocks+1 -1 do
            begin
                dec_up_blocks[block_idx].forward(work, x, work3, batch, cur_h, cur_w, num_groups);
                inc(block_idx);
                QNNCopy(x, work, batch * ch_out * cur_h * cur_w);
                if assigned(vae_progress_callback) then begin
                    vae_progress_callback(progress, total_blocks);
                    inc(progress)
                end;
            end;
        if level > 0 then
            begin
                new_h := cur_h * 2;
                new_w := cur_w * 2;
                QNNUpSampleNearest(work, x, batch, ch_out, cur_h, cur_w, 2, 2);
                QNNConv2d(x, work, dec_upsample[up_idx].conv_weight, dec_upsample[up_idx].conv_bias, ch_out, ch_out, new_h, new_w, 3, 3, 1, 1, batch);
                inc(up_idx);
                cur_h := new_h;
                cur_w := new_w
            end;
        dec(level)
    end;
    out_ch := base_channels;
    QNNGroupNorm(work, x, dec_norm_out_weight, dec_norm_out_bias, batch, out_ch, cur_h, cur_w, num_groups);
    QNNSiluInplace(work, batch * out_ch * cur_h * cur_w);
    QNNConv2d(x, work, dec_conv_out_weight, dec_conv_out_bias, out_ch, 3, cur_h, cur_w, 3, 3, 1, 1, batch);
    QNNCopy(dst, x, 4*cur_w*cur_h);
    if assigned(phase_callback) then
        phase_callback('VAE Decode', true);
end;

//procedure TVAE.load(var f: file);
//var i, ch, mid_ch:longint;
//begin
//    z_channels      := readInt(f);
//    latent_channels := z_channels * 4;  (* 2x2 patchify *)
//    base_channels   := readInt(f);
//    num_res_blocks  := readInt(f);
//    num_groups      := readInt(f);
//    max_h           := readInt(f);
//    max_w           := readInt(f);
//    scaling_factor  := 0.0;
//    shift_factor    := 0.0;
//
//    ch_mult    := QNN_VAE_CH_MULT;
//    //ch_mult[0] := QNN_VAE_CH_MULT_0;
//    //ch_mult[1] := QNN_VAE_CH_MULT_1;
//    //ch_mult[2] := QNN_VAE_CH_MULT_2;
//    //ch_mult[3] := QNN_VAE_CH_MULT_3;
//
//    eps := EPSILON;  (* batch_norm_eps from config *)
//
//    (* Read encoder conv_in *)
//    enc_conv_in_weight := readSingles(f, base_channels*3*3*3);
//    enc_conv_in_bias   := readSingles(f, base_channels);
//
//    (* Read encoder down blocks *)
//    setLength(enc_down_blocks, 4*num_res_blocks);
//    for i := 0 to high(enc_down_blocks) do
//        enc_down_blocks[i].load(f);
//
//
//    (* Read encoder downsamples *)
//    setLength(enc_downsample, 3);;
//    for i:=0 to high(enc_downsample) do begin
//        ch := base_channels*ch_mult[i];
//        enc_downsample[i].channels := ch;
//        enc_downsample[i].conv_weight := readSingles(f, ch *ch*3*3);
//        enc_downsample[i].conv_bias   := readSingles(f, ch);
//    end;
//
//    (* Read encoder mid block *)
//    enc_mid_block1.load(f);
//    enc_mid_attn.load(f);
//    enc_mid_block2.load(f);
//
//    (* Read encoder output *)
//    mid_ch := base_channels*ch_mult[3];
//    enc_norm_out_weight := readSingles(f, mid_ch);
//    enc_norm_out_bias   := readSingles(f, mid_ch);
//    enc_conv_out_weight := readSingles(f, z_channels*2*mid_ch*3*3);
//    enc_conv_out_bias   := readSingles(f, z_channels*2);
//
//    (* Read decoder conv_in *)
//    dec_conv_in_weight := readSingles(f, mid_ch*z_channels*3*3);
//    dec_conv_in_bias   := readSingles(f, mid_ch);
//
//    (* Read decoder mid block *)
//    dec_mid_block1.load(f);
//    dec_mid_attn.load(f);
//    dec_mid_block2.load(f);
//
//    (* Read decoder up blocks *)
//    setLength(dec_up_blocks, 4 * (num_res_blocks + 1));
//    for i := 0 to high(dec_up_blocks) do
//        dec_up_blocks[i].load(f);
//
//    (* Read decoder upsamples *)
//    setLength(dec_upsample,3);
//    for i := 0 to high(dec_upsample) do begin
//        ch := base_channels * ch_mult[3 - i];
//        dec_upsample[i].channels := ch;
//        dec_upsample[i].conv_weight := readSingles(f, ch * ch * 3 * 3);
//        dec_upsample[i].conv_bias := readSingles(f, ch);
//    end;
//
//    (* Read decoder output *)
//    dec_norm_out_weight := readSingles(f, base_channels);
//    dec_norm_out_bias   := readSingles(f, base_channels);
//    dec_conv_out_weight := readSingles(f, 3*base_channels * 3 * 3);
//    dec_conv_out_bias   := readSingles(f, 3);
//
//    (* Read batch norm stats *)
//    bn_mean := readSingles(f, latent_channels);
//    bn_var  := readSingles(f, latent_channels);
//
//    (* Allocate working memory *)
//    work_size := 4*mid_ch*max_h*max_w;
//    work1     := TMemoryBlock.create(work_size);
//    work2     := TMemoryBlock.create(work_size);
//    work3     := TMemoryBlock.create(work_size);
//
//end;

procedure TVAE.load(const sf: TSafetensorFiles; const zChannels: longint;
  const scalingFactor, shiftFactor: QNNFloat);
var
    num_down_blocks, block_idx, level, ch, prev_ch, r, in_ch, i, mid_ch
     , num_up_blocks, up_idx, lc, max_spatial, max_channels: longint;
    bn_mean_t, bn_var_t: PSafeTensor;
begin

    z_channels := zChannels;
    latent_channels := z_channels * 4;
    base_channels := 128;
    num_res_blocks := 2;
    num_groups := 32;
    max_h := QNN_VAE_MAX_DIM;
    max_w := QNN_VAE_MAX_DIM;
    scaling_factor := scalingFactor;
    shift_factor := shiftFactor;
    if (scaling_factor <> 0.0) then eps := 1e-6 else eps := 1e-4;
    ch_mult := QNN_VAE_CH_MULT;
    enc_conv_in_weight := sf.getTensorDataMemBlock('encoder.conv_in.weight', useMMap);
    enc_conv_in_bias := sf.getTensorDataMemBlock('encoder.conv_in.bias', useMMap);
    num_down_blocks := 4 * num_res_blocks;
    setLength(enc_down_blocks, num_down_blocks);
    block_idx := 0;
    for level := 0 to 4 -1 do begin
        ch := base_channels * ch_mult[level];
        if (level = 0) then
            prev_ch := base_channels
        else
            prev_ch := base_channels * ch_mult[level-1];
        for r := 0 to num_res_blocks -1 do  begin
            if ((r = 0) and (level > 0)) then
                in_ch := prev_ch
            else
                in_ch := ch;
            enc_down_blocks[block_idx].load(sf, format('encoder.down_blocks.%d.resnets.%d', [level, r]), in_ch, ch, useMMap);
            inc(block_idx)
        end
    end;
    setLength(enc_downsample, 3);
    for i := 0 to high(enc_downsample) do begin
        ch := base_channels * ch_mult[i];
        enc_downsample[i].channels := ch;
        enc_downsample[i].conv_weight := sf.getTensorDataMemBlock(format('encoder.down_blocks.%d.downsamplers.0.conv.weight', [i]), useMMap);
        enc_downsample[i].conv_bias   := sf.getTensorDataMemBlock(format('encoder.down_blocks.%d.downsamplers.0.conv.bias', [i]), useMMap)
    end;
    mid_ch := base_channels * ch_mult[3];
    enc_mid_block1.load(sf, 'encoder.mid_block.resnets.0', mid_ch, mid_ch, useMMap);
    enc_mid_attn.load(sf, 'encoder.mid_block.attentions.0', mid_ch, useMMap);
    enc_mid_block2.load(sf, 'encoder.mid_block.resnets.1', mid_ch, mid_ch, useMMap);
    enc_norm_out_weight := sf.getTensorDataMemBlock('encoder.conv_norm_out.weight', useMMap);
    enc_norm_out_bias := sf.getTensorDataMemBlock('encoder.conv_norm_out.bias', useMMap);
    enc_conv_out_weight := sf.getTensorDataMemBlock('encoder.conv_out.weight', useMMap);
    enc_conv_out_bias := sf.getTensorDataMemBlock('encoder.conv_out.bias', useMMap);
    if assigned(sf.getTensor('quant_conv.weight')) then begin
        quant_conv_weight := sf.getTensorDataMemBlock('quant_conv.weight', useMMap);
        quant_conv_bias := sf.getTensorDataMemBlock('quant_conv.bias', useMMap)
    end;
    dec_conv_in_weight := sf.getTensorDataMemBlock('decoder.conv_in.weight', useMMap);
    dec_conv_in_bias := sf.getTensorDataMemBlock('decoder.conv_in.bias', useMMap);
    dec_mid_block1.load(sf, 'decoder.mid_block.resnets.0', mid_ch, mid_ch, useMMap);
    dec_mid_attn.load(sf, 'decoder.mid_block.attentions.0', mid_ch, useMMap);
    dec_mid_block2.load(sf, 'decoder.mid_block.resnets.1', mid_ch, mid_ch, useMMap);
    num_up_blocks := 4 * (num_res_blocks + 1);
    setLength(dec_up_blocks, num_up_blocks);
    block_idx := 0;
    level := 3;
    while level >= 0 do begin
        ch := base_channels * ch_mult[level];
        if (level = 3) then
            prev_ch := mid_ch
        else
            prev_ch := base_channels * ch_mult[level+1];
        for r := 0 to num_res_blocks+1 -1 do begin
            if (r = 0) then
                in_ch := prev_ch
            else
                in_ch := ch;
            up_idx := 3-level;
            dec_up_blocks[block_idx].load(sf, format('decoder.up_blocks.%d.resnets.%d', [up_idx, r]), in_ch, ch, useMMap);
            inc(block_idx);
        end;
        dec(level)
    end;
    setLength(dec_upsample, 3);
    for i := 0 to high(dec_upsample) do begin
      ch := base_channels * ch_mult[3-i];
      dec_upsample[i].channels := ch;
      dec_upsample[i].conv_weight := sf.getTensorDataMemBlock(format('decoder.up_blocks.%d.upsamplers.0.conv.weight', [i]), useMMap);
      dec_upsample[i].conv_bias := sf.getTensorDataMemBlock(format('decoder.up_blocks.%d.upsamplers.0.conv.bias', [i]), useMMap)
    end;
    dec_norm_out_weight := sf.getTensorDataMemBlock('decoder.conv_norm_out.weight', useMMap);
    dec_norm_out_bias   := sf.getTensorDataMemBlock('decoder.conv_norm_out.bias', useMMap);
    dec_conv_out_weight := sf.getTensorDataMemBlock('decoder.conv_out.weight', useMMap);
    dec_conv_out_bias   := sf.getTensorDataMemBlock('decoder.conv_out.bias', useMMap);
    bn_mean_t := sf.getTensor('bn.running_mean');
    if assigned(bn_mean_t) then begin
        bn_mean := bn_mean_t.asSingle(useMMap);
        bn_var_t := sf.getTensor('bn.running_var');
        if assigned(bn_var_t) then
            bn_var := bn_var_t.asSingle(useMMap)
        else
            bn_var := default(TMemoryBlock) // nil
    end;
    if assigned(sf.getTensor('post_quant_conv.weight')) then begin
        post_quant_conv_weight := sf.getTensorDataMemBlock('post_quant_conv.weight', useMMap);
        post_quant_conv_bias := sf.getTensorDataMemBlock('post_quant_conv.bias', useMMap)
    end;
    if not boolean(bn_mean) and (scaling_factor = 0.0) then begin
        lc := latent_channels;
        bn_mean := TMemoryBlock.Create(lc, 'VAE_LOAD_BN_MEAN');
        bn_var := TMemoryBlock.Create(lc, 'VAE_LOAD_BN_VAR');
        QNNFill(bn_var, 1.0, lc);
    end;
    max_spatial := max_h * max_w;
    max_channels := base_channels;
    work_size := 4 * max_channels * max_spatial;

    work1 := TMemoryBlock.Create(work_size, 'VAE_WORK1');//calloc(work_size, sizeof(single));//;
    work2 := TMemoryBlock.Create(work_size, 'VAE_WORK2');//calloc(work_size, sizeof(single));//;
    work3 := TMemoryBlock.Create(work_size, 'VAE_WORK3');//calloc(work_size, sizeof(single));//
end;

class procedure TVAE.padRightBottom(const dst, src: TMemoryBlock; batch,
  channels, H, W: longint);
var
    Hp, Wp, b, c: longint;
    in_plane, out_plane: IntPtr;
    s, d: TMemoryBlock;
    y: longint;
begin
    Hp := H+1;
    Wp := W+1;
    in_plane := H * W;
    out_plane := Hp * Wp;
    QNNFill(dst, 0, batch*channels*out_plane);
    for b := 0 to batch -1 do
        for c := 0 to channels -1 do
            begin
                s := src + (b* channels+c) * in_plane;
                d := dst + (b* channels+c) * out_plane;
                for y := 0 to H -1 do
                    QNNCopy(d + y*Wp, s + y*W, W)
            end
end;

procedure TVAE.free;
var i:longint;
begin

    enc_conv_in_weight.free;
    enc_conv_in_bias  .free;
    for i := 0 to high(enc_down_blocks) do
      enc_down_blocks[i].free;
    enc_down_blocks := nil;
    setLength(enc_downsample, 3);
    for i := 0 to high(enc_downsample) do begin
      enc_downsample[i].free;
      enc_downsample[i].free;
    end;
    enc_downsample:= nil;
    enc_mid_block1      .free;
    enc_mid_attn        .free;
    enc_mid_block2      .free;
    enc_norm_out_weight .free;
    enc_norm_out_bias   .free;
    enc_conv_out_weight .free;
    enc_conv_out_bias   .free;
    quant_conv_weight   .free;
    quant_conv_bias     .free;
    dec_conv_in_weight  .free;
    dec_conv_in_bias    .free;
    dec_mid_block1      .free;
    dec_mid_attn        .free;
    dec_mid_block2      .free;
    for i := 0 to high(dec_up_blocks) do
       dec_up_blocks[i].free;
    dec_up_blocks := nil;
    for i := 0 to high(dec_upsample) do begin
      dec_upsample[i].conv_weight.free;
      dec_upsample[i].conv_bias.free;
    end;
    dec_upsample := nil;
    dec_norm_out_weight .free;
    dec_norm_out_bias   .free;
    dec_conv_out_weight .free;
    dec_conv_out_bias   .free;
    bn_mean             .free;
    bn_var              .free;
    post_quant_conv_weight.free;
    post_quant_conv_bias.free;
    bn_mean             .free;
    bn_var              .free;
    work1               .free;
    work2               .free;
    work3               .free;
    fillchar(self, sizeof(self), #0)
end;
end.

