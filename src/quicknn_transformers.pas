unit quicknn_transformers;

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
{$T+}

{$define USE_MULTITHREADING}

interface
uses SysUtils, safetensor, quicknn_common, quicknn_kernels, quickjson
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  ;


type

  TSingles = TMemoryBlock;

  TBF16s   = TMemoryBlock;

  // AdaLN-Zero modulation parameters (shared across blocks)
  TModeWeight = record
    mod_weight: TSingles;        (* [hidden * 6] for double, [hidden * 3] for single *)
  end;


  (* Double-stream block (MM-DiT style) *)
  PDoubleBlock = ^TDoubleBlock;

  { TDoubleBlock }

  TDoubleBlock = record
    (* Image stream - separate Q, K, V weights (f32 and bf16) *)
    // [hidden, hidden]
    img_q_weight: TSingles;
    img_k_weight: TSingles;
    img_v_weight: TSingles;
    img_q_weight_bf16: TBF16s;
    img_k_weight_bf16: TBF16s;
    img_v_weight_bf16: TBF16s;

    //[head_dim] - QK norm on Q and K
    img_norm_q_weight: TSingles;
    img_norm_k_weight: TSingles;

    // [hidden, hidden]
    img_proj_weight: TSingles;
    img_proj_weight_bf16: TBF16s;

    //[mlp_hidden, hidden]
    img_mlp_gate_weight: TSingles;
    img_mlp_gate_weight_bf16: TBF16s;
    img_mlp_up_weight: TSingles;
    img_mlp_up_weight_bf16: TBF16s;

    //[hidden, mlp_hidden]
    img_mlp_down_weight: TSingles;
    img_mlp_down_weight_bf16: TBF16s;

    //Text stream - separate Q, K, V weights
    //[hidden, hidden]
    txt_q_weight: TSingles;
    txt_k_weight: TSingles;
    txt_v_weight: TSingles;
    txt_q_weight_bf16: TBF16s;
    txt_k_weight_bf16: TBF16s;
    txt_v_weight_bf16: TBF16s;

    //[head_dim] - QK norm on Q and K
    txt_norm_q_weight: TSingles;
    txt_norm_k_weight: TSingles;

    //[hidden, hidden]
    txt_proj_weight: TSingles;
    txt_proj_weight_bf16: TBF16s;

    //[mlp_hidden, hidden]
    txt_mlp_gate_weight: TSingles;
    txt_mlp_gate_weight_bf16: TBF16s;
    txt_mlp_up_weight: TSingles;
    txt_mlp_up_weight_bf16: TBF16s;

    //[hidden, mlp_hidden]
    txt_mlp_down_weight: TSingles;
    txt_mlp_down_weight_bf16: TBF16s;
    ismmap : boolean;

    procedure load(const safeTensorfiles:TSafeTensorFiles; const idx ,h ,mlp:longint; const use_bf16:boolean = true; const use_mmap:boolean = true);
    procedure free;
  end;

  //Single-stream block (Parallel DiT style, fused)
  PSingleBlock = ^TSingleBlock;

  { TSingleBlock }

  TSingleBlock = record
    //Fused QKV + FFN input projection
    //[hidden*3 + mlp_hidden*2, hidden]
    qkv_mlp_weight: TSingles;
    qkv_mlp_weight_bf16: TBF16s;

    //QK normalization
    //[head_dim]
    norm_q_weight: TSingles;
    norm_k_weight: TSingles;

    //Fused attention out + FFN down projection
    //[hidden, hidden + mlp_hidden]
    proj_mlp_weight: TSingles;
    proj_mlp_weight_bf16: TBF16s;
    procedure load(const safeTensorfiles:TSafeTensorFiles; const idx ,h ,mlp:longint; const use_bf16:boolean = true; const use_mmap:boolean = true);
    procedure free;
  end;

  (* Timestep embedding MLP
   * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
   * linear_1: [hidden, 256] - projects sinusoidal to hidden
   * linear_2: [hidden, hidden] - another linear layer *)

  { TTimeEmbed }

  TTimeEmbed = record
    fc1_weight: TSingles; // [hidden, 256]
    fc2_weight: TSingles; // [hidden, hidden]
    sincos_dim: longint; // 256 for FLUX.2-klein
    procedure forward(const dst, sinCos:TMemoryBlock; const hidden: longint);
  end;

const MAX_TF_SHARDS = 4;

type

  //Full transformer context

  { TTransformerFlux }

  TTransformerFlux = record
    hidden_size: longint;         // 3072
    num_heads: longint;           // 24
    head_dim: longint;            // 128
    mlp_hidden: longint;          // hidden * 3 = 9216
    num_double_layers: longint;   // 5
    num_single_layers: longint;   // 20
    text_dim: longint;            // 7680
    latent_channels: longint;     // 128
    rope_theta: single;           // 2000
    rope_dim: longint;            // 128
    use_bf16: boolean;            // Use bf16 weights (1) or f32 (0)

    //Input projections
    img_in_weight: TSingles;       // [hidden, latent_channels]
    txt_in_weight: TSingles;       // [hidden, text_dim]
    img_in_weight_bf16: TBF16s;    // [hidden, latent_channels]
    txt_in_weight_bf16: TBF16s;    // [hidden, text_dim]

    time_embed: TTimeEmbed;
    time_freq: TSingles;                // [hidden/2]
    adaln_double_img_weight: TSingles;    //[hidden * 6, hidden] for double block img stream
    adaln_double_img_weight_bf16: TBF16s;
    adaln_double_txt_weight: TSingles;    //[hidden * 6, hidden] for double block txt stream
    adaln_double_txt_weight_bf16: TBF16s;
    adaln_single_weight: TSingles;        // [hidden * 3, hidden] for single block
    adaln_single_weight_bf16: TBF16s;

    //Transformer block
    double_blocks: TArray<TDoubleBlock>;
    single_blocks: TArray<TSingleBlock>;


    final_norm_weight: TSingles;         // [hidden]
    final_proj_weight: TSingles;         // [latent_channels, hidden]
    final_proj_weight_bf16: TBF16s;

    rope_freqs: TSingles;                // [max_seq, head_dim/2, 2]  - legacy 1D
    rope_cos: TSingles;                  // [max_seq, axis_dim] - 2D cos frequencies
    rope_sin: TSingles;                  // [max_seq, axis_dim] - 2D sin frequencies

    max_seq_len: longint;
    axis_dim: longint;                  // 32 for FLUX (128 head_dim / 4 axes)

    img_hidden: TSingles;                // [max_img_seq, hidden]
    txt_hidden: TSingles;                // [max_txt_seq, hidden]

    q: TSingles;                         // [max_seq, hidden]
    k: TSingles;
    v: TSingles;

    attn_out: TSingles;                  // [max_seq, hidden]
    mlp_buffer: TSingles;                // [max_seq, mlp_hidden]

    work1: TSingles;
    work2: TSingles;
    work_size: size_t;

    //Pre-allocated attention workspaces to avoid malloc in hot pat
    attn_q_t: TSingles;                    //  [max_seq, hidden] transposed Q
    attn_k_t: TSingles;                    //  [max_seq, hidden] transposed K
    attn_v_t: TSingles;                    //  [max_seq, hidden] transposed V
    attn_out_t: TSingles;                  //  [max_seq, hidden] transposed output
    attn_scores: TSingles;                 //  [num_heads, seq, seq] attention scores (BLAS/Metal only)
    attn_scores_alloc: size_t;            //  Currently allocated size in bytes
    work_seq_alloc: longint;              //  Currently allocated sequence length for work buffers
    attn_cat_k: TSingles;                  //  [max_seq, hidden] concatenated K
    attn_cat_v: TSingles;                  //  [max_seq, hidden] concatenated V

    //Single-block work buffers (pre-allocated to avoid malloc in hot path)
    single_q: TSingles;                 // [max_seq, hidden]
    single_k: TSingles;                 // [max_seq, hidden]
    single_v: TSingles;                 // [max_seq, hidden]
    single_mlp_gate: TSingles;          // [max_seq, mlp_hidden]
    single_mlp_up: TSingles;            // [max_seq, mlp_hidden]
    single_attn_out: TSingles;          // [max_seq, hidden]
    single_concat: TSingles;            // [max_seq, hidden + mlp_hidden]

    //FFN work buffers (shared by double and single blocks)
    ffn_gate: TSingles;                 // [max_seq, mlp_hidden]
    ffn_up: TSingles;                   // [max_seq, mlp_hidden]

    //Double-block work buffers
    t_emb_silu: TSingles;               // [hidden]
    double_mod_img: TSingles;           // [hidden * 6]
    double_mod_txt: TSingles;           // [hidden * 6]
    double_img_attn_out: TSingles;      // [max_seq, hidden]
    double_txt_attn_out: TSingles;      // [max_seq, hidden]

    //Cached 2D RoPE embeddings (to avoid malloc/compute each step)
    cached_img_rope_cos: TSingles;      // [img_seq * axis_dim * 4]
    cached_img_rope_sin: TSingles;
    cached_img_h: longint;
    cached_img_w: longint;
    cached_ref_rope_cos: TSingles;      // [ref_seq * axis_dim * 4] for img2img
    cached_ref_rope_sin: TSingles;
    cached_ref_h: longint;
    cached_ref_w: longint;
    cached_ref_t_offset: longint;

    cached_txt_rope_cos: TSingles;      // [txt_seq * head_dim] */
    cached_txt_rope_sin: TSingles;
    cached_txt_seq: longint;

    //Combined RoPE cache for img2img (target + reference)
    cached_combined_rope_cos: TSingles;  // [(img_seq + ref_seq) * axis_dim * 4]
    cached_combined_rope_sin: TSingles;
    cached_combined_img_h: longint;
    cached_combined_img_w: longint;
    cached_combined_ref_h: longint;
    cached_combined_ref_w: longint;
    cached_combined_t_offset: longint;

    use_mmap: boolean;
    sf_files:TSafeTensorFiles;
    num_sf_files: longint;
    constructor LoadFromDir(const modelDir: string);
    procedure reInitialize(const totalSeq:longint);
    class function loadShards(const modelDir: string): TSafeTensorFiles; static;
    procedure getCachedRefRoPE(const patch_h, patch_w, offset:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedImgRoPE(const patch_h, patch_w:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedTxtRoPE(const txt_seq:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedCombinedRoPE(const img_h, img_w, ref_h, ref_w, offset:longint; var cosOut:TSingles; var sinOut:TSingles);
    procedure multiHeadForward(const dst, Q, K, V:PQNNFloat; const seq{, heads, head_dim}:longint);
    procedure jointAttentionForward(
              const img_out, txt_out,
                    img_Q  , img_K, img_V,
                    txt_Q  , txt_K, txt_V : PQNNFloat;
              const img_seq, txt_seq{, heads, head_dim}: longint);
    procedure ffnSwigluForward(const dst, x:PQNNFloat;
              const gate_weight, up_weight, down_weight:TSingles;
              const gate_weight_bf16, up_weight_bf16, down_weight_bf16:TBF16s;
              const seq, hidden, mlpHidden: longint);

    procedure doubleBlockForward(
              const img_hidden, txt_hidden :PQNNFloat; const blockIdx:longint; const img_mod, txt_mod,
                    img_rope_cos, img_rope_sin,
                    txt_rope_cos, txt_rope_sin: PQNNFloat;
              const img_seq, txt_seq : longint);

    procedure singleBlockForward(const hidden:PQNNFloat;
                                 const blockIdx:longint;
                                 const t_emb, adaln_weight,
                                 img_rope_cos, img_rope_sin,
                                 txt_rope_cos, txt_rope_sin:PQNNFloat;
                                 const seq, img_offset:longint);
  end;

implementation

{ TDoubleBlock }

procedure TDoubleBlock.load(
  const safeTensorfiles: TSafeTensorFiles; const idx, h, mlp: longint;
  const use_bf16: boolean; const use_mmap: boolean);
const TRANSFORMER_BLOCK : array[0..15] of string = (
      'transformer_blocks.%d.attn.norm_q.weight',
      'transformer_blocks.%d.attn.norm_k.weight',
      'transformer_blocks.%d.attn.to_q.weight',
      'transformer_blocks.%d.attn.to_k.weight',
      'transformer_blocks.%d.attn.to_v.weight',
      'transformer_blocks.%d.attn.to_out.0.weight',
      'transformer_blocks.%d.ff.linear_in.weight',
      'transformer_blocks.%d.ff.linear_out.weight',
      'transformer_blocks.%d.attn.norm_added_q.weight',
      'transformer_blocks.%d.attn.norm_added_k.weight',
      'transformer_blocks.%d.attn.add_q_proj.weight',
      'transformer_blocks.%d.attn.add_k_proj.weight',
      'transformer_blocks.%d.attn.add_v_proj.weight',
      'transformer_blocks.%d.attn.to_add_out.weight',
      'transformer_blocks.%d.ff_context.linear_in.weight',
      'transformer_blocks.%d.ff_context.linear_out.weight'
      );
begin
  // as in forward order 1st do the norm q and k weights;

  img_norm_q_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[0], [idx]), use_mmap);
  img_norm_k_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[1], [idx]), use_mmap);
  ismmap:= use_mmap;
  if use_bf16 then begin
  (* Image Q, K, V projections - skip f32 if bf16 available *)
    img_q_weight_bf16 := safeTensorfiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[2], [idx]), use_mmap);
    img_k_weight_bf16 := safeTensorfiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[3], [idx]), use_mmap);
    img_v_weight_bf16 := safeTensorfiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[4], [idx]), use_mmap);
    img_proj_weight_bf16 := safeTensorfiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[5], [idx]), use_mmap);
  (* Image FFN - linear_in contains gate and up fused - skip f32 if bf16 available *)
    img_mlp_gate_weight_bf16 := safeTensorfiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[6], [idx]), use_mmap);
    assert(use_mmap or (img_mlp_gate_weight_bf16.count >= mlp*h*2), 'ERROR : incorrect size of '+format(TRANSFORMER_BLOCK[6], [idx]));
    img_mlp_up_weight_bf16 := PBF16(img_mlp_gate_weight_bf16) + mlp*h;
    img_mlp_down_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[7], [idx]), use_mmap);
    // during forward attention norm added q and v should be here,
    // this is just loading so will be loaded after this block
    //txt_norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[8], [idx]), use_mmap);
    //txt_norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[9], [idx]), use_mmap);

    txt_q_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[10], [idx]), use_mmap);
    txt_k_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[11], [idx]), use_mmap);
    txt_v_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[12], [idx]), use_mmap);
  (* Text Q, K, V projections - skip f32 if bf16 available *)
    txt_proj_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[13], [idx]), use_mmap);
    txt_mlp_gate_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[14], [idx]), use_mmap);
    assert(use_mmap or (txt_mlp_gate_weight_bf16.count >= mlp*h*2), 'ERROR : incorrect size of '+format(TRANSFORMER_BLOCK[14], [idx]));
    txt_mlp_up_weight_bf16 := PBF16(txt_mlp_gate_weight_bf16) + mlp * h;
    txt_mlp_down_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[15], [idx]), use_mmap)
  end else begin
  (* Image Q, K, V projections - skip f32 if bf16 available *)
    img_q_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[2], [idx]), use_mmap);
    img_k_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[3], [idx]), use_mmap);
    img_v_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[4], [idx]), use_mmap);
    img_proj_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[5], [idx]), use_mmap);
  (* Image FFN - linear_in contains gate and up fused - skip f32 if bf16 available *)
    img_mlp_gate_weight := safeTensorfiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[6], [idx]), use_mmap);
    assert(use_mmap or (img_mlp_gate_weight.count >= mlp*h*2), 'ERROR : incorrect size of '+format(TRANSFORMER_BLOCK[6], [idx]));
    img_mlp_up_weight := PSingle(img_mlp_gate_weight_bf16) + mlp*h;
    img_mlp_down_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[7], [idx]), use_mmap);
    // during forward attention norm added q and v should be here,
    // this is just loading so will be loaded after this block
    //txt_norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[8], [idx]), use_mmap);
    //txt_norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[9], [idx]), use_mmap);

    txt_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[10], [idx]), use_mmap);
    txt_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[11], [idx]), use_mmap);
    txt_v_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[12], [idx]), use_mmap);
  (* Text Q, K, V projections - skip f32 if bf16 available *)
    txt_proj_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[13], [idx]), use_mmap);
    txt_mlp_gate_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[14], [idx]), use_mmap);
    assert(use_mmap or (txt_mlp_gate_weight.count >= mlp*h*2), 'ERROR : incorrect size of '+format(TRANSFORMER_BLOCK[14], [idx]));
    txt_mlp_up_weight := PSingle(txt_mlp_gate_weight_bf16) + mlp * h;
    txt_mlp_down_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[15], [idx]), use_mmap)
  end;

  (* Text stream - QK norm weights (always f32) *)
  txt_norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[8], [idx]), false {use_mmap}); // always allocate
  txt_norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[9], [idx]), false {use_mmap}); // always allocate

end;

procedure TSingleBlock.load(
  const safeTensorfiles: TSafeTensorFiles; const idx, h, mlp: longint;
  const use_bf16: boolean; const use_mmap: boolean);
const
    TRANSFORMER_BLOCK : array[0..3] of string = (
      'single_transformer_blocks.%d.attn.norm_q.weight',
      'single_transformer_blocks.%d.attn.norm_k.weight',
      'single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight',
      'single_transformer_blocks.%d.attn.to_out.weight'
    );
begin

  (* QK norm weights (always f32, small) *)
  norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[0], [idx]), use_mmap);
  norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[1], [idx]), use_mmap);

  if use_bf16 then begin
  (* Fused QKV+MLP input projection - skip f32 if bf16 available *)
    qkv_mlp_weight_bf16  := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[2], [idx]), use_mmap);
  (* Fused attn out + MLP down projection - skip f32 if bf16 available *)
    proj_mlp_weight_bf16 := safeTensorFiles.getTensorDataMemBlockBF16(format(TRANSFORMER_BLOCK[3], [idx]), use_mmap);
  end else begin
    qkv_mlp_weight  := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[2], [idx]), use_mmap);
    proj_mlp_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[3], [idx]), use_mmap);
  end;
end;

procedure TDoubleBlock.free;
begin
  //freemem(img_norm_q_weight);      img_norm_q_weight := nil;
  //freemem(img_norm_k_weight);      img_norm_k_weight := nil;
  //freemem(img_q_weight);           img_q_weight := nil;
  //freemem(img_k_weight);           img_k_weight := nil;
  //freemem(img_v_weight);           img_v_weight := nil;
  //freemem(img_proj_weight);        img_proj_weight := nil;
  //freemem(img_mlp_gate_weight);    img_mlp_gate_weight := nil;
  //freemem(img_mlp_up_weight);      img_mlp_up_weight := nil;
  //freemem(img_mlp_down_weight);    img_mlp_down_weight := nil;
  //
  //img_q_weight_bf16 := nil;
  //img_k_weight_bf16 := nil;
  //img_v_weight_bf16 := nil;
  //img_proj_weight_bf16 := nil;
  //img_mlp_gate_weight_bf16 := nil;
  //img_mlp_up_weight_bf16 := nil;
  //img_mlp_down_weight_bf16 := nil;
  //
  //freemem(txt_norm_q_weight);   txt_norm_q_weight := nil;
  //freemem(txt_norm_k_weight);   txt_norm_k_weight := nil;
  //freemem(txt_q_weight);        txt_q_weight := nil;
  //freemem(txt_k_weight);        txt_k_weight := nil;
  //freemem(txt_v_weight);        txt_v_weight := nil;
  //freemem(txt_proj_weight);     txt_proj_weight := nil;
  //freemem(txt_mlp_gate_weight); txt_mlp_gate_weight := nil;
  //freemem(txt_mlp_up_weight);   txt_mlp_up_weight := nil;
  //freemem(txt_mlp_down_weight); txt_mlp_down_weight := nil;
  //
  //txt_q_weight_bf16 := nil;
  //txt_k_weight_bf16 := nil;
  //txt_v_weight_bf16 := nil;
  //txt_proj_weight_bf16 := nil;
  //txt_mlp_gate_weight_bf16 := nil;
  //txt_mlp_up_weight_bf16 := nil;
  //txt_mlp_down_weight_bf16 := nil
end;

procedure TSingleBlock.free;
begin
  //freemem(norm_q_weight);   norm_q_weight   := nil;
  //freemem(norm_k_weight);   norm_k_weight   := nil;
  //freemem(qkv_mlp_weight);  qkv_mlp_weight  := nil;
  //freemem(proj_mlp_weight); proj_mlp_weight := nil;
  //(* bf16 pointers are direct mmap pointers - just clear, don't free *)
  //
  //qkv_mlp_weight_bf16 :=  nil;
  //proj_mlp_weight_bf16 := nil;
end;

{ TTimeEmbed }

procedure TTimeEmbed.forward(const dst, sinCos: TMemoryBlock; const hidden: longint);
begin
  if length(workspace)< 1*hidden then
    setLength(workspace, hidden);

  QNNLinearNoBias(pointer(workspace), sinCos, fc1_weight, 1, sincos_dim, hidden);
  QNNSilu(pointer(workspace), hidden);
  QNNLinearNoBias(dst, pointer(workspace), fc2_weight, 1, hidden, hidden);
end;

{ TTransformerFlux }

constructor TTransformerFlux.LoadFromDir(const modelDir: string);
var json : TJSON;
  headsCount, headDim, layerCount, singleCount, JointAttDim, inChannels:longint;
  mlpRatio, ropeTheta : single;
  eps : single;
begin
  json := TJSON.LoadFromFile(modelDir+'/transformer/config.json');
  headsCount    := json['num_attention_heads'];
  headDim       := json['attention_head_dim'];
  layerCount    := json['num_layers'];
  singleCount   := json['num_single_layers'];
  jointAttDim   := json['joint_attention_dim'];
  inChannels    := json['in_channels'];
  mlpRatio      := json['mlp_ratio'].value;
  ropeTheta     := json['rope_theta'].value;
  eps           := json['eps'].Value;

  head_dim          := headDim;
  hidden_size       := headsCount * headDim;

  if mlpRatio>0 then
    mlp_hidden        := trunc(hidden_size * mlpRatio)
  else
    mlp_hidden        := hidden_size *  3;


  num_double_layers := ifthen(layerCount > 0, layerCount, 5);
  num_single_layers := ifthen(singleCount > 0, singleCount, 20);
  text_dim          := ifthen(jointAttDim > 0, jointAttDim, 7680);
  latent_channels   := ifthen(inChannels > 0, inChannels, 128);

  if ropeTheta > 0 then
    rope_theta        := ropeTheta
  else
    rope_theta        := 2000.0;

  rope_dim          := headDim;
  axis_dim          := headDim div 4;  // 4 RoPE axes

  sf_files := loadShards(modelDir);


end;

procedure TTransformerFlux.reInitialize(const totalSeq: longint);
var fused_dim, hidden, mlp:longint;
begin
  if totalSeq<=work_seq_alloc then exit;

  hidden := hidden_size;
  mlp    := mlp_hidden;

  img_hidden := TMemoryBlock.Create(totalSeq * hidden);
  txt_hidden := TMemoryBlock.Create(totalSeq * hidden);
  (* work2 needs to hold fused QKV+MLP output: seq * (hidden*3 + mlp*2) + mod_params (hidden*3)
  * fused_dim = hidden*3 + mlp*2 = 3072*3 + 9216*2 = 27648 *)
  fused_dim := hidden * 3 + mlp * 2;

  work_size           := totalSeq*fused_dim + hidden*3 ;

  work1               := TMemoryBlock.Create(totalSeq * hidden);
  work2               := TMemoryBlock.Create(work_size);
  attn_q_t            := TMemoryBlock.Create(totalSeq * hidden);
  attn_k_t            := TMemoryBlock.Create(totalSeq * hidden);
  attn_v_t            := TMemoryBlock.Create(totalSeq * hidden);
  attn_out_t          := TMemoryBlock.Create(totalSeq * hidden);
  attn_cat_k          := TMemoryBlock.Create(totalSeq * hidden);
  attn_cat_v          := TMemoryBlock.Create(totalSeq * hidden);
  single_q            := TMemoryBlock.Create(totalSeq * hidden);
  single_k            := TMemoryBlock.Create(totalSeq * hidden);
  single_v            := TMemoryBlock.Create(totalSeq * hidden);
  single_mlp_gate     := TMemoryBlock.Create(totalSeq * mlp);
  single_mlp_up       := TMemoryBlock.Create(totalSeq * mlp);
  single_attn_out     := TMemoryBlock.Create(totalSeq * hidden );
  single_concat       := TMemoryBlock.Create(totalSeq * (hidden + mlp));
  ffn_gate            := TMemoryBlock.Create(totalSeq * mlp   );
  ffn_up              := TMemoryBlock.Create(totalSeq * mlp   );
  double_img_attn_out := TMemoryBlock.Create(totalSeq * hidden);
  double_txt_attn_out := TMemoryBlock.Create(totalSeq * hidden);
  work_seq_alloc := totalSeq;
end;

class function TTransformerFlux.loadShards(const modelDir: string):TSafeTensorFiles;
var json, wm:TJSON; i:integer;
  fn : string;
begin
  fn := modelDir+'/transformer/diffusion_pytorch_model.safetensors.index.json';
  if FileExists(fn) then begin
    json := TJSON.LoadFromFile(fn);
    if json.keyExist('weight_map') then begin
      setLength(result, json['weight_map'].count);
      wm:= json['weight_map'];
      for i:=0 to high(result) do
        result[i] := TSafeTensorFile.open('/transformer/'+wm.childObjs[i].Value);
    end;
  end else
    result := [TSafeTensorFile.open(modelDir+'/transformer/diffusion_pytorch_model.safetensors')];
end;

procedure TTransformerFlux.getCachedRefRoPE(const patch_h, patch_w,
  offset: longint; var cosOut: TSingles; var sinOut: TSingles);
var
    seq, axisDim: longint;
    size: NativeInt;
begin
    seq := patch_h * patch_w;
    axisDim := axis_dim;
    if (cached_ref_h = patch_h) and (cached_ref_w = patch_w) and (cached_ref_t_offset = offset) and (boolean(cached_ref_rope_cos) and boolean(cached_ref_rope_sin)) then
      begin
        cosOut := cached_ref_rope_cos;
        sinOut := cached_ref_rope_sin;
        exit()
      end;
    //if assigned(cached_ref_rope_cos) then
    //    free();
    //if assigned(cached_ref_rope_sin) then
    //    free();
    size := seq * axisDim * 4 ;
    cached_ref_rope_cos := TMemoryBlock.Create(size, dtF32);
    cached_ref_rope_sin := TMemoryBlock.Create(size, dtF32);
    cached_ref_h := patch_h;
    cached_ref_w := patch_w;
    cached_ref_t_offset := offset;
    QNNComputeRoPE2DOffset(cached_ref_rope_cos, cached_ref_rope_sin, patch_h, patch_w, axisDim, rope_theta, offset);
    cosOut := cached_ref_rope_cos;
    sinOut := cached_ref_rope_sin
end;

procedure TTransformerFlux.getCachedImgRoPE(const patch_h, patch_w: longint;
  var cosOut: TSingles; var sinOut: TSingles);
var
    seq, axisDim: longint;
    size: NativeInt;
begin
    seq := patch_h * patch_w;
    axisDim := axis_dim;
    if (cached_img_h = patch_h) and (cached_img_w = patch_w) and (boolean(cached_img_rope_cos) and boolean(cached_img_rope_sin)) then
      begin
        cosOut := cached_img_rope_cos;
        sinOut := cached_img_rope_sin;
        exit()
      end;
    //if assigned(cached_img_rope_cos) then
    //    free();
    //if assigned(cached_img_rope_sin) then
    //    free();
    size := seq * axisDim * 4 ;
    cached_img_rope_cos := TMemoryBlock.Create(size, dtF32);
    cached_img_rope_sin := TMemoryBlock.Create(size, dtF32);
    cached_img_h := patch_h;
    cached_img_w := patch_w;
    QNNComputeRoPE2D(cached_img_rope_cos, cached_img_rope_sin, patch_h, patch_w, axisDim, rope_theta);
    cosOut := cached_img_rope_cos;
    sinOut := cached_img_rope_sin
end;

procedure TTransformerFlux.getCachedTxtRoPE(const txt_seq: longint; var cosOut: TSingles; var sinOut: TSingles);
var
    headDim: longint;
    size: NativeInt;
begin
    headDim := head_dim;
    size := txt_seq * headDim;

    if (cached_txt_seq = txt_seq) and (boolean(cached_txt_rope_cos) and boolean(cached_txt_rope_sin)) then
      begin
        cosOut := cached_txt_rope_cos;
        sinOut := cached_txt_rope_sin;
        exit()
      end;
    //if assigned(cached_img_rope_cos) then
    //    free();
    //if assigned(cached_img_rope_sin) then
    //    free();
    cached_txt_rope_cos := TMemoryBlock.Create(size, dtF32);
    cached_txt_rope_sin := TMemoryBlock.Create(size, dtF32);
    cached_txt_seq := txt_seq;
    QNNComputeRoPEText(cached_txt_rope_cos, cached_img_rope_sin, txt_seq, headDim, rope_theta);
    cosOut := cached_txt_rope_cos;
    sinOut := cached_txt_rope_sin
end;

procedure TTransformerFlux.getCachedCombinedRoPE(const img_h, img_w, ref_h, ref_w, offset: longint; var cosOut: TSingles; var sinOut: TSingles);
var
    img_seq, ref_seq, combined_seq, axisDim: longint;
    combined_size, img_size, ref_size: NativeInt;
    img_rope_cos, img_rope_sin, ref_rope_cos, ref_rope_sin : TSingles;
begin
    img_seq := img_h * img_w;
    ref_seq := ref_h * ref_w;
    combined_seq := img_seq+ref_seq;
    axisDim := axis_dim;
    if (cached_combined_img_h = img_h) and
       (cached_combined_img_w = img_w) and
       (cached_combined_ref_h = ref_h) and
       (cached_combined_ref_w = ref_w) and
       (cached_combined_t_offset = offset) and boolean(cached_combined_rope_cos) and boolean(cached_combined_rope_sin) then
        begin
            cosOut := cached_combined_rope_cos;
            sinOut := cached_combined_rope_sin;
            exit()
        end;
    //if cached_combined_rope_cos then
    //    free(cached_combined_rope_cos);
    //if cached_combined_rope_sin then
    //    free(cached_combined_rope_sin);
    combined_size := combined_seq * axisDim * 4;
    cached_combined_rope_cos := TMemoryBlock.Create(combined_size, dtF32);
    cached_combined_rope_sin := TMemoryBlock.Create(combined_size, dtF32);
    cached_combined_img_h := img_h;
    cached_combined_img_w := img_w;
    cached_combined_ref_h := ref_h;
    cached_combined_ref_w := ref_w;
    cached_combined_t_offset := offset;

    getCachedImgRoPE(img_h, img_w, img_rope_cos,  img_rope_sin);

    getCachedRefRoPE(ref_h, ref_w, offset,  ref_rope_cos,  ref_rope_sin);

    img_size := img_seq * axisDim * 4 * sizeof(QNNFloat);
    ref_size := ref_seq * axisDim * 4 * sizeof(QNNFloat);

    move(PQNNFloat(img_rope_cos)^, (PQNNFloat(cached_combined_rope_cos)                        )^, img_size);
    move(PQNNFloat(ref_rope_cos)^, (PQNNFloat(cached_combined_rope_cos) + img_seq * axisDim * 4)^, ref_size);
    move(PQNNFloat(img_rope_sin)^, (PQNNFloat(cached_combined_rope_sin)                        )^, img_size);
    move(PQNNFloat(ref_rope_sin)^, (PQNNFloat(cached_combined_rope_sin) + img_seq * axisDim * 4)^, ref_size);

    cosOut := cached_combined_rope_cos;
    sinOut := cached_combined_rope_sin
end;

procedure TTransformerFlux.multiHeadForward(const dst, Q, K, V: PQNNFloat; const seq{, heads, head_dim}: longint);
var
  hidden, headDim : longint;
  scale : QNNFloat;
  scores : PQNNFloat;

{$ifdef FPC}
procedure worker(const start, finish:IntPtr; const data:pointer);
{$else}
   worker : TGroupProcNested;
begin
  worker := procedure (const start, finish:IntPtr; const data:pointer)
{$endif}
var
  i  : longint;
  qh, kh, vh, oh, sh : PQNNFloat;
begin
    for i := start to finish do begin
        qh := Q      + i*headDim;
        kh := K      + i*headDim;
        vh := V      + i*headDim;
        oh := dst    + i*headDim;
        sh := scores + i*seq*seq;

        cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq, seq, headDim,
                    scale,
                    qh {Q      + i*headDim}, hidden,
                    kh {K      + i*headDim}, hidden,
                    0.0,
                    sh {scores + i*seq*seq}, seq);
        QNNSoftmax(sh {scores + i*seq*seq}, seq, seq);

        cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq, headDim, seq,
                    1.0,
                    sh {scores + i*seq*seq}, seq,
                    vh {V      + i*headDim}, hidden,
                    0.0,
                    oh {dst    + i*headDim}, hidden);
    end;
end;

{$ifdef fpc}
begin
{$endif}

  headDim := head_dim;
  hidden := num_heads * head_dim;  //but we use self.hidden_size
  scores := attn_scores;
  scale := 1/sqrt(head_dim);

{$ifdef USE_MULTITHREADING}
  mp.&for(worker, 0, num_heads-1) ;
{$else}
  worker(0, num_heads-1, nil);
{$endif}

  //QNNFlashAttention(dst, Q, K, V, seq, seq, num_heads, head_dim, 1/sqrt(head_dim));


end;

procedure TTransformerFlux.jointAttentionForward(const img_out, txt_out, img_Q,
  img_K, img_V, txt_Q, txt_K, txt_V: PQNNFloat; const img_seq, txt_seq: longint
  );
var
  totalSeq, hidden, headDim:longint;
  scale : QNNFloat;
  cat_k, cat_v, scores : PQNNFloat;
  {$ifdef fpc}
  procedure worker(const start, finish:IntPtr; const data:pointer);
{$else}
  worker : TGroupProcNested;
begin
  worker := procedure (const start, finish:IntPtr; const data:pointer)
{$endif}
  var
    i: longint;
    imgQh, imgSh, imgOh,
    txtQh, txtSh, txtOh,
    kh, vh : PQNNFloat;
  begin
    for i:= start to finish do begin
            imgQh  := img_Q   + i*headDim;
            txtQh  := txt_K   + i*headDim;
            kh     := cat_k   + i*headDim;
            vh     := cat_v   + i*headDim;
            imgOh := img_out + i*headDim;
            txtOh := txt_out + i*headDim;
            imgSh := scores  + i*totalSeq*totalSeq;
            txtSh := imgSh  + img_seq*totalSeq; // txt_sh is an offset from img_sh

            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        img_seq, totalSeq, headDim,
                        scale,
                        imgQh, hidden,
                        kh, hidden,
                        0.0,
                        imgSh, totalSeq);
            QNNSoftmax(imgSh, img_seq, totalSeq);
            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        img_seq, headDim, totalSeq,
                        1.0,
                        imgSh, totalSeq,
                        vh, hidden,
                        0.0,
                        imgOh, hidden);

            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        txt_seq, totalSeq, headDim,
                        scale,
                        txtQh, hidden,
                        kh, hidden,
                        0.0,
                        txtSh, totalSeq);
            QNNSoftmax(txtSh, txt_seq, totalSeq);
            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        txt_seq, headDim, totalSeq,
                        1.0,
                        txtSh, totalSeq,
                        vh, hidden,
                        0.0,
                        txtOh, hidden);

    end;
  end;
{$ifdef fpc}
begin
{$endif}

  totalSeq := img_seq + txt_seq;
  headDim := head_dim;
  hidden   := num_heads*head_dim;
  scores := attn_scores;
  scale  := 1/sqrt(head_dim);

  (* Use pre-allocated buffers for K/V concatenation *)
  cat_k := attn_cat_k;
  cat_v := attn_cat_v;

  //Concatenate K, V from both streams in [seq, heads, head_dim] format
  //IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
  QNNCopy(cat_K, txt_K, txt_seq*hidden);
  QNNCopy(cat_V, txt_V, txt_seq*hidden);
  QNNCopy(cat_K + txt_seq*hidden, img_K, img_seq*hidden);
  QNNCopy(cat_V + txt_seq*hidden, img_V, img_seq*hidden);

{$ifdef USE_MULTITHREADING}
  mp.&for(worker, 0, num_heads-1);
{$else}
  worker(0, num_heads-1, nil)
{$endif}

  //QNNFlashAttention(img_out, img_q, cat_k, cat_v, img_seq, totalSeq, num_heads, head_dim, scale);
  //QNNFlashAttention(txt_out, txt_q, cat_k, cat_v, txt_seq, totalSeq, num_heads, head_dim, scale);

end;

procedure TTransformerFlux.ffnSwigluForward(const dst, x: PQNNFloat;
  const gate_weight, up_weight, down_weight: TSingles; const gate_weight_bf16,
  up_weight_bf16, down_weight_bf16: TBF16s; const seq, hidden,
  mlpHidden: longint);
begin
  assert(boolean(gate_weight) or boolean(gate_weight_bf16),
    'ERROR ffnForward no gate_weight or gate_weight_bf16 assigned!');

  //if boolean(gate_weight_bf16) then
  //  QNNLinearNoBias_BF16(ffn_gate, x, gate_weight_bf16, seq, hidden, mlpHidden)
  //else
  //  QNNLinearNoBias(ffn_gate,      x, gate_weight     , seq, hidden, mlpHidden);

  //if boolean(up_weight_bf16) then
  //  QNNLinearNoBias_BF16(ffn_up, x, up_weight_bf16, seq, hidden, mlpHidden)
  //else
  //  QNNLinearNoBias(ffn_up,      x, up_weight     , seq, hidden, mlpHidden);

  QNNLINEAR_BF16_OR_F32(ffn_gate, x, gate_weight, gate_weight_bf16, seq, hidden, mlpHidden);
  QNNLINEAR_BF16_OR_F32(ffn_up  , x, up_weight  , up_weight_bf16  , seq, hidden, mlpHidden);

  QNNSiluMul(ffn_gate, ffn_up, seq*mlp_hidden);

  //if boolean(down_weight_bf16) then
  //  QNNLinearNoBias_BF16(dst, ffn_gate, down_weight_bf16, seq, mlpHidden, hidden)
  //else
  //  QNNLinearNoBias(dst, ffn_gate     , down_weight     , seq, mlpHidden, hidden)

  QNNLINEAR_BF16_OR_F32(dst, ffn_gate, down_weight, down_weight_bf16, seq, mlpHidden, hidden)

end;

(* ========================================================================
  Double-Stream Block (MM-DiT)
  ========================================================================

  One MM-DiT (Multi-Modal DiT) double block. Processes image and text as
  separate streams with their own Q/K/V projections, but performs joint
  attention (K and V are concatenated from both streams so each modality
  attends to the other). Each stream gets its own AdaLN modulation with
  6 parameters: shift1, scale1, gate1 (pre-attention), shift2, scale2,
  gate2 (pre-FFN). The gating mechanism controls information flow from
  the attention and FFN residual paths. *)

procedure TTransformerFlux.doubleBlockForward(const img_hidden,
  txt_hidden: PQNNFloat; const blockIdx: longint; const img_mod, txt_mod,
  img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin: PQNNFloat;
  const img_seq, txt_seq: longint);

const AXIS_DIM = 32;

var
  hidden : longint;
  img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2,
  txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2,
  img_norm, img_q, img_k, img_v, txt_norm, txt_q, txt_k, txt_v,
  img_proj, txt_proj :PQNNFloat;
  block : PDoubleBlock;
begin
  hidden := num_heads*head_dim;

  (* Extract pre-computed modulation parameters *)
  img_shift1 := img_mod;
  img_scale1 := img_mod + hidden;
  img_gate1  := img_mod + hidden*2;
  img_shift2 := img_mod + hidden*3;
  img_scale2 := img_mod + hidden*4;
  img_gate2  := img_mod + hidden*5;

  txt_shift1 := txt_mod;
  txt_scale1 := txt_mod + hidden;
  txt_gate1  := txt_mod + hidden*2;
  txt_shift2 := txt_mod + hidden*3;
  txt_scale2 := txt_mod + hidden*4;
  txt_gate2  := txt_mod + hidden*5;

  img_norm := work1;
  txt_norm := img_norm + img_seq*hidden;

  img_q    := work2;
  img_k    := img_q + img_seq*hidden;
  img_v    := img_k + img_seq*hidden;

  txt_q    := img_v + img_seq*hidden;
  txt_k    := txt_q + txt_seq*hidden;
  txt_v    := txt_k + txt_seq*hidden;

  block := @double_blocks[blockIdx];

  QNNAdaLN(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden);
  QNNLINEAR_BF16_OR_F32(img_q, img_norm, block.img_q_weight, block.img_q_weight_bf16, img_seq, hidden, hidden);
  QNNLINEAR_BF16_OR_F32(img_k, img_norm, block.img_k_weight, block.img_k_weight_bf16, img_seq, hidden, hidden);
  QNNLINEAR_BF16_OR_F32(img_v, img_norm, block.img_v_weight, block.img_v_weight_bf16, img_seq, hidden, hidden);
  QNNQKRMSNorm(img_q, img_k, block.img_norm_q_weight, block.img_norm_k_weight, img_seq, num_heads, head_dim);
  QNNApplyRoPE2D(img_q, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);
  QNNApplyRoPE2D(img_v, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);

  QNNAdaLN(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden);
  QNNLINEAR_BF16_OR_F32(txt_q, txt_norm, block.txt_q_weight, block.txt_q_weight_bf16, txt_seq, hidden, hidden);
  QNNLINEAR_BF16_OR_F32(txt_k, txt_norm, block.txt_k_weight, block.txt_k_weight_bf16, txt_seq, hidden, hidden);
  QNNLINEAR_BF16_OR_F32(txt_v, txt_norm, block.txt_v_weight, block.txt_v_weight_bf16, txt_seq, hidden, hidden);
  QNNQKRMSNorm(txt_q, txt_k, block.txt_norm_q_weight, block.txt_norm_k_weight, txt_seq, num_heads, head_dim);
  QNNApplyRoPE2D(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);
  QNNApplyRoPE2D(txt_v, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);

  jointAttentionForward(double_img_attn_out, double_txt_attn_out, img_q, img_k, img_v, txt_q, txt_k, txt_v, img_seq, txt_seq{, num_heads, head_dim});

  img_proj := work1;
  txt_proj := img_proj + img_seq*hidden;

  QNNLINEAR_BF16_OR_F32(img_proj, double_img_attn_out, block.img_proj_weight, block.img_proj_weight_bf16, img_seq, hidden, hidden);
  QNNLINEAR_BF16_OR_F32(txt_proj, double_txt_attn_out, block.txt_proj_weight, block.txt_proj_weight_bf16, txt_seq, hidden, hidden);

  QNNGatedAdd(img_hidden, img_gate1, img_proj, img_seq, hidden);
  QNNGatedAdd(txt_hidden, txt_gate1, txt_proj, txt_seq, hidden);

  QNNAdaLN(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden);
  ffnSwigluForward(img_proj, img_norm,
                    block.img_mlp_gate_weight, block.img_mlp_up_weight,
                    block.img_mlp_down_weight,
                    block.img_mlp_gate_weight_bf16, block.img_mlp_up_weight_bf16,
                    block.img_mlp_down_weight_bf16,
                    img_seq, hidden, mlp_hidden);
  QNNGatedAdd(img_hidden, img_gate2, img_proj, img_seq, hidden);

  QNNAdaLN(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden);
  ffnSwigluForward(txt_proj, txt_norm,
                     block.txt_mlp_gate_weight, block.txt_mlp_up_weight,
                     block.txt_mlp_down_weight,
                     block.txt_mlp_gate_weight_bf16, block.txt_mlp_up_weight_bf16,
                     block.txt_mlp_down_weight_bf16,
                     txt_seq, hidden, mlp_hidden);
  QNNGatedAdd(txt_hidden, txt_gate2, txt_proj, txt_seq, hidden);

end;

(* ========================================================================
   Single-stream DiT block operating on concatenated [text, image] tokens.
   ========================================================================

   Unlike double blocks, text and image share the same self-attention and are
   processed as one sequence. Uses a fused projection that outputs [Q, K, V,
   gate, up] in one matmul for efficiency. Text and image portions get
   different RoPE (text: axis 3, image: axes 1,2). After attention, the SwiGLU
   MLP output is concatenated with the attention output before the output
   projection -- this parallel attention+MLP design saves a serial step.*)

procedure TTransformerFlux.singleBlockForward(const hidden: PQNNFloat;
  const blockIdx: longint; const t_emb, adaln_weight, img_rope_cos,
  img_rope_sin, txt_rope_cos, txt_rope_sin: PQNNFloat; const seq,
  img_offset: longint);
begin

end;

//var
//  db : TDoubleBlock;
//  sb: TSingleBlock;
//  tr : TTransformerFlux;
initialization
  //tr := TTransformerFlux.LoadFromDir('C:\development\flux2.c\FLUX.2-klein-base-4B');
  //
  //db.load(tr.sf_files, 0, tr.mlp_hidden, tr.hidden_size);

end.

