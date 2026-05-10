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
uses SysUtils, safetensor, quicknn_common, quicknn_kernels, quicknn_vae, quicknn_qwen3, quickjson
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  ;


type
  PSingles = ^TSingles;
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
  PTimeEmbed = ^TTimeEmbed;
  TTimeEmbed = record
    fc1_weight: TSingles; // [hidden, 256]
    fc2_weight: TSingles; // [hidden, hidden]
    sincos_dim: longint; // 256 for FLUX.2-klein
    procedure forward(const dst, sinCos:TMemoryBlock; const hidden: longint; const outSilu:PQNNFloat = nil);
    procedure free();
  end;

const MAX_TF_SHARDS = 4;

type

  //Full transformer context

  { TTransformerFlux }
  PTransformerFlux = ^TTransformerFlux;

  TTransformerFlux = record
  const
    CONFIG_FILE = '/transformer/config.json';
  public
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
    attn_scores_alloc: size_t;            //  Currently allocated size
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
    eps : single;
    class function parse(const modelDir: string):TTransformerFlux; static;
    constructor load(const modelDir: string; const useMMAp:boolean = true);
    function isLoaded():boolean;
    procedure reInitialize(const totalSeq:longint);
    procedure free();
    procedure freeMMapCache();
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

    // for text to image
    function forward(const img_latent: PQNNFloat; const img_h, img_w: longint; const txt_emb: PQNNFloat; const txt_seq: longint; timestep: QNNFloat):TMemoryBlock; overload;
    // for reference image with text to image?
    function forward(const img_latent: PQNNFloat; const img_h, img_w: longint; const ref_latent: PQNNFloat; const ref_h, ref_w, t_offset: longint; const txt_emb: PQNNFloat; const txt_seq: longint; timestep: QNNFloat):TMemoryBlock; overload;
    // for multi-reference images with text to image
    function forward(const img_latent: PQNNFloat; const img_h, img_w: longint; const refs: TArray<TImageRef>; const txt_emb: PQNNFloat; const txt_seq: longint; const timestep: single):TMemoryBlock; overload;

    function sampleEuler(const z: TMemoryBlock; const batch, channels, h, w: longint;
      const text_emb: PQNNFloat; const text_seq: longint;
      const schedule: PQNNFloat; const num_steps: longint; const progress_callback:TStepCallback): TMemoryBlock; overload;

    function sampleEuler(const z: PQNNFloat; const batch, channels, h, w: longint; const text_emb_cond: PQNNFloat;
             const text_seq_cond: longint;
             const text_emb_uncond: PQNNFloat; const text_seq_uncond: longint;
             const guidance_scale: QNNFloat;
             const schedule: PQNNFloat; const num_steps: longint;
             const progress_callback: TStepCallback):TMemoryBlock; overload;

    function sampleEuler(const z: TMemoryBlock;
      const batch, channels, h, w: longint; const ref_latent: PQNNFloat;
      const ref_h, ref_w, t_offset: longint; const text_emb_cond: PQNNFloat;
      const text_seq_cond: longint; const text_emb_uncond: PQNNFloat;
      const text_seq_uncond: longint; const guidance_scale: QNNFloat;
      const schedule: PQNNFloat; const num_steps: longint;
      const progress_callback: TStepCallback): TMemoryBlock; overload;
  end;

  TQNNCtx = record
    tokenixer : TQNNTokenizer;
    qwen3Encoder : TQWEN3Encoder;
    vae : TVAE;
    transformer : TTransformerFlux;

    (* Configuration *)
    max_width          : longint ;
    max_height         : longint ;
    default_steps      : longint ;
    default_guidance   : QNNFloat ;
    is_distilled       : boolean ;  (* 1 = distilled (4-step), 0 = base (50-step CFG) *)
    text_dim           : longint ;      (* Text embedding dimension (7680 for 4B, varies for 9B) *)
    is_non_commercial  : boolean ; (* 1 if model has non-commercial license (9B) *)
    num_heads          : longint ;     (* Transformer attention heads (24 for 4B, 32 for 9B) *)

    (* Z-Image specific config (from transformer/config.json) *)
    is_zimage          : boolean ;     (* 1 = Z-Image S3-DiT, 0 = Flux MMDiT *)
    zi_dim             : longint   ;            (* Hidden dim (3840) *)
    zi_n_layers        : longint   ;       (* Main transformer layers (30) *)
    zi_n_refiner       : longint   ;      (* Noise/context refiner layers (2) *)
    zi_cap_feat_dim    : longint   ;   (* Caption feature dim (2560) *)
    zi_in_channels     : longint   ;    (* VAE latent channels (16) *)
    zi_patch_size      : longint   ;     (* Spatial patch size (2) *)
    zi_rope_theta      : QNNFloat ;   (* RoPE theta (256.0) *)
    zi_axes_dims       : array[0..2] of longint   ;   (* RoPE axis dims [32, 48, 48] *)
    zi_latent_channels : longint   ;(* Patchified latent channels (64 = 16*2*2) *)

    (* VAE config (read from vae/config.json) *)
    vae_z_channels     : longint   ;    (* Latent channels before patchify (32 Flux, 16 Z-Image) *)
    vae_scaling        : QNNFloat ;     (* Scaling factor (0.3611 for Z-Image, 0 = use batch norm) *)
    vae_shift          : QNNFloat ;       (* Shift factor (0.1159 for Z-Image, 0 = use batch norm) *)

    (* Model info *)
    model_name, model_version, model_dir : string  ;  (* For reloading text encoder if released *)

    (* Memory mode *)
    use_mmap           : boolean   ;  (* Use mmap for text encoder (lower memory, slower) *)
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
    img_mlp_up_weight := PSingle(img_mlp_gate_weight) + mlp*h;
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
    txt_mlp_up_weight := PSingle(txt_mlp_gate_weight) + mlp * h;
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
  img_norm_q_weight.free;
  img_norm_k_weight.free;
  img_q_weight.free;
  img_k_weight.free;
  img_v_weight.free;
  img_proj_weight.free;
  img_mlp_gate_weight.free;
  img_mlp_up_weight.free;
  img_mlp_down_weight.free;
  //
  img_q_weight_bf16.free;
  img_k_weight_bf16.free;
  img_v_weight_bf16.free;
  img_proj_weight_bf16.free;
  img_mlp_gate_weight_bf16.free;
  img_mlp_up_weight_bf16.free;
  img_mlp_down_weight_bf16.free;

  txt_norm_q_weight.free;
  txt_norm_k_weight.free;
  txt_q_weight.free;
  txt_k_weight.free;
  txt_v_weight.free;
  txt_proj_weight.free;
  txt_mlp_gate_weight.free;
  txt_mlp_up_weight.free;
  txt_mlp_down_weight.free;

  txt_q_weight_bf16.free;
  txt_k_weight_bf16.free;
  txt_v_weight_bf16.free;
  txt_proj_weight_bf16.free;
  txt_mlp_gate_weight_bf16.free;
  txt_mlp_up_weight_bf16.free;
  txt_mlp_down_weight_bf16.free;
  self := default(TDoubleBlock);
end;

procedure TSingleBlock.free;
begin
  norm_q_weight  .free;
  norm_k_weight  .free;
  qkv_mlp_weight .free;
  proj_mlp_weight.free;
  //(* bf16 pointers are direct mmap pointers - just clear, don't free *)
  //
  qkv_mlp_weight_bf16 .free;
  proj_mlp_weight_bf16.free;

  self := default(TSingleBlock);
end;

{ TTimeEmbed }

procedure TTimeEmbed.forward(const dst, sinCos: TMemoryBlock;
  const hidden: longint; const outSilu: PQNNFloat);
begin
  //if length(workspace)< 1*hidden then
  //  setLength(workspace, hidden);

  QNNLinearNoBias(outSilu, sinCos, fc1_weight, 1, sincos_dim, hidden);
  QNNSilu(outSilu, hidden);
  QNNLinearNoBias(dst, outSilu, fc2_weight, 1, hidden, hidden);
end;

procedure TTimeEmbed.free();
begin
  fc1_weight.free; // [hidden, 256]
  fc2_weight.free; // [hidden, hidden]
  self := default(TTimeEmbed)
end;

{ TTransformerFlux }

class function TTransformerFlux.parse(const modelDir: string): TTransformerFlux;
var json : TJSON;
  headsCount, headDim, layerCount, singleCount, JointAttDim, inChannels:longint;
  mlpRatio, ropeTheta : single;
  //eps : single;
begin
  result := default(TTransformerFlux);
  json := TJSON.LoadFromFile(modelDir+CONFIG_FILE);
  headsCount    := json['num_attention_heads'];
  headDim       := json['attention_head_dim'];
  layerCount    := json['num_layers'];
  singleCount   := json['num_single_layers'];
  jointAttDim   := json['joint_attention_dim'];
  inChannels    := json['in_channels'];
  mlpRatio      := json['mlp_ratio'].value;
  ropeTheta     := json['rope_theta'].value;
  result.eps    := json['eps'].Value;

  result.num_heads         := headsCount;
  result.head_dim          := headDim;
  result.hidden_size       := headsCount * headDim;

  if mlpRatio>0 then
    result.mlp_hidden        := trunc(result.hidden_size * mlpRatio)
  else
    result.mlp_hidden        := result.hidden_size *  3;


  result.num_double_layers := ifthen(layerCount > 0, layerCount, 5);
  result.num_single_layers := ifthen(singleCount > 0, singleCount, 20);
  result.text_dim          := ifthen(jointAttDim > 0, jointAttDim, 7680);
  result.latent_channels   := ifthen(inChannels > 0, inChannels, 128);

  if ropeTheta > 0 then
    result.rope_theta        := ropeTheta
  else
    result.rope_theta        := 2000.0;

  result.rope_dim          := headDim;
  result.axis_dim          := headDim div 4;  // 4 RoPE axes

  result.sf_files := loadShards(modelDir);
end;

constructor TTransformerFlux.load(const modelDir: string; const useMMAp: boolean);
var cfgfile : string;
begin
  self := default(TTransformerFlux);
  cfgfile := modelDir;
  if FileExists(cfgfile+CONFIG_FILE) then
    self := parse(cfgfile)
  else begin
    hidden_size := 3072;
    num_heads   := 24;
    head_dim    := 128;
    mlp_hidden  := 9216;
    num_double_layers := 5;
    num_single_layers := 20;
    text_dim    := 7680;
    latent_channels := 128;
    rope_theta  := 2000.0;
    rope_dim    := 128;
    axis_dim    := 32;
  end;
  max_seq_len   := 52000;
  use_mmap := useMMap;
  sf_files := loadShards(modelDir);

  img_in_weight := sf_files.getTensorDataMemBlock('x_embedder.weight', use_mmap);
  txt_in_weight := sf_files.getTensorDataMemBlock('context_embedder.weight', use_mmap);
  if use_bf16 then
      begin
          img_in_weight_bf16 := sf_files.getTensorDataMemBlockBF16('x_embedder.weight', use_mmap);
          txt_in_weight_bf16 := sf_files.getTensorDataMemBlockBF16('context_embedder.weight', use_mmap)
      end;
  time_embed.sincos_dim := 256;
  time_embed.fc1_weight := sf_files.getTensorDataMemBlock('time_guidance_embed.timestep_embedder.linear_1.weight', use_mmap);
  time_embed.fc2_weight := sf_files.getTensorDataMemBlock('time_guidance_embed.timestep_embedder.linear_2.weight', use_mmap);
  adaln_double_img_weight := sf_files.getTensorDataMemBlock('double_stream_modulation_img.linear.weight', use_mmap);
  adaln_double_txt_weight := sf_files.getTensorDataMemBlock('double_stream_modulation_txt.linear.weight', use_mmap);
  adaln_single_weight := sf_files.getTensorDataMemBlock('single_stream_modulation.linear.weight', use_mmap);
  if use_bf16 then
      begin
          adaln_double_img_weight_bf16 := sf_files.getTensorDataMemBlockBF16('double_stream_modulation_img.linear.weight', use_mmap);
          adaln_double_txt_weight_bf16 := sf_files.getTensorDataMemBlockBF16('double_stream_modulation_txt.linear.weight', use_mmap);
          adaln_single_weight_bf16 := sf_files.getTensorDataMemBlockBF16('single_stream_modulation.linear.weight', use_mmap)
      end;
  setLength(double_blocks, num_double_layers);
  setLength(single_blocks, num_single_layers);
  final_norm_weight := sf_files.getTensorDataMemBlock('norm_out.linear.weight', use_mmap);
  final_proj_weight := sf_files.getTensorDataMemBlock('proj_out.weight', use_mmap);
  if use_bf16 then
      final_proj_weight_bf16 := sf_files.getTensorDataMemBlockBF16('proj_out.weight', use_mmap);
  rope_freqs := TSingles.Create(max_seq_len * head_dim);
  if rope_freqs.isAssigned() then
      QNNComputeRoPE(rope_freqs, max_seq_len, head_dim, rope_theta);

  (* Small constant-size buffers - always allocate *)
  t_emb_silu     := TSingles.Create(hidden_size);
  double_mod_img := TSingles.Create(hidden_size * 6);
  double_mod_txt := TSingles.Create(hidden_size * 6);
end;

function TTransformerFlux.isLoaded(): boolean;
begin
  result := assigned(sf_files)
end;

procedure TTransformerFlux.reInitialize(const totalSeq: longint);
var fused_dim, hidden, mlp:longint; attn_scores_need:size_t;
begin
  attn_scores_need := num_heads*totalSeq*totalSeq;

  if attn_scores_need > attn_scores_alloc then begin
      if attn_scores.isAssigned() then
        attn_scores.reSize(attn_scores_need)
      else
        attn_scores := TMemoryBlock.Create(attn_scores_need);
      attn_scores_alloc := attn_scores_need;
  end;

  if totalSeq<=work_seq_alloc then exit;

  hidden := hidden_size;
  mlp    := mlp_hidden;


  img_hidden          .free;
  txt_hidden          .free;
  work1               .free;
  work2               .free;
  attn_q_t            .free;
  attn_k_t            .free;
  attn_v_t            .free;
  attn_out_t          .free;
  attn_cat_k          .free;
  attn_cat_v          .free;
  single_q            .free;
  single_k            .free;
  single_v            .free;
  single_mlp_gate     .free;
  single_mlp_up       .free;
  single_attn_out     .free;
  single_concat       .free;
  ffn_gate            .free;
  ffn_up              .free;
  double_img_attn_out .free;
  double_txt_attn_out .free;




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

procedure TTransformerFlux.free();
var i:longint;
begin
    img_in_weight.free;
    txt_in_weight.free;
    img_in_weight_bf16.free;
    txt_in_weight_bf16.free;
    time_embed.fc1_weight.free;
    time_embed.fc2_weight.free;
    for i:=0 to high(double_blocks) do double_blocks[i].free;
    for i:=0 to high(single_blocks) do single_blocks[i].free;

    setLength(double_blocks, 0);
    setLength(single_blocks, 0);

    final_norm_weight.free;
    final_proj_weight.free;
    final_proj_weight_bf16.free;
    rope_freqs.free;
    img_hidden.free;
    txt_hidden.free;
    work1.free;
    work2.free;
    adaln_double_img_weight.free;
    adaln_double_txt_weight.free;
    adaln_single_weight.free;
    adaln_double_img_weight_bf16.free;
    adaln_double_txt_weight_bf16.free;
    adaln_single_weight_bf16.free;
    attn_q_t.free;
    attn_k_t.free;
    attn_v_t.free;
    attn_out_t.free;
    attn_scores.free;
    attn_cat_k.free;
    attn_cat_v.free;
    single_q.free;
    single_k.free;
    single_v.free;
    single_mlp_gate.free;
    single_mlp_up.free;
    single_attn_out.free;
    single_concat.free;
    ffn_gate.free;
    ffn_up.free;
    t_emb_silu.free;
    double_mod_img.free;
    double_mod_txt.free;
    double_img_attn_out.free;
    double_txt_attn_out.free;
    cached_img_rope_cos.free;
    cached_img_rope_sin.free;
    cached_ref_rope_cos.free;
    cached_ref_rope_sin.free;
    cached_txt_rope_cos.free;
    cached_txt_rope_sin.free;
    cached_combined_rope_cos.free;
    cached_combined_rope_sin.free;
    for i:=0 to high(sf_files) do
      sf_files[i].close();
    self := default(TTransformerFlux)
end;

procedure TTransformerFlux.freeMMapCache();
var i:longint;
begin
  if use_mmap then begin
    for i:=0 to high(double_blocks) do double_blocks[i].free;
    for i:=0 to high(single_blocks) do single_blocks[i].free;
    setLength(double_blocks, 0);
    setLength(single_blocks, 0);
  end;
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
    if cached_ref_rope_cos.isAssigned() then cached_ref_rope_cos.free();
    if cached_ref_rope_sin.isAssigned() then cached_ref_rope_sin.free();
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
    if cached_img_rope_cos.isAssigned() then cached_img_rope_cos.free;
    if cached_img_rope_sin.isAssigned() then cached_img_rope_sin.free;

    cached_img_rope_cos := TMemoryBlock.Create(size, dtF32);
    cached_img_rope_sin := TMemoryBlock.Create(size, dtF32);
    cached_img_h := patch_h;
    cached_img_w := patch_w;
    QNNComputeRoPE2D(cached_img_rope_cos, cached_img_rope_sin, patch_h, patch_w, axisDim, rope_theta);
    cosOut := cached_img_rope_cos;
    sinOut := cached_img_rope_sin
end;

procedure TTransformerFlux.getCachedTxtRoPE(const txt_seq: longint;
  var cosOut: TSingles; var sinOut: TSingles);
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
    if cached_txt_rope_cos.isAssigned() then cached_txt_rope_cos.free();
    if cached_txt_rope_sin.isAssigned() then cached_txt_rope_sin.free();
    cached_txt_rope_cos := TMemoryBlock.Create(size, dtF32);
    cached_txt_rope_sin := TMemoryBlock.Create(size, dtF32);
    cached_txt_seq := txt_seq;
    QNNComputeRoPEText(cached_txt_rope_cos, cached_txt_rope_sin, txt_seq, axis_dim, rope_theta);
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
    if cached_combined_rope_cos.isAssigned() then cached_combined_rope_cos.free();
    if cached_combined_rope_sin.isAssigned() then cached_combined_rope_sin.free();
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

    img_size := img_seq * axisDim * 4;
    ref_size := ref_seq * axisDim * 4;

    QNNCopy(PQNNFloat(cached_combined_rope_cos)                        ,img_rope_cos, img_size);
    QNNCopy(PQNNFloat(cached_combined_rope_cos) + img_size             ,ref_rope_cos, ref_size);
    QNNCopy(PQNNFloat(cached_combined_rope_sin)                        ,img_rope_sin, img_size);
    QNNCopy(PQNNFloat(cached_combined_rope_sin) + img_size             ,ref_rope_sin, ref_size);

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
  img_K, img_V, txt_Q, txt_K, txt_V: PQNNFloat; const img_seq, txt_seq: longint);
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
            txtQh  := txt_Q   + i*headDim;
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

const AXISDIM = 32;

var
  img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2,
  txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2,
  img_norm, img_q, img_k, img_v, txt_norm, txt_q, txt_k, txt_v,
  img_proj, txt_proj :PQNNFloat;
  block : PDoubleBlock;
begin

  (* Extract pre-computed modulation parameters *)
  img_shift1 := img_mod;
  img_scale1 := img_mod + hidden_size;
  img_gate1  := img_mod + hidden_size*2;
  img_shift2 := img_mod + hidden_size*3;
  img_scale2 := img_mod + hidden_size*4;
  img_gate2  := img_mod + hidden_size*5;

  txt_shift1 := txt_mod;
  txt_scale1 := txt_mod + hidden_size;
  txt_gate1  := txt_mod + hidden_size*2;
  txt_shift2 := txt_mod + hidden_size*3;
  txt_scale2 := txt_mod + hidden_size*4;
  txt_gate2  := txt_mod + hidden_size*5;

  img_norm := work1;
  txt_norm := img_norm + img_seq*hidden_size;

  img_q    := work2;
  img_k    := img_q + img_seq*hidden_size;
  img_v    := img_k + img_seq*hidden_size;

  txt_q    := img_v + img_seq*hidden_size;
  txt_k    := txt_q + txt_seq*hidden_size;
  txt_v    := txt_k + txt_seq*hidden_size;

  block := @double_blocks[blockIdx];

  QNNAdaLN(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden_size);
  QNNLINEAR_BF16_OR_F32(img_q, img_norm, block.img_q_weight, block.img_q_weight_bf16, img_seq, hidden_size, hidden_size);
  QNNLINEAR_BF16_OR_F32(img_k, img_norm, block.img_k_weight, block.img_k_weight_bf16, img_seq, hidden_size, hidden_size);
  QNNLINEAR_BF16_OR_F32(img_v, img_norm, block.img_v_weight, block.img_v_weight_bf16, img_seq, hidden_size, hidden_size);
  QNNQKRMSNorm(img_q, img_k, block.img_norm_q_weight, block.img_norm_k_weight, img_seq, num_heads, head_dim);

  QNNApplyRoPE2D(img_q, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXISDIM);
  QNNApplyRoPE2D(img_k, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXISDIM);
//printStat(txt_hidden, img_seq*hidden_size);
  QNNAdaLN(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden_size);
//printStat(txt_norm, img_seq*hidden_size);
  QNNLINEAR_BF16_OR_F32(txt_q, txt_norm, block.txt_q_weight, block.txt_q_weight_bf16, txt_seq, hidden_size, hidden_size);
  QNNLINEAR_BF16_OR_F32(txt_k, txt_norm, block.txt_k_weight, block.txt_k_weight_bf16, txt_seq, hidden_size, hidden_size);
  QNNLINEAR_BF16_OR_F32(txt_v, txt_norm, block.txt_v_weight, block.txt_v_weight_bf16, txt_seq, hidden_size, hidden_size);
  QNNQKRMSNorm(txt_q, txt_k, block.txt_norm_q_weight, block.txt_norm_k_weight, txt_seq, num_heads, head_dim);

//printStat(txt_q, txt_seq * hidden_size);
//printStat(txt_k, txt_seq * hidden_size);
//printStat(txt_v, txt_seq * hidden_size);

  QNNApplyRoPE2D(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXISDIM);
  QNNApplyRoPE2D(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXISDIM);

//printStat(txt_q, txt_seq * hidden_size);
//printStat(txt_k, txt_seq * hidden_size);
//printStat(txt_v, txt_seq * hidden_size);

  jointAttentionForward(double_img_attn_out, double_txt_attn_out, img_q, img_k, img_v, txt_q, txt_k, txt_v, img_seq, txt_seq{, num_heads, head_dim});

  img_proj := work1;
  txt_proj := img_proj + img_seq*hidden_size;

  QNNLINEAR_BF16_OR_F32(img_proj, double_img_attn_out, block.img_proj_weight, block.img_proj_weight_bf16, img_seq, hidden_size, hidden_size);
  QNNLINEAR_BF16_OR_F32(txt_proj, double_txt_attn_out, block.txt_proj_weight, block.txt_proj_weight_bf16, txt_seq, hidden_size, hidden_size);

  QNNGatedAdd(img_hidden, img_gate1, img_proj, img_seq, hidden_size);
  QNNGatedAdd(txt_hidden, txt_gate1, txt_proj, txt_seq, hidden_size);

  QNNAdaLN(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden_size);
  ffnSwigluForward(img_proj, img_norm,
                    block.img_mlp_gate_weight, block.img_mlp_up_weight,
                    block.img_mlp_down_weight,
                    block.img_mlp_gate_weight_bf16, block.img_mlp_up_weight_bf16,
                    block.img_mlp_down_weight_bf16,
                    img_seq, hidden_size, mlp_hidden);
  QNNGatedAdd(img_hidden, img_gate2, img_proj, img_seq, hidden_size);

  QNNAdaLN(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden_size);
  ffnSwigluForward(txt_proj, txt_norm,
                     block.txt_mlp_gate_weight, block.txt_mlp_up_weight,
                     block.txt_mlp_down_weight,
                     block.txt_mlp_gate_weight_bf16, block.txt_mlp_up_weight_bf16,
                     block.txt_mlp_down_weight_bf16,
                     txt_seq, hidden_size, mlp_hidden);
  QNNGatedAdd(txt_hidden, txt_gate2, txt_proj, txt_seq, hidden_size);

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

const AXIS_DIM = 32;
var
  fused_dim, mod_size, s, img_seq, txt_seq: longint;
  mod_params, shift, scale, gate, norm, fused_out, q, k, v, mlp_gate,
    mlp_up, row, img_q, img_k, attn_out, concat, proj_out: PQNNFloat;
  block : PSingleBlock;
begin

    fused_dim := hidden_size*3 + mlp_hidden*2;
    img_seq := seq - img_offset;
    mod_size := hidden_size * 3;
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for i := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[i];
    //        t_emb_silu[i] := x / (1.0+expf(-x))
    //    end;
    mod_params := PQNNFloat(work2) + seq*fused_dim;

    QNNLinearNoBias(mod_params, t_emb_silu, adaln_weight, 1, hidden_size, mod_size);
    shift := mod_params;
    scale := mod_params + hidden_size;
    gate  := mod_params + hidden_size * 2;
    norm  := work1;
    QNNAdaLN(norm, hidden, shift, scale, seq, hidden_size);
    fused_out := work2;
    block := @single_blocks[blockIdx];
    QNNLINEAR_BF16_OR_F32(fused_out, norm, block.qkv_mlp_weight, block.qkv_mlp_weight_bf16, seq, hidden_size, fused_dim);
    q := single_q;
    k := single_k;
    v := single_v;
    mlp_gate := single_mlp_gate;
    mlp_up   := single_mlp_up;
    for s := 0 to seq -1 do begin
        row := fused_out+s * fused_dim;
        QNNCopy(q        + s*hidden_size, row                             , hidden_size);
        QNNCopy(k        + s*hidden_size, row + hidden_size               , hidden_size);
        QNNCopy(v        + s*hidden_size, row + hidden_size*2             , hidden_size);
        QNNCopy(mlp_gate + s*mlp_hidden , row + hidden_size*3             , mlp_hidden );
        QNNCopy(mlp_up   + s*mlp_hidden , row + hidden_size*3 + mlp_hidden, mlp_hidden )
    end;
    QNNQKRMSNorm(q, k, block.norm_q_weight, block.norm_k_weight, seq, num_heads, head_dim);
    txt_seq := img_offset;
    QNNApplyRoPE2D(q, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);
    QNNApplyRoPE2D(k, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);
    img_q := q+img_offset * hidden_size;
    img_k := k+img_offset * hidden_size;
    QNNApplyRoPE2D(img_q, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);
    QNNApplyRoPE2D(img_k, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);
    attn_out := single_attn_out;
    multiHeadForward(attn_out, q, k, v, seq);
    QNNSiluMul(mlp_gate, mlp_up, seq*mlp_hidden);
    concat := single_concat;
    for s := 0 to seq -1 do begin
        QNNCopy(concat + s*(hidden_size+mlp_hidden)              , attn_out + s*hidden_size, hidden_size);
        QNNCopy(concat + s*(hidden_size+mlp_hidden) + hidden_size, mlp_gate + s*mlp_hidden , mlp_hidden )
    end;
    proj_out := work1;
    QNNLINEAR_BF16_OR_F32(proj_out, concat, block.proj_mlp_weight, block.proj_mlp_weight_bf16, seq, hidden_size+mlp_hidden, hidden_size);
    QNNGatedAdd(hidden, gate, proj_out, seq, hidden_size)
end;

function TTransformerFlux.forward(const img_latent: PQNNFloat; const img_h,
  img_w: longint; const txt_emb: PQNNFloat; const txt_seq: longint;
  timestep: QNNFloat): TMemoryBlock;

var
    img_seq, total_seq, double_mod_size, i: longint;
    //x: single;
    t_emb, img_transposed, concat_hidden, img_rope_cos, txt_rope_cos, img_rope_sin, txt_rope_sin, output_nlc : TSingles;
    final_mod, final_scale, final_shift, final_norm : PQNNFloat;
    t_sincos : array[0..255] of QNNFloat;
begin
    img_seq := img_h*img_w;
    total_seq := img_seq+txt_seq;
    reInitialize(total_seq);

    t_emb := TMemoryBlock.Create(hidden_size);
    QNNTimestepEmbedding(t_sincos, timestep * 1000.0, time_embed.sincos_dim, 10000.0);

    time_embed.forward(t_emb, t_sincos,  hidden_size, t_emb_silu);

    getCachedImgRoPE(img_h, img_w, img_rope_cos, img_rope_sin);

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);

    img_transposed := TMemoryBlock.Create(img_seq * latent_channels);
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        img_transposed[pos*latent_channels + c] := img_latent[c*img_seq + pos];
    QNNMatTranspose(img_transposed, img_latent, latent_channels, img_seq);

    QNNLINEAR_BF16_OR_F32(img_hidden, img_transposed, img_in_weight, img_in_weight_bf16, img_seq, latent_channels, hidden_size);
    img_transposed.free;

    QNNLINEAR_BF16_OR_F32(txt_hidden, txt_emb, txt_in_weight, txt_in_weight_bf16, txt_seq, text_dim, hidden_size);
//printStat(txt_emb, txt_seq*hidden_size);
//printStat(txt_hidden, txt_seq*hidden_size);

    double_mod_size := hidden_size * 6;

    //for j := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[j];
    //        t_emb_silu[j] := x / (1.0+expf(-x))
    //    end;
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    QNNLinearNoBias(double_mod_img, t_emb_silu, adaln_double_img_weight, 1, hidden_size, double_mod_size);
    QNNLinearNoBias(double_mod_txt, t_emb_silu, adaln_double_txt_weight, 1, hidden_size, double_mod_size);

    for i := 0 to num_double_layers {high(double_blocks)}-1 do begin
      if use_mmap and not (boolean(double_blocks[i].img_q_weight) or boolean(double_blocks[i].img_q_weight_bf16)) then
          double_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
      // todo TTransformerFlux.forward : {} nest to a param means it's redundant, it is a record member var, remove/refactor
      doubleBlockForward(img_hidden, txt_hidden{}, i, double_mod_img{}, double_mod_txt{}, img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin, img_seq, txt_seq);
      if use_mmap then
        double_blocks[i].free;
//img_hidden.printStat;
//txt_hidden.printStat;
      if assigned(substep_callback) then
          substep_callback(SUBSTEP_DOUBLE_BLOCK, i, num_double_layers);
    end;
    concat_hidden := TMemoryBlock.Create(total_seq * hidden_size);
    QNNCopy(concat_hidden                      , txt_hidden, txt_seq*hidden_size);
    QNNCopy(PQNNFloat(concat_hidden) + txt_seq*hidden_size, img_hidden, img_seq*hidden_size);
    for i := 0 to num_single_layers -1 {high(single_blocks)} do
        begin
            if use_mmap and not(boolean(single_blocks[i].qkv_mlp_weight) or boolean(single_blocks[i].qkv_mlp_weight_bf16)) then
                single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
            singleBlockForward(concat_hidden, i, t_emb, adaln_single_weight, img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);
            if use_mmap then
                single_blocks[i].free;
            //    free_single_block_weights(@single_blocks[i]);
            if assigned(substep_callback) then
                substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers)
        end;
    QNNCopy(img_hidden, PQNNFloat(concat_hidden) + txt_seq*hidden_size, img_seq*hidden_size);
    concat_hidden.free;
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for i := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[i];
    //        t_emb_silu[i] := x / (1.0+expf(-x))
    //    end;
    final_mod := double_mod_img;
    QNNLinearNoBias(final_mod, t_emb_silu, final_norm_weight, 1, hidden_size, hidden_size * 2);
    final_scale := final_mod;
    final_shift := final_mod+hidden_size;
    final_norm := work1;
    QNNAdaLN(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden_size, -6);
    output_nlc := TMemoryBlock.create(img_seq * latent_channels);
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.Create(img_seq * latent_channels);
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        result[c * img_seq+pos] := output_nlc[pos * latent_channels+c];
    QNNMatTranspose(result, output_nlc, img_seq, latent_channels);
    output_nlc.free;
    t_emb.free;
    //iris_timing_transformer_double := iris_timing_transformer_double + double_time;
    //iris_timing_transformer_single := iris_timing_transformer_single + single_time;
    //iris_timing_transformer_final := iris_timing_transformer_final + final_time;
    //iris_timing_transformer_total := iris_timing_transformer_total + (double_time+single_time+final_time);
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);

end;

function TTransformerFlux.forward(const img_latent: PQNNFloat; const img_h,
  img_w: longint; const ref_latent: PQNNFloat; const ref_h, ref_w,
  t_offset: longint; const txt_emb: PQNNFloat; const txt_seq: longint;
  timestep: QNNFloat): TMemoryBlock;
var
    img_seq, ref_seq, combined_img_seq, total_seq, pos, c, i, double_mod_size: longint;
    t_sincos: array[0..255] of QNNFloat;

    final_mod, final_scale, final_shift, final_norm, output_nlc_ptr, result_ptr, combined_transposed_ptr: PQNNFloat;

    t_emb, combined_transposed, combined_hidden, txt_rope_cos, txt_rope_sin
    , combined_rope_cos, combined_rope_sin, concat_hidden, output_nlc : TMemoryBlock;
begin
    if not assigned(ref_latent) then
        exit(forward(img_latent, img_h, img_w, txt_emb, txt_seq, timestep));
    img_seq := img_h*img_w;
    ref_seq := ref_h*ref_w;
    combined_img_seq := img_seq + ref_seq;
    total_seq := combined_img_seq + txt_seq;
    reInitialize(total_seq);
    t_emb := TMemoryBlock.Create(hidden_size);
    QNNTimestepEmbedding(t_sincos, timestep * 1000.0, time_embed.sincos_dim, 10000.0);
    time_embed.forward(t_emb, t_sincos, hidden_size, t_emb_silu);
    getCachedCombinedRoPE(img_h, img_w, ref_h, ref_w, t_offset, combined_rope_cos,  combined_rope_sin);

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);
    combined_transposed := TMemoryBlock.Create(combined_img_seq * latent_channels);
    combined_transposed_ptr := combined_transposed;
    for pos := 0 to img_seq -1 do
        for c := 0 to latent_channels -1 do
            combined_transposed_ptr[pos * latent_channels+c] := img_latent[c * img_seq+pos];
    for pos := 0 to ref_seq -1 do
        for c := 0 to latent_channels -1 do
            combined_transposed_ptr[(img_seq+pos) * latent_channels+c] := ref_latent[c * ref_seq+pos];

    combined_hidden := TMemoryBlock.Create(combined_img_seq*hidden_size);
    QNNLINEAR_BF16_OR_F32(combined_hidden, combined_transposed, img_in_weight, img_in_weight_bf16, combined_img_seq, latent_channels, hidden_size);

    QNNLINEAR_BF16_OR_F32(txt_hidden, txt_emb, txt_in_weight, txt_in_weight_bf16, txt_seq, text_dim, hidden_size);
    double_mod_size := hidden_size * 6;

    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for j := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[j];
    //        t_emb_silu[j] := x / (1.0+expf(-x))
    //    end;
    QNNLinearNoBias(double_mod_img, t_emb_silu, adaln_double_img_weight, 1, hidden_size, double_mod_size);
    QNNLinearNoBias(double_mod_txt, t_emb_silu, adaln_double_txt_weight, 1, hidden_size, double_mod_size);
    for i := 0 to num_double_layers -1 do
        begin
            if use_mmap then
                double_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
            doubleBlockForward(combined_hidden, txt_hidden, i, double_mod_img, double_mod_txt, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, combined_img_seq, txt_seq);
            //if use_mmap then
            //    free_double_block_weights( and double_blocks[i]);
            if assigned(substep_callback) then
                substep_callback(SUBSTEP_DOUBLE_BLOCK, i, num_double_layers)
        end;
    concat_hidden := TMemoryBlock.Create(total_seq * hidden_size);
    QNNCopy(concat_hidden, txt_hidden, txt_seq * hidden_size);
    QNNCopy(PQNNFloat(concat_hidden)+txt_seq * hidden_size, combined_hidden, combined_img_seq*hidden_size);
    for i := 0 to num_single_layers -1 do
        begin
            if use_mmap then
                single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
            singleblockforward(concat_hidden, i, t_emb, adaln_single_weight, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);
            if assigned(substep_callback) then
                substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers)
        end;
    img_hidden :=  TMemoryBlock.Create(img_seq * hidden_size);
    QNNCopy(img_hidden, PQNNFloat(concat_hidden)+txt_seq * hidden_size, img_seq * hidden_size);
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for i := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[i];
    //        t_emb_silu[i] := x div (1.0+expf(-x))
    //    end;
    final_mod := double_mod_img;
    QNNLinearNoBias(final_mod, t_emb_silu, final_norm_weight, 1, hidden_size, hidden_size * 2);
    final_scale := final_mod;
    final_shift := final_mod + hidden_size;
    final_norm := work1;
    QNNAdaLN(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden_size, -6);

    output_nlc := TMemoryBlock.create(img_seq * latent_channels);
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.create(img_seq * latent_channels);
    output_nlc_ptr := output_nlc;
    result_ptr := result;
    for pos := 0 to img_seq -1 do
        for c := 0 to latent_channels -1 do
            result_ptr[c * img_seq+pos] := output_nlc_ptr[pos * latent_channels+c];
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.forward(const img_latent: PQNNFloat; const img_h, img_w: longint; const refs: TArray<TImageRef>; const txt_emb: PQNNFloat; const txt_seq: longint; const timestep: single):TMemoryBlock;
var
    img_seq, axis_dim, total_ref_seq, r, combined_img_seq,
      total_seq, rope_offset, ref_seq, pos, c,
      trans_offset, i, double_mod_size: longint;

    t_emb, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, output_nlc : TMemoryBlock;
      combined_transposed, combined_hidden, concat_hidden,
      img_hidden, final_mod, final_scale, final_shift, final_norm, result_ptr, output_ptr: PQNNFloat;
    t_sincos : array[0..255] of QNNFloat;
begin
    if not assigned(refs) then
        exit(forward(img_latent, img_h, img_w, txt_emb, txt_seq, timestep));
    if (length(refs) = 1) then
        exit(forward(img_latent, img_h, img_w, refs[0].latent, refs[0].h, refs[0].w, refs[0].offset, txt_emb, txt_seq, timestep));

    img_seq := img_h * img_w;
    axis_dim := 32;
    total_ref_seq := 0;
    for r := 0 to high(refs) do
        inc(total_ref_seq, refs[r].h*refs[r].w);
    combined_img_seq := img_seq+total_ref_seq;
    total_seq := combined_img_seq+txt_seq;
    reInitialize(total_seq);
    t_emb := TMemoryBlock.Create(hidden_size);

    QNNTimestepEmbedding(t_sincos, timestep * 1000.0, time_embed.sincos_dim, 10000.0);
    time_embed.forward(t_emb, t_sincos,  hidden_size, t_emb_silu);
    combined_rope_cos := TMemoryblock.Create(combined_img_seq * axis_dim * 4);
    combined_rope_sin := TMemoryblock.Create(combined_img_seq * axis_dim * 4);
    QNNComputeRoPE2D(combined_rope_cos, combined_rope_sin, img_h, img_w, axis_dim, rope_theta);
    rope_offset := img_seq * axis_dim * 4;
    for r := 0 to high(refs) do
        begin
            ref_seq := refs[r].h*refs[r].w;
            QNNComputeRoPE2DOffset(PQNNFloat(combined_rope_cos) + rope_offset, PQNNFloat(combined_rope_sin) + rope_offset, refs[r].h, refs[r].w, axis_dim, rope_theta, refs[r].offset);
            inc(rope_offset, ref_seq * axis_dim * 4)
        end;

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);
    combined_transposed := TMemoryBlock.Create(combined_img_seq * latent_channels);
    for pos := 0 to img_seq -1 do
        for c := 0 to latent_channels -1 do
            combined_transposed[pos * latent_channels+c] := img_latent[c * img_seq+pos];
    trans_offset := img_seq;
    for r := 0 to high(refs) do
        begin
            ref_seq := refs[r].h * refs[r].w;
            for pos := 0 to ref_seq -1 do
                for c := 0 to latent_channels -1 do
                    combined_transposed[(trans_offset + pos)*latent_channels + c] := refs[r].latent[c*ref_seq + pos];
            trans_offset := trans_offset + ref_seq
        end;
    combined_hidden := TMemoryBlock.Create(combined_img_seq * hidden_size);
    QNNLINEAR_BF16_OR_F32(combined_hidden, combined_transposed, img_in_weight, img_in_weight_bf16, combined_img_seq, latent_channels, hidden_size);
    QNNLINEAR_BF16_OR_F32(txt_hidden, txt_emb, txt_in_weight, txt_in_weight_bf16, txt_seq, text_dim, hidden_size);
    double_mod_size := hidden_size * 6;
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for j := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[j];
    //        tf.t_emb_silu[j] := x / (1.0+expf(-x))
    //    end;
    QNNLinearNoBias(double_mod_img, t_emb_silu, adaln_double_img_weight, 1, hidden_size, double_mod_size);
    QNNLinearNoBias(double_mod_txt, t_emb_silu, adaln_double_txt_weight, 1, hidden_size, double_mod_size);
    for i := 0 to num_double_layers -1 do
        begin
            if use_mmap then
                double_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
            doubleBlockForward(combined_hidden, txt_hidden, i, double_mod_img, double_mod_txt, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, combined_img_seq, txt_seq);
            if assigned(substep_callback) then
                substep_callback(SUBSTEP_DOUBLE_BLOCK, i, num_double_layers)
        end;
    concat_hidden := TMemoryBlock.Create(total_seq * hidden_size);
    QNNCopy(concat_hidden, txt_hidden, txt_seq*hidden_size);
    QNNCopy(concat_hidden + txt_seq*hidden_size, combined_hidden, combined_img_seq*hidden_size);
    for i := 0 to num_single_layers -1 do
        begin
            if use_mmap then
                single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
            singleBlockForward(concat_hidden, i, t_emb, adaln_single_weight, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);

            if assigned(substep_callback) then
              substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers)
        end;
    img_hidden := TMemoryBlock.Create(img_seq * hidden_size);
    QNNCopy(img_hidden, concat_hidden + txt_seq*hidden_size, img_seq*hidden_size);
    QNNSilu(t_emb_silu, t_emb, hidden_size);
    //for i := 0 to hidden_size -1 do
    //    begin
    //        x := t_emb[i];
    //        tf.t_emb_silu[i] := x / (1.0+expf(-x))
    //    end;
    final_mod := double_mod_img;
    QNNLinearNoBias(final_mod, t_emb_silu, final_norm_weight, 1, hidden_size, hidden_size * 2);
    final_scale := final_mod;
    final_shift := final_mod+hidden_size;
    final_norm := work1;
    QNNAdaLN(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden_size);

    output_nlc := TMemoryBlock.Create(img_seq*latent_channels);
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.Create(img_seq * latent_channels);
    result_ptr := result;
    output_ptr := output_nlc;
    for pos := 0 to img_seq -1 do
        for c := 0 to latent_channels -1 do
            result_ptr[c * img_seq+pos] := output_ptr[pos * latent_channels+c];
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const text_emb: PQNNFloat; const text_seq: longint;
  const schedule: PQNNFloat; const num_steps: longint;
  const progress_callback: TStepCallback): TMemoryBlock;
var
    latent_size, step: longint;
    v_cond:TMemoryBlock;
    t_curr, t_next, dt: QNNFloat;
    img : TQNNImage;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryBlock.Create(latent_size);

    QNNcopy(result, z, latent_size);
    for step := 0 to num_steps -1 do
        begin
            t_curr := schedule[step];
            t_next := schedule[step+1];
            dt := t_next - t_curr;
            if assigned(step_callback) then
                step_callback(step+1, num_steps);
            v_cond :=forward(result, h, w, text_emb, text_seq, t_curr);
            QNNFusedScaleAdd(result, v_cond, result, dt, latent_size);
            v_cond.free();
            if assigned(progress_callback) then
                progress_callback(step+1, num_steps);
            if assigned(step_image_callback) and assigned(step_image_vae) and (step+1 < num_steps) then
                begin
                    img := PVAE(step_image_vae).decode(result, 1, h, w);
                    if assigned(img.data) then
                        begin
                            step_image_callback(step+1, num_steps, img);
                            img.free
                        end
                end
        end;
    freeMMapCache();
end;

function TTransformerFlux.sampleEuler(const z: PQNNFloat; const batch,
  channels, h, w: longint; const text_emb_cond: PQNNFloat;
  const text_seq_cond: longint; const text_emb_uncond: PQNNFloat;
  const text_seq_uncond: longint; const guidance_scale: QNNFloat;
  const schedule: PQNNFloat; const num_steps: longint;
  const progress_callback: TStepCallback): TMemoryBlock;
var
    latent_size, step, i: longint;
    t_curr, t_next, dt, v: QNNFloat;
    v_uncond, v_cond: TMemoryBlock;
    v_uncondPtr, v_condPtr, resultPtr: PQNNFloat;
    img: TQNNImage;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryblock.create(latent_size);
    resultPtr := result;
    QNNCopy(result, z, latent_size);
    for step := 0 to num_steps -1 do
        begin
            t_curr := schedule[step];
            t_next := schedule[step+1];
            dt := t_next-t_curr;
            if assigned(step_callback) then
                step_callback(step+1, num_steps);
            v_uncond := forward(result, h, w, text_emb_uncond, text_seq_uncond, t_curr);
            v_uncondPtr := v_uncond;
            v_cond := forward(result, h, w, text_emb_cond, text_seq_cond, t_curr);
            v_condPtr := v_cond;
            for i := 0 to latent_size -1 do
                begin
                    v := v_uncondPtr[i]+guidance_scale * (v_condPtr[i]-v_uncondPtr[i]);
                    resultPtr[i] := resultPtr[i] + (dt * v)
                end;
            v_uncond.free;
            v_cond.free;
            if assigned(progress_callback) then
                progress_callback(step+1, num_steps);
            if assigned(step_image_callback) and assigned(step_image_vae) and (step+1 < num_steps) then
                begin
                    img := PVAE(step_image_vae).decode(result, 1, h, w);
                    if assigned(img.data) then
                        begin
                            step_image_callback(step+1, num_steps, img);
                            img.free
                        end
                end
        end;
    freeMMapCache();

end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const ref_latent: PQNNFloat; const ref_h, ref_w,
  t_offset: longint; const text_emb_cond: PQNNFloat;
  const text_seq_cond: longint; const text_emb_uncond: PQNNFloat;
  const text_seq_uncond: longint; const guidance_scale: QNNFloat;
  const schedule: PQNNFloat; const num_steps: longint;
  const progress_callback: TStepCallback):TMemoryBlock;
var
    latent_size, step, i: longint;
    t_curr, t_next, dt, v: QNNFloat;
    v_uncond,  v_cond: TMemoryBlock;
    v_condPtr, v_uncondPtr, resultPtr:PQNNFloat;
    img:TQNNImage;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryBlock.Create(latent_size);
    resultPtr := result;
    QNNcopy(result, z, latent_size);

    for step := 0 to num_steps -1 do
        begin
            t_curr := schedule[step];
            t_next := schedule[step+1];
            dt := t_next-t_curr;
            if assigned(step_callback) then
                step_callback(step+1, num_steps);
            v_uncond := forward(result, h, w, ref_latent, ref_h, ref_w, t_offset, text_emb_uncond, text_seq_uncond, t_curr);
            v_cond := forward(result, h, w, ref_latent, ref_h, ref_w, t_offset, text_emb_cond, text_seq_cond, t_curr);
            v_condPtr := v_cond;
            v_uncondPtr := v_uncond;

            for i := 0 to latent_size -1 do
                begin
                    v := v_uncondPtr[i] + guidance_scale * (v_condPtr[i] - v_uncondPtr[i]);
                    resultPtr[i] := resultPtr[i] + (dt * v)
                end;
            v_uncond.free;
            v_cond.free;
            if assigned(progress_callback) then
                progress_callback(step+1, num_steps);
            if assigned(step_image_callback) and assigned(step_image_vae) and (step+1 < num_steps) then
                begin
                    img := PVAE(step_image_vae).decode(result, 1, h, w);
                    if assigned(img.data) then
                        begin
                            step_image_callback(step+1, num_steps, img);
                            img.free
                        end
                end
        end;
    freeMMapCache()
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

