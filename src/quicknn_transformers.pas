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

{.$define USE_MULTITHREADING}

interface
uses SysUtils, math, safetensor, quicknn_common, quicknn_vae, quicknn_qwen3, quickjson
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  ;

const
  flux2_latent_rgb_proj : TArray<TArray<single>> = [
    [ 0.000736   , -0.008385 , -0.019710 ],
    [ -0.001352  , -0.016392 , 0.020693  ],
    [ -0.006376  , 0.002428  , 0.036736  ],
    [ 0.039384   , 0.074167  , 0.119789  ],
    [ 0.007464   , -0.005705 , -0.004734 ],
    [ -0.004086  , 0.005287  , -0.000409 ],
    [ -0.032835  , 0.050802  , -0.028120 ],
    [ -0.003158  , -0.000835 , 0.000406  ],
    [ -0.112840  , -0.084337 , -0.023083 ],
    [ 0.001462   , -0.006656 , 0.000549  ],
    [ -0.009980  , -0.007480 , 0.009702  ],
    [ 0.032540   , 0.000214  , -0.061388 ],
    [ 0.011023   , 0.000694  , 0.007143  ],
    [ -0.001468  , -0.006723 , -0.001678 ],
    [ -0.005921  , -0.010320 , -0.003907 ],
    [ -0.028434  , 0.027584  , 0.018457  ],
    [ 0.014349   , 0.011523  , 0.000441  ],
    [ 0.009874   , 0.003081  , 0.001507  ],
    [ 0.002218   , 0.005712  , 0.001563  ],
    [ 0.053010   , -0.019844 , 0.008683  ],
    [ -0.002507  , 0.005384  , 0.000938  ],
    [ -0.002177  , -0.011366 , 0.003559  ],
    [ -0.000261  , 0.015121  , -0.003240 ],
    [ -0.003944  , -0.002083 , 0.005043  ],
    [ -0.009138  , 0.011336  , 0.003781  ],
    [ 0.011429   , 0.003985  , -0.003855 ],
    [ 0.010518   , -0.005586 , 0.010131  ],
    [ 0.007883   , 0.002912 , -0.001473  ],
    [ -0.003318  , -0.003160 , 0.003684  ],
    [ -0.034560  , -0.008740 , 0.012996  ],
    [ 0.000166   , 0.001079  , -0.012153 ],
    [ 0.017772   , 0.000937  , -0.011953 ]
  ];

  flux2_latent_rgb_bias : TArray<single> = [-0.028738, -0.098463, -0.107619];

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
    procedure forward(const dst, sinCos:TMemoryBlock; const hidden: longint; const outSilu:TMemoryBlock);
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
    attn_scores_alloc: Int64;            //  Currently allocated size
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
    procedure getCachedRefRoPE(const patch_h, patch_w, t_offset:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedImgRoPE(const patch_h, patch_w:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedTxtRoPE(const txt_seq:longint; var cosOut : TSingles; var sinOut:TSingles);
    procedure getCachedCombinedRoPE(const img_h, img_w, ref_h, ref_w, t_offset:longint; var cosOut:TSingles; var sinOut:TSingles);
    procedure multiHeadForward(const dst, Q, K, V:TMemoryBlock; const seq{, heads, head_dim}:longint);
    procedure jointAttentionForward(
              const img_out, txt_out,
                    img_Q  , img_K, img_V,
                    txt_Q  , txt_K, txt_V : TMemoryBlock;
              const img_seq, txt_seq{, heads, head_dim}: longint);
    procedure ffnSwigluForward(const dst, x:TMemoryBlock;
              const gate_weight, up_weight, down_weight:TSingles;
              const gate_weight_bf16, up_weight_bf16, down_weight_bf16:TBF16s;
              const seq, hidden, mlpHidden: longint);

    procedure doubleBlockForward(
              const img_hidden, txt_hidden :TMemoryBlock; const blockIdx:longint; const img_mod, txt_mod,
                    img_rope_cos, img_rope_sin,
                    txt_rope_cos, txt_rope_sin: TMemoryBlock;
              const img_seq, txt_seq : longint);

    procedure singleBlockForward(const hidden:TMemoryBlock;
                                 const blockIdx:longint;
                                 const t_emb, adaln_weight,
                                 img_rope_cos, img_rope_sin,
                                 txt_rope_cos, txt_rope_sin:TMemoryBlock;
                                 const seq, img_offset:longint);

    // for text to image
    function forward(const img_latent: TMemoryBlock; const img_h, img_w: longint; const txt_emb: TMemoryBlock; const txt_seq: longint; timestep: QNNFloat):TMemoryBlock; overload;
    // for reference image with text to image?
    function forward(const img_latent: TMemoryBlock; const img_h, img_w: longint; const ref_latent: TMemoryBlock; const ref_h, ref_w, t_offset: longint; const txt_emb: TMemoryBlock; const txt_seq: longint; timestep: QNNFloat):TMemoryBlock; overload;
    // for multi-reference images with text to image
    function forward(const img_latent: TMemoryBlock; const img_h, img_w: longint; const refs: TArray<TImageRef>; const txt_emb: TMemoryBlock; const txt_seq: longint; const timestep: single):TMemoryBlock; overload;

    // simplest for distilled
    function sampleEuler(const z: TMemoryBlock; const batch, channels, h, w: longint;
      const text_emb: TMemoryBlock; const text_seq: longint;
      const schedule: TMemoryBlock; const num_steps: longint;
      const progress_callback:TProgressCallback): TMemoryBlock; overload;

    // simplist distilled with one latent image
    function sampleEuler(const z: TMemoryBlock;
      const batch, channels, h, w: longint;
      const ref_latent: TMemoryBlock; const ref_h, ref_w, t_offset: longint;
      const text_emb: TMemoryBlock; const text_seq: longint;
      const schedule: TMemoryBlock; const num_steps: longint;
      const progress_callback:TProgressCallback): TMemoryBlock; overload;

    // non distilled with uncoditioning and guidance
    function sampleEuler(const z: TMemoryBlock;
      const batch, channels, h, w: longint;
      const text_emb_cond: TMemoryBlock ; const text_seq_cond  : longint;
      const text_emb_uncond: TMemoryBlock; const text_seq_uncond: longint;
      const guidance_scale: QNNFloat;   const schedule: TMemoryBlock;
      const num_steps: longint;
      const progress_callback: TProgressCallback):TMemoryBlock; overload;

    // non distiled with unconditioning, guidance and one latent image
    function sampleEuler(const z: TMemoryBlock;
      const batch, channels, h, w: longint;
      const ref_latent: TMemoryBlock; const ref_h, ref_w, t_offset: longint;
      const text_emb_cond: TMemoryBlock; const text_seq_cond: longint;
      const text_emb_uncond: TMemoryBlock; const text_seq_uncond: longint;
      const guidance_scale: QNNFloat; const schedule: TMemoryBlock;
      const num_steps: longint;
      const progress_callback: TProgressCallback): TMemoryBlock; overload;

    // simplest disilled with multi images
    function sampleEuler(const z:TMemoryBlock; const batch, channels, h, w : longint;
      const refs : TArray<TImageRef>;
      const text_emb : TMemoryBlock; const text_seq:longint;
      const schedule:TMemoryBlock; const num_steps:longint;
      const progress_callback:TProgressCallback):TMemoryBlock;    overload;

    // non disilled with unconditioning, gudance and multi images
    function sampleEuler(const z:TMemoryBlock; const batch, channels, h, w : longint;
      const refs : TArray<TImageRef>;
      const text_emb_cond : TMemoryBlock; const text_seq_cond:longint;
      const text_emb_uncond : TMemoryBlock; const text_seq_uncond:longint;
      const gudance_scale:QNNFloat; const schedule:PQNNFloat; num_steps:longint;
      const progress_callback:TProgressCallback):TMemoryBlock;  overload;
  end;

  const
      ZI_SEQ_MULTI_OF  = 32   ;  (* Pad sequences to multiples of 32 *)
      ZI_NORM_EPS      = 1e-5;  (* RMSNorm epsilon *)
      ZI_BF16_SDPA_SEQ = 1024 ;  (* Prefer bf16 SDPA at large sequence lengths *)
      ZI_MAX_SHARDS    = 32   ;

type
  PBlockZI = ^TBlockZI;

  { TBlockZi }

  TBlockZI =  record
      (* Attention *)
      attn_q_weight : TSingles;       (* [dim, dim] *)
      attn_k_weight : TSingles;       (* [dim, dim] *)
      attn_v_weight : TSingles;       (* [dim, dim] *)
      attn_out_weight : TSingles;     (* [dim, dim] *)
      attn_norm_q : TSingles;         (* [n_heads, head_dim] for QK norm *)
      attn_norm_k : TSingles;         (* [n_heads, head_dim] *)
      attn_norm1 : TSingles;          (* [dim] RMSNorm before attention *)
      attn_norm2 : TSingles;          (* [dim] RMSNorm after attention *)

      (* FFN (SwiGLU) *)
      ffn_w1 : TSingles;              (* [ffn_dim, dim] gate projection *)
      ffn_w2 : TSingles;              (* [dim, ffn_dim] down projection *)
      ffn_w3 : TSingles;              (* [ffn_dim, dim] up projection *)
      ffn_norm1 : TSingles;           (* [dim] RMSNorm before FFN *)
      ffn_norm2 : TSingles;           (* [dim] RMSNorm after FFN *)

      (* AdaLN modulation (NULL for context_refiner blocks) *)
      adaln_weight : TSingles;        (* [4*dim, adaln_dim] *)
      adaln_bias : TSingles;          (* [4*dim] *)
      procedure load(const files:TSafeTensorFiles; const prefix:string; const has_modulation:boolean; const useMMap:boolean = true);
      procedure free();
  end;

  (* Final layer weights *)
  PFinalZI = ^TFinalZI;
  TFinalZI = record
      adaln_weight : TSingles;        (* [dim, adaln_dim] *)
      adaln_bias   : TSingles;        (* [dim] *)
      norm_weight  : TSingles;        (* NULL (no affine) or [dim] *)
      linear_weight : TSingles;       (* [out_ch, dim] *)
      linear_bias   : TSingles;       (* [out_ch] *)
  end;

  (* Z-Image transformer context *)

  { TTransformerZi }

  TTransformerZI = record
      (* Architecture config *)
      dim : longint;                    (* 3840 *)
      n_heads : longint;                (* 30 *)
      head_dim : longint;               (* 128 *)
      n_layers : longint;               (* 30 *)
      n_refiner : longint;              (* 2 *)
      ffn_dim : longint;                (* 8*dim/3 = 10240 *)
      in_channels : longint;            (* 16 *)
      patch_size : longint;             (* 2 *)
      adaln_dim : longint;              (* min(dim, 256) = 256 *)
      rope_theta : single;           (* 256.0 *)
      axes_dims : array[0..2] of longint;           (* [32, 48, 48] *)
      axes_lens : array[0..2] of longint;           (* [1024, 512, 512] *)

      (* Embedders *)
      t_emb_mlp0_weight    : TSingles;   (* [mid_size, 256] *)
      t_emb_mlp0_bias      : TSingles;   (* [mid_size] *)
      t_emb_mlp2_weight    : TSingles;   (* [adaln_dim, mid_size] *)
      t_emb_mlp2_bias      : TSingles;   (* [adaln_dim] *)
      t_emb_mid_size       : longint;         (* intermediate timestep MLP size *)

      cap_emb_norm         : TSingles;    (* [cap_feat_dim] RMSNorm weight *)
      cap_emb_linear_w     : TSingles;    (* [dim, cap_feat_dim] *)
      cap_emb_linear_b     : TSingles;    (* [dim] *)
      cap_feat_dim         : longint;     (* 2560 *)

       x_emb_weight        : TSingles;    (* [dim, patch_feat] where patch_feat = ps*ps*in_ch *)
       x_emb_bias          : TSingles;    (* [dim] *)

      x_pad_token          : TSingles;    (* [dim] *)
      cap_pad_token        : TSingles;    (* [dim] *)

      (* Transformer blocks *)
      noise_refiner   : TArray<TBlockZi>;  (* [n_refiner] *)
      context_refiner : TArray<TBlockZi>;(* [n_refiner] *)
      layers : TArray<TBlockZi>;         (* [n_layers] *)

      (* Final layer *)
      final_layer : TFinalZi;

      (* CPU mmap mode: keep shard files open and use direct f32 pointers. *)
      mmap_f32_weights : boolean;
      sf_files : TSafeTensorFiles;
      num_sf_files : longint;

      (* Precomputed RoPE frequencies (complex pairs) *)
      rope_cos  : array[0..2] of TSingles; (* [axes_lens[i], axes_dims[i]/2] *)
      rope_sin  : array[0..2] of TSingles; (* [axes_lens[i], axes_dims[i]/2] *)

      (* Working memory *)
      work_x      : TSingles;   (* Main token buffer *)
      work_tmp    : TSingles;   (* Temporary buffer *)
      work_qkv    : TSingles;   (* Q, K, V buffers *)
      work_attn   : TSingles;   (* Attention scores *)
      work_ffn    : TSingles;   (* FFN intermediate *)
      work_alloc  : NativeInt; (* Total allocated *)
      max_seq : longint;                (* Max sequence length allocated for *)
      procedure attentionForward(const dst, src : TMemoryBlock;
                          const block : TBlockZi; const pos_ids : Plongint;
                          const mask : PLongint; const seq : longint);
      procedure ffnForward(const dst, src:TMemoryBlock; const block:TBlockZi; const seq:longint);
      procedure blockForward(const dst : TMemoryBlock; const block : TBlockZi;
                              const pos_ids:PLongint; const mask:PLongint;
                              const t_emb:TMemoryBlock; const seq:longint);
      procedure preComputeRope();
      procedure finalComputeScale(const scale, t_emb:TMemoryBlock);
      procedure finalForward(const dst, src, t_emb:TMemoryBlock; const seq: longint);
      function forward(const latent : TMemoryBlock; const latent_h, latent_w:longint; const timestep : QNNFloat; const cap_feats: TMemoryBlock; const cap_seq_len:longint):TMemoryBlock;
      procedure applyRope(const dst:PQNNFloat; const pos_ids:PLongint; const seq, n_heads:longint);
      procedure timeStepEmbed(const dst:TMemoryBlock; const t:QNNFloat);
      function sampleEuler(const z:TMemoryBlock; const batch{, channels}, h, w, patch_size:longint; const cap_feats:TMemoryBlock;const cap_seq:longint; const schedule:PQNNFloat; const num_steps:longint; const progress_callback:TStepCallback = nil):TMemoryBlock;
      procedure load(const modelDir: string; const adim, an_heads, an_layers, an_refiner, acap_feat_dim, ain_channels, apatch_size: longint; const arope_theta: QNNFloat; const aAxes_dims: Plongint; const useMMAP:boolean = true);
      function isLoaded():boolean;
      procedure free();
  end;

procedure patchifyZi(const dst : PQNNFloat; const src:PQNNFloat; const in_ch, Height, Width, ps:longint);
procedure UnPatchifyZi(const dst:PQNNFloat; const src: PQNNFloat; const in_ch, Height, Width, ps:longint);
function loadShards(const modelDir: string):TSafeTensorFiles;

implementation
uses quicknn_kernels;

function loadShards(const modelDir: string):TSafeTensorFiles;
var json, wm:TJSON; i:integer;
  fn : string; files:TArray<string>;
begin
  files := nil;
  result := nil;
  fn := modelDir+'/transformer/diffusion_pytorch_model.safetensors.index.json';
  if FileExists(fn) then begin
    json := TJSON.LoadFromFile(fn);
    if json.keyExist('weight_map') then begin
      wm:= json['weight_map'];
      for i:=0 to wm.count-1 do
        if indexOf(wm.childObjs[i].Value, files)<0 then begin
          insert(string(wm.childObjs[i].Value), files, length(files));
          setLength(result, length(result)+1);
          result[high(result)] := TSafeTensorFile.open(modelDir+'/transformer/'+wm.childObjs[i].Value);
        end;
    end;
  end else
    result := [TSafeTensorFile.open(modelDir+'/transformer/diffusion_pytorch_model.safetensors')];
end;


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
    img_mlp_up_weight_bf16 := img_mlp_gate_weight_bf16 + mlp*h;
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
    txt_mlp_up_weight_bf16 := txt_mlp_gate_weight_bf16 + mlp * h;
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
    img_mlp_up_weight := img_mlp_gate_weight + mlp*h;
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
    txt_mlp_up_weight := txt_mlp_gate_weight + mlp * h;
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
  const hidden: longint; const outSilu: TMemoryBlock);
begin
  //if length(workspace)< 1*hidden then
  //  setLength(workspace, hidden);

  QNNLinearNoBias(outSilu, sinCos, fc1_weight, 1, sincos_dim, hidden);
  QNNSiluInplace(outSilu, hidden);
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


  result.num_double_layers := quicknn_common.ifthen(layerCount > 0, layerCount, 5);
  result.num_single_layers := quicknn_common.ifthen(singleCount > 0, singleCount, 20);
  result.text_dim          := quicknn_common.ifthen(jointAttDim > 0, jointAttDim, 7680);
  result.latent_channels   := quicknn_common.ifthen(inChannels > 0, inChannels, 128);

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
  if use_bf16 then begin
      img_in_weight_bf16 := sf_files.getTensorDataMemBlockBF16('x_embedder.weight', use_mmap);
      txt_in_weight_bf16 := sf_files.getTensorDataMemBlockBF16('context_embedder.weight', use_mmap)
  end;
  time_embed.sincos_dim := 256;
  time_embed.fc1_weight := sf_files.getTensorDataMemBlock('time_guidance_embed.timestep_embedder.linear_1.weight', use_mmap);
  time_embed.fc2_weight := sf_files.getTensorDataMemBlock('time_guidance_embed.timestep_embedder.linear_2.weight', use_mmap);
  adaln_double_img_weight := sf_files.getTensorDataMemBlock('double_stream_modulation_img.linear.weight', use_mmap);
  adaln_double_txt_weight := sf_files.getTensorDataMemBlock('double_stream_modulation_txt.linear.weight', use_mmap);
  adaln_single_weight := sf_files.getTensorDataMemBlock('single_stream_modulation.linear.weight', use_mmap);
  if use_bf16 then begin
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
  rope_freqs := TSingles.Create([max_seq_len , head_dim], 'FLUX_LOAD_ROPE_FREQS');
  if rope_freqs.isAssigned() then
      QNNComputeRoPE(rope_freqs, max_seq_len, head_dim, rope_theta);

  (* Small constant-size buffers - always allocate *)
  t_emb_silu     := TSingles.Create([hidden_size    ], 'FLUX_LOAD_t_emb_silu');
  double_mod_img := TSingles.Create([hidden_size , 6], 'FLUX_LOAD_double_mod_img');
  double_mod_txt := TSingles.Create([hidden_size , 6], 'FLUX_LOAD_double_mod_txt');
end;

function TTransformerFlux.isLoaded(): boolean;
begin
  result := assigned(sf_files)
end;



procedure TTransformerFlux.reInitialize(const totalSeq: longint);
var fused_dim, hidden, mlp:longint; attn_scores_need:TArray<int64>;
begin

  if isUsingBlas then begin
    attn_scores_need := [num_heads, totalSeq, totalSeq];

    if product(attn_scores_need) > attn_scores_alloc then begin
      if attn_scores.isAssigned() then
        attn_scores.reSize(attn_scores_need)
      else
        attn_scores := TMemoryBlock.Create(attn_scores_need, 'FLUX_INIT_ATTN_SCORES');
      attn_scores_alloc := product(attn_scores_need);
    end;
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




  img_hidden := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_IMG_HIDDEN');
  txt_hidden := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_TXT_HIDDEN');
  (* work2 needs to hold fused QKV+MLP output: seq * (hidden*3 + mlp*2) + mod_params (hidden*3)
  * fused_dim = hidden*3 + mlp*2 = 3072*3 + 9216*2 = 27648 *)
  fused_dim := hidden * 3 + mlp * 2;

  work_size           := totalSeq*fused_dim + hidden*3 ;

  work1               := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_work1          ');
  work2               := TMemoryBlock.Create([work_size       ], 'FLUX_INIT_work2          ');
  attn_q_t            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_q_t       ');
  attn_k_t            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_k_t       ');
  attn_v_t            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_v_t       ');
  attn_out_t          := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_out_t     ');
  attn_cat_k          := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_cat_k     ');
  attn_cat_v          := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_attn_cat_v     ');
  single_q            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_single_q       ');
  single_k            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_single_k       ');
  single_v            := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_single_v       ');
  single_mlp_gate     := TMemoryBlock.Create([totalSeq, mlp]   , 'FLUX_INIT_single_mlp_gate');
  single_mlp_up       := TMemoryBlock.Create([totalSeq, mlp]   , 'FLUX_INIT_single_mlp_up  ');
  single_attn_out     := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_single_attn_out');
  single_concat       := TMemoryBlock.Create([totalSeq, (hidden + mlp)], 'FLUX_INIT_single_concat');
  ffn_gate            := TMemoryBlock.Create([totalSeq, mlp   ], 'FLUX_INIT_ffn_gate');
  ffn_up              := TMemoryBlock.Create([totalSeq, mlp   ], 'FLUX_INIT_ffn_up');
  double_img_attn_out := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_double_img_attn_out');
  double_txt_attn_out := TMemoryBlock.Create([totalSeq, hidden], 'FLUX_INIT_double_txt_attn_out');
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
    sf_files := nil;
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

procedure TTransformerFlux.getCachedRefRoPE(const patch_h, patch_w,
  t_offset: longint; var cosOut: TSingles; var sinOut: TSingles);
var
    seq, axisDim: longint;
    size: NativeInt;
begin
    seq := patch_h * patch_w;
    axisDim := axis_dim;
    if (cached_ref_h = patch_h) and
       (cached_ref_w = patch_w) and
       (cached_ref_t_offset = t_offset) and
       cached_ref_rope_cos.isAssigned() and cached_ref_rope_sin.isAssigned() then
      begin
        cosOut := cached_ref_rope_cos;
        sinOut := cached_ref_rope_sin;
        exit()
      end;
    if cached_ref_rope_cos.isAssigned() then cached_ref_rope_cos.free();
    if cached_ref_rope_sin.isAssigned() then cached_ref_rope_sin.free();
    size := seq * axisDim * 4 ;
    cached_ref_rope_cos := TMemoryBlock.Create([seq, axisDim * 4], 'FLUX_GET_CACHED_ROPE_COS');
    cached_ref_rope_sin := TMemoryBlock.Create([seq, axisDim * 4], 'FLUX_GET_CACHED_ROPE_SIN');
    cached_ref_h := patch_h;
    cached_ref_w := patch_w;
    cached_ref_t_offset := t_offset;
    QNNComputeRoPE2DOffset(cached_ref_rope_cos, cached_ref_rope_sin, patch_h, patch_w, axisDim, rope_theta, t_offset);
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

    cached_img_rope_cos := TMemoryBlock.Create([seq, axisDim, 4], 'FLUX_GET_CACHED_IMG_ROPE_COS');
    cached_img_rope_sin := TMemoryBlock.Create([seq, axisDim, 4], 'FLUX_GET_CACHED_IMG_ROPE_SIN');
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
    cached_txt_rope_cos := TMemoryBlock.Create([txt_seq, headDim], 'FLUX_GET_CACHED_TXT_ROPE_COS');
    cached_txt_rope_sin := TMemoryBlock.Create([txt_seq, headDim], 'FLUX_GET_CACHED_TXT_ROPE_SIN');
    cached_txt_seq := txt_seq;
    QNNComputeRoPEText(cached_txt_rope_cos, cached_txt_rope_sin, txt_seq, axis_dim, rope_theta);
    cosOut := cached_txt_rope_cos;
    sinOut := cached_txt_rope_sin
end;

procedure TTransformerFlux.getCachedCombinedRoPE(const img_h, img_w, ref_h,
  ref_w, t_offset: longint; var cosOut: TSingles; var sinOut: TSingles);
var
    img_seq, ref_seq, combined_seq, axisDim: longint;
    combined_size, img_size, ref_size: NativeInt;
    img_rope_cos, img_rope_sin, ref_rope_cos, ref_rope_sin : TSingles;
begin
    img_seq := img_h * img_w;
    ref_seq := ref_h * ref_w;
    combined_seq := img_seq+ref_seq;
    axisDim := axis_dim;
    combined_size := combined_seq * axisDim * 4;
    img_size := img_seq * axisDim * 4;
    ref_size := ref_seq * axisDim * 4;
    if (cached_combined_img_h = img_h) and
       (cached_combined_img_w = img_w) and
       (cached_combined_ref_h = ref_h) and
       (cached_combined_ref_w = ref_w) and
       (cached_combined_t_offset = t_offset) and
       cached_combined_rope_cos.isAssigned() and cached_combined_rope_sin.isAssigned() then
        begin
            cosOut := cached_combined_rope_cos;
            sinOut := cached_combined_rope_sin;
            exit()
        end;
    if cached_combined_rope_cos.isAssigned() then cached_combined_rope_cos.free();
    if cached_combined_rope_sin.isAssigned() then cached_combined_rope_sin.free();
    cached_combined_rope_cos := TMemoryBlock.Create([combined_seq, axisDim, 4], 'FLUX_GET_CACHED_COMBINED_ROPE_COS');
    cached_combined_rope_sin := TMemoryBlock.Create([combined_seq, axisDim, 4], 'FLUX_GET_CACHED_COMBINED_ROPE_SIN');
    cached_combined_img_h := img_h;
    cached_combined_img_w := img_w;
    cached_combined_ref_h := ref_h;
    cached_combined_ref_w := ref_w;
    cached_combined_t_offset := t_offset;

    getCachedImgRoPE(img_h, img_w, img_rope_cos,  img_rope_sin);

    getCachedRefRoPE(ref_h, ref_w, t_offset,  ref_rope_cos,  ref_rope_sin);


    QNNCopy(cached_combined_rope_cos             ,img_rope_cos, img_size);
    QNNCopy(cached_combined_rope_cos + img_size  ,ref_rope_cos, ref_size);
    QNNCopy(cached_combined_rope_sin             ,img_rope_sin, img_size);
    QNNCopy(cached_combined_rope_sin + img_size  ,ref_rope_sin, ref_size);

    cosOut := cached_combined_rope_cos;
    sinOut := cached_combined_rope_sin
end;




procedure TTransformerFlux.multiHeadForward(const dst, Q, K, V: TMemoryBlock; const seq: longint);
var
  hidden, headDim : longint;
  scale : QNNFloat;
  scores : TMemoryBlock;

{$ifdef FPC}
procedure worker(const start, finish:IntPtr; const data:pointer);
{$else}
   worker : TGroupProcNested;
begin
  worker := procedure (const start, finish:IntPtr; const data:pointer)
{$endif}
var
  i  : longint;
  qh, kh, vh, oh, sh : TMemoryBlock;
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
        QNNSoftmaxRows(sh {scores + i*seq*seq}, seq, seq);

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
  scale := 1/sqrt(head_dim);
  scores := attn_scores; // because unlice FPC delphi does not capture Self in lambda referenced procedures
  if not isUsingBlas then
    QNNFlashAttention(dst, Q, K, V, num_heads, seq, seq, head_dim, scale{1/sqrt(head_dim)})
  else
  {$ifdef _USE_MULTITHREADING}       // CAUTION : do not multithread
    mp.&for(worker, 0, num_heads-1) ;
  {$else}
    worker(0, num_heads-1, nil);
  {$endif}


end;

procedure TTransformerFlux.jointAttentionForward(const img_out, txt_out, img_Q,
  img_K, img_V, txt_Q, txt_K, txt_V: TMemoryBlock; const img_seq,
  txt_seq: longint);
var
  totalSeq, hidden, headDim:longint;
  scale : QNNFloat;
  cat_k, cat_v, scores : TMemoryBlock;
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
    kh, vh : TMemoryBlock;
  begin
    for i:= start to finish do begin
            imgQh := img_Q        + i*headDim;
            txtQh := txt_Q        + i*headDim;
            kh    := cat_k        + i*headDim;
            vh    := cat_v        + i*headDim;
            imgOh := img_out      + i*headDim;
            txtOh := txt_out      + i*headDim;
            imgSh := scores       + i*totalSeq*totalSeq;
            txtSh := imgSh        + img_seq*totalSeq; // txt_sh is an offset from img_sh

            cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        img_seq, totalSeq, headDim,
                        scale,
                        imgQh, hidden,
                        kh, hidden,
                        0.0,
                        imgSh, totalSeq);
            QNNSoftmaxRows(imgSh, img_seq, totalSeq);
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
            QNNSoftmaxRows(txtSh, txt_seq, totalSeq);
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
  scale  := 1/sqrt(head_dim);

  (* Use pre-allocated buffers for K/V concatenation *)
  cat_k := attn_cat_k;
  cat_v := attn_cat_v;
  scores := attn_scores; // because delphi does not capture Self in lambda procedures

  //Concatenate K, V from both streams in [seq, heads, head_dim] format
  //IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
  QNNCopy(attn_cat_k, txt_K, txt_seq*hidden);
  QNNCopy(attn_cat_v, txt_V, txt_seq*hidden);
  QNNCopy(attn_cat_k + txt_seq*hidden, img_K, img_seq*hidden);
  QNNCopy(attn_cat_v + txt_seq*hidden, img_V, img_seq*hidden);

  if not isUsingBlas then begin
    QNNFlashAttention(img_out, img_q, attn_cat_k, attn_cat_v, num_heads, img_seq, totalSeq, head_dim, scale);
    QNNFlashAttention(txt_out, txt_q, attn_cat_k, attn_cat_v, num_heads, txt_seq, totalSeq, head_dim, scale);
  end else begin
  {$ifdef _USE_MULTITHREADING}    // CAUTION : do not multithread
    mp.&for(worker, 0, num_heads-1);
  {$else}
    worker(0, num_heads-1, nil)
  {$endif}
  end

end;

procedure TTransformerFlux.ffnSwigluForward(const dst, x: TMemoryBlock;
  const gate_weight, up_weight, down_weight: TSingles; const gate_weight_bf16,
  up_weight_bf16, down_weight_bf16: TBF16s; const seq, hidden,
  mlpHidden: longint);
begin
  assert(boolean(gate_weight) or boolean(gate_weight_bf16),
    'ERROR ffnSwigluForward no gate_weight or gate_weight_bf16 assigned!');

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
  txt_hidden: TMemoryBlock; const blockIdx: longint; const img_mod, txt_mod,
  img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin: TMemoryBlock;
  const img_seq, txt_seq: longint);

const AXISDIM = 32;

var
  img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2,
  txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2,
  img_norm, img_q, img_k, img_v, txt_norm, txt_q, txt_k, txt_v,
  img_proj, txt_proj :TMemoryBlock;
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

procedure TTransformerFlux.singleBlockForward(const hidden: TMemoryBlock;
  const blockIdx: longint; const t_emb, adaln_weight, img_rope_cos,
  img_rope_sin, txt_rope_cos, txt_rope_sin: TMemoryBlock; const seq,
  img_offset: longint);

const AXIS_DIM = 32;
var
  fused_dim, mod_size, s, img_seq, txt_seq: longint;
  mod_params, shift, scale, gate, norm, fused_out,
    row, img_q, img_k, proj_out: TMemoryBlock;
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
    mod_params := work2 + seq*fused_dim;

    QNNLinearNoBias(mod_params, t_emb_silu, adaln_weight, 1, hidden_size, mod_size);
    shift := mod_params;
    scale := mod_params + hidden_size;
    gate  := mod_params + hidden_size * 2;
    norm  := work1;
    QNNAdaLN(norm, hidden, shift, scale, seq, hidden_size);
    fused_out := work2;
    block := @single_blocks[blockIdx];
    QNNLINEAR_BF16_OR_F32(fused_out, norm, block.qkv_mlp_weight, block.qkv_mlp_weight_bf16, seq, hidden_size, fused_dim);

    for s := 0 to seq -1 do begin
        row := fused_out + s*fused_dim;
        QNNCopy(single_q        + s*hidden_size       , row                             , hidden_size);
        QNNCopy(single_k        + s*hidden_size       , row + hidden_size               , hidden_size);
        QNNCopy(single_v        + s*hidden_size       , row + hidden_size*2             , hidden_size);
        QNNCopy(single_mlp_gate + s*mlp_hidden , row + hidden_size*3             , mlp_hidden );
        QNNCopy(single_mlp_up   + s*mlp_hidden , row + hidden_size*3 + mlp_hidden, mlp_hidden )
    end;
    QNNQKRMSNorm(single_q, single_k, block.norm_q_weight, block.norm_k_weight, seq, num_heads, head_dim);
    txt_seq := img_offset;
    QNNApplyRoPE2D(single_q, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);
    QNNApplyRoPE2D(single_k, txt_rope_cos, txt_rope_sin, txt_seq, num_heads, head_dim, AXIS_DIM);
    img_q := single_q+img_offset * hidden_size;
    img_k := single_k+img_offset * hidden_size;
    QNNApplyRoPE2D(img_q, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);
    QNNApplyRoPE2D(img_k, img_rope_cos, img_rope_sin, img_seq, num_heads, head_dim, AXIS_DIM);
    multiHeadForward(single_attn_out, single_q, single_k, single_v, seq);
    QNNSiluMul(single_mlp_gate, single_mlp_up, seq*mlp_hidden);
    for s := 0 to seq -1 do begin
        QNNCopy(single_concat + s*(hidden_size+mlp_hidden)              , single_attn_out + s*hidden_size, hidden_size);
        QNNCopy(single_concat + s*(hidden_size+mlp_hidden) + hidden_size, single_mlp_gate + s*mlp_hidden , mlp_hidden )
    end;
    proj_out := work1;
    QNNLINEAR_BF16_OR_F32(proj_out, single_concat, block.proj_mlp_weight, block.proj_mlp_weight_bf16, seq, hidden_size+mlp_hidden, hidden_size);
    QNNGatedAdd(hidden, gate, proj_out, seq, hidden_size)
end;

function TTransformerFlux.forward(const img_latent: TMemoryBlock; const img_h,
  img_w: longint; const txt_emb: TMemoryBlock; const txt_seq: longint;
  timestep: QNNFloat): TMemoryBlock;

var
    img_seq, total_seq, double_mod_size, i: longint;
    //x: single;
    t_emb, img_transposed, concat_hidden, img_rope_cos, txt_rope_cos, img_rope_sin, txt_rope_sin, output_nlc : TSingles;
    final_mod, final_scale, final_shift, final_norm : TMemoryBlock;
    t_sincos : TMemoryBlock;//array[0..255] of QNNFloat;
begin
    img_seq := img_h*img_w;
    total_seq := img_seq+txt_seq;
    reInitialize(total_seq);

    t_emb := TMemoryBlock.Create([hidden_size], 'FLUX_FW_TIME_EMB');
    t_sincos := QNNTimestepEmbedding(timestep * 1000.0, time_embed.sincos_dim, 10000.0);

    time_embed.forward(t_emb, t_sincos,  hidden_size, t_emb_silu);
    t_sincos.free;

    getCachedImgRoPE(img_h, img_w, img_rope_cos, img_rope_sin);

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);

    img_transposed := TMemoryBlock.Create([img_seq, latent_channels], 'FLUX_FW_IMG_TRANSPOSED');
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
    concat_hidden := TMemoryBlock.Create([total_seq, hidden_size], 'FLUX_FW_CONCAT_HIDDEN');
    QNNCopy(concat_hidden                                 , txt_hidden, txt_seq*hidden_size);
    QNNCopy(concat_hidden + txt_seq*hidden_size, img_hidden, img_seq*hidden_size);
    for i := 0 to num_single_layers -1 {high(single_blocks)} do begin
        if use_mmap and not(boolean(single_blocks[i].qkv_mlp_weight) or boolean(single_blocks[i].qkv_mlp_weight_bf16)) then
            single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
        singleBlockForward(concat_hidden, i, t_emb, adaln_single_weight, img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);
        if use_mmap then
            single_blocks[i].free;
        //    free_single_block_weights(@single_blocks[i]);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers) ;
    end;


    QNNCopy(img_hidden, concat_hidden + txt_seq*hidden_size, img_seq*hidden_size);
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
    final_shift := final_mod + hidden_size;
    final_norm := work1;
    QNNAdaLN(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden_size, -6);
    output_nlc := TMemoryBlock.create([img_seq, latent_channels], 'FLUX_FW_OUT_NLC');
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.Create([img_seq, latent_channels], 'FLUX_FW_RESULT');
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        result[c * img_seq+pos] := output_nlc[pos * latent_channels+c];
    QNNMatTranspose(result, output_nlc, img_seq, latent_channels);
    output_nlc.free;
    t_emb.free;

    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.forward(const img_latent: TMemoryBlock; const img_h,
  img_w: longint; const ref_latent: TMemoryBlock; const ref_h, ref_w,
  t_offset: longint; const txt_emb: TMemoryBlock; const txt_seq: longint;
  timestep: QNNFloat): TMemoryBlock;
var
    img_seq, ref_seq, combined_img_seq, total_seq, pos, c, i, double_mod_size: longint;
    t_sincos: TMemoryBlock;//array[0..255] of QNNFloat;

    //output_nlc_ptr, result_ptr, combined_transposed_ptr, img_latent_ptr, ref_latent_ptr: PQNNFloat;

    final_mod, final_scale, final_shift, final_norm, t_emb, combined_transposed, combined_hidden, txt_rope_cos, txt_rope_sin
    , combined_rope_cos, combined_rope_sin, concat_hidden, output_nlc : TMemoryBlock;
begin
    if not ref_latent.isAssigned() then
        exit(forward(img_latent, img_h, img_w, txt_emb, txt_seq, timestep));
    img_seq := img_h*img_w;
    ref_seq := ref_h*ref_w;
    combined_img_seq := img_seq + ref_seq;
    total_seq := combined_img_seq + txt_seq;
    reInitialize(total_seq);
    t_emb := TMemoryBlock.Create([hidden_size], 'FLUX_FW_TIME_EMBED');
    t_sincos := QNNTimeStepEmbedding(timestep * 1000.0, time_embed.sincos_dim, 10000.0);
    time_embed.forward(t_emb, t_sincos, hidden_size, t_emb_silu);
    t_sincos.free;
    getCachedCombinedRoPE(img_h, img_w, ref_h, ref_w, t_offset, combined_rope_cos,  combined_rope_sin);

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);
    combined_transposed := TMemoryBlock.Create([combined_img_seq, latent_channels], 'FLUX_FW_COMBINED_TRANSPOSED');
    //combined_transposed_ptr := combined_transposed;
    //img_latent_ptr := img_latent;
    //ref_latent_ptr := ref_latent;
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        combined_transposed_ptr[pos*latent_channels + c] := img_latent_ptr[c*img_seq + pos];

    //for pos := 0 to ref_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        combined_transposed_ptr[(img_seq + pos)*latent_channels + c] := ref_latent_ptr[c*ref_seq + pos];

    QNNMatTranspose(combined_transposed, img_latent, latent_channels, img_seq);
    QNNMatTranspose(combined_transposed + img_seq*latent_channels, ref_latent, latent_channels, ref_seq);


    combined_hidden := TMemoryBlock.Create([combined_img_seq, hidden_size], 'FLUX_FW_COMBINED_HIDDEN');
    QNNLINEAR_BF16_OR_F32(combined_hidden, combined_transposed, img_in_weight, img_in_weight_bf16, combined_img_seq, latent_channels, hidden_size);
    combined_transposed.free;
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
    for i := 0 to num_double_layers -1 do begin
        if use_mmap then
            double_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
        doubleBlockForward(combined_hidden, txt_hidden, i, double_mod_img, double_mod_txt, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, combined_img_seq, txt_seq);
        //if use_mmap then
        //    free_double_block_weights( and double_blocks[i]);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_DOUBLE_BLOCK, i, num_double_layers)
    end;
    concat_hidden := TMemoryBlock.Create([total_seq, hidden_size], 'FLUX_FW_CONACT_HIDDEN');
    QNNCopy(concat_hidden, txt_hidden, txt_seq * hidden_size);
    QNNCopy(concat_hidden + txt_seq*hidden_size, combined_hidden, combined_img_seq*hidden_size);
    for i := 0 to num_single_layers -1 do begin
        if use_mmap then
            single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
        singleblockforward(concat_hidden, i, t_emb, adaln_single_weight, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers)
    end;
    img_hidden :=  TMemoryBlock.Create([img_seq, hidden_size], 'FLUX_FW_IMG_HIDDEN');
    QNNCopy(img_hidden, concat_hidden + txt_seq*hidden_size, img_seq * hidden_size);
    concat_hidden.free;
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
    img_hidden.free;
    output_nlc := TMemoryBlock.create([img_seq, latent_channels], 'FLUX_FW_OUTPUT_NLC');
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.create([img_seq, latent_channels], 'FLUX_FW_RESULT');
    //output_nlc_ptr := output_nlc;
    //result_ptr := result;
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        result_ptr[c * img_seq+pos] := output_nlc_ptr[pos * latent_channels+c];
    QNNMatTranspose(result, output_nlc, img_seq, latent_channels);
    output_nlc.free;
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.forward(const img_latent: TMemoryBlock; const img_h,
  img_w: longint; const refs: TArray<TImageRef>; const txt_emb: TMemoryBlock;
  const txt_seq: longint; const timestep: single): TMemoryBlock;
var
    img_seq, axis_dim, total_ref_seq, r, combined_img_seq,
      total_seq, rope_offset, pos, c,
      {ref_seq, trans_offset,} i, double_mod_size: longint;

    t_emb, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, output_nlc,
      combined_transposed, combined_hidden, concat_hidden, img_hidden, t_sincos,
      final_mod, final_scale, final_shift, final_norm : TMemoryBlock;
    //result_ptr, output_nlc_ptr, combined_transposed_ptr, ref_latent_ptr, img_latent_ptr : PQNNFloat;
    //t_sincos : array[0..255] of QNNFloat;
begin
    if not assigned(refs) then
        exit(forward(img_latent, img_h, img_w, txt_emb, txt_seq, timestep));
    if (length(refs) = 1) then
        exit(forward(img_latent, img_h, img_w, refs[0].latent, refs[0].h, refs[0].w, refs[0].t_offset, txt_emb, txt_seq, timestep));

    img_seq := img_h * img_w;
    axis_dim := 32;
    total_ref_seq := 0;
    for r := 0 to high(refs) do
        inc(total_ref_seq, refs[r].h*refs[r].w);
    combined_img_seq := img_seq+total_ref_seq;
    total_seq := combined_img_seq+txt_seq;
    reInitialize(total_seq);
    t_emb := TMemoryBlock.Create([hidden_size], 'FLUX_FW_TIME_EMB');

    t_sincos := QNNTimestepEmbedding(timestep * 1000.0, time_embed.sincos_dim, 10000.0);
    time_embed.forward(t_emb, t_sincos,  hidden_size, t_emb_silu);
    t_sincos.free;
    combined_rope_cos := TMemoryblock.Create([combined_img_seq, axis_dim, 4], 'FLUX_FW_COMBINED_ROPE_COS');
    combined_rope_sin := TMemoryblock.Create([combined_img_seq, axis_dim, 4], 'FLUX_FW_COMBINED_ROPE_SIN');
    QNNComputeRoPE2D(combined_rope_cos, combined_rope_sin, img_h, img_w, axis_dim, rope_theta);
    rope_offset := img_seq * axis_dim * 4;
    for r := 0 to high(refs) do begin
        QNNComputeRoPE2DOffset(combined_rope_cos + rope_offset, combined_rope_sin + rope_offset, refs[r].h, refs[r].w, axis_dim, rope_theta, refs[r].t_offset);
        inc(rope_offset, refs[r].h*refs[r].w * axis_dim * 4)
    end;

    getCachedTxtRoPE(txt_seq, txt_rope_cos, txt_rope_sin);
    combined_transposed := TMemoryBlock.Create([combined_img_seq, latent_channels], 'FLUX_FW_COMBINED_TRANSPOSED');
    //combined_transposed_ptr := combined_transposed;
    //img_latent_ptr := img_latent;
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        combined_transposed_ptr[pos * latent_channels+c] := img_latent_ptr[c * img_seq+pos];
    QNNMatTranspose(combined_transposed, img_latent, latent_channels, img_seq);
    //trans_offset := img_seq;
    for r := 0 to high(refs) do
        begin
            //ref_latent_ptr := refs[r].latent;
            //ref_seq := refs[r].h * refs[r].w;
            //for pos := 0 to ref_seq -1 do
            //    for c := 0 to latent_channels -1 do
            //        combined_transposed_ptr[(trans_offset + pos)*latent_channels + c] := ref_latent_ptr[c*ref_seq + pos];
            QNNMatTranspose(combined_transposed + r*img_seq*latent_channels, refs[r].latent, latent_channels, refs[r].h*refs[r].w);
            //trans_offset := trans_offset + ref_seq
        end;
    combined_hidden := TMemoryBlock.Create([combined_img_seq, hidden_size], 'FLUX_FW_COMBINED_HIDDEN');
    QNNLINEAR_BF16_OR_F32(combined_hidden, combined_transposed, img_in_weight, img_in_weight_bf16, combined_img_seq, latent_channels, hidden_size);
    combined_transposed.free;
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
    concat_hidden := TMemoryBlock.Create([total_seq, hidden_size], 'FLUX_FW_CONCAT)HIDDEN');
    QNNCopy(concat_hidden, txt_hidden, txt_seq*hidden_size);
    QNNCopy(concat_hidden + txt_seq*hidden_size, combined_hidden, combined_img_seq*hidden_size);
    combined_hidden.free;
    for i := 0 to num_single_layers -1 do begin
        if use_mmap then
            single_blocks[i].load(sf_files, i, hidden_size, mlp_hidden, use_bf16);
        singleBlockForward(concat_hidden, i, t_emb, adaln_single_weight, combined_rope_cos, combined_rope_sin, txt_rope_cos, txt_rope_sin, total_seq, txt_seq);

        if assigned(substep_callback) then
          substep_callback(SUBSTEP_SINGLE_BLOCK, i, num_single_layers)
    end;
    combined_rope_cos.free;
    combined_rope_sin.free;
    img_hidden := TMemoryBlock.Create([img_seq, hidden_size], 'FLUX_FW_IMG_HIDDEN');
    QNNCopy(img_hidden, concat_hidden + txt_seq*hidden_size, img_seq*hidden_size);
    concat_hidden.free;
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
    img_hidden.free;
    output_nlc := TMemoryBlock.Create([img_seq, latent_channels], 'FLUX_FW_OUTPUT_NLC');
    QNNLINEAR_BF16_OR_F32(output_nlc, final_norm, final_proj_weight, final_proj_weight_bf16, img_seq, hidden_size, latent_channels);
    result := TMemoryBlock.Create([img_seq, latent_channels], 'FLUX_FW_RESULT');
    //result_ptr := result;
    //output_nlc_ptr := output_nlc;
    //for pos := 0 to img_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        result_ptr[c * img_seq+pos] := output_nlc_ptr[pos * latent_channels+c];
    QNNMatTranspose(result, output_nlc, img_seq, latent_channels);
    output_nlc.free;
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const text_emb: TMemoryBlock;
  const text_seq: longint; const schedule: TMemoryBlock;
  const num_steps: longint; const progress_callback: TProgressCallback
  ): TMemoryBlock;
var
    latent_size, step: longint;
    v_cond:TMemoryBlock;
    t_curr, t_next, dt: QNNFloat;
    img : TQNNImage;
    schedule_ptr : PQNNFloat;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryBlock.Create([batch, channels, h, w], 'FLUX_SAMPLE_EULER_RESULT');

    QNNcopy(result, z, latent_size);
    schedule_ptr := schedule;
    for step := 0 to num_steps -1 do begin
        t_curr := schedule_ptr[step];
        t_next := schedule_ptr[step+1];
        dt := t_next - t_curr;
        if assigned(step_callback) then
            step_callback(step, num_steps);
        v_cond :=forward(result, h, w, text_emb, text_seq, t_curr);
        QNNFusedScaleAdd(result, v_cond, result, dt, latent_size);
        v_cond.free();
        if assigned(progress_callback) then
            progress_callback(step, num_steps, result);
        if assigned(step_image_callback) and assigned(vae_ptr) and (step+1 < num_steps) then  begin
            img := PVAE(vae_ptr).decode(result, 1, h, w);
            if assigned(img.data) then begin
                step_image_callback(step, num_steps, img);
                img.free
            end
        end
    end;
    freeMMapCache();
end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const ref_latent: TMemoryBlock; const ref_h, ref_w,
  t_offset: longint; const text_emb: TMemoryBlock; const text_seq: longint;
  const schedule: TMemoryBlock; const num_steps: longint;
  const progress_callback: TProgressCallback): TMemoryBlock;
var
  latent_size, step: LongInt;
  t_curr, t_next, dt: QNNFloat;
  v : TMemoryBlock;
  img : TQNNImage;
  schedule_ptr : PQNNFloat;
begin
  //assert(not ref_latent.is_Nan());
  //ref_latent.printStat;
  latent_size := batch * channels * h * w;

  (* Working buffer *)
  result := TMemoryBlock.Create([batch, channels, h, w], 'FLUX_SAMPLE_EULER_RESULT');
  QNNcopy(result, z, latent_size);
  schedule_ptr := schedule;
  for step := 0 to num_steps -1 do begin
      t_curr := schedule_ptr[step];
      t_next := schedule_ptr[step+1];
      dt     := t_next - t_curr;

      (* Notify step start *)
      if assigned(step_callback) then
          step_callback(step, num_steps);
      (* Predict velocity with reference image conditioning *)
      v := forward(result, h, w,
                  ref_latent, ref_h, ref_w,
                  t_offset,
                  text_emb, text_seq,
                  t_curr);
      (* Euler step: z_next = result + dt * v *)
      QNNFusedScaleAdd(result, v, result, dt, latent_size);

      v.free;

      if assigned(progress_callback) then
          progress_callback(step, num_steps, result);

      (* Step image callback - decode and display intermediate result *)
      if assigned(step_image_callback) and assigned(vae_ptr) and (step + 1 < num_steps) then begin
          img := PVAE(vae_ptr).decode(result, 1, h, w);
          step_image_callback(step, num_steps, img);
          img.free;
      end
  end;

  freeMMapCache();
end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const text_emb_cond: TMemoryBlock;
  const text_seq_cond: longint; const text_emb_uncond: TMemoryBlock;
  const text_seq_uncond: longint; const guidance_scale: QNNFloat;
  const schedule: TMemoryBlock; const num_steps: longint;
  const progress_callback: TProgressCallback): TMemoryBlock;
var
    latent_size, step, i: longint;
    t_curr, t_next, dt, v: QNNFloat;
    v_uncond, v_cond: TMemoryBlock;
    v_uncondPtr, v_condPtr, resultPtr, schedule_ptr: PQNNFloat;
    img: TQNNImage;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryblock.create([batch, channels, h, w], 'FLUX_SAMPLE_EULER_RESULT');
    resultPtr := result;
    QNNCopy(result, z, latent_size);
    schedule_ptr := schedule;
    for step := 0 to num_steps -1 do begin
        t_curr := schedule_ptr[step];
        t_next := schedule_ptr[step+1];
        dt := t_next-t_curr;
        if assigned(step_callback) then
            step_callback(step, num_steps);
        v_uncond := forward(result, h, w, text_emb_uncond, text_seq_uncond, t_curr);
        v_uncondPtr := v_uncond;
        v_cond := forward(result, h, w, text_emb_cond, text_seq_cond, t_curr);
        v_condPtr := v_cond;
        // todo sampleEuuler refactor as fused axpy
        for i := 0 to latent_size -1 do begin
            v := v_uncondPtr[i]+guidance_scale * (v_condPtr[i]-v_uncondPtr[i]);
            resultPtr[i] := resultPtr[i] + (dt * v)
        end;
        v_uncond.free;
        v_cond.free;
        if assigned(progress_callback) then
            progress_callback(step, num_steps, result);
        if assigned(step_image_callback) and assigned(vae_ptr) and (step+1 < num_steps) then begin
            img := PVAE(vae_ptr).decode(result, 1, h, w);
            if assigned(img.data) then begin
                step_image_callback(step, num_steps, img);
                img.free
            end
        end
    end;
    freeMMapCache();

end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const ref_latent: TMemoryBlock; const ref_h, ref_w,
  t_offset: longint; const text_emb_cond: TMemoryBlock;
  const text_seq_cond: longint; const text_emb_uncond: TMemoryBlock;
  const text_seq_uncond: longint; const guidance_scale: QNNFloat;
  const schedule: TMemoryBlock; const num_steps: longint;
  const progress_callback: TProgressCallback): TMemoryBlock;
var
    latent_size, step, i: longint;
    t_curr, t_next, dt, v: QNNFloat;
    v_uncond,  v_cond: TMemoryBlock;
    v_condPtr, v_uncondPtr, resultPtr, schedule_ptr:PQNNFloat;
    img:TQNNImage;
begin
    latent_size := batch * channels * h * w;
    result := TMemoryBlock.Create([batch, channels, h, w], 'FLUX_SAMPLE_EULER_RESULT');
    resultPtr := result;
    QNNcopy(result, z, latent_size);
    schedule_ptr := schedule;
    for step := 0 to num_steps -1 do begin
        t_curr := schedule_ptr[step];
        t_next := schedule_ptr[step+1];
        dt := t_next-t_curr;
        if assigned(step_callback) then
            step_callback(step, num_steps);
        v_uncond := forward(result, h, w, ref_latent, ref_h, ref_w, t_offset, text_emb_uncond, text_seq_uncond, t_curr);
        v_cond := forward(result, h, w, ref_latent, ref_h, ref_w, t_offset, text_emb_cond, text_seq_cond, t_curr);
        v_condPtr := v_cond;
        v_uncondPtr := v_uncond;

        for i := 0 to latent_size -1 do begin
            v := v_uncondPtr[i] + guidance_scale * (v_condPtr[i] - v_uncondPtr[i]);
            resultPtr[i] := resultPtr[i] + (dt * v)
        end;
        v_uncond.free;
        v_cond.free;
        if assigned(progress_callback) then
            progress_callback(step, num_steps, result);
        if assigned(step_image_callback) and assigned(vae_ptr) and (step+1 < num_steps) then begin
            img := PVAE(vae_ptr).decode(result, 1, h, w);
            if assigned(img.data) then begin
                step_image_callback(step, num_steps, img);
                img.free
            end
        end
    end;
    freeMMapCache()
end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const refs: TArray<TImageRef>;
  const text_emb: TMemoryBlock; const text_seq: longint;
  const schedule: TMemoryBlock; const num_steps: longint;
  const progress_callback: TProgressCallback): TMemoryBlock;
var
  i, step, latent_size:longint;
  t_curr, t_next, dt: QNNFloat;
  v:TMemoryBlock;
  img:TQNNImage;
  schedule_ptr : PQNNFloat;
begin
  latent_size := batch * channels * h * w;

  result := TMemoryBlock.create([batch, channels, h, w], 'FLUX_SAMPLE_EULER_RESULT');
  QNNCopy(result, z, latent_size);
  schedule_ptr := schedule;
  for step := 0 to num_steps -1 do begin
      t_curr := schedule_ptr[step];
      t_next := schedule_ptr[step+1];
      dt := t_next - t_curr;


      if assigned(step_callback) then
          step_callback(step, num_steps);

      (* Predict velocity with multiple reference images *)
      v := forward(result, h, w, refs, text_emb, text_seq, t_curr);

      (* Euler step *)
      QNNFusedScaleAdd(result, v, result, dt, latent_size);
      v.free;

      if assigned(progress_callback) then
          progress_callback(step, num_steps, result);

      if assigned(step_image_callback) and assigned(vae_ptr) and (step + 1 < num_steps) then begin
          img := PVAE(vae_ptr).decode(result, 1, h, w);
          step_image_callback(step, num_steps, img);
          img.free;
      end;
  end;

  freeMMapCache();

end;

function TTransformerFlux.sampleEuler(const z: TMemoryBlock; const batch,
  channels, h, w: longint; const refs: TArray<TImageRef>;
  const text_emb_cond: TMemoryBlock; const text_seq_cond: longint;
  const text_emb_uncond: TMemoryBlock; const text_seq_uncond: longint;
  const gudance_scale: QNNFloat; const schedule: PQNNFloat; num_steps: longint;
  const progress_callback: TProgressCallback): TMemoryBlock;
begin

end;

{ TBlockZi }

procedure TBlockZi.load(const files: TSafeTensorFiles; const prefix: string;
  const has_modulation: boolean; const useMMap: boolean);
begin

  (* Attention weights *)
  attn_q_weight := files.getTensorDataMemBlock(prefix+'.attention.to_q.weight', useMMap);
  attn_k_weight := files.getTensorDataMemBlock(prefix+'.attention.to_k.weight', useMMap);
  attn_v_weight := files.getTensorDataMemBlock(prefix+'.attention.to_v.weight', useMMap);
  attn_out_weight := files.getTensorDataMemBlock(prefix+'.attention.to_out.0.weight', useMMap);

  (* QK norm *)
  attn_norm_q := files.getTensorDataMemBlock(prefix+'.attention.norm_q.weight', useMMap);
  attn_norm_k := files.getTensorDataMemBlock(prefix+'.attention.norm_k.weight', useMMap);

  (* Pre/post attention norms *)
  attn_norm1 := files.getTensorDataMemBlock(prefix+'.attention_norm1.weight', useMMap);
  attn_norm2 := files.getTensorDataMemBlock(prefix+'.attention_norm2.weight', useMMap);

  (* FFN weights *)
  ffn_w1 := files.getTensorDataMemBlock(prefix+'.feed_forward.w1.weight', useMMap);
  ffn_w2 := files.getTensorDataMemBlock(prefix+'.feed_forward.w2.weight', useMMap);
  ffn_w3 := files.getTensorDataMemBlock(prefix+'.feed_forward.w3.weight', useMMap);

  (* FFN norms *)
  ffn_norm1 := files.getTensorDataMemBlock(prefix+'.ffn_norm1.weight', useMMap);
  ffn_norm2 := files.getTensorDataMemBlock(prefix+'.ffn_norm2.weight', useMMap);

  (* AdaLN modulation (only for modulated blocks) *)
  if has_modulation then begin
      adaln_weight := files.getTensorDataMemBlock(prefix+'.adaLN_modulation.0.weight', useMMap);
      adaln_bias := files.getTensorDataMemBlock(prefix+'.adaLN_modulation.0.bias', useMMap);
  end else begin
//      adaln_weight := nil;
//      adaln_bias := nil;
  end

end;

procedure TBlockZi.free();
begin
  (* Attention weights *)
  attn_q_weight      .free;
  attn_k_weight      .free;
  attn_v_weight      .free;
  attn_out_weight    .free;
  attn_norm_q        .free;
  attn_norm_k        .free;
  attn_norm1         .free;
  attn_norm2         .free;
  ffn_w1             .free;
  ffn_w2             .free;
  ffn_w3             .free;
  ffn_norm1          .free;
  ffn_norm2          .free;
  if adaln_weight.isAssigned() then adaln_weight.free;
  if adaln_bias.isAssigned() then adaln_bias.free;
end;

{ TTransformerZi }

(* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. *)
// compute rope ?
procedure sinusoidalEmbeddingZi(const dst:PQNNFloat; const t:QNNFloat; const dim:longint);
var
  half, i:longint;
  freq, angle : QNNFloat;
const log_max_period:QNNFloat = 9.2103403719761827360719658187375;//ln(10000.0);
begin
    half := dim div 2;
    //log_max_period = ln(10000.0);
    for i := 0 to half-1 do begin
        freq  := exp(-log_max_period*i/half);
        angle := t*freq;
        dst[i] := cos(angle);
        dst[i + half] := sin(angle);
    end
end;

procedure TTransformerZI.attentionForward(const dst, src: TMemoryBlock;
  const block: TBlockZi; const pos_ids: Plongint; const mask: PLongint;
  const seq: longint);
var
  Q, K, V, attn_out
    {, qi, kj, oi, vj}: TMemoryBlock;
  scores : TMemoryBlock;
  d, h, i, j: longint;
  scale, dot, s : QNNFloat;
  dd:TArray<single>;
begin

    Q := work_qkv;
    K := Q + seq*dim;
    V := K + seq*dim;

    (* Q, K, V projections *)
    QNNMatMulNT(Q, src, block.attn_q_weight, seq, dim, dim);
    QNNMatMulNT(K, src, block.attn_k_weight, seq, dim, dim);
    QNNMatMulNT(V, src, block.attn_v_weight, seq, dim, dim);

    (* QK normalization *)
    QNNRMSNormSeq(Q, block.attn_norm_q, seq, n_heads, head_dim);
    QNNRMSNormSeq(K, block.attn_norm_k, seq, n_heads, head_dim);


    (* Apply RoPE *)
    applyRope(Q, pos_ids, seq, n_heads);
    applyRope(K, pos_ids, seq, n_heads);

    (* Scaled dot-product attention per head *)
    scale := 1.0 / sqrt(head_dim);
    attn_out := work_ffn;

    scores := work_attn;
    for h := 0 to n_heads-1 do begin

        (* Compute Q @ K^T for this head *)
        //for i := 0 to seq-1 do begin
        //    qi := Q + i * dim + h * head_dim;
        //    for j := 0 to seq-1 do begin
        //        kj := K + j * dim + h * head_dim;
        //        dot := 0;
        //        for d := 0 to head_dim-1 do
        //            dot := dot + qi[d]*kj[d];
        //        scores[i*seq + j] := dot*scale;
        //    end
        //end;
        cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq, seq, head_dim, scale, Q + h*head_dim, dim, K + h*head_dim, dim, 0, scores, seq);

        (* Apply mask: set padding positions to -inf *)
        if assigned(mask) then
            QNNMaskFill(scores, mask, Single.NegativeInfinity, seq, seq);
            //for i := 0 to seq-1 do
            //    for j := 0 to seq-1 do
            //        if mask[j]=0 then
            //            scores[i*seq + j] := -1e9;

        (* Softmax *)
        QNNSoftmaxRows(scores, seq, seq);

        (* Scores @ V *)
        //for i := 0 to seq-1 do begin
        //    oi := attn_out + i*dim + h*head_dim;
        //    QNNFill(oi, 0, head_dim);
        //    for j  := 0 to seq-1 do begin
        //        s  := scores[i * seq + j];
        //        vj := V + j * dim + h * head_dim;
        //        for d := 0 to head_dim-1 do
        //            oi[d] := oi[d] + s*vj[d];
        //    end;
        //end;
        cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq, head_dim, seq, 1.0, scores, seq, V + h*head_dim, dim, 0, attn_out + h*head_dim, dim)
    end;
    QNNMatMulNT(dst, attn_out, block.attn_out_weight, seq, dim, dim);
end;

procedure TTransformerZI.ffnForward(const dst, src: TMemoryBlock;
  const block: TBlockZi; const seq: longint);
var
  gate, up : TMemoryBlock;
  N:longint;
begin
    N := seq*ffn_dim;
    gate := work_ffn;
    up := gate + N;

    (* W1 (gate) and W3 (up) projections *)
    QNNMatMulNT(gate, src, block.ffn_w1, seq, dim, ffn_dim);
    QNNMatMulNT(up, src, block.ffn_w3, seq, dim, ffn_dim);

    (* SiLU(gate) * up *)
    QNNSiluInplace(gate, N);
    //for i := 0 to n-1 do gate[i] := gate[i]*up[i];
    QNNMulInplace(gate, up, N);

    (* W2 (down) projection *)
    QNNMatMulNT(dst, gate, block.ffn_w2, seq, ffn_dim, dim);
end;

procedure TTransformerZI.blockForward(const dst: TMemoryBlock;
  const block: TBlockZi; const pos_ids: PLongint; const mask: PLongint;
  const t_emb: TMemoryBlock; const seq: longint);
var
    _mod: array[0..4 * {dim}3840-1 ] of QNNFloat;
    modMem: TMemoryBlock;
    n, i, s: longint;
    scale_msa, gate_msa, scale_mlp, gate_mlp: TMemoryBlock;
    norm_out, attn_out, ffn_out, scaled : TMemoryBlock;
begin
    n := seq * dim;
    attn_out := work_tmp;
    norm_out := attn_out+n;
    scaled   := attn_out+2 * n;
    ffn_out  := attn_out+3 * n;
    if not work_tmp.isAssigned() or (max_seq < seq) then
        exit();
    modMem.assignPtr(PQNNFloat(@_mod[0]), [length(_mod)]);
    if block.adaln_weight.isAssigned() then
        begin
            assert(t_emb.isAssigned(), 'ERROR : blockForward t_emb not assigned!');
            QNNMatMulNT(modMem, t_emb, block.adaln_weight, 1, adaln_dim, 4 * dim);
            //for i := 0 to 4 * dim -1 do
            //    _mod[i] := _mod[i] + block.adaln_bias[i];
            QNNAddInplace(modMem, block.adaln_bias, 4*dim);
            scale_msa := modMem;
            gate_msa  := modMem + dim;
            scale_mlp := modMem + 2 * dim;
            gate_mlp  := modMem + 3 * dim;
            //for i := 0 to dim -1 do
            //    begin
            //        scale_msa[i] := 1.0+scale_msa[i];
            //        gate_msa[i] := tanh(gate_msa[i]);
            //        scale_mlp[i] := 1.0+scale_mlp[i];
            //        gate_mlp[i] := tanh(gate_mlp[i])
            //    end;
            QNNBiasInplace(scale_msa, 1.0, dim);
            QNNTanh(gate_msa, dim);
            QNNBiasInplace(scale_mlp, 1.0, dim);
            QNNTanh(gate_mlp, dim);
            QNNRMSNormRows(norm_out, dst, block.attn_norm1, seq, dim);

            for s := 0 to seq -1 do
                //for i := 0 to dim -1 do
                //    scaled[s * dim+i] := norm_out[s * dim+i] * scale_msa[i];
                QNNMul(scaled + s*dim, norm_out + s*dim, scale_msa, dim);

            attentionForward(attn_out, scaled, block, pos_ids, mask, seq);
            QNNRMSNormRows(norm_out, attn_out, block.attn_norm2, seq, dim);
            for s := 0 to seq -1 do
                //for i := 0 to dim -1 do
                //    dst[s * dim+i] := dst[s * dim+i] + (gate_msa[i] * norm_out[s * dim+i]);
                QNNFusedMulAdd(dst + s*dim, gate_msa, norm_out + s*dim, dst + s*dim, dim);
            QNNRMSNormRows(norm_out, dst, block.ffn_norm1, seq, dim);
            for s := 0 to seq -1 do
                QNNMul(scaled + s*dim, norm_out + s*dim, scale_mlp, dim);
                //for i := 0 to dim -1 do
                //    scaled[s * dim+i] := norm_out[s * dim+i] * scale_mlp[i];
            ffnForward(ffn_out, scaled, block, seq);
            QNNRMSNormRows(norm_out, ffn_out, block.ffn_norm2, seq, dim);
            for s := 0 to seq -1 do
                QNNFusedMulAdd(dst + s*dim, gate_mlp, norm_out + s*dim, dst + s*dim, dim);
                //for i := 0 to dim -1 do
                //    dst[s * dim+i] := dst[s * dim+i] + (gate_mlp[i] * norm_out[s * dim+i])
        end
    else
        begin
            QNNRMSNormRows(norm_out, dst, block.attn_norm1, seq, dim);
            attentionForward(attn_out, norm_out, block, pos_ids, mask, seq);
            QNNRMSNormRows(norm_out, attn_out, block.attn_norm2, seq, dim);
            //for i := 0 to N -1 do
            //    dst[i] := dst[i] + norm_out[i];
            QNNAddInplace(dst, norm_out, N);
            QNNRMSNormRows(norm_out, dst, block.ffn_norm1, seq, dim);
            ffnForward(ffn_out, norm_out, block, seq);
            QNNRMSNormRows(norm_out, ffn_out, block.ffn_norm2, seq, dim);
            //for i := 0 to N -1 do
            //    dst[i] := dst[i] + norm_out[i]
            QNNAddInplace(dst, norm_out, N);

        end
end;

procedure TTransformerZI.preComputeRope;
var
  ax, d, half_d, pos, max_pos, i:longint;
  freq, angle:QNNFloat;
  ropeCos, ropeSin : PQNNFloat;
begin
    for ax := 0 to 3-1 do begin
        d       := axes_dims[ax];
        half_d  := d div 2;
        max_pos := axes_lens[ax];

        rope_cos[ax] := TMemoryBlock.create([max_pos, half_d], 'ZI_PRECOMPUTE_ROPE_COS_'+IntToStr(ax));
        rope_sin[ax] := TMemoryBlock.create([max_pos, half_d], 'ZI_PRECOMPUTE_ROPE_SIN_'+IntToStr(ax));
        ropeCos := rope_cos[ax];
        ropeSin := rope_sin[ax];
        for pos := 0 to max_pos-1 do begin
          for i := 0 to half_d-1 do begin
                freq  := 1.0 / math.power(rope_theta, (2*i)/d);
                angle := pos*freq;
                ropeCos[pos*half_d + i] := cos(angle);
                ropeSin[pos*half_d + i] := sin(angle);
          end;
        end
    end
end;

procedure TTransformerZI.finalComputeScale(const scale, t_emb: TMemoryBlock);
var
  silu_emb : array[0..255] of QNNFLoat;
  siluEmbMem : TMemoryBlock;
begin
    siluEmbMem.assignPtr(PQNNFloat(@silu_emb[0]), [length(silu_emb)]);
    QNNCopy(siluEmbMem, t_emb, adaln_dim);
    QNNSiluInplace(siluEmbMem, adaln_dim);

    QNNMatMulNT(scale, siluEmbMem, final_layer.adaln_weight, 1, adaln_dim, dim);
    //for (int i = 0; i < dim; i++)
    //    scale[i] = 1.0 + scale[i] + adaln_bias[i];
    QNNFusedBiasAdd(scale, final_layer.adaln_bias, scale, 1.0, dim);
end;

procedure TTransformerZI.finalForward(const dst, src, t_emb: TMemoryBlock;
  const seq: longint);
var
  out_dim, s : longint;
  scale, normed : TMemoryBlock;
  //normed_ptr : PQNNFloat;
  //mean, variance, inv_std : QNNFloat;
begin
    out_dim := patch_size * patch_size * in_channels;

    scale := TMemoryBlock.Create([dim], 'ZI_FINAL_SCALE');
    finalComputeScale(scale, t_emb);

    (* LayerNorm (no affine) -> scale *)
    normed := TMemoryBlock.Create([seq, dim], 'ZI_FINAL_NORMED');
    for s := 0 to seq-1 do begin
        //xr := src + s * dim;
        //nr := normed + s * dim;

        (* Compute mean and variance *)
        //mean := QNNMean(dim, xr);
        //variance := QNNVariance(dim, xr, mean);
        //inv_std = 1.0 / sqrt(variance + 1e-6); (* Final LayerNorm uses 1e-6 *)
        //QNNFusedBiasMulScale(nr, xr, scale, -mean, inv_std, dim);
        QNNNorm(dim, normed + s*dim, src + s*dim, scale)
    end;

    (* Linear projection: dim -> out_dim *)
    QNNMatMulNT(dst, normed, final_layer.linear_weight, seq, dim, out_dim);
    for s := 0 to seq-1 do
        //for i = 0 to out_dim-1 do
        //    dst[s*out_dim + i] := dst[s*out_dim + i] + linear_bias[i];
        QNNAddInplace(dst + s*out_dim, final_layer.linear_bias, out_dim);
    scale.free;
    normed.free;
end;

(* ========================================================================
 * Patchify / Unpatchify
 * ======================================================================== *)

(* Converts latent [in_ch, H, W] to patch sequence [n_patches, ps*ps*in_ch].
 * Gathers each ps x ps spatial block into a flat vector, ordering as
 * (ph, pw, channel). This is the inverse of unpatchify and creates the
 * token sequence the transformer operates on. *)
procedure patchifyZi(const dst : PQNNFloat; const src:PQNNFloat; const in_ch, Height, Width, ps:longint);
var
  H_tokens, W_tokens, patch_feat, patch_idx, h, w, di, c, ph, pw, sx, sy:longint;
  d : PQNNFloat;
begin
    H_tokens := Height div ps;
    W_tokens := Width div ps;
    patch_feat := ps * ps * in_ch;

    for h := 0 to H_tokens-1 do begin
      for w := 0 to W_tokens-1 do begin
        patch_idx := h*W_tokens + w;
        d         := dst + patch_idx*patch_feat;
        di        := 0;

        (* Gather patch: iterate (ph, pw, c) *)
        for ph := 0 to ps-1 do
          for pw := 0 to ps-1 do
            for c := 0 to in_ch-1 do begin
                sy := h * ps + ph;
                sx := w * ps + pw;
                d[di] := src[c*Height*Width + sy*Width + sx];
                inc(di)
            end
      end
    end
end;

(* Unpatchify: [n_patches, patch_feat_dim] -> [in_ch, H, W] *)
procedure UnPatchifyZi(const dst: PQNNFloat; const src: PQNNFloat; const in_ch,
  Height, Width, ps: longint);
var
  H_tokens, W_Tokens, patch_feat, h, w, patch_idx, si, ph, pw, sy, sx, c : longint;
  d : PQNNFloat;
begin
    H_tokens := Height div ps;
    W_tokens := Width div ps;
    patch_feat := ps * ps * in_ch;

    for h := 0 to H_tokens-1 do
        for w := 0 to W_tokens-1 do begin
            patch_idx := h*W_tokens + w;
            d         := src + patch_idx * patch_feat;
            si := 0;
            for ph := 0 to ps-1 do
                for pw := 0 to ps-1 do
                    for c := 0 to in_ch-1 do begin
                        sy := h * ps + ph;
                        sx := w * ps + pw;
                        dst[c*Height*Width + sy*Width + sx] := d[si];
                        inc(si)
                    end
       end
end;


function TTransformerZI.forward(const latent: TMemoryBlock; const latent_h,
  latent_w: longint; const timestep: QNNFloat; const cap_feats: TMemoryBlock;
  const cap_seq_len: longint): TMemoryBlock;
var patch_feat, H_tokens, W_tokens, img_seq, refiner_total
    , img_pad, cap_pad, img_padded, cap_padded, unified_seq, needed, i, s, h, w, out_ch : longint;
  t_emb : array[0..255] of QNNFloat;
  img_patches, img_emb, cap_emb, cap_normed, cap_padded_feats, img_out, final_out, t_embMem : TMemoryBlock;
  cap_pos, img_pos, unified_pos : TArray<longint>;
  unified : TMemoryBlock;
begin
    t_embMem.assignPtr(PQNNFloat(@t_emb[0]), [length(t_emb)]);
    patch_feat := patch_size * patch_size * in_channels;  (* 64 *)

    H_tokens      := latent_h div patch_size;
    W_tokens      := latent_w div patch_size;
    img_seq       := H_tokens * W_tokens;
    refiner_total := n_refiner * 2;

    (* Pad sequences to multiples of ZI_SEQ_MULTI_OF *)
    img_pad := (ZI_SEQ_MULTI_OF - (img_seq mod ZI_SEQ_MULTI_OF)) mod ZI_SEQ_MULTI_OF;
    cap_pad := (ZI_SEQ_MULTI_OF - (cap_seq_len mod ZI_SEQ_MULTI_OF)) mod ZI_SEQ_MULTI_OF;
    img_padded := img_seq + img_pad;
    cap_padded := cap_seq_len + cap_pad;
    unified_seq := img_padded + cap_padded;

    (* Ensure working memory is sufficient *)
    needed := unified_seq * dim * 4 + unified_seq * dim * 3 +  (* QKV *) unified_seq * unified_seq + (* attention scores *) unified_seq * ffn_dim * 2;
    if needed > work_alloc then begin
        work_x   .free;
        work_tmp .free;
        work_qkv .free;
        work_attn.free;
        work_ffn .free;

        work_x     := TMemoryBlock.Create([unified_seq, dim]        , 'ZI_FW_work_x');
        work_tmp   := TMemoryBlock.Create([unified_seq, dim, 4]     , 'ZI_FW_work_tmp');
        work_qkv   := TMemoryBlock.Create([unified_seq, dim, 3]     , 'ZI_FW_work_qkv');
        work_attn  := TMemoryBlock.Create([unified_seq, unified_seq], 'ZI_FW_work_attn');
        work_ffn   := TMemoryBlock.Create([unified_seq, ffn_dim, 2] , 'ZI_FW_work_ffn');
        work_alloc := needed;
        max_seq := unified_seq;
    end;

    (* 1. Timestep embedding *)

    timeStepEmbed(t_embMem, timestep);

    (* 2. Patchify image -> [img_seq, patch_feat] *)
    img_patches := TMemoryBlock.Create([img_padded, patch_feat], 'ZI_FW_IMG_PATCHES');
    //img_patches_ptr := img_patches;

    patchifyZi(img_patches, latent, in_channels, latent_h, latent_w, patch_size);

    (* Pad image patches (repeat last token) *)
    // todo ZI.Forward  use QNNBroadcast instead of copy loop
    for i := img_seq to img_padded-1 do
        QNNCopy(img_patches + i * patch_feat, img_patches + (img_seq - 1) * patch_feat, patch_feat);

    (* Embed image: [img_padded, patch_feat] -> [img_padded, dim] *)
    img_emb := TMemoryBlock.Create([img_padded, dim], 'ZI_FW_IMG_EMB');
    //img_emb_ptr := img_emb;
    QNNMatMulNT(img_emb, img_patches, x_emb_weight, img_padded, patch_feat, dim);
    for s := 0 to img_padded-1 do
        QNNAddInplace(img_emb + s*dim, x_emb_bias, dim);
        //for i := 0 to dim-1 do
        //    img_emb[s * dim + i] := img_emb[s * dim + i] + x_emb_bias[i];
    img_patches.free;


    (* Apply pad token to image padding positions *)
    // todo ZI.Forward  use QNNBroadcast instead of copy loop
    for s := img_seq to img_padded-1 do
        QNNCopy(img_emb + s*dim, x_pad_token, dim);

    (* 3. Caption embedding: RMSNorm -> Linear *)
    cap_emb := TMemoryBlock.Create([cap_padded, dim], 'ZI_FW_CAP_EMB');
    //cap_emb_ptr := cap_emb;
    cap_normed := TMemoryBlock.Create([cap_padded, cap_feat_dim], 'ZI_FW_CAP_NORMED');

    (* Pad caption features (repeat last token) *)
    cap_padded_feats := TMemoryBlock.Create([cap_padded, cap_feat_dim], 'ZI_FW_CAP_PADDED_FEATS');
    //cap_padded_feats_ptr := cap_padded_feats;
    QNNCopy(cap_padded_feats, cap_feats, cap_seq_len * cap_feat_dim);
    (* Apply pad token to image padding positions *)
       // todo ZI.Forward  use QNNBroadcast instead of copy loop
    for s := cap_seq_len to cap_padded-1 do
        QNNCopy(cap_padded_feats + s * cap_feat_dim,
               cap_feats + (cap_seq_len - 1) * cap_feat_dim,
               cap_feat_dim );

    QNNRMSNormRows(cap_normed, cap_padded_feats, cap_emb_norm, cap_padded, cap_feat_dim);
    QNNMatMulNT(cap_emb, cap_normed, cap_emb_linear_w, cap_padded, cap_feat_dim, dim);

    cap_padded_feats.free;
    cap_normed.free;

    for s := 0 to cap_seq_len-1 do //cap_padded-1 do  // todo should be to cap_seq_len-1, see the next fot to know why
        //for (int i = 0; i < dim; i++)
        //    cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
        QNNAddInplace(cap_emb + s*dim, cap_emb_linear_b, dim);

    (* Apply pad token to caption padding positions *)
    for s := cap_seq_len to cap_padded-1 do
        QNNCopy(cap_emb + s*dim, cap_pad_token, dim);

    (* 4. Build position IDs *)

    (* Image position IDs: (T=cap_padded+1, H=h_idx, W=w_idx)
     * All image tokens share the same T position (one frame). *)
    setLength(img_pos, img_padded * 3);//(int *)calloc(img_padded * 3, sizeof(int));

    for h := 0 to H_tokens-1 do begin
        for w := 0 to W_tokens-1 do begin
            i := h * W_tokens + w;
            img_pos[i * 3 + 0] := cap_padded + 1;  (* T (same for all) *)
            img_pos[i * 3 + 1] := h;                (* H *)
            img_pos[i * 3 + 2] := w;                (* W *)
        end
    end;
    (* Padding tokens get (0, 0, 0) *)

    (* Caption position IDs: (T=1+seq_idx, H=0, W=0) *)
    setLength(cap_pos, cap_padded * 3);
    for s := 0 to cap_padded-1 do begin
        cap_pos[s * 3 + 0] := 1 + s;  (* T *)
        cap_pos[s * 3 + 1] := 0;       (* H *)
        cap_pos[s * 3 + 2] := 0;       (* W *)
    end;

    (* 5. Noise refiner: image-only self-attention with modulation *)
    for i := 0 to n_refiner-1 do begin
        blockForward(img_emb, noise_refiner[i], pointer(img_pos), nil, t_embMem, img_padded);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    end;

    (* 6. Context refiner: caption-only self-attention without modulation *)
    for i := 0 to n_refiner-1 do begin
        blockForward(cap_emb, context_refiner[i], pointer(cap_pos), nil, default(TMemoryBlock), cap_padded);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_DOUBLE_BLOCK, n_refiner + i, refiner_total);
    end;

    (* 7. Build unified sequence: [image_tokens, caption_tokens] *)
    unified := work_x;
    QNNCopy(unified, img_emb, img_padded * dim);
    QNNCopy(unified + img_padded * dim, cap_emb, cap_padded * dim);
    img_emb.free;
    cap_emb.free;

    (* Unified position IDs *)
    setLength(unified_pos, unified_seq * 3);
    move(img_pos[0], unified_pos[0]             , img_padded * 3 * sizeof(longint));
    move(cap_pos[0], unified_pos[img_padded * 3], cap_padded * 3 * sizeof(longint));

    (* 8. Main transformer layers *)
    for i := 0 to n_layers-1 do begin
        blockForward(unified, layers[i], pointer(unified_pos), nil, t_embMem, unified_seq);
        if assigned(substep_callback) then
            substep_callback(SUBSTEP_SINGLE_BLOCK, i, n_layers);
    end;

    setLength(unified_pos, 0);

    (* 9. Final layer: extract image tokens only, then project *)
    img_out := TMemoryBlock.Create([img_seq, dim], 'ZI_FW_IMG_OUT');
    QNNCopy(img_out, unified, img_seq * dim);

    out_ch := patch_size * patch_size * in_channels;  (* 64 *)
    final_out := TMemoryBlock.Create([img_seq * out_ch], 'ZI_FW_FINALE_OUT');
    finalForward(final_out, img_out, t_embMem, img_seq);
    img_out.free();
    if assigned(substep_callback) then
        substep_callback(SUBSTEP_FINAL_LAYER, 0, 1);
    (* 10. Unpatchify: [n_patches, 64] -> [16, latent_h, latent_w] *)
    result := TMemoryBlock.Create([in_channels, latent_h, latent_w], 'ZI_FW_RESULT');
    UnpatchifyZi(result, final_out, in_channels, latent_h, latent_w, patch_size);
    final_out.free();
end;

procedure TTransformerZI.applyRope(const dst: PQNNFloat;
  const pos_ids: PLongint; const seq, n_heads: longint);
var
  offset, d, half_d, s, pos, h, i, ax : longint;
  cos_tab, sin_tab, head, ropeCos, ropeSin : PQNNFloat;
  x0, x1, c, sn :QNNFloat;
begin
  offset := 0;

  for ax := 0 to 3-1 do begin
      d      := axes_dims[ax];
      half_d := d div 2;
      ropeCos := rope_cos[ax];
      ropeSin := rope_sin[ax];

      for s := 0 to seq-1 do begin
          pos := pos_ids[s * 3 + ax];
          if (pos < 0) or (pos >= axes_lens[ax]) then continue;

          cos_tab := ropeCos + pos * half_d;
          sin_tab := ropeSin + pos * half_d;

          for h := 0 to n_heads-1 do begin
              head := dst + (s * n_heads + h) * head_dim + offset;
              for i := 0 to half_d-1 do begin
                  x0 := head[2 * i];
                  x1 := head[2 * i + 1];
                  c  := cos_tab[i];
                  sn := sin_tab[i];
                  head[2 * i]     := x0*c - x1*sn;
                  head[2 * i + 1] := x1*c + x0*sn;
              end
          end
      end;
      inc(offset, d);
  end

end;

(* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. *)
procedure TTransformerZI.timeStepEmbed(const dst: TMemoryBlock;
  const t: QNNFloat);
var
  sin_emb : array[0..255] of QNNFloat;
  hidden, sinEmbMem : TMemoryBlock;
begin
    sinEmbMem.assignPtr(@sin_emb[0], [length(sin_emb)]);
    sinusoidalEmbeddingZi(PQNNFloat(@sin_emb[0]), t*1000.0, 256);

    (* MLP: Linear(256 -> t_emb_mid_size) + SiLU + Linear(t_emb_mid_size -> adaln_dim) *)
    hidden := TMemoryBlock.Create([t_emb_mid_size], 'ZI_TIMESTEP_EMBED');

    (* Linear 0 *)
    QNNMatMulNT(hidden, sinEmbMem, t_emb_mlp0_weight, 1, 256, t_emb_mid_size);
    QNNAddInplace(hidden, t_emb_mlp0_bias, t_emb_mid_size);
    //for i := 0 to t_emb_mid_size-1 do
    //    hidden[i] := hidden[i] + t_emb_mlp0_bias[i];

    (* SiLU *)
    QNNSiluInplace(hidden, t_emb_mid_size);

    (* Linear 2 *)
    QNNMatMulNT(dst, hidden, t_emb_mlp2_weight, 1, t_emb_mid_size, adaln_dim);
    QNNAddInplace(dst, t_emb_mlp2_bias, adaln_dim);
    //for i := 0 to adaln_dim-1 do
    //    dst[i] := dst[i] + t_emb_mlp2_bias[i];

    hidden.free();
end;

function TTransformerZI.sampleEuler(const z: TMemoryBlock; const batch, h, w,
  patch_size: longint; const cap_feats: TMemoryBlock; const cap_seq: longint;
  const schedule: PQNNFloat; const num_steps: longint;
  const progress_callback: TStepCallback): TMemoryBlock;
var
    latent_size, step_h, step_w, step_ch, step_latent_size, step, i, decode_h, decode_w: longint;
    sigma, sigma_next, dt, timestep: QNNFloat;
    step_latent, model_out, decode_latent: TMemoryBlock;
    img: TQNNImage;
begin
    latent_size := batch * in_channels{channels} * h * w;
    result := TMemoryBlock.Create([batch, in_channels{channels}, h, w], 'ZI_SAMPLE_EULER_RESULT');
    QNNCopy(result, z, latent_size);
//    step_latent := nil;
    step_h := h;
    step_w := w;
    if assigned(step_image_callback) and assigned(vae_ptr) and (num_steps > 1) and (patch_size > 1) then begin
      if (h mod patch_size = 0) and (w mod patch_size = 0) then begin
        step_ch := in_channels{channels} *patch_size * patch_size;
        step_h := h div patch_size;
        step_w := w div patch_size;
        step_latent_size := batch * step_ch * step_h * step_w;
        step_latent := TMemoryBlock.create([batch, step_ch, step_h, step_w], 'ZI_SAMPEL_EULER_STEP_LATENT')
      end
    end;
    for step := 0 to num_steps -1 do
        begin
            sigma := schedule[step];
            sigma_next := schedule[step+1];
            dt := sigma_next - sigma;
            timestep := 1.0-sigma;
            if assigned(step_callback) then
                step_callback(step, num_steps);
            model_out := forward(result, h, w, timestep, cap_feats, cap_seq);

            //for i := 0 to latent_size -1 do
            //    result[i] := result[i] + (dt * (-model_out[i]));
            QNNFusedScaleAdd(result, model_out, result, -dt, latent_size);
            model_out.free;
            if assigned(progress_callback) then
                progress_callback(step+1, num_steps);
            if assigned(step_image_callback) and assigned(vae_ptr) and (step+1 < num_steps) then begin
              decode_latent := result;
              decode_h := h;
              decode_w := w;
              if patch_size > 1 then begin
                  if not step_latent.isAssigned() then
                      continue;
                  QNNPatchify(step_latent, result, batch, in_channels{channels}, h, w, patch_size);
                  decode_latent := step_latent;
                  decode_h := step_h;
                  decode_w := step_w
              end;
              img := PVAE(vae_ptr).decode(decode_latent, 1, decode_h, decode_w);
              step_image_callback(step, num_steps, img);
              img.free
            end
        end;
    step_latent.free;
end;

procedure TTransformerZI.load(const modelDir: string; const adim, an_heads,
  an_layers, an_refiner, acap_feat_dim, ain_channels, apatch_size: longint;
  const arope_theta: QNNFloat; const aAxes_dims: Plongint; const useMMAP: boolean
  );
var i:longint;
  json, wm : TJSON;
  sf : PSafeTensor;
  fn : string;
begin
  (* Set config *)
  dim := adim;
  n_heads := an_heads;
  head_dim := dim div n_heads;
  n_layers := an_layers;
  n_refiner := an_refiner;
  ffn_dim := (8 * dim div 3 + 255) div 256 * 256;  (* Round up to 256 *)
  in_channels := ain_channels;
  patch_size := apatch_size;
  if dim < 256 then adaln_dim := dim else adaln_dim := 256;
  rope_theta := arope_theta;
  cap_feat_dim := acap_feat_dim;

  for i := 0 to 3-1 do begin
      axes_dims[i] := aAxes_dims[i];
      axes_lens[i] := 1024;  (* Default max positions *)
  end;

  (* Open safetensors files *)

  sf_files := loadShards(modelDir);


  num_sf_files := length(sf_files);

  (* Determine FFN dimension from weights *)
  sf := sf_files.getTensor('layers.0.feed_forward.w1.weight');
  if assigned(sf) then
      ffn_dim := sf.shape[0];

  (* Determine t_embedder mid_size from weights *)
  t_emb_mid_size := 1024;  (* Default *)
  sf := sf_files.getTensor('t_embedder.mlp.0.weight');
  if assigned(sf) then
      t_emb_mid_size := sf.shape[0];

  (* BLAS/CPU fast-load mode: keep mmap files open and use direct f32 pointers. *)
  mmap_f32_weights := useMMAp;

  (* Load timestep embedder *)
  t_emb_mlp0_weight := sf_files.getTensorDataMemBlock('t_embedder.mlp.0.weight', useMMAp);
  t_emb_mlp0_bias := sf_files.getTensorDataMemBlock('t_embedder.mlp.0.bias', useMMap);
  t_emb_mlp2_weight := sf_files.getTensorDataMemBlock('t_embedder.mlp.2.weight', useMMap);
  t_emb_mlp2_bias := sf_files.getTensorDataMemBlock('t_embedder.mlp.2.bias', useMMap);

  (* Load caption embedder: RMSNorm + Linear *)
  cap_emb_norm := sf_files.getTensorDataMemBlock('cap_embedder.0.weight', useMMap);
  cap_emb_linear_w := sf_files.getTensorDataMemBlock('cap_embedder.1.weight', useMMap);
  cap_emb_linear_b := sf_files.getTensorDataMemBlock('cap_embedder.1.bias', useMMap);

  (* Load image embedder *)
  x_emb_weight := sf_files.getTensorDataMemBlock(format('all_x_embedder.%d-1.weight', [patch_size]), false);
  x_emb_bias := sf_files.getTensorDataMemBlock(format('all_x_embedder.%d-1.bias', [patch_size]), false);

  (* Pad tokens *)
  x_pad_token := sf_files.getTensorDataMemBlock('x_pad_token', useMMap);
  cap_pad_token := sf_files.getTensorDataMemBlock('cap_pad_token', useMMap);

  (* Load noise refiner blocks *)
  setLength(noise_refiner, n_refiner);

  for i := 0 to high(noise_refiner) do
      noise_refiner[i].load(sf_files, format('noise_refiner.%d', [i]), true, useMMap);

  (* Load context refiner blocks (no modulation) *)
  setLength(context_refiner, n_refiner);

  for i := 0 to n_refiner-1 do
      context_refiner[i].load(sf_files, format('context_refiner.%d', [i]), false, useMMap);

  (* Load main transformer blocks *)
  setLength(layers, n_layers);

  for i := 0 to n_layers-1 do
      layers[i].load(sf_files, format('layers.%d', [i]), true, useMMap);

  (* Load final layer *)
  final_layer.adaln_weight := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.adaLN_modulation.1.weight', [patch_size]), useMMAP);
  final_layer.adaln_bias := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.adaLN_modulation.1.bias', [patch_size]), useMMAP);
  //final_layer.norm_weight := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.norm_final.weight', [patch_size]), useMMAP);
  final_layer.linear_weight := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.linear.weight', [patch_size]), useMMAP);
  final_layer.linear_bias := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.linear.bias', [patch_size]), useMMAP);

  (* Precompute RoPE tables *)
  precomputeRope();

  (* Allocate initial working memory (will be resized as needed) *)
  work_alloc := 0;
  //work_x := nil;
  //work_tmp := nil;
  //work_qkv := nil;
  //work_attn := nil;
  //work_ffn := nil;
  max_seq := 0;

end;

function TTransformerZI.isLoaded(): boolean;
begin
  result := assigned(sf_files)
end;

procedure TTransformerZI.free;
var i:longint;
begin
  //if not mmap_f32_weights then begin
      t_emb_mlp0_weight.free;
      t_emb_mlp0_bias.free;
      t_emb_mlp2_weight.free;
      t_emb_mlp2_bias.free;
      cap_emb_norm.free;
      cap_emb_linear_w.free;
      cap_emb_linear_b.free;
      x_emb_weight.free;
      x_emb_bias.free;
      x_pad_token.free;
      cap_pad_token.free;
  //end;

  if assigned(noise_refiner) then begin
      for i := 0 to high(noise_refiner) do
          noise_refiner[i].free();
      setLength(noise_refiner, 0);
  end;
  if assigned(context_refiner) then begin
      for i := 0 to high(context_refiner) do
          context_refiner[i].free;
      setLength(context_refiner, 0);
  end;
  if assigned(layers) then begin
      for i := 0 to high(layers) do
          layers[i].free;
      setLength(layers, 0);
  end;

  final_layer.adaln_weight.free;
  final_layer.adaln_bias.free;
  //final_layer.norm_weight.free;
  final_layer.linear_weight.free;
  final_layer.linear_bias.free;

  for i := 0 to high(sf_files) do
      sf_files[i].close();
  num_sf_files := 0;

  for i := 0 to 3-1 do begin
      rope_cos[i].free;
      rope_sin[i].free;
  end;

  work_x.free;
  work_tmp.free;
  work_qkv.free;
  work_attn.free;
  work_ffn.free;
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

