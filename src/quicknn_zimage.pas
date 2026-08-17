unit quicknn_zimage;

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
uses quicknn_common, safetensor, quicknn_transformers, quicknn_qwen3, quicknn_vae;


type

  //TQNNZImage = TQNNCtx;
  { TQNNZImageHelper }

  TQNNZImage = record
    //tokenizer : TQNNTokenizer;
    qwen3Encoder : TQWEN3Encoder;
    vae : TVAE;
    transformer : TTransformerZI;

    (* Configuration *)
    max_width          : longint ;
    max_height         : longint ;
    default_steps      : longint ;
    default_guidance   : QNNFloat ;
    is_distilled       : boolean ;  (* 1 = distilled (4-step), 0 = base (50-step CFG) *)
    text_dim           : longint ;  (* Text embedding dimension (7680 for 4B, varies for 9B) *)
    is_non_commercial  : boolean ;  (* 1 if model has non-commercial license (9B) *)
    num_heads          : longint ;  (* Transformer attention heads (24 for 4B, 32 for 9B) *)

    (* Z-Image specific config (from transformer/config.json) *)
    is_zimage          : boolean ;   (* 1 = Z-Image S3-DiT, 0 = Flux MMDiT *)
    zi_dim             : longint   ; (* Hidden dim (3840) *)
    zi_n_layers        : longint   ; (* Main transformer layers (30) *)
    zi_n_refiner       : longint   ; (* Noise/context refiner layers (2) *)
    zi_cap_feat_dim    : longint   ; (* Caption feature dim (2560) *)
    zi_in_channels     : longint   ; (* VAE latent channels (16) *)
    zi_patch_size      : longint   ; (* Spatial patch size (2) *)
    zi_rope_theta      : QNNFloat  ; (* RoPE theta (256.0) *)
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
    constructor load(const aModel_dir:string; const OnStatus: TPhaseCallback = nil);
    procedure loadVAE();
    function generate(const prompt: rawbytestring; params: TGenerateParams): TQNNImage;
    function generateFromEmbeddings(const text_emb:TMemoryBlock; const text_seq:longint; params: TGenerateParams):TQNNImage;
    function encodeText(const prompt: rawbytestring; var out_seq_len: longint): TMemoryBlock;
    procedure free();
  end;

implementation
uses
  sysutils, quickjson, quicknn_kernels, quicknn_sample
  {$ifdef MSWINDOWS}
  , windows
  {$endif}
;

{ TQNNZImage }

constructor TQNNZImage.load(const aModel_dir: string;
  const OnStatus: TPhaseCallback);
var
  json : TJSON;
  modelArc : string;
  hidden_size : longint;
  sf : TSafeTensorFile;
  arr:TArray<longint>;
begin
  self := default(TQNNZImage);
  phase_callback := onStatus;
  max_width     := QNN_VAE_MAX_DIM;
  max_height    := QNN_VAE_MAX_DIM;
  model_version := '1.0';
  model_dir     := aModel_dir;

  (* Autodetect model type from model_index.json.
   * Distilled model has "is_distilled": true, base model does not.
   * Z-Image has "_class_name": "ZImagePipeline". *)
  modelArc := format('%s/model_index.json', [model_dir]);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', false);
  if fileExists(modelArc) then begin
    json      := TJSON.LoadFromFile(modelArc);
    modelArc := json['_class_name'];
    assert((pos('ZImagePipeline',modelArc)=1) or (pos('Z-Image', modelArc)=1), 'ERROR : Model is not a Z-Image pipeline');
    //is_zimage := (modelArc='ZImagePipeline') or (modelArc='Z-Image');
    is_distilled := json.get('is_distilled', false);
  end else begin // assuming default FLUX.2-klein-4b
    is_distilled := true;
  end;
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);

  modelArc := format('%s/transformer/config.json', [model_dir]);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', false);
  json := TJSON.LoadFromFile(modelArc);
  if json.keyExist('cap_feat_dim') then is_zimage:=true;

  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);

  (* dim (hidden size) *)
  zi_dim := json.get('zi-dim', 3840);
  num_heads := zi_dim div 128;  (* head_dim = 128 *)

  (* n_layers *)
  zi_n_layers := json.get('n_layers', 30);

  (* n_refiner_layers *)
  zi_n_refiner :=json.get('n_refiner_layers',  2);

  (* cap_feat_dim *)
  zi_cap_feat_dim := json.get('cap_feat_dim', 2560);

  (* in_channels *)
  zi_in_channels := json.get('in_channels', 16);

  (* patch_size *)
  zi_patch_size := json.get('patch_size', 2);

  (* rope_theta *)
  zi_rope_theta := json.get('rope_theta', 256.0);

  (* axes_dims - parse JSON array [32, 48, 48] *)
  arr := json['axes_dims'];

  zi_axes_dims[0] := arr[0];
  zi_axes_dims[1] := arr[1];
  zi_axes_dims[2] := arr[2];
  //zi_axes_dims[0] := 32;
  //zi_axes_dims[1] := 48;
  //zi_axes_dims[2] := 48;

  (* Derived values *)
  zi_latent_channels := zi_in_channels * zi_patch_size * zi_patch_size;
  text_dim := zi_cap_feat_dim;  (* 2560 for Z-Image *)


  hidden_size := zi_dim;
  is_non_commercial := false;  (* Z-Image is Apache 2.0 *)
  default_steps := 9;       (* 8 NFE = 9 scheduler steps *)
  default_guidance := 0.0; (* No CFG for Z-Image-Turbo *)
  is_distilled := true;        (* Treat as distilled (no CFG) *)
  modelArc := '6B';  (* Z-Image-Turbo is 6B *)
  if hidden_size<>3840 then
    if hidden_size > 3840  then
      modelArc := 'large'
    else
      modelArc := 'small';
  model_name := 'Z-Image-Turbo-'+modelArc;

  // transformer files will load later while inferencing
  if not fileExists(format('%s/transformer/config.json', [model_dir])) then
     assert(fileExists(format('%s/transformer/diffusion_pytorch_model.safetensors', [model_dir])), 'ERROR : Transformer model not found (missing config.json and safetensors)');
  rng_seed(getTickCount64());
end;

procedure TQNNZImage.loadVAE();
var
  modelArc:string;
  json: TJSON;
  sf:TSafeTensorFile;
begin
  modelArc := format('%s/vae/config.json', [model_dir]);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', false);
  json := TJSON.LoadFromFile(modelarc);
  vae_z_channels := json.get('latent_channels', QNN_VAE_Z_CHANNELS);
  vae_scaling    := json.get('scaling_factor', 0.0);
  vae_shift      := json.get('shift_factor', 0.0);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);

  modelArc:= format('%s/vae/diffusion_pytorch_model.safetensors',[model_dir]);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', false);
  sf := TSafetensorFile.open(modelArc);
  vae.load([sf], vae_z_channels, vae_scaling, vae_shift);
  sf.close();
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);
end;

function TQNNZImage.generate(const prompt: rawbytestring;
  params: TGenerateParams): TQNNImage;
var
  text_seq:longint;
  text_emb : TMemoryBlock;
begin
  params.guidance := 0;  // always zero or no?
  (* Encode text (Z-Image mode: extraction mode 1, single layer) *)
  text_emb := encodeText(prompt, text_seq);

  result := generateFromEmbeddings(text_emb, text_seq, params);
  text_emb.free

end;

function TQNNZImage.generateFromEmbeddings(const text_emb: TMemoryBlock;
  const text_seq: longint; params: TGenerateParams): TQNNImage;
var
  pre_h, pre_w, post_h, post_w, latent_ch, image_seq_len: longint;
  z, schedule, denoised, latent : TMemoryBlock;
begin

  (* Validate dimensions *)
  if params.width <= 0 then params.width := 1024;   (* Z-Image default: 1024x1024 *)
  if params.height <= 0 then params.height := 1024;
  if params.num_steps <= 0 then params.num_steps := default_steps;

  (* Ensure dimensions are divisible by 16 *)
  params.width := (params.width div 16) * 16;
  params.height := (params.height div 16) * 16;
  if params.width < 64 then params.width := 64;
  if params.height < 64 then params.height := 64;
  assert((params.width <= QNN_VAE_MAX_DIM) and (params.height <= QNN_VAE_MAX_DIM), 'Image size must be maximum of 1792 X 1792');

  (* Release text encoder to free memory before loading transformer *)
  qwen3Encoder.free;


  (* Load Z-Image transformer on-demand (persistent across generations). *)
  if assigned(phase_callback) then phase_callback('Loading Z-Image transformer', false);
  transformer.load(model_dir, zi_dim, zi_dim div 128, zi_n_layers, zi_n_refiner, zi_cap_feat_dim, zi_in_channels, zi_patch_size, zi_rope_theta, @zi_axes_dims[0], use_mmap);
  if assigned(phase_callback) then phase_callback('Loading Z-Image transformer', true);

  (* Z-Image latent dimensions:
   * The transformer works at pre-patchification: [in_ch, H/8, W/8]
   * where in_ch=16, and patchification happens inside the transformer.
   * VAE decode expects post-patchification: [latent_ch, H/16, W/16]
   * where latent_ch = in_ch * ps * ps = 64. *)
  pre_h := params.height div 8;    (* H/8: pre-patchification spatial *)
  pre_w := params.width div 8;
  post_h := pre_h div zi_patch_size;     (* H/16: post-patchification spatial *)
  post_w := pre_w div zi_patch_size;
  image_seq_len := post_h * post_w;

  (* Initialize noise at pre-patchification dimensions: [in_channels, H/8, W/8] *)
  if params.seed<0 then params.seed := randseed;

  z := init_noise(1, transformer.in_channels, pre_h, pre_w, params.seed);

  (* Get Z-Image schedule (default FlowMatch; linear/power if explicitly requested). *)
  if params.schedule=QNN_SCHEDULE_DEFAULT then params.schedule:=QNN_SCHEDULE_FLOWMATCH;
  schedule := selected_schedule(params, image_seq_len);

  (* Sample using Z-Image Euler method.
   * The transformer takes [in_channels, pre_h, pre_w] and returns same shape. *)
  denoised := transformer.sampleEuler(
      z, 1, {in_channels,} pre_h, pre_w,
      zi_patch_size,
      text_emb, text_seq,
      schedule, params.num_steps
  );
//denoised.printCompare(readTensor());
  transformer.free;
  z.free;
  schedule.free;
//denoised.printCompare(readTensor());
  (* Patchify transformer output for VAE decode:
   * [1, in_channels, H/8, W/8] -> [1, latent_ch, H/16, W/16]
   * where latent_ch = in_channels * zi_patch_size * zi_patch_size = 64 *)
  latent_ch := zi_in_channels * zi_patch_size * zi_patch_size;
  latent := TMemoryBlock.Create([latent_ch, post_h, post_w ], 'ZI_GENERATE_FROM_EMBEDDINGS_LATENT');
//printCompare(latent_ch*post_h*post_w, denoised, readTensor());
  QNNPatchify(latent, denoised, 1, zi_in_channels, pre_h, pre_w, zi_patch_size);
//printCompare(latent_ch*post_h*post_w, latent, readTensor());

  denoised.free;
  (* Decode latent to image *)
//printCompare(latent_ch* post_h* post_w, latent, readTensor());
  loadVAE();

  if assigned(phase_callback) then phase_callback('decoding image', false);
  result := vae.decode(latent, 1, post_h, post_w);
  if assigned(phase_callback) then phase_callback('decoding image', true);
  vae.free;
  latent.free;
end;

function TQNNZImage.encodeText(const prompt: rawbytestring; var out_seq_len: longint): TMemoryBlock;
var num_real_tokens:longint;
begin
  if model_dir<>'' then begin
    if assigned(phase_callback) then phase_callback('Loading Qwen3 encoder', false);
    qwen3Encoder.load(model_dir, use_mmap);
    out_seq_len:= QWEN3_MAX_SEQ_LEN;;
    if assigned(phase_callback) then phase_callback('Loading Qwen3 encoder', true);
  end;
  qwen3Encoder.model.setExtractionMode(is_zimage);
  if assigned(phase_callback) then phase_callback('encoding text', false);
  result := qwen3encoder.encodeText(prompt, num_real_tokens);
  if is_zimage then
      out_seq_len := num_real_tokens
  else
    out_seq_len := QWEN3_MAX_SEQ_LEN;
  if assigned(phase_callback) then phase_callback('encoding text', true);

end;

procedure TQNNZImage.free();
begin
  qwen3Encoder.free;
  transformer.free;
  vae.free;
end;

end.

