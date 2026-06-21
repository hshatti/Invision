unit quicknn_flux;

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
  SysUtils, math, quicknn_common, quicknn_transformers, quicknn_qwen3, quicknn_vae,safetensor, quickjson
  {$ifdef FPC}
  {$else}
  , windows
  {$endif}
  ;

type

  //TQNNFlux = type TQNNCtx;

  { TQNNFlux }

  TQNNFlux = record
    //tokenizer : TQNNTokenizer;
    qwen3Encoder : TQWEN3Encoder;
    vae : TVAE;
    transformer : TTransformerFlux;

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
    function generateLatent(const text_emb: TMemoryBlock; const text_seq, height, width, num_steps: longint; const seed:int64; const progress_callback:TStepCallback):TMemoryBlock;
    function generate(const prompt: rawbytestring; params: TGenerateParams): TQNNImage;
    function encodeText(const prompt: rawbytestring; var out_seq_len: longint): TMemoryBlock;
    procedure free();
  end;

implementation
uses quicknn_sample, quicknn_kernels;

{ TQNNFlux }

constructor TQNNFlux.load(const aModel_dir: string;
  const OnStatus: TPhaseCallback);
var
  json : TJSON;
  modelArc : string;
  hidden_size : longint;
  sf : TSafeTensorFile;
begin
  self := default(TQNNFlux);
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
    assert(pos('flux',LowerCase(modelArc))=1, 'ERROR : Model is not a Flux pipeline');
    //is_zimage := (modelArc='ZImagePipeline') or (modelArc='Z-Image');
    is_distilled := json.get('is_distilled', false);
  end else begin // assuming default FLUX.2-klein-4b
    is_distilled := true;
  end;
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);

  modelArc := format('%s/transformer/config.json', [model_dir]);
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', false);
  json := TJSON.LoadFromFile(modelArc);
  num_heads := json.get('num_attention_heads', 24);    // default 4B
  text_dim  := json.get('joint_attention_dim', 7680);  // default 4B: 3 * 2560
  hidden_size  := num_heads * 128;

  vae_scaling := 0;
  vae_shift   := 0;
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);

  is_non_commercial :=  hidden_size>3072; // shouldn't it the opposite? mans it's commercial if above 3072 ?
  if is_non_commercial then
     model_name     := '9B'
  else
    model_name := '4B';
  if is_distilled then begin
    default_steps := 4;
    default_guidance := 1.0;
    model_name := 'FLUX.2-klein-'+model_name
  end else begin
    default_steps := 50;
    default_guidance := 4.0;
    model_name := 'FLUX.2-klein-base-'+model_name;
  end;

  // transformer files will load later while inferencing
  if not fileExists(format('%s/transformer/config.json', [model_dir])) then
     assert(fileExists(format('%s/transformer/diffusion_pytorch_model.safetensors', [model_dir])), 'ERROR : Transformer model not found (missing config.json and safetensors)');
  rng_seed(getTickCount64());


end;

procedure TQNNFlux.loadVAE();
var
  modelArc: String;
  json: TJSON;
  sf: TSafeTensorFile;
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
  vae.useMMap:=use_mmap;
  vae.load([sf], vae_z_channels, vae_scaling, vae_shift);
  sf.close();
  if assigned(phase_callback) then phase_callback('Loading ['+modelArc+']', true);
end;

function TQNNFlux.generateLatent(const text_emb: TMemoryBlock;
  const text_seq, height, width, num_steps: longint; const seed: int64;
  const progress_callback: TStepCallback): TMemoryBlock;
var
  latent_h, latent_w, channels: longint;
  z, schedule : TMemoryBlock;
begin
  (* Compute latent dimensions *)
  latent_h := height div 16;
  latent_w := width div 16;
  channels := QNN_LATENT_CHANNELS;

  (* Initialize noise *)
  z := init_noise(1, channels, latent_h, latent_w, seed);

  (* Get schedule (4 steps for klein distilled) *)
  schedule := schedule_linear(num_steps);

  (* Sample (FLUX.2-klein is guidance-distilled, no CFG needed) *)
  result := transformer.sampleEuler(z, 1, channels, latent_h, latent_w,
                                    text_emb, text_seq,
                                    schedule, num_steps,
                                    progress_callback);

  z.free;
  schedule.free;

end;

//type TSingleArray= array of single;
//    TFloatArray = array[0..MaxLongint div 32] of single;
//    PFloatArray = ^TFloatArray;

function TQNNFlux.generate(const prompt: rawbytestring; params: TGenerateParams): TQNNImage;
var
  text_emb, text_emb_uncond : TMemoryBlock;
  latent_h, latent_w, text_seq, image_seq_len, text_seq_uncond : longint;
  z, schedule, latent : TMemoryBlock;
begin
  if params.width<=0 then params.width := QNN_DEFAULT_WIDTH;
  if params.height<=0 then params.height := QNN_DEFAULT_HEIGHT;
  if params.num_steps<=0 then params.num_steps:= default_steps;
  if params.guidance<=0 then params.guidance:= default_guidance;
  params.width  := max(16 * params.width div 16, 64);
  params.height := max(16 * params.height div 16, 64);
  assert((params.width<=QNN_VAE_MAX_DIM) and (params.height<=QNN_VAE_MAX_DIM), 'ERROR : Image maximum dimensions exceeded (1792x1792)!') ;
  text_emb := encodeText(prompt, text_seq);

  if not is_distilled then
     text_emb_uncond := encodeText('', text_seq_uncond);
  qwen3Encoder.free;

  latent_h := params.height div 16;
  latent_w := params.width div 16;
  image_seq_len := latent_h*latent_w;

  if params.seed <= 0 then
     params.seed := GetTickCount64();
  if params.schedule=QNN_SCHEDULE_DEFAULT then params.schedule:=QNN_SCHEDULE_SIGMOID;

  z :=  init_noise(1, QNN_LATENT_CHANNELS, latent_h, latent_w, params.seed);
  schedule := selected_schedule(params, image_seq_len);

  if not transformer.isLoaded() then begin
     if assigned(phase_callback) then
        phase_callback('Loading FLUX.2 transformer', false);
     transformer := TTransformerFlux.load(model_dir, use_mmap);
     if assigned(phase_callback) then
        phase_callback('Loading FLUX.2 transformer', true);
  end;
  if is_distilled then
      latent := TTransformerFlux(transformer).sampleEuler(z, 1, QNN_LATENT_CHANNELS, latent_h, latent_w, text_emb, text_seq, schedule, params.num_steps, nil {step callback})
  else
      latent := TTransformerFlux(transformer).sampleEuler(z, 1, QNN_LATENT_CHANNELS, latent_h, latent_w, text_emb, text_seq, text_emb_uncond, text_seq_uncond, params.guidance, schedule, params.num_steps, nil {step callback});

  z.free;
  schedule.free;
  text_emb.free;
  text_emb_uncond.free;
  transformer.free;

  loadVAE();
  result := vae.decode(latent, 1, latent_h, latent_w);
  vae.free;

  latent.free;
  // todo continue fro here with sampeling



end;

function TQNNFlux.encodeText(const prompt: rawbytestring; var out_seq_len: longint): TMemoryBlock;
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


procedure TQNNFlux.free();
begin
  qwen3Encoder.free;
  transformer.free;
  vae.free;
end;

type
  PArr = ^TArr;
  TArr = array[0..1000] of single;
var p : TGenerateParams;
    noise : TMemoryBlock;

    a : PQNNFloat;
initialization
  //p := default(TGenerateParams);
  //p.num_steps:=10000000;
  //p.schedule:=QNN_SCHEDULE_FLOWMATCH;
  //noise := init_noise(1, 3, 256, 256, 10);
  //a := noise;
  //printStat(noise, 1*3*256*256 );
  //readln

end.

