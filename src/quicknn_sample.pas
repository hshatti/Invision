unit quicknn_sample;

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
  SysUtils, math, quicknn_common;

function schedule_linear(const num_steps:longint):TMemoryBlock;
function schedule_power(const num_steps:longint; const alpha:QNNFloat):TMemoryBlock;
function schedule_sigmoid(const num_steps:longint; const shift:QNNFloat):TMemoryBlock;
function schedule_resolution(const num_steps, height, width: longint):TMemoryBlock;
function compute_empirical_mu(const image_seq_len, num_steps: longint):single;
function generalized_time_snr_shift(const t, mu, sigma: QNNFloat):QNNFloat;
function schedule_flux(const num_steps, image_seq_len: longint):TMemoryBlock;
function schedule_zimage(const num_steps, image_seq_len:longint):TMemoryBlock;
function init_noise(const batch, channels, h, w:longint; const seed: int64):TMemoryBlock;
procedure rng_seed(seed:uint64);
function selected_schedule(const params:TGenerateParams; const  image_seq_len:longint):TMemoryBlock;

implementation

function schedule_linear(const num_steps:longint):TMemoryBlock;
var i:longint;
  resultPtr: PQNNFloat;
begin
    result := TMemoryblock.Create([num_steps + 1], 'SCHEDULER_LINEAR');
    resultPtr := result;
    for i := 0 to num_steps do
        resultPtr[i] := 1.0 - i/num_steps;
end;


(*
 * Power schedule: denser steps at the start (high noise), sparser at the end.
 * schedule[i] = 1 - (i/n)^alpha
 * alpha=1.0 is linear, alpha=2.0 is quadratic, etc.
 *)

function schedule_power(const num_steps:longint; const alpha:QNNFloat):TMemoryBlock;
var i:longint; resultPtr: PQNNFloat;
begin
    result := TMemoryBlock.Create([num_steps + 1], 'SCHEDULER_POWER');
    resultPtr:= result;
    for i := 0 to num_steps do
        resultPtr[i] := 1.0 - power(i/num_steps, alpha);
end;

(*
 * Shifted sigmoid schedule (better for flow matching)
 * shift controls where the inflection point is
 *)
function schedule_sigmoid(const num_steps:longint; const shift:QNNFloat):TMemoryBlock;
var
  i:longint;
  t,x : QNNFloat;
  resultPtr : PQNNFloat;
begin
    result := TMemoryBlock.Create([num_steps + 1], 'SCHEDULER_SIGMOID');
    resultPtr := result;
    for i := 0 to num_steps do begin
        t := i / num_steps;
        // Shifted sigmoid: more steps at the end
        x := (t - 0.5)*10.0 + shift;
        resultPtr[i] := 1.0 - 1.0/(1.0 + exp(-x));
    end;
    // Ensure endpoints
    resultPtr[0] := 1.0;
    resultPtr[num_steps] := 0.0;

end;

(*
 * Resolution-dependent schedule (as used in FLUX.2)
 * Higher resolutions use more steps at the start
*)
function schedule_resolution(const num_steps, height, width: longint):TMemoryBlock;
var
    i, pixels: longint;
    resultPtr: PQNNFloat;
    shift, t: QNNFloat;
begin
    result := TMemoryBlock.Create([num_steps+1], 'SCHEDULER_RESOLUTION');
    resultPtr := result;
    pixels := height*width;
    shift := 0.0;
    if pixels >=1024*1024 then
      shift := 1.0
    else if pixels >= 512*512 then
      shift := 0.5;
    for i := 0 to num_steps do begin
      t := power(i/num_steps, 1.0 + shift*0.5);
      resultPtr[i] := 1.0 - t
    end
end;

(*
 * FLUX.2 official schedule with empirical mu calculation
 * Matches Python's get_schedule() function from official flux2 code
 *
 * Compute the empirical shift parameter mu for the resolution-dependent
 * noise schedule. The constants a1, b1, a2, b2 are fitted from the Flux
 * training distribution and control how the SNR schedule adapts to different
 * image resolutions. Higher resolution images need more denoising steps
 * at high noise levels. Interpolates between two linear fits based on
 * step count, with a cutoff at 4300 tokens. *)

function compute_empirical_mu(const image_seq_len, num_steps: longint): single;
var
    a1, a2, b1, b2, m_200, m_10, a, b: QNNFloat;
begin
    a1 := 8.73809524e-05; b1 := 1.89833333;
    a2 := 0.00016927; b2 := 0.45666666;
    if image_seq_len > 4300 then
        exit(a2 * image_seq_len+b2);
    m_200 := a2 * image_seq_len+b2;
    m_10 := a1 * image_seq_len+b1;
    a := (m_200 - m_10) / 190.0;
    b := m_200 - 200.0 * a;
    exit(a * num_steps+b)
end;

(* Apply exponential SNR (Signal-to-Noise Ratio) shift to a timestep.
 * Maps t in [0,1] through t/(t + (1-t)*exp(-mu)), shifting the schedule
 * toward more time spent at higher noise levels. The boundary guards
 * (t<=0, t>=1) prevent division by zero. *)
function generalized_time_snr_shift(const t, mu, sigma: QNNFloat):QNNFloat;
begin
    if (t <= 0.0) then
        exit(0.0);
    if t >= 1.0 then
        exit(1.0);
    exit(exp(mu)/(exp(mu)+power(1.0/t - 1.0, sigma)))
end;

function schedule_flux(const num_steps, image_seq_len: longint): TMemoryBlock;
var
    resultPtr: PQNNFloat;
    t, mu: QNNFloat;
    i: longint;
begin
    result := TMemoryblock.Create([num_steps+1], 'SCHEDULER_FLUX');
    resultPtr := result;
    mu := compute_empirical_mu(image_seq_len, num_steps);
    for i := 0 to num_steps do begin
      t := 1.0 - i/num_steps;
      resultPtr[i] := generalized_time_snr_shift(t, mu, 1.0)
    end
end;

(*
 * Z-Image schedule: FlowMatchEulerDiscreteScheduler with static shift.
 *
 * Matches diffusers FlowMatchEulerDiscreteScheduler.set_timesteps()
 * for use_dynamic_shifting=False, invert_sigmas=False.
 *
 * Diffusers behavior (with num_train_timesteps=1000, shift=3.0):
 * 1) Build training sigmas from [1.0 .. 1/1000], then shift once.
 * 2) For inference, linspace from sigma_max to sigma_min (already shifted).
 * 3) Shift again.
 * 4) Append terminal sigma=0.
 *
 * Returns array of num_steps+1 sigma values.
 * The sampling loop uses these as sigmas directly with Euler:
 *   dt = sigma_next - sigma (negative for denoising)
 *   The transformer timestep = (1 - sigma), range [0, 1]
 *)

function schedule_zimage(const num_steps, image_seq_len:longint):TMemoryBlock;
var i:longint;
    resultPtr:PQNNFloat;
    shift, sigma_max, sigma_min, sigma_train_min, u, raw:QNNFloat;
begin
     result := TMemoryBlock.Create([num_steps + 1], 'SCHEDULER_ZIMAGE');
     resultPtr := result;
     shift := 3.0;
     sigma_max       := 1.0;
     sigma_train_min := 1.0 / 1000.0;  (* num_train_timesteps=1000 *)
     sigma_min       := shift * sigma_train_min / (1.0 + (shift - 1.0)*sigma_train_min);

     (* Diffusers set_timesteps(): linspace(sigma_max, sigma_min, num_steps),
      * then apply static shift one more time. *)
     for i := 0 to num_steps-1 do begin
        if num_steps > 1 then
          u := i / (num_steps - 1)
        else
          u:= 0.0;
         raw := sigma_max + (sigma_min - sigma_max) * u;
         resultPtr[i] := shift * raw / (1.0 + (shift - 1.0) * raw);
     end;
     (* Diffusers appends terminal sigma=0 (invert_sigmas=False). *)
     resultPtr[num_steps] := 0.0;
end;

var rng_state:array[0..3] of UInt64 = (
                                          UInt64($853c49e6748fea9b),
                                          UInt64($da3e39cb94b95bdb),
                                          UInt64($647c4677a2884327),
                                          UInt64($c6e7918d2e2969f5)
                                        );

function rotl(const x:UInt64; const k:longint):UInt64; inline;
begin
    result := (x shl k) or (x shr (64 - k))
end;

{$R-} // disable range check
{$Q-} // disable arithmetic overflow check
function xoshiro256ss(): UInt64;
var t : UInt64;
begin
  result := rotl(rng_state[1] * 5, 7) * 9;
  t := rng_state[1] shl 17;
  rng_state[2] :=rng_state[2] xor rng_state[0];
  rng_state[3] :=rng_state[3] xor rng_state[1];
  rng_state[1] :=rng_state[1] xor rng_state[2];
  rng_state[0] :=rng_state[0] xor rng_state[3];
  rng_state[2] :=rng_state[2] xor t;
  rng_state[3] := rotl(rng_state[3], 45);
end;

procedure rng_seed(seed:uint64);
var i:longint;
    z: UInt64;
begin
  (* SplitMix64 to initialize state from seed *)
  for i := 0 to 3 do begin
      seed := seed + UInt64($9e3779b97f4a7c15);
      z := seed;
      z := (z xor (z shr 30)) * UInt64($bf58476d1ce4e5b9);
      z := (z xor (z shr 27)) * UInt64($94d049bb133111eb);
      rng_state[i] := z xor (z shr 31);
  end;
end;
{$Q+}
{$R+}


function rand:single;
begin
  result :=  (xoshiro256ss() shr 11) * (1.0 / 9007199254740992.0);
end;

function random_normal:single;
var u1, u2:QNNFloat;
begin
  (* Box-Muller transform *)
  u1 := rand();
  u2 := rand();
  (* Avoid log(0) *)
  while u1 = 0.0 do u1 := rand();
  result := sqrt(-2.0 * ln(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
end;

procedure randn(const dst:PQNNFloat; const N:longint);
var i:longint;
begin
  for i:=0 to N-1 do dst[i]:=random_normal();
end;

function init_noise(const batch, channels, h, w:longint; const seed: int64):TMemoryBlock;
const NOISE_MAX_LATENT_DIM = 1792 div 16 ; (* 1792/16 = 112 *)
var
    target_size, sy, sx, src_idx, dst_idx, b, ty, tx, max_h, max_w, max_size, c: longint;
    maxNoise : TMemoryBlock;
    resultPtr, maxNoisePtr :PQNNFloat;
begin
    target_size := batch * channels * h * w;
    result := TMemoryBlock.Create([batch, channels, h, w], 'INIT_NOISE_RESULT');
    resultPtr := result;

    if seed >= 0 then
       rng_seed(seed);


    (* If target is max size or larger, just generate directly *)
    if (h >= NOISE_MAX_LATENT_DIM) and (w >= NOISE_MAX_LATENT_DIM) then begin
        randn(result, target_size);
        exit;
    end;

    (* Generate result at max latent size *)
    max_h := NOISE_MAX_LATENT_DIM;
    max_w := NOISE_MAX_LATENT_DIM;
    max_size := batch * channels * max_h * max_w;
    maxNoise := TMemoryBlock.Create([max_size], 'INIT_NOISE_MAX_NOISE');
    maxNoisePtr := maxNoise;
    randn(maxNoisePtr, max_size);

    (* Subsample to target size using nearest-neighbor *)
    for b := 0 to batch-1 do
      for c := 0 to channels-1 do
        for ty := 0 to h-1 do
          for tx := 0 to w-1 do begin
            (* Map target position to source position *)
            sy := ty * max_h div h;
            sx := tx * max_w div w;

            src_idx := ((b * channels + c) * max_h + sy) * max_w + sx;
            dst_idx := ((b * channels + c) * h + ty) * w + tx;
            resultPtr[dst_idx] := maxNoisePtr[src_idx];
          end;
    maxNoise.free;
end;

function selected_schedule(const params:TGenerateParams; const  image_seq_len:longint):TMemoryBlock;
begin
  case params.schedule of
    QNN_SCHEDULE_LINEAR:    exit(schedule_linear(params.num_steps));
    QNN_SCHEDULE_POWER:     exit(schedule_power(params.num_steps, params.powerAlpha));
    QNN_SCHEDULE_FLOWMATCH: exit(schedule_zimage(params.num_steps, image_seq_len));
    QNN_SCHEDULE_SIGMOID:   exit(schedule_flux(params.num_steps, image_seq_len));
  else
    exit(schedule_flux(params.num_steps, image_seq_len));
  end;
end;


initialization

end.

