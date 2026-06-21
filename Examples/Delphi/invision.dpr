program invision;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  safetensor, quicknn_kernels, quicknn_common, quicknn_tokenizer,
  quicknn_transformers, quicknn_vae, quicknn_sample,
  quicknn_qwen3, quicknn_flux, quicknn_zimage, sixel;

var
  flux:TQNNFLux;
  zimage : TQNNZImage;
  params:TGenerateParams;
  img : TQNNImage;

procedure afterblockForward(typ :TSubstepType; i, total: longint);
begin
  case typ of
    SUBSTEP_DOUBLE_BLOCK: write('D');
    SUBSTEP_SINGLE_BLOCK: if i mod 4=0 then write('S');
    SUBSTEP_FINAL_LAYER: writeln('F');

  end;
end;

procedure afterstep(i, total:longint);
begin
  if i= total-1 then
    writeln(#13' ',i+1,'/',total)
  else if i=0 then
    write(#13#10' ',i+1,'/',total)
  else
    write(#13' ',i+1,'/',total)

end;

procedure afterphase(const status:string; const done:boolean);
begin
  if done then
    writeln(' Finished')
  else
    write(status, '...')
end;

begin
  params := default(TGenerateParams);
  params.width :=512;
  params.height:=512;
  substep_callback:=afterblockForward;
  step_callback := afterstep;
  text_progress_callback:=afterstep;
  vae_progress_callback := afterstep;
  phase_callback := afterphase;

//  flux := TQNNFLux.load('c:\development\flux2.c\flux-klein-model');
//  flux.use_mmap:=true;
//////  flux := TQNNFLux.load('c:\development\flux2.c\FLUX.2-klein-9B');
////  img := flux.generate(utf8string('تفاحة'), params);
////  img := flux.generate('a pink cute rabit wearing a blue hat.', params);
////  img := flux.generate(utf8string('تفاحة'), params);
//  img := flux.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);

  zimage := TQNNZImage.load('c:\development\flux2.c\Z-Image-Turbo');
  zimage.use_mmap:=true;
////  img := zimage.generate(utf8string('تفاحة'), params);
  img := zimage.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  // cast the prompt to utf8string for international non english prompts
  printSixel(pointer(img.data), img.width, img.height, true);
  img.free;
  zimage.free;
//  flux.free;
  readln;
end.
