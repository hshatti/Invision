program invision;

{$ifdef FPC}
  {$mode delphi}
{$endif}

uses
  safetensor, quicknn_kernels, quicknn_common, quicknn_tokenizer,
  qwen3_tokenizer, quicknn_transformers, quicknn_vae, quicknn_sample,
  quicknn_qwen3, quicknn_flux, sixel;

var
  flux:TQNNFLux;
  params:TGenerateParams;
  img : TQNNImage;

procedure afterblockForward(typ :TSubstepType; i, total: longint);
begin
  case typ of
    SUBSTEP_DOUBLE_BLOCK: write('D');
    SUBSTEP_SINGLE_BLOCK: write('S');
    SUBSTEP_FINAL_LAYER: writeln('F');

  end;
end;

procedure afterstep(i, total:longint);
begin
  writeln(' ',i,'/',total)
end;

procedure afterphase(const status:string; const done:boolean);
begin
  if done then
    writeln(' Finished')
  else
    write(status, '...')
end;

begin
  flux.use_mmap:=true;
  flux := TQNNFLux.load('c:\development\flux2.c\flux-klein-model');
  params := default(TGenerateParams);
  params.width:=1280;
  params.width:=720;

  substep_callback:=afterblockForward;
  step_callback := afterstep;
  vae_progress_callback := afterstep;
  phase_callback := afterphase;

  img := flux.generate('an image of duffy duck drinking mate tea.', params);
  printSixel(pointer(img.data), img.width, img.height, true);


  readln;
end.

