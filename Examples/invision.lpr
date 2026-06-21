program invision;

{$ifdef FPC}
  {$mode delphi}
{$endif}

uses
  safetensor, quicknn_kernels, quicknn_common, quicknn_tokenizer,
  quicknn_transformers, quicknn_vae, quicknn_sample,
  quicknn_qwen3
  , quicknn_flux
  , quicknn_zimage
  , sixel
  , termesc;

const memAllocated:IntPtr = 0;

var
  flux:TQNNFlux;
  zimage : TQNNZImage;
  params:TGenerateParams;
  img : TQNNImage;

procedure afterblockForward(typ :TSubstepType; i, total: longint);
begin
  case typ of
    SUBSTEP_DOUBLE_BLOCK: write('D');
    SUBSTEP_SINGLE_BLOCK: if i mod 5 =0 then write('S');
    SUBSTEP_FINAL_LAYER: writeln('F');

  end;
end;

procedure afterstep(i, total:longint);
begin
  //if i=total-1 then
  //  write(#13' ',i+1,'/',total, ' ')
  //else
  if i=0 then
    write(#13#10' ',i+1,'/',total,' ')
  else
    write(#13' ',i+1,'/',total,' ')

end;

procedure afterphase(const status:string; const done:boolean);
begin
  if done then
    writeln(' Finished')
  else
    write(status, '...', tab)
end;

procedure memoryChange(const status:string; const old, new:IntPtr);
begin
  saveCursorPos();
  memAllocated := memAllocated + new - old;
  cursorAbsPos(100, 1);
  write(memAllocated/1000000000:1:3, ' GB');
  restorCursorPos();
end;

begin
  params := default(TGenerateParams);
  params.width :=512;
  params.height:=512;
  params.seed:= 666;
  //params.seed:=1781898218;
  params.powerAlpha := 2;
  onMemoryUpdate:=memoryChange;
  substep_callback:=afterblockForward;
  step_callback := afterstep;
  text_progress_callback:=afterstep;
  vae_progress_callback := afterstep;

  {$define ZIMAGE}
  {$ifdef ZIMAGE}
  zimage := TQNNZImage.load('c:\development\flux2.c\Z-Image-Turbo', afterphase);
  zimage.use_mmap:=true;
  img := zimage.generate('a realistic pink grisly bear wearing a blue hat holding a (I''m Sad) signboard.', params);
  //img := zimage.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
  //img := zimage.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  //img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
  //img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
  //img := flux.generate('تفاحة', params);
  zimage.free;
  {$else}
  //flux := TQNNFLux.load('c:\development\flux2.c\FLUX.2-klein-9B');
  flux := TQNNFlux.load('c:\development\flux2.c\flux-klein-model', afterphase);
  flux.use_mmap := true;
  img := flux.generate('a realistic pink grisly bear wearing a blue hat holding "will code for food" signboard.', params);
  //img := flux.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
  //img := flux.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  //img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
  //img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
  //img := flux.generate('تفاحة', params);
  //flux.free;
  {$endif}
  writeln('');
  img.saveToFile('pascal.png');
  printSixel(pointer(img.data), img.width, img.height, true);

  img.free;
  readln;
end.

