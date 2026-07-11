program invision;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  safetensor, quicknn_kernels, quicknn_common, quicknn_tokenizer,
  quicknn_transformers, quicknn_vae, quicknn_sample,
  quicknn_qwen3, quicknn_flux, quicknn_zimage, sixel, quicknn_downloader;

var
  flux:TQNNFLux;
  zimage : TQNNZImage;
  params:TGenerateParams;
  img, src : TQNNImage;
  imgs : TArray<TQNNImage>;

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
  //params.seed:= 666;
  //params.seed:=1781898218;
  params.powerAlpha := 2;
  substep_callback:=afterblockForward;
  step_callback := afterstep;
  text_progress_callback:=afterstep;
  vae_progress_callback := afterstep;
  phase_callback := afterphase;

  {.$define ZIMAGE}
  {$ifdef ZIMAGE}
  zimage := TQNNZImage.load('c:\development\flux2.c\Z-Image-Turbo', afterphase);
  zimage.use_mmap:=true;
  img := zimage.generate('A pink grizly bear wearing a blue fedora in a cozy room with green furniture and holding a sign with "I will code for  food!" signboard.', params);
  //img := zimage.generate('a realistic pink grisly bear wearing a blue hat holding a (I''m Sad) signboard.', params);
  //img := zimage.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);
  //img := zimage.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
  //img := zimage.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  //img := flux.generate('√—‰» —Ê»Ê Ì Ì€‰Ì ›Ì «·›÷«¡', params);
  //img := flux.generate('√—‰» —Ê»Ê Ì Ì ‰«Ê· «·⁄‘«¡ „⁄ ﬁÿ…', params);
  //img := flux.generate(' ›«Õ…', params);
  zimage.free;
  {$else}
  src := TQNNImage.loadFromFile('c:\development\flux2.c\bear.png');
  printSixel(pointer(src.data), src.width, src.height, true);
  //flux := TQNNFLux.load('c:\development\flux2.c\FLUX.2-klein-9B');
  flux := TQNNFlux.load('c:\development\flux2.c\flux-klein-model', afterphase);
  flux.use_mmap := true;
  //img := flux.generate('a realistic pink grisly bear wearing a blue hat holding (will code for food) signboard.', params);
  //img := flux.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);

  //img := flux.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
//  img := flux.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  img := flux.generate('Make the bear wear a headset', params, src);
//  setLength(imgs, 2);
//  imgs[0] := TQNNImage.loadFromFile('c:\development\flux2.c\car.jpg');
//  imgs[1] := TQNNImage.loadFromFile('c:\development\flux2.c\beach.jpg');

//  img := flux.generate('A pink grizly bear wearing a blue fedora in a cozy room with green furniture and holding a sign with "I will code for  food!" signboard.', params);
//  img := flux.generate('Latters has eyes, noses and mouths, The letter "A" dancing and singling with the letter "Z" in the party of letters', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
//  img := flux.generate(UTF8String('√—‰» —Ê»Ê Ì Ì€‰Ì ›Ì «·›÷«¡'), params);
//  img := flux.generate(UTF8String('√—‰» —Ê»Ê Ì Ì ‰«Ê· «·⁄‘«¡ „⁄ ﬁÿ…'), params);
//  img := flux.generate(UTF8String(' ›«Õ…'), params);
  flux.free;
  {$endif}
  writeln('');
  img.saveToFile('pascal.png');

  printSixel(pointer(img.data), img.width, img.height, true);
  img.free;
  readln;
end.
