program Envision;

{$ifdef FPC}
  {$mode delphi}
{$endif}

uses
  {$ifndef MSWINDOWS}
  cthreads,
  {$endif}
  SysUtils, Generics.Collections, TypInfo
  , quicknn_common
  , quicknn_transformers
  , quicknn_vae
  , quicknn_flux
  , quicknn_zimage
  , sixel
  , termesc
  , nchrono ;

const memAllocated:IntPtr = 0;

var
  dict : TDictionary<string, TMemoryBlock>;
  flux:TQNNFlux;
  zimage : TQNNZImage;
  params:TGenerateParams;
  img, src : TQNNImage;
  imgs : TArray<TQNNImage>;
  imgFile: String;
  t : int64;

procedure afterblockForward(const typ :TSubstepType; const i, total: longint);
begin
  case typ of
    SUBSTEP_DOUBLE_BLOCK: write('D');
    SUBSTEP_SINGLE_BLOCK: if i mod 5 =0 then write('S');
    SUBSTEP_FINAL_LAYER: writeln('F [', ((clock-t) / CLOCKS_PER_SEC):1:3, '] Seconds');
  end;

  if i=0 then t:= clock();
end;

procedure previewProgress(const step, num_steps:longint; const latent:TMemoryBlock);
begin
  if isConsole then
    PVAE(vae_ptr).preview(latent, flux2_latent_rgb_proj, latent.height(), latent.width(), latent.channels, 1, flux2_latent_rgb_bias, 2)
      .printSixel(); // flux2 patchsize is 2   );
end;

procedure afterstep(const i, total:longint);
begin
  //if i=total-1 then
  //  write(#13' ',i+1,'/',total, ' ')
  //else
  if i=0 then begin
    write(#13#10' ',i+1,'/',total,' ')
  end
  else
    write(#13' ',i+1,'/',total,' ')
end;

procedure afterphase(const status:string; const done:boolean);
begin
  if done then begin
    writeln(' Finished [', ((clock-t) / CLOCKS_PER_SEC):1:3, '] Seconds')
  end
  else begin
    t := clock();
    write(status, '...', tab)
  end;
end;

procedure DEBUGAllMems();
var m : TPair<string, TMemoryBlock>;
  sum:intPtr;
begin
  sum :=0;
  for m in dict do begin

    ////if m.value.name<>'' then
    writeln(m.value.name, ' = ', m.value.size*DATATYPE_BITS[m.Value.DataType] div 8);
    inc(sum, m.value.size*DATATYPE_BITS[m.Value.DataType] div 8)
  end;
  writeln('SUM MEM =', sum,' ALLOCATED =', memAllocated);
  if sum<>memAllocated then begin
    SysUtils.beep;
    readln;
  end;
end;

procedure memoryChange(const status:string; const old, new:IntPtr; const mem:TMemoryBlock);
var delta: IntPtr;
  o : IntPtr;
begin
  assert(trim(mem.name)<>'');
  saveCursorPos();
  delta := new - old;
  if not ((new>0) and (old>0)) then begin
    if not dict.ContainsKey(mem.name) then begin
      assert(status='new');
      dict.Add(mem.name, mem);
    end;
    if new=0 then begin
      assert(dict.ContainsKey(mem.name));
      dict.Remove(mem.name);
    end;
  end;
  //s := dict[mem.name].size;
  if (memAllocated+delta)<0 then
    readln;
  o := memAllocated;
  InterlockedExchangeAdd64(memAllocated, delta);
  //DEBUGAllMems();
  cursorAbsPos(100, 1);
  write(memAllocated/1000000000:1:3, ' GB');
  restorCursorPos();
end;

var
  l_w, l_h : longint;
  l : TMemoryBlock;
  PROMPT: String;
begin
  dict := TDictionary<string, TMemoryBlock>.create();
  onMemoryUpdate:=memoryChange;
  params := default(TGenerateParams);
  params.width :=512;
  params.height:=512;
  //params.seed:= 666;
  //params.seed:=1781898218;
  params.powerAlpha := 2;
  params.num_steps := 2;
  substep_callback:=afterblockForward;
  step_callback := afterstep;
  text_progress_callback:=afterstep;
  vae_progress_callback := afterstep;
  PROMPT := 'a cute realistic panda holding a "I will code for food!" signboard';
  //PROMPT := 'A Delphi with gargoyles on the top of it, a fron view of the whole delphi building';
  //PROMPT := 'cartimaphoble hembrashel';
  {$define _ZIMAGE}
  {$ifdef ZIMAGE}
  zimage := TQNNZImage.load('../../models/Z-Image-Turbo', afterphase);
  zimage.use_mmap:=true;
  img := zimage.generate(PROMPT, params);
  //img := zimage.generate('a realistic pink grisly bear wearing a blue hat holding a (I''m Sad) signboard.', params);
  //img := zimage.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);
  //img := zimage.generate('a cute pink raccoon holding a "I''m Sad" signboard', params);
  //img := zimage.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  //img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
  //img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
  //img := flux.generate('تفاحة', params);
  imgFile := GetCurrentDir()+ DirectorySeparator+FormatDateTime('YYYY_MM_DD_hhnnsszzz', Now())+ '_pascal.png';
  writeln('Saving to [', imgFile ,']');
  img.saveToFile(imgFile, 'Envision', '{"program" : "Invision , a (text to image/ image to image) generator example written in Object Pascal", "model" : "'+zimage.model_name+'"'+'"prompt" : "'+StringReplace(PROMPT, '"', '\"', [rfReplaceAll])+'", "seed" : '+IntToStr(params.seed)+'}');
  zimage.free;
  {$else}

  //src := TQNNImage.loadFromFile('bear.png');
  //src.print();
  //printSixel(pointer(src.data), src.width, src.height, true);

  // load the directory name containing models (download it first)
  // could be any of [FLUX.2-klein-9B, FLUX.2-klein-base-4B, FLUX.2-klein-base-9B, ...],
  flux := TQNNFlux.load('../../models/FLUX.2-klein-4B', afterphase);

  // i'm poor, be memory efficient :(
  flux.use_mmap := true;

  //img := flux.generate('a realistic pink grisly bear wearing a blue hat holding (will code for food) signboard.', params);
  //img := flux.generate('On a desk, a laptop displays a wallpaper of a countryside landscape with a small river. The landscape extends beyond the computer screen, invading the desk, the walls, and the room. The river overflows the screen, and plants and trees grow beyond its boundaries, blending seamlessly with the surrounding environment. The style is a stunning 3D rendering, with deep, cinematic lighting.', params);

  img := flux.generate(PROMPT, params, previewProgress);
  //img := flux.generate('a young lady with sunglasses, red hair and little frickles', params);
  //img := flux.generate('A mechanical dog made of brass gears and copper pipes, steampunk style, highly detailed.', params);
  //img := flux.generate('Tom and Jerry', params);
  //setLength(imgs, 2);
  //imgs[0] := TQNNImage.loadFromFile('c:\development\flux2.c\car.jpg');
  //imgs[1] := TQNNImage.loadFromFile('c:\development\flux2.c\beach.jpg');

  //img := flux.generate('A pink grizly bear wearing a blue fedora in a cozy room with green furniture and holding a sign with "I will code for  food!" signboard.', params);
  //img := flux.generate('een robotkonijn dat zingt in de ruimte', params);
  //img := flux.generate('أرنب روبوتي يغني في الفضاء', params);
  //img := flux.generate('أرنب روبوتي يتناول العشاء مع قطة', params);
  //img := flux.generate('تفاحة', params);
  imgFile := GetCurrentDir()+ DirectorySeparator+FormatDateTime('YYYY_MM_DD_hhnnsszzz', Now())+ '_pascal.png';
  writeln('Saving to [', imgFile ,']');
  img.saveToFile(imgFile, 'program', 'Envision, a (txt2img/img2img) generator example written in Object Pascal (Delphi and FPC)');
  img.addPngMeta(imgFile, 'json',
                          '{"model" : "'+flux.model_name+'"'+
                          ', "prompt" : "'+StringReplace(PROMPT, '"', '\"', [rfReplaceAll])+
                          '", "seed" : '+IntToStr(params.seed)+
                          ', "steps" : '+intToStr(params.num_steps)+
                          ', "schedueler" : "'+copy(GetEnumName(TypeInfo(TQNNSchedule), ord(params.schedule)), 5)+'"}');
  flux.free;
  {$endif}
  img.printSixel();

  img.free;
  //DEBUGAllMems();
  freeAndNil(dict);
  readln;
end.

