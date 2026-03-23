unit quicknn_tokenizer;
{$ifdef FPC}
  {$PackRecords C}
  {$mode delphi}
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

interface

uses
  SysUtils, quicknn_common;

type
  TQNNBPEMerge = record
      left, right, &result, priority : longint;
  end;

  { TQNNTokenizer }

  TQNNTokenizer = record
   private
      vocabsDic      : TVocabs;
   public
      (* Vocabulary *)
      vocabs         : TArray<RawByteString>;

      (* BPE merges *)
      merges         : TArray<TQNNBPEMerge>;

      (* Special token IDs *)
      pad_id, unk_id, bos_id, eos_id : longint;

      (* Configuration *)
      max_length     : longint;
      add_bos, add_eos :boolean;
      constructor Load(const fileName:string);
      function BPEEncodeWord(const str:RawByteString):TArray<longint>;
      function tokenize(const aText: RawByteString; maxLen: longint=0):TArray<longint>;
      function deTokenize(const tokens:TArray<longint>): RawByteString;
      procedure free();
  end;


implementation

function getListOfWords(const str:RawByteString):TArray<RawByteString>;
var i, start:NativeInt;
begin
  i:=1;
  start :=1;
  result :=nil;
  while i <= length(str) do begin
    case str[i] of
      TAB, LF, CR, SP, // skip white spaces
        '!'..'/', ':'..'@', '['..'`', '{'..'~': // insert punctuation
        begin
          inc(i);
          insert(copy(str, start, i-start), result, length(result));
          start := i;
        end;

      //TAB, LF, CR, SP : // skip white spaces
      //  begin
      //    inc(i);
      //    inc(start)
      //  end
      else
        begin
          while (i<=length(str)) and not (str[i] in WORD_SEPERATORS) do // a word found, iterate to get number of chars
            inc(i, UTF8CharSize(str[i]));
          insert(copy(str, start, i-start), result, length(result));
          start := i;
        end;
    end;
  end;

end;

{ TQNNTokenizer }

constructor TQNNTokenizer.Load(const fileName: string);
const TOK_MAGIC = 'FTOK';
var
  F:File;
  s:string;
  config : array[0..7] of longword;
  merge  : array[0..2] of longword;
  i : integer;
  len : word;
begin
  if not assigned(vocabsDic) then
    vocabsDic := TVocabs.create;
  assert(FileExists(filename),'ERROR : File not found '+fileName);
  assignfile(f, filename);
  reset(f, 1);
  try
    setLength(s, length(TOK_MAGIC));
    blockread(f, s[1], length(TOK_MAGIC));
    assert(s=TOK_MAGIC, 'ERROR : Invalid magic header');
    blockread(f, config, sizeof(config));
    pad_id     := config[2];
    unk_id     := config[3];
    bos_id     := config[4];
    eos_id     := config[5];
    max_length := config[6];
    add_bos    := boolean(config[7] and 1);
    add_eos    := boolean((config[7] shr 1) and 1);
    vocabsDic.Clear;
    SetLength(vocabs, config[0]);
    SetLength(merges, config[1]);
    for i := 0 to high(vocabs) do begin
      blockread(f, len, sizeof(len));
      setLength(vocabs[i], len);
      blockRead(f, vocabs[i][1], len);
      vocabsDic.AddOrSetValue(vocabs[i], i);
    end;

    for i:=0 to high(merges) do begin
      blockread(f, merge, sizeOf(merge));
      merges[i].left     := merge[0];
      merges[i].right    := merge[1];
      merges[i].&result  := merge[2];
      merges[i].priority := i;

    end;

  finally
    closeFile(f);
  end
end;

function TQNNTokenizer.BPEEncodeWord(const str: RawByteString): TArray<longint>;
var i, m, n, id, len, best_idx, best_priority : longint;
  s: RawByteString;
  changed: boolean;
begin
   assert(assigned(vocabsDic),'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
   setLength(result, length(str));
   i := 1;
   n := 0;
   while i<=Length(str) do begin
     len := UTF8CharSize(str[i]);
     s := copy(str, i, len);
     if vocabsDic.TryGetValue(s, id) then begin
       result[n] := id;
       inc(n);
     end else begin
       for m:=1 to len do begin
         if not vocabsDic.TryGetValue(format('<0x%.2X>', [ord(s[m])]), id) then
           id := unk_id;
         result[n] := id;
         inc(n);
       end;
     end;
     inc(i, len);
   end;

   changed := true;
   while changed and (n>1) do begin
     changed := false;
     best_idx := -1;
     best_priority := length(merges)+1;
     for i := 0 to n-2 do
       for m := 0 to high(merges) do
         if (merges[m].left = result[i]) and (merges[m].right = result[i+1]) then
           if merges[m].priority<best_priority then begin
             best_priority := merges[m].priority;
             best_idx := i
           end;
     if best_idx>=0 then
       for m:=0 to high(merges) do
         if (merges[m].left=result[best_idx]) and (merges[m].right=result[best_idx+1]) then begin
           result[best_idx] := merges[m].&result;
           // shift result left
           for i := best_idx+1 to n-2 do
             result[i] := result[i+1];
            dec(n);
            changed := true;
            break
         end;
   end;
   setlength(result, n)

end;

function ifthen(const cond:boolean; const ifTrue, ifFalse:longint):longint;
begin
  if cond then result:= ifTrue else result := iffalse
end;

function TQNNTokenizer.tokenize(const aText: RawByteString; maxLen: longint): TArray<longint>;
var
  words : TArray<RawByteString>;
  word_tokens : TArray<longint>;
  total, w, i : longint;

  procedure appendToken(const token:longint);
  begin
    insert(token, result, length(result));
    inc(total);
  end;

begin
  assert(assigned(vocabsDic),'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  result := nil;
  if maxLen=0 then maxLen := max_length;

  words := getListOfWords(aText);
  total := 0;

  if add_bos and (bos_id>=0) then
    appendToken(bos_id);

  w := 0;
  while (w < length(words)) and (length(result)<maxLen - ifthen(add_eos, 1, 0)) do begin
    word_tokens := BPEEncodeWord(words[w]);
    i := 0;
    while (i < length(word_tokens)) and (length(result)<maxLen - ifthen(add_eos, 1, 0)) do begin
      appendToken(word_tokens[i]);
      inc(i)
    end;

    inc(w)
  end;

  if add_eos and (eos_id >= 0) and (length(result) < maxLen) then
    appendToken(eos_id);

end;

function TQNNTokenizer.deTokenize(const tokens: TArray<longint>): RawByteString;
var
  i, j, id, total_len: Integer;
begin
  assert(assigned(vocabsDic),'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  result :='';
  if not assigned(tokens) then exit;
  total_len := 0;
  for i:=0 to high(tokens) do begin
    id := tokens[i];
    if (id>=0) and (id<length(vocabs)) then
      inc(total_len, length(vocabs[id]))
  end;
  setLength(result, total_len);

  j:=1;

  for i:=0 to high(tokens) do begin
    id := tokens[i];
    if (id>=0) and (id<length(vocabs)) then begin
      if (id = bos_id) or (id=eos_id) or (id=pad_id) then continue;
      move(vocabs[id][1], result[j], length(vocabs[id]));
      inc(j, length(vocabs[id]))
    end;
  end;

  setLength(result, j-1)
end;

procedure TQNNTokenizer.free();
begin
  freeAndNil(vocabsDic);
  vocabs := nil;
  merges := nil
end;

function createSimpleTokenizer():TQNNTokenizer;
var i:longint;
  buf : RawByteString;
begin

  result.pad_id := 256;
  result.unk_id := 257;
  result.bos_id := 258;
  result.eos_id := 259;
  result.max_length := QNN_MAX_SEQ_LEN;
  result.add_bos := true;
  result.add_eos := true;

  setLength(result.vocabs, 256 + 4);
  result.vocabsDic := TVocabs.create;
  (* Create vocabulary *)
  for i := 0 to 255 do begin

    if (i >= 32) and ( i < 127) then
      buf := ansichar(i)
    else
      buf := format('<0x%.2X>', [i]);

     result.vocabs[i] := buf;
     result.vocabsDic.TryAdd(result.vocabs[i], i);
  end;
  result.vocabs[256] := '<pad>';
  result.vocabs[257] := '<unk>';
  result.vocabs[258] := '<bos>';
  result.vocabs[259] := '<eos>';

  result.vocabsDic.tryAdd('<pad>', 256);
  result.vocabsDic.tryAdd('<unk>', 257);
  result.vocabsDic.tryAdd('<bos>', 258);
  result.vocabsDic.tryAdd('<eos>', 259);

end;

// const
//  TXT_ANSI = 'the quick brown fox jumps over the lazy dog '; // ansi
//  TXT_UTF8  ='أبجد هوز حطي كلمن';                            // utf-8
//var t: TQNNTokenizer;
//    tokens : TArray<longint>;
//    s : RawByteString;
initialization

  //t := createSimpleTokenizer();
  //
  //tokens := t.tokenize(TXT_ANSI+TXT_UTF8);
  //s := t.deTokenize(tokens);
  //t.free;
  //writeln(s);
  //readln

finalization


end.

