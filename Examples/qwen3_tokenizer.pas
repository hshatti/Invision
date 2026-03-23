unit qwen3_tokenizer;

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
  SysUtils, quicknn_common, quickjson;

type
  TQNNBPEMerge = record
      left  : string;
      right : string;
      rank  : longint;  (* Lower rank = higher priority (merge first) *)
  end;

  { TQNNTokenizer }

  TQNNTokenizer = record
      (* Vocabulary: id -> token string *)
      vocabs    : TArray<string>;

      (* Hash table: token string -> id *)
      vocabsDic : TVocabs;

      (* BPE merges *)
      merges    : TArray<TQNNBPEMerge>;

      (* Merge rank lookup: "left right" -> rank *)
      merge_ranks :TVocabs;
      constructor load(const jsonFilename:string);
      function getToken(const id:longint):string;
      function getId(const token:string):longint;
      function getMergeRank(const left, right:string) : longint;
      function BPEEncodeWord(const str:string):TArray<string>;
      function tokenize(const str:string; max_len:longint=0):TArray<longint>;
      function tokenizeChat(const prompt:string; const skipThinkTags:boolean=false; max_len:longint=0):TArray<longint>;
      function deTokenize(const tokens: TArray<longint>): string;
      procedure free;

      class operator initialize({$ifdef fpc}var{$else}out{$endif} val : TQNNTokenizer);
  end;
  function padTokens(const tokens:TArray<longint>; const max_len:longint; const attentionMask : TArray<longint>):TArray<longint>;

implementation

const
  QWEN3_MAX_SEQ_LEN = 512;
  QWEN3_PAD_ID = 151643      ;(* <|endoftext|> *)
  QWEN3_IM_START_ID = 151644 ;(* <|im_start|> *)
  QWEN3_IM_END_ID = 151645   ;(* <|im_end|> *)
  QWEN3_THINK_START_ID = 151667 ;(* <think> *)
  QWEN3_THINK_END_ID = 151668   ;(* </think> *)


var
  byteToUnicode : array [0..255] of longint;
  UnicodeToByte : array [0..511] of longint;

const
  byteEncoderInitializd : boolean = false;

procedure initByteEncoder();
var
    i,offset: longint;
begin
    if byteEncoderInitializd then
        exit();
    fillchar(byteToUnicode, sizeof(byteToUnicode), 0);
    fillchar(UnicodeToByte, sizeof(UnicodeToByte), 0);
    for i := 33 to 126 do begin
      byteToUnicode[i] := i;
      UnicodeToByte[i] := i
    end;
    for i := 161 to 172 do begin
      byteToUnicode[i] := i;
      UnicodeToByte[i] := i
    end;
    for i := 174 to 255 do begin
      byteToUnicode[i] := i;
      UnicodeToByte[i] := i
    end;
    offset := 256;
    for i := 0 to 255 do
      if (byteToUnicode[i] = 0) and (i <> 33) then begin
        byteToUnicode[i] := offset;
        UnicodeToByte[offset] := i;
        inc(offset)
      end;
    byteToUnicode[0] := 256;
    UnicodeToByte[256] := 0;
    byteEncoderInitializd := true
end;

function encodeByteToUTF8(const b: byte; var str: string):longint;
var
    cp: longint;
begin
    initByteEncoder();
    cp := byteToUnicode[b];
    if cp < 128 then begin
      setLength(str, length(str)+1);
      str[length(str)] := char(cp);
      exit(1)
    end
    else if cp < 2048 then begin
      setLength(str, length(str)+2);
      str[length(str)-1] := char(($C0 or (cp shr 6)));
      str[length(str)  ] := char(($80 or (cp and $3F)));
      exit(2)
    end;
    //
    setLength(str, length(str)+1);
    str[length(str)] := '?';
    result := 1
end;

function padTokens(const tokens: TArray<longint>; const max_len: longint;
  const attentionMask: TArray<longint>): TArray<longint>;
var i:longint;
begin
  setLength(result, max_len);

  for i:=0 to high(result) do begin
    if i<length(tokens) then begin
      result[i] := tokens[i];
      if assigned(attentionMask) then
          attentionMask[i]:=1;
    end else begin
      result[i] := QWEN3_PAD_ID;
      if assigned(attentionMask) then
          attentionMask[i]:=0;
    end;
  end;
end;

{ TQNNTokenizer }

constructor TQNNTokenizer.load(const jsonFilename: string);
var json, vocab, merge, addedTokens : TJSON;
    i, id: longint;
    content:string;
begin
  if not assigned(vocabsDic) then
      vocabsDic := TVocabs.Create;
  if not assigned(merge_ranks) then
      merge_ranks := TVocabs.Create;
  vocabsDic.Clear;
  merge_ranks.clear;
  json := TJSON.LoadFromFile(jsonFilename);
  assert(json.keyExist('model'), 'ERROR : ['+jsonFilename+'] has no "model" section!');
  assert(json['model'].keyExist('vocab'), 'ERROR : ['+jsonFilename+'] has no "vocab" section!');
  assert(json['model'].keyExist('merges'), 'ERROR : ['+jsonFilename+'] has no "merges" section!');
  vocab := json['model']['vocab'];
  merge := json['model']['merges'];
  addedTokens := json['added_tokens'];
  setLength(vocabs, vocab.count()+addedTokens.count);// make space for added tokens
  setLength(merges, merge.count());// make space for added tokens
  for i:=0 to vocab.count-1 do begin
      id := vocab[i].value;
      vocabs[id] := vocab[i].name;
      vocabsDic.TryAdd(vocab[i].name, id);
  end;

  for i:=0 to high(merges) do begin
    merges[i].left  := merge[i][0].value;
    merges[i].right := merge[i][1].value;
    merges[i].rank := i;
    if (merges[i].left<>'') and (merges[i].right<>'') then begin
        merge_ranks.TryAdd(merges[i].left + ' ' + merges[i].right, i);
    end;
  end;

  for i:=0 to addedTokens.count-1 do begin
    id := addedTokens[i]['id'].Value;
    content := addedTokens[i]['content'];
    if (content='') or (id<0) or (id>=length(vocabs)) then continue; // skip IDs that has no allocated position
    vocabs[id] := content;
    vocabsDic.TryAdd(vocabs[id], id);
  end;
end;

function TQNNTokenizer.getToken(const id: longint): string;
begin
  result := '';
  if assigned(vocabs) and (id>=0) and (id<length(vocabs)) then
    result := vocabs[id]
end;

function TQNNTokenizer.getId(const token: string): longint;
begin
  if not assigned(vocabsDic) or (token='') then exit(-1);
  if not vocabsDic.TryGetValue(token, result) then
      result := -1;
end;

function TQNNTokenizer.getMergeRank(const left, right: string): longint;
begin
  assert(assigned(merge_ranks), 'ERROR : tokenizer is empty, load tokenizer file 1st!');
  if not merge_ranks.TryGetValue(left+' '+right, result) then result := -1;
end;

function TQNNTokenizer.BPEEncodeWord(const str: string): TArray<string>;
  procedure appendToken(const a:string);
  begin
    insert(a, result, length(result));
  end;

var i, m, n, id, len, best_idx, best_rank, rank : longint;
  s: string;
  changed: boolean;

begin
  assert(assigned(vocabsDic),'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  result := nil;
  i := 1;
  n := 0;

  while i<=Length(str) do begin
    len := UTF8CharSize(str[i]);
    //s := copy(str, i, len);
    appendToken(copy(str, i, len));
    inc(i, len);
  end;

  changed := true;
  while (changed) do begin
      changed := false;

      (* Find best merge (lowest rank) *)
      best_rank := length(merges) + 1;


      for i:=0 to high(result)-1 do begin
          rank := getMergeRank(result[i], result[i+1]);
          if (rank >= 0) and (rank < best_rank) then begin
              best_rank := rank;
              best_idx := i;
              changed := true
          end
      end;
      if not changed then break;
      (* Apply best merge *)
      (* Merge left and right at best_idx*)
      s := result[best_idx]+result[best_idx+1];
      result[best_idx] := s;
      delete(result, best_idx+1, 1)
  end ;

end;

function pretokenize(const text: string):TArray<string>;
var
    count, len: longint;
    p, start: PAnsiChar;
    lower: ansiChar;
    chunk: string;
begin
    result := nil;
    chunk := '';
    count := 0;
    p := PAnsiChar(text);
    while p^<>NUL do
        begin
            start := p;
            if (p[0] = '''') and (p[1]<>NUL) then begin
                lower := LowerCase(p[1]);
                if lower in ['s', 't', 'm', 'd'] then
                    p := p + 2
                else
                    if (lower in ['r', 'v', 'l']) and (p[2]<>NUL) and (LowerCase(p[2]) in ['e', 'l']) then
                        p := p + 3
                else
                    inc(p)
            end
            else if (p[0] in ALPHABETS) or (PByte(p)^ >= 128) then
                while (p[0]<>NUL) and ((p[0] in ALPHABETS) or (PByte(p)^ >= 128)) do begin
                    if byte(p[0]) >= 128 then  begin
                        if (byte(p[0]) and $E0) = $C0 then
                            inc(p, 2)
                        else if (byte(p[0]) and $F0) = $E0 then
                           inc(p, 3)
                        else if (byte(p[0]) and $F8) = $F0 then
                           inc(p, 4)
                        else
                           inc(p)
                    end
                    else
                        inc(p)
                end
            else if p[0] in NUMERICS then
                while p[0] in NUMERICS do
                    inc(p)
            else if (p[0] = ' ') and (p[1]<>NUL) and ((p[1] in ALPHABETS) or (byte(p[1]) >= 128)) then begin
                inc(p);
                while (p[0]<>NUL) and ((p[0] in ALPHABETS) or (byte(p[0]) >= 128)) do begin
                    if byte(p[0]) >= 128 then begin
                        if (PByte(p)^ and $E0) = $C0 then
                            inc(p, 2)
                        else
                            if (byte(p[0]) and $F0) = $E0 then
                                inc(p, 3)
                        else
                            if (byte(p[0]) and $F8) = $F0 then
                                inc(p, 4)
                        else
                            inc(p)
                    end
                    else
                        inc(p)
                end
            end
            else if (p[0] = ' ') and (p[1] in NUMERICS) then begin
                inc(p);
                while p[0] in NUMERICS do
                    inc(p)
            end
            else if p[0] in WHITE_SPACES then
                while p[0] in WHITE_SPACES do
                    inc(p)
            else
                inc(p);
            if p > start then begin
                len := p-start;
                setLength(chunk, len);
                move(start[0] , chunk[1], len);
                setLength(result, length(result)+1);
                result[count] := chunk;
                inc(count)
            end
        end
end;

function textToUTF8(const str:string):string;
var i, len : longint;
begin
  initByteEncoder();
  result :='';
  for i:= 1 to length(str) do begin
      encodeByteToUTF8(byte(str[i]), result);
  end;
end;

function TQNNTokenizer.tokenize(const str: string; max_len: longint): TArray<longint>;
var
    chunks: TArray<string>;
    total, c: longint;
    byte_text: string;
    bpe_tokens: TArray<string>;
    i: longint;
    id: longint;
begin
    if (max_len <= 0) then
        max_len := QWEN3_MAX_SEQ_LEN;
    chunks := pretokenize(str);
    total := 0;
    c := 0;
    while (c < length(chunks)) and (total < max_len) do begin
        byte_text := textToUTF8(chunks[c]);
        if byte_text <> '' then begin
          bpe_tokens := BPEEncodeWord(byte_text);
          i := 0;
          while (i < length(bpe_tokens)) and (total < max_len) do begin
            if vocabsDic.TryGetValue(bpe_tokens[i], id) then begin
              setLength(result, length(result)+1);
              result[total] := id;
              inc(total)
            end;
            inc(i)
          end;
        end;
        inc(c)
    end
end;

(* Apply the Qwen3 chat template and tokenize the result. Template:
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * For Flux, also appends <think>\n\n</think>\n\n to match the training
 * template that triggers direct generation. For Z-Image the think tags
 * are skipped (controlled by skipThinkTags). *)

function TQNNTokenizer.tokenizeChat(const prompt: string; const skipThinkTags: boolean; max_len: longint): TArray<longint>;
var
    tokens :TArray<longint>;
    total, i:longint;
begin
  if (max_len <= 0) then
      max_len := QWEN3_MAX_SEQ_LEN;
  result := [QWEN3_IM_START_ID];
  total := 1;

  tokens := tokenize('user'#10, max_len - total);
  for i:=0 to high(tokens) do begin
    if total>=max_len then break;
    inc(total);
    setLength(result, total);
    result[total-1] := tokens[i];
  end;

  tokens := tokenize(prompt, max_len - total);
  for i:=0 to high(tokens) do begin
    if total>=max_len then break;
    inc(total);
    setLength(result, total);
    result[total-1] := tokens[i];
  end;
  if total<max_len then begin
    inc(total);
    setLength(result, total);
    result[total-1] := QWEN3_IM_END_ID;
  end;

  tokens := tokenize(#10, max_len-total);
  for i:=0 to high(tokens) do begin
    if total>=max_len then break;
    inc(total);
    setLength(result, total);
    result[total-1] := tokens[i];
  end;

  if total<max_len then begin
    inc(total);
    setLength(result, total);
    result[total-1] := QWEN3_IM_START_ID;
  end;

  tokens := tokenize('assistant'#10, max_len-total);
  for i:=0 to high(tokens) do begin
    if total>=max_len then break;
    inc(total);
    setLength(result, total);
    result[total-1] := tokens[i];
  end;

  if not skipThinkTags then begin
    if total<max_len then begin
      inc(total);
      setLength(result, total);
      result[total-1] := QWEN3_THINK_START_ID;
    end;

    tokens := tokenize(#10#10, max_len-total);
    for i:=0 to high(tokens) do begin
      if total>=max_len then break;
      inc(total);
      setLength(result, total);
      result[total-1] := tokens[i];
    end;

    if total<max_len then begin
      inc(total);
      setLength(result, total);
      result[total-1] := QWEN3_THINK_END_ID;
    end;

    tokens := tokenize(#10#10, max_len-total);
    for i:=0 to high(tokens) do begin
      if total>=max_len then break;
      inc(total);
      setLength(result, total);
      result[total-1] := tokens[i];
    end;
  end;


end;

function TQNNTokenizer.deTokenize(const tokens: TArray<longint>): string;
var i, j, id, total_len:longint;
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

  j:=0;

  for i:=0 to high(tokens) do begin
    id := tokens[i];
    if (id>=0) and (id<length(vocabs)) then begin
      move(vocabs[id][1], result[j+1], length(vocabs[id]));
      inc(j, length(vocabs[id]))
    end;
  end;
//
//  setLength(result, j)
end;

procedure TQNNTokenizer.free;
begin
  if assigned(vocabsDic) then FreeAndNil(vocabsDic);
  if assigned(merges) then FreeAndNil(merges);
  vocabs := nil;
end;

class operator TQNNTokenizer.initialize({$ifdef fpc}var{$else}out{$endif} val : TQNNTokenizer);
begin
  val := default(TQNNTokenizer)
end;

// const
//  TXT_ANSI = 'the quick brown fox jumps over the lazy dog. '; // ansi
//  TXT_UTF8  ='أبجد هوز حطي كلمن';                            // utf-8
//var t: TQNNTokenizer;
//    tokens : TArray<longint>;
//    pretokens : TArray<string>;
//    s : string;
//    f : uint64;
initialization

  //t := createSimpleTokenizer();
  //f := GetTickCount64;
  //t := TQNNTokenizer.load('c:\development\flux2.c\flux-klein-model\tokenizer\tokenizer.json');
  //writeln('loading took ', (GetTickCount64-f)/1000:1:3, ' seconds');
  //pretokens := pretokenize(TXT_ANSI+TXT_UTF8);
  //tokens := t.tokenize(TXT_ANSI+TXT_UTF8);
  //s := t.deTokenize(tokens);
  //t.free;
  //writeln(s);
  //readln

  readln

end.

