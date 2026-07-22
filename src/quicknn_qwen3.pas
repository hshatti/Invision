unit quicknn_qwen3;

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
  SysUtils, quicknn_common, safetensor, quickjson;

(* ========================================================================
 * Architecture Constants
 * ======================================================================== *)

const
(* Fixed constants (same across model sizes) *)
  QWEN3_VOCAB_SIZE       =151936    ;
  QWEN3_MAX_SEQ_LEN      =512       ;
  QWEN3_RMS_NORM_EPS     =1e-6      ;
  QWEN3_ROPE_THETA       =1000000.0 ;

(* Output layers to extract (0-indexed)
 * Python uses hidden_states[9,18,27] which are outputs AFTER layers 8,17,26
 * since hidden_states[0] is embedding and hidden_states[i] is output after layer i-1 *)
  QWEN3_OUTPUT_LAYER_1   =8      ;
  QWEN3_OUTPUT_LAYER_2   =17     ;
  QWEN3_OUTPUT_LAYER_3   =26     ;

type
  TQNNBPEMerge = record
      left  : rawbytestring;
      right : rawbytestring;
      rank  : longint;  (* Lower rank = higher priority (merge first) *)
  end;

  { TQNNTokenizer }
  PQNNTokenizer = ^TQNNTokenizer;
  TQNNTokenizer = record
      (* Vocabulary: id -> token string *)
      vocabs    : TArray<rawbytestring>;

      (* Hash table: token string -> id *)
      vocabsDic : TVocabs;

      (* BPE merges *)
      merges    : TArray<TQNNBPEMerge>;

      (* Merge rank lookup: "left right" -> rank *)
      merge_ranks :TVocabs;
      constructor load(const jsonFilename:string);
      function getToken(const id:longint):rawbytestring;
      function getId(const token:rawbytestring):longint;
      function getMergeRank(const left, right:rawbytestring) : longint;
      function BPEEncodeWord(const str:rawbytestring):TArray<rawbytestring>;
      function tokenize(const str:rawbytestring; max_len:longint=0):TArray<longint>;
      function tokenizeChat(const prompt:rawbytestring; const skipThinkTags:boolean=false; max_len:longint=0):TArray<longint>;
      function deTokenize(const tokens: TArray<longint>): rawbytestring;
      procedure free;

      class operator initialize({$ifdef fpc}var{$else}out{$endif} val : TQNNTokenizer);
  end;

  PQWEN3Model = ^TQWEN3Model;
  PQWEN3Layer = ^TQWEN3Layer;

  { TQWEN3Attention }

  TQWEN3Attention = record
      model : PQWEN3Model;
      layer : PQWEN3Layer;
      q_proj_weight : TMemoryBlock;    //[num_heads * head_dim, hidden] = [4096, 2560]
      k_proj_weight : TMemoryBlock;    //[num_kv_heads * head_dim, hidden] = [1024, 2560]
      v_proj_weight : TMemoryBlock;    //[num_kv_heads * head_dim, hidden] = [1024, 2560]
      o_proj_weight : TMemoryBlock;    //[hidden, num_heads * head_dim] = [2560, 4096]
      q_norm_weight : TMemoryBlock;    //[head_dim] = [128]
      k_norm_weight : TMemoryBlock;    //[head_dim] = [128]

      // incase of GPU
      q_proj_weight_bf16: TMemoryBlock;
      k_proj_weight_bf16: TMemoryBlock;
      v_proj_weight_bf16: TMemoryBlock;
      o_proj_weight_bf16: TMemoryBlock;
      q_norm_weight_bf16: TMemoryBlock;
      k_norm_weight_bf16: TMemoryBlock;
      procedure forward(const seq_len:longint; const attention_mask : TArray<longint>);

  end;

  { TQWEN3MLP }

  TQWEN3MLP = record
      model : PQWEN3Model;
      layer : PQWEN3Layer;
      gate_proj_weight   :TMemoryBlock;  // [intermediate, hidden] = [9728, 2560]
      up_proj_weight     :TMemoryBlock;    // [intermediate, hidden] = [9728, 2560]
      down_proj_weight   :TMemoryBlock;  // [hidden, intermediate] = [2560, 9728]

      //GPU
      gate_proj_weight_bf16 : TMemoryBlock;
      up_proj_weight_bf16   : TMemoryBlock;
      down_proj_weight_bf16 : TMemoryBlock;
      procedure forward(const seq_len : longint);
  end;

  { TQWEN3Layer }

  TQWEN3Layer = record
      model : PQWEN3Model;
      input_layernorm_weight          : TMemoryBlock;  // [hidden]
      post_attention_layernorm_weight : TMemoryBlock;  // [hidden]

      attn :TQWEN3Attention;
      mlp : TQWEN3MLP;
      // BF16 layer norm weights (for GPU path) - unused currently, kept for future
      input_layernorm_weight_bf16          : TMemoryBlock;
      post_attention_layernorm_weight_bf16 : TMemoryBlock;
      procedure forward(const seq_len: longint; const attentionMask : TArray<longint>);
      procedure load(const layer_idx: longint; const useMMAP:boolean);
      procedure free();
  end;

  { TQWEN3Model }

  TQWEN3Model=record
      (* Architecture (from config.json) *)
      hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, vocab_size, text_dim, extraction_mode : longint;      (* 0 = Flux (layers 8,17,26 concat), 1 = Z-Image (layer -2) *)
      rope_theta : QNNFloat;

      (* Embedding layer *)
      embed_tokens : TMemoryBlock;      (* [vocab_size, hidden] *)

      (* Transformer layers *)
      layers : TArray<TQWEN3Layer>;

      (* Final layer norm *)
      norm_weight : TMemoryBlock;       (* [hidden] *)

      (* RoPE precomputed *)
      rope_cos : TMemoryBlock;          (* [max_seq_len, head_dim/2] *)
      rope_sin : TMemoryBlock;          (* [max_seq_len, head_dim/2] *)

      (* Working memory *)
      hidden_state : TMemoryBlock;      (* [seq_len, hidden] *)
      residual : TMemoryBlock;          (* [seq_len, hidden] *)
      q_buf : TMemoryBlock;             (* [seq_len, num_heads * head_dim] *)
      k_buf : TMemoryBlock;             (* [seq_len, num_kv_heads * head_dim] *)
      v_buf : TMemoryBlock;             (* [seq_len, num_kv_heads * head_dim] *)
      attn_scores : TMemoryBlock;       (* [num_heads, seq_len, seq_len] *)
      attn_out : TMemoryBlock;          (* [seq_len, num_heads * head_dim] *)
      mlp_gate : TMemoryBlock;          (* [seq_len, intermediate] *)
      mlp_up : TMemoryBlock;            (* [seq_len, intermediate] *)
      mlp_out : TMemoryBlock;           (* [seq_len, hidden] *)
      norm_buf : TMemoryBlock;          (* [seq_len, hidden] *)

      (* Output layers storage (for extracting layers 9, 18, 27) *)
      layer_outputs : array[0..2] of TMemoryBlock;  (* [seq_len, hidden] each *)

      (* Pre-allocated attention work buffers (avoid per-call allocation) *)
      attn_q_head : TMemoryBlock;       (* [seq_len, head_dim] *)
      attn_v_head : TMemoryBlock;       (* [seq_len, head_dim] *)
      attn_out_head : TMemoryBlock;     (* [seq_len, head_dim] *)

      (* Mmap mode: keep safetensors files open, load layer weights on-demand *)
      use_mmap, use_bf16 : boolean;
      sf_files : TSafeTensorFiles;
      procedure load(const modelDir:string; const useMMAP:boolean);

      function parse(const modelDir:string):boolean;
      procedure setDefaults();
      procedure loadSafeTensors(const modelDir: string);
      procedure reInitialize();
      procedure free();
      procedure setExtractionMode(const mode:boolean);
      function forward(const input_ids: TArray<longint>; const attention_mask: TArray<longint>; seq_len: longint):TMemoryBlock;
      (* BF16 GPU acceleration *)
  end;

  { TQWEN3Encoder }

  TQWEN3Encoder = record
      tokenizer : TQNNTokenizer;
      model : TQWEN3Model;
      procedure load(const modelDir:string; const useMMAP:boolean);
      function encodeText(const prompt: rawbytestring; var out_seq_len:longint):TMemoryBlock;
      procedure free;
  end;

  function padTokens(const tokens:TArray<longint>; const max_len:longint; const attentionMask : TArray<longint>):TArray<longint>;

implementation
uses quicknn_kernels;

const
  //QWEN3_MAX_SEQ_LEN = 512;
  QWEN3_PAD_ID = 151643      ;(* <|endoftext|> *)
  QWEN3_IM_START_ID = 151644 ;(* <|im_start|> *)
  QWEN3_IM_END_ID = 151645   ;(* <|im_end|> *)
  QWEN3_THINK_START_ID = 151667 ;(* <think> *)
  QWEN3_THINK_END_ID = 151668   ;(* </think> *)


var
  byteToUnicode : array [0..255] of longint;
  UnicodeToByte : array [0..511] of longint;

var
  byteEncoderInitializd : boolean = false;

procedure QNNHeadRSMNorm(const dst, src, weight: TMemoryBlock; const seqLen, numHeads, headDim:longint);
var i , off:longint;
begin
  for i:=0 to seqLen-1 do begin
    off := i*numHeads*headDim;
    QNNRMSNormRows(dst+off , src+off, weight, numHeads, headDim);
  end;
end;

procedure QWEN3ApplyRoPE(const Q, K, cosCache, sinCache: TMemoryBlock; const seqLen, numQHeads, numKHeads, headDim:longint);
begin
  QNNApplyRoPE3(Q, cosCache, sinCache, seqLen, numQHeads, headDim);
  QNNApplyRoPE3(K, cosCache, sinCache, seqLen, numKHeads, headDim);
end;

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

function encodeByteToUTF8(const b: byte; var str: rawbytestring):longint;
var
    cp: longint;
begin
    initByteEncoder();
    cp := byteToUnicode[b];
    if cp < 128 then begin
      setLength(str, length(str)+1);
      str[length(str)] := ansichar(cp);
      exit(1)
    end
    else if cp < 2048 then begin
      setLength(str, length(str)+2);
      str[length(str)-1] := ansichar(($C0 or (cp shr 6)));
      str[length(str)  ] := ansichar(($80 or (cp and $3F)));
      exit(2)
    end;
    //
    setLength(str, length(str)+1);
    str[length(str)] := '?';
    result := 1
end;

function padTokens(const tokens: TArray<longint>; const max_len: longint; const attentionMask: TArray<longint>): TArray<longint>;
var i:longint;
begin
  setLength(result, max_len);

  move(tokens[0], result[0], length(tokens)*sizeof(tokens[0]));
  FillDWord(result[length(tokens)], length(result)-length(tokens), QWEN3_PAD_ID);

  if assigned(attentionMask) then begin
    FillDWord(attentionMask[0], length(tokens), 1);
    FillDWord(attentionMask[length(tokens)], length(result)-length(tokens), 0);
  end;

  //for i:=0 to high(result) do begin
  //  if i<length(tokens) then begin
  //    result[i] := tokens[i];
  //    if assigned(attentionMask) then
  //        attentionMask[i]:=1;
  //  end else begin
  //    result[i] := QWEN3_PAD_ID;
  //    if assigned(attentionMask) then
  //        attentionMask[i]:=0;
  //  end;
  //end;
end;

{ TQNNTokenizer }

constructor TQNNTokenizer.load(const jsonFilename: string);
var json, vocab, merge, addedTokens : TJSON;
    i, id: longint;
    content : rawbytestring;
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
  setLength(merges, merge.count());// make space for added merges
  for i:=0 to vocab.count-1 do begin
      id := vocab[i].value;
      vocabs[id] := vocab[i].name;
      vocabsDic.TryAdd(vocab[i].name, id);
  end;

  for i:=0 to high(merges) do begin
    merges[i].left  := ansistring(merge[i][0].value);
    merges[i].right := ansistring(merge[i][1].value);
    merges[i].rank := i;
    if (merges[i].left<>'') and (merges[i].right<>'') then begin
        merge_ranks.TryAdd(merges[i].left + ' ' + merges[i].right, i);
    end;
  end;

  for i:=0 to addedTokens.count-1 do begin
    id := addedTokens[i]['id'].Value;
    content := ansistring(addedTokens[i]['content']);
    if (content='') or (id<0) or (id>=length(vocabs)) then continue; // skip IDs that has no allocated position
    vocabs[id] := content;
    vocabsDic.TryAdd(vocabs[id], id);
  end;
end;

function TQNNTokenizer.getToken(const id: longint): rawbytestring;
begin
  result := '';
  if assigned(vocabs) and (id>=0) and (id<length(vocabs)) then
    result := vocabs[id]
end;

function TQNNTokenizer.getId(const token: rawbytestring): longint;
begin
  if not assigned(vocabsDic) or (token='') then exit(-1);
  if not vocabsDic.TryGetValue(token, result) then
      result := -1;
end;

function TQNNTokenizer.getMergeRank(const left, right: rawbytestring): longint;
begin
  assert(assigned(merge_ranks), 'ERROR : tokenizer is empty, load tokenizer file 1st!');
  if not merge_ranks.TryGetValue(left+' '+right, result) then result := -1;
end;

function TQNNTokenizer.BPEEncodeWord(const str: rawbytestring): TArray<rawbytestring>;
  procedure appendToken(const a:rawbytestring);
  begin
    insert(a, result, length(result));
  end;

var i, m, n, id, len, best_idx, best_rank, rank : longint;
  s: rawbytestring;
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

function pretokenize(const text: rawbytestring):TArray<rawbytestring>;
var
    count, len: longint;
    p, start: PAnsiChar;
    lower: ansiChar;
    chunk: rawbytestring;
begin
    result := nil;
    chunk := '';
    count := 0;
    p := PAnsiChar(text);
    while p^<>NUL do
        begin
            start := p;
            if (p[0] = '''') and (p[1]<>NUL) then begin
                lower := LCase(p[1]);
                if lower in ['s', 't', 'm', 'd'] then
                    p := p + 2
                else
                    if (lower in ['r', 'v', 'l']) and (p[2]<>NUL) and (LCase(p[2]) in ['e', 'l']) then
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

function textToUTF8(const str:rawbytestring):rawbytestring;
var i, len : longint;
begin
  initByteEncoder();
  result :='';
  for i:= 1 to length(str) do begin
      encodeByteToUTF8(byte(str[i]), result);
  end;
end;

function TQNNTokenizer.tokenize(const str: rawbytestring; max_len: longint): TArray<longint>;
var
    chunks: TArray<rawbytestring>;
    total, c: longint;
    byte_text: rawbytestring;
    bpe_tokens: TArray<rawbytestring>;
    i: longint;
    id: longint;
    found : boolean;
begin
    if (max_len <= 0) then
        max_len := QWEN3_MAX_SEQ_LEN;
    chunks := pretokenize(str);
    total := 0;
    c := 0;
    result := nil;
    while (c < length(chunks)) and (total < max_len) do begin
        byte_text := textToUTF8(chunks[c]);
        if byte_text <> '' then begin
          bpe_tokens := BPEEncodeWord(byte_text);
          i := 0;
          while (i < length(bpe_tokens)) and (total < max_len) do begin
            found := vocabsDic.TryGetValue(bpe_tokens[i], id);
            if found and (id>0) then begin
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

function TQNNTokenizer.tokenizeChat(const prompt: rawbytestring; const skipThinkTags: boolean; max_len: longint): TArray<longint>;
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

function TQNNTokenizer.deTokenize(const tokens: TArray<longint>): rawbytestring;
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
  if assigned(vocabsDic) then begin
    vocabsDic.free;
    vocabsDic := nil
  end;
  if assigned(merges) then setLength(merges, 0);
  if assigned(vocabs) then setLength(vocabs, 0);
  if assigned(merge_ranks) then begin
    merge_ranks.free;
    merge_ranks := nil
  end;
end;

class operator TQNNTokenizer.initialize({$ifdef fpc}var{$else}out{$endif} val : TQNNTokenizer);
begin
  fillchar(val, sizeof(TQNNTokenizer), #0)
end;

{ TQWEN3Attention }

procedure TQWEN3Attention.forward(const seq_len: longint; const attention_mask: TArray<longint>);
var scale :QNNFloat;
    kv_dim, q_dim, heads_per_kv, kv_h, h, i, j : longint;
    q_strided, k_strided, v_strided, out_strided : TMemoryBlock;
    scores : TMemoryBlock;
begin
  assert(assigned(model) and assigned(layer),'ERROR : failed attention forward, no model loaded!');
  kv_dim := model.num_kv_heads * model.head_dim;
  q_dim  := model.num_heads * model.head_dim;
  scale := 1.0 / sqrt(model.head_dim);

  QNNLinearNoBias(model.q_buf, model.norm_buf, layer.attn.q_proj_weight, seq_len, model.hidden_size, q_dim);
  QNNlinearNoBias(model.k_buf, model.norm_buf, layer.attn.k_proj_weight, seq_len, model.hidden_size, kv_dim);
  QNNLinearNoBias(model.v_buf, model.norm_buf, layer.attn.v_proj_weight, seq_len, model.hidden_size, kv_dim);


  (* Q/K RMS normalization (per-head) *)
  QNNHeadRSMNorm(model.q_buf, model.q_buf, layer.attn.q_norm_weight, seq_len, model.num_heads, model.head_dim);
  QNNHeadRSMNorm(model.k_buf, model.k_buf, layer.attn.k_norm_weight, seq_len, model.num_kv_heads, model.head_dim);

  (* Apply RoPE *)
  QNNApplyRoPE3(model.q_buf, model.rope_cos, model.rope_sin, seq_len, model.num_heads   , model.head_dim);
  QNNApplyRoPE3(model.k_buf, model.rope_cos, model.rope_sin, seq_len, model.num_kv_heads, model.head_dim);

  heads_per_kv := model.num_heads div model.num_kv_heads;

  for h := 0 to model.num_heads-1 do begin
    kv_h := h div heads_per_kv;  (* Which KV head to use *)
    scores := model.attn_scores + h*seq_len*seq_len;

    (* Q accessed directly with strided lda (avoids copy)
     * Q[s,d] = q_buf[s * q_dim + h * head_dim + d] *)
    q_strided := model.q_buf + h*model.head_dim;

    (* K accessed directly with strided lda + CblasTrans (avoids transpose)
     * K[s,d] = k_buf[s * kv_dim + kv_h * head_dim + d] *)
    k_strided := model.k_buf + kv_h*model.head_dim;

    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, model.head_dim,
                        scale, q_strided, q_dim, k_strided, kv_dim,
                        0.0, scores, seq_len);

    (* Apply causal mask and attention mask, then softmax *)
    //for i := 0 to seq_len-1 do begin          // todo [QWEN3Attention.Forward] optimize for GPU?
    //    for j := 0 to seq_len-1 do begin
    //        if (j > i) or assigned(attention_mask) and (attention_mask[j] = 0) then begin
    //            scores_ptr[i*seq_len + j] := Single.NegativeInfinity;//-1e9;
    //        end;
    //    end;
    //    QNNSoftmax(scores_ptr + i*seq_len, seq_len);
    //end;
    QNNMatTriangularFill(seq_len, scores, QNNFloat.NegativeInfinity, pointer(attention_mask));
    QNNSoftmaxRows(scores, seq_len, seq_len);
    (* V can be accessed directly with strided lda (avoids copy)
     * V[s,d] = v_buf[s * kv_dim + kv_h * head_dim + d] *)
    v_strided := model.v_buf + kv_h*model.head_dim;

    (* Output can be written directly with strided ldc (avoids copy)
     * out[s,d] = attn_out[s * q_dim + h * head_dim + d] *)
    out_strided := model.attn_out + h*model.head_dim;

    (* out = scores @ V using strided BLAS (avoids V copy and output copy)
     * scores: [seq_len, seq_len], V: [seq_len, head_dim] with ldb=kv_dim *)
     cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 seq_len, model.head_dim, seq_len,
                 1.0, scores, seq_len, v_strided, kv_dim,
                 0.0, out_strided, q_dim);
  end;

  //model.attn_out.printCompare(readTensor);
  QNNLinearNoBias(model.hidden_state, model.attn_out, layer.attn.o_proj_weight, seq_len, q_dim, model.hidden_size);

end;

{ TQWEN3MLP }

procedure TQWEN3MLP.forward(const seq_len: longint);
begin
  assert(assigned(model) and assigned(layer),'ERROR : failed MLP forward, no model loaded!');
  (* Gate and Up projections *)
  QNNLinearNoBias(model.mlp_gate, model.norm_buf, layer.mlp.gate_proj_weight, seq_len, model.hidden_size, model.intermediate_size);
  QNNLinearNoBias(model.mlp_up,   model.norm_buf, layer.mlp.up_proj_weight  , seq_len, model.hidden_size, model.intermediate_size);

  (* SwiGLU: silu(gate) * up - fused for better performance *)
  QNNSiluMul(model.mlp_gate, model.mlp_up, seq_len * model.intermediate_size);

  (* Down projection *)
  QNNLinearNoBias(model.mlp_out,  model.mlp_gate, layer.mlp.down_proj_weight, seq_len, model.intermediate_size, model.hidden_size);
end;

{ TQWEN3Layer }

procedure TQWEN3Layer.forward(const seq_len: longint; const attentionMask: TArray<longint>);
var i:longint;
begin

  (* Save residual *)
  QNNCopy(model.residual, model.hidden_state, seq_len * model.hidden_size);

  (* Pre-attention LayerNorm *)
  QNNRMSNormRows(model.norm_buf, model.hidden_state, input_layernorm_weight, seq_len, model.hidden_size);

  (* Self-attention *)
  attn.forward(seq_len, attentionMask);

  (* Residual connection *)
  QNNAddInplace(model.hidden_state, model.residual, seq_len*model.hidden_size);
  //for i := 0 to seq_len * model.hidden_size do
  //    model.hidden_state[i] += model.residual[i];


  (* Save residual *)
  QNNCopy(model.residual, model.hidden_state, seq_len * model.hidden_size);

  (* Pre-MLP LayerNorm *)
  QNNRMSNormRows(model.norm_buf, model.hidden_state, post_attention_layernorm_weight, seq_len, model.hidden_size);

  (* MLP *)
  mlp.forward(seq_len);

  (* Residual connection *)
  QNNAdd(model.hidden_state, model.residual, model.mlp_out, seq_len*model.hidden_size);

  //for i := 0 to seq_len * hidden_size-1 do
  //    model.hidden_state[i] = model.residual[i] + model.mlp_out[i];
end;

procedure TQWEN3Layer.load(const layer_idx: longint; const useMMAP: boolean);
begin
  assert(assigned(model), 'ERROR : failed to load QWEN3 Layer, no model assigned yet');
  //Input layernorm
  input_layernorm_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.input_layernorm.weight', [layer_idx]), useMMAP);
  // Post-attention layernorm                                              layer->post_attention_layernorm_weight := sf_files.getTensorDataMemBlock(snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
  post_attention_layernorm_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.post_attention_layernorm.weight', [layer_idx]), useMMAP);

   //Attention weights
  attn.model:=model;
  attn.layer:=@self;
  attn.q_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.q_proj.weight', [layer_idx]), useMMAP);
  attn.k_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.k_proj.weight', [layer_idx]), useMMAP);
  attn.v_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.v_proj.weight', [layer_idx]), useMMAP);
  attn.o_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.o_proj.weight', [layer_idx]), useMMAP);

  // Q/K norm
  attn.q_norm_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.q_norm.weight', [layer_idx]), useMMAP);
  attn.k_norm_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.self_attn.k_norm.weight', [layer_idx]), useMMAP);

  // MLP weights
  mlp.model:=model;
  mlp.layer:=@self;
  mlp.gate_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.mlp.gate_proj.weight', [layer_idx]), useMMAP);
  mlp.up_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.mlp.up_proj.weight', [layer_idx]), useMMAP);
  mlp.down_proj_weight := model.sf_files.getTensorDataMemBlock(format('model.layers.%d.mlp.down_proj.weight', [layer_idx]), useMMAP);
end;

procedure TQWEN3Layer.free();
begin
  input_layernorm_weight.free;
  (* Post-attention layernorm *)
  post_attention_layernorm_weight.free;

  (* Attention weights *)
  attn.q_proj_weight.free;
  attn.k_proj_weight.free;
  attn.v_proj_weight.free;
  attn.o_proj_weight.free;

  (* Q/K norm *)
  attn.q_norm_weight.free;
  attn.k_norm_weight.free;

  (* MLP weights *)
  mlp.gate_proj_weight.free;
  mlp.up_proj_weight.free;
  mlp.down_proj_weight.free;
end;

{ TQWEN3Model }

procedure TQWEN3Model.load(const modelDir: string; const useMMAP: boolean);
var half_dim, max_seq, i:longint;
begin
  use_mmap:=useMMAP;
  parse(modelDir);

  loadSafeTensors(modelDir);

  embed_tokens := sf_files.getTensorDataMemBlock('model.embed_tokens.weight', useMMAP);
  if not useMMap then
    for i:=0 to high(layers) do
      layers[i].load(i, useMMAP);
  norm_weight  := sf_files.getTensorDataMemBlock('model.norm.weight', useMMAP);

  half_dim := head_dim div 2;
  max_seq := QWEN3_MAX_SEQ_LEN;

  rope_cos     := TMemoryBlock.Create([max_seq, half_dim], 'QWEN3_LOAD_ROPE_COS' );
  rope_sin     := TMemoryBlock.Create([max_seq, half_dim], 'QWEN3_LOAD_ROPE_SIN');
  QNNComputeRoPE2(rope_cos, rope_sin, max_seq, head_dim, rope_theta);
  reInitialize();


end;

function TQWEN3Model.parse(const modelDir: string): boolean;
var
  json :TJSON;
  num_layers, i : longint;
  fn : string;
begin
  fn := modelDir+'/config.json';
  result := fileExists(fn);
  if result then begin
    json := TJSON.LoadFromFile(fn);

    hidden_size       := json['hidden_size'];
    num_heads         := json['num_attention_heads'];
    intermediate_size := json.get('intermediate_size', 9728);
    num_kv_heads      := json.get('num_key_value_heads', 8);
    head_dim          := json.get('head_dim', 128);
    vocab_size        := json.get('vocab_size', QWEN3_VOCAB_SIZE);
    num_layers        := json.get('num_hidden_layers', 36);
    rope_theta        := json.get('rope_theta', QWEN3_ROPE_THETA);
    setLength(layers, num_layers);
    for i:=0 to high(layers) do
      layers[i].model := @self;
    text_dim          := hidden_size*3; // for flux *3(default) , for zi-image *1
 end else
   setDefaults();
end;

// Defaults for qwen3-4b case
procedure TQWEN3Model.setDefaults();
var num_layers, i:longint;
begin
  hidden_size       := 2560;
  num_heads         := 32;
  intermediate_size := 9728;
  num_kv_heads      := 8;
  head_dim          := 128;
  vocab_size        := QWEN3_VOCAB_SIZE;
  num_layers        := 36;
  rope_theta        := QWEN3_ROPE_THETA;
  setLength(layers, 36);
  for i:=0 to high(layers) do
    layers[i].model := @self;
  text_dim          := hidden_size*3;
end;

function indexOf(const arr:TArray<string>; const str:string):longint;
var i:longint;
begin
  for i:=0 to high(arr) do
    if arr[i]=str then exit(i);
  result := -1
end;

procedure TQWEN3Model.loadSafeTensors(const modelDir:string);
var
  i, fileCount: longint;
  json : TJSON;
  path : string;
  arr  : TArray<TJSON>;
  sfNames: TArray<string>;
begin
  path := modelDir + '/model.safetensors.index.json';
  if FileExists(path) then begin
    json := TJSON.LoadFromFile(path);
    arr := json['weight_map'].childObjs;
    for i:=0 to high(arr) do
      if indexOf(sfNames, string(arr[i].Value))<0 then
        insert(string(arr[i].value), sfNames, length(sfNames));
    setLength(sf_files, length(sfNames));
    for i:=0 to high(sf_files) do
      sf_files[i] := TSafeTensorFile.open(modelDir+'/'+sfNames[i]);
    exit
  end;
  setLength(sf_files, 2);
  sf_files[0] := TSafeTensorFile.open(modelDir+'/model-00001-of-00002.safetensors');
  sf_files[1] := TSafeTensorFile.open(modelDir+'/model-00002-of-00002.safetensors');

end;

procedure TQWEN3Model.reInitialize();
var i, seq_len:longint;
begin
  extraction_mode:=0;
  seq_len := QWEN3_MAX_SEQ_LEN;
  hidden_state  := TMemoryBlock.Create([seq_len , hidden_size],             'QWEN3_INIT_hidden_state' );
  residual      := TMemoryBlock.Create([seq_len , hidden_size],             'QWEN3_INIT_residual'     );
  q_buf         := TMemoryBlock.Create([seq_len , num_heads , head_dim],    'QWEN3_INIT_q_buf'        );
  k_buf         := TMemoryBlock.Create([seq_len , num_kv_heads , head_dim], 'QWEN3_INIT_k_buf'        );
  v_buf         := TMemoryBlock.Create([seq_len , num_kv_heads , head_dim], 'QWEN3_INIT_v_buf'        );
  attn_scores   := TMemoryBlock.Create([num_heads , seq_len , seq_len],     'QWEN3_INIT_attn_scores'  );
  attn_out      := TMemoryBlock.Create([seq_len , num_heads , head_dim],    'QWEN3_INIT_attn_out'     );
  mlp_gate      := TMemoryBlock.Create([seq_len , intermediate_size],       'QWEN3_INIT_mlp_gate'     );
  mlp_up        := TMemoryBlock.Create([seq_len , intermediate_size],       'QWEN3_INIT_mlp_up'       );
  mlp_out       := TMemoryBlock.Create([seq_len , hidden_size],             'QWEN3_INIT_mlp_out'      );
  norm_buf      := TMemoryBlock.Create([seq_len , hidden_size],             'QWEN3_INIT_norm_buf'     );

  attn_q_head   := TMemoryBlock.Create([seq_len , head_dim],                'QWEN3_INIT_attn_q_head'  );
  attn_v_head   := TMemoryBlock.Create([seq_len , head_dim],                'QWEN3_INIT_attn_v_head'  );
  attn_out_head := TMemoryBlock.Create([seq_len , head_dim],                'QWEN3_INIT_attn_out_head');

  for i := 0 to high(layer_outputs) do
      layer_outputs[i] := TMemoryBlock .Create([seq_len,hidden_size], 'QWEN3_LAYER_OUT_'+intToStr(i));

end;

procedure TQWEN3Model.free();
var i:longint;
begin

  embed_tokens  .free();
  norm_weight   .free();
  rope_cos      .free();
  rope_sin      .free();


  hidden_state  .free();
  residual      .free();
  q_buf         .free();
  k_buf         .free();
  v_buf         .free();
  attn_scores   .free();
  attn_out      .free();
  mlp_gate      .free();
  mlp_up        .free();
  mlp_out       .free();
  norm_buf      .free();
  attn_q_head   .free();
  attn_v_head   .free();
  attn_out_head .free();

  for i:=0 to high(layers) do
    layers[i].free();
  layers := nil;
  for i := 0 to high(layer_outputs) do
      layer_outputs[i].free();
  for i:=0 to high(sf_files) do
    sf_files[i].close();
  sf_files := nil;
  self := default(TQWEN3Model)
end;

procedure TQWEN3Model.setExtractionMode(const mode: boolean);
begin
  extraction_mode := ord(mode);
  if mode then
    text_dim := hidden_size
  else
    text_dim := hidden_size * 3;
end;

(* ========================================================================
 * Forward Pass
 * ======================================================================== *)

(* Main Qwen3 forward pass. Runs embedding lookup then processes through
 * transformer layers. Flux mode: saves hidden states at layers 8, 17, 26
 * and concatenates them -> [seq, 3*hidden]. Z-Image mode: saves only
 * hidden_states[-2] (layer 34) -> [seq, hidden]. Stops early at the last
 * needed extraction layer to skip ~9 unnecessary layers of compute*)
function TQWEN3Model.forward(const input_ids: TArray<longint>; const attention_mask: TArray<longint>; seq_len: longint): TMemoryBlock;
var
    zimage : boolean;
    last_layer, s, token_id, layer_idx: longint;
begin

    zimage := boolean(extraction_mode);
    if zimage then
        last_layer := length(layers)-2
    else
        last_layer := QWEN3_OUTPUT_LAYER_3;
    for s := 0 to seq_len -1 do
        begin
            token_id := input_ids[s];
            if (token_id >= 0) and (token_id<vocab_size) then
                QNNCopy(hidden_state + s*hidden_size, embed_tokens + token_id*hidden_size, hidden_size)
            else
                QNNFill(hidden_state + s*hidden_size, 0, hidden_size)
        end;

    for layer_idx := 0 to last_layer do
        begin
            if use_mmap then
              layers[layer_idx].load(layer_idx, use_mmap);
            layers[layer_idx].forward(seq_len, attention_mask);
            if use_mmap then
              layers[layer_idx].free;
            if zimage then begin
                if layer_idx = last_layer then
                    QNNCopy(layer_outputs[0], hidden_state, seq_len * hidden_size)
            end else begin
                if layer_idx = QWEN3_OUTPUT_LAYER_1 then
                    QNNCopy(layer_outputs[0], hidden_state, seq_len * hidden_size)
                else if layer_idx = QWEN3_OUTPUT_LAYER_2 then
                    QNNCopy(layer_outputs[1], hidden_state, seq_len * hidden_size)
                else if layer_idx = QWEN3_OUTPUT_LAYER_3 then
                    QNNCopy(layer_outputs[2], hidden_state, seq_len * hidden_size)
            end;
            if assigned(text_progress_callback) then
                text_progress_callback(layer_idx, last_layer+1)
        end;
    result := TMemoryBlock.Create([seq_len , text_dim], 'QWEN3_RESULT');

    if zimage then
        QNNCopy(result, layer_outputs[0], seq_len * hidden_size)
    else
        for s := 0 to seq_len -1 do
            begin
                QNNCopy(result + s*text_dim                , layer_outputs[0] + s*hidden_size, hidden_size);
                QNNCopy(result + s*text_dim +   hidden_size, layer_outputs[1] + s*hidden_size, hidden_size);
                QNNCopy(result + s*text_dim + 2*hidden_size, layer_outputs[2] + s*hidden_size, hidden_size)
            end;
end;

{ TQWEN3Encoder }

procedure TQWEN3Encoder.load(const modelDir: string; const useMMAP: boolean);
begin
  tokenizer := TQNNTokenizer.load(modelDir+'/tokenizer/tokenizer.json');
  model.load(modelDir+'/text_encoder', useMMAP);
end;

function TQWEN3Encoder.encodeText(const prompt: rawbytestring; var out_seq_len: longint): TMemoryBlock;
var
    skip_think_tags:boolean;
    tokens, attention_mask, padded_tokens: TArray<longint>;
begin
    attention_mask:=nil;;
    if not assigned(tokenizer.vocabsDic) or not assigned(model.layers) then
      exit(default(TMemoryBlock));
    skip_think_tags := model.extraction_mode = 1;
    tokens := tokenizer.tokenizeChat(prompt, skip_think_tags, QWEN3_MAX_SEQ_LEN);
    out_seq_len := length(tokens);
    setLength(attention_mask, QWEN3_MAX_SEQ_LEN);
    padded_tokens := padTokens(tokens, QWEN3_MAX_SEQ_LEN, attention_mask);
    result := model.forward(padded_tokens, attention_mask, QWEN3_MAX_SEQ_LEN);
end;

procedure TQWEN3Encoder.free;
begin
  tokenizer.free;
  model.free;
end;

//const
//  TXT_ANSI = 'the quick brown fox jumps over the lazy dog. '; // ansi
//  TXT_UTF8  ='أبجد هوز حطي كلمن';                            // utf-8
var t: TQNNTokenizer;
  m : TQWEN3Model;
//    tokens : TArray<longint>;
    //pretokens : TArray<string>;
//    s : string;
    //f : uint64;
initialization

  //t := createSimpleTokenizer();
  //f := GetTickCount64;
  //t := TQNNTokenizer.load('c:\development\flux2.c\flux-klein-model\tokenizer\tokenizer.json');
  //m.loadSafeTensors('c:\development\flux2.c\flux-klein-model\text_encoder');
  //readln;

  //writeln('loading took ', (GetTickCount64-f)/1000:1:3, ' seconds');
  //pretokens := pretokenize(TXT_ANSI+TXT_UTF8);
  //tokens := t.tokenize(TXT_ANSI+TXT_UTF8);
  //s := t.deTokenize(tokens);
  //t.free;
  //writeln(s);
  //readln
  //readln

end.

