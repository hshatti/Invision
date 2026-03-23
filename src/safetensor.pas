unit safetensor;
{$ifdef FPC}
  {$PackRecords C}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$modeswitch nestedprocvars}
  {$ifdef CPUX64}
    {$asmmode intel}
  {$endif}
{$endif}
{$C+} // enable assertions
{$H+}
{$pointermath on}

interface

uses
  SysUtils
  {$if defined(MSWINDOWS)}
  , Windows
  {$else} // POSIX
  , unixbase
  {$endif}
  , quicknn_common
  , quickjson
  ;


const
  SAFETENSORS_MAX_TENSORS = 512;

type
  pchar = ^char;
  {$if not defined(int64_t)}
  Pint64_t = ^int64_t;
  int64_t = Int64;
  {$endif}

  {$if not defined(int32_t)}
  Pint32_t = ^int32_t;
  int32_t = LongInt;
  {$endif}

  {$if not defined(int16_t)}
  Pint16_t = ^int16_t;
  int16_t = smallint;
  {$endif}

  {$if not defined(uint16_t)}
  Puint16_t = ^uint16_t;
  uint16_t = word;
  {$endif}

  {$if not defined(uint64_t)}
  Puint64_t = ^uint64_t;
  uint64_t = UInt64;
  {$endif}

  {$if not defined(uint32_t)}
  Puint32_t = ^uint32_t;
  uint32_t = LongWord;
  {$endif}

  {$if not defined(size_t)}
  Psize_t = ^size_t;
  size_t = IntPtr;
  {$endif}

  {$if not defined(usize_t)}
  Pusize_t = ^usize_t;
  usize_t = UIntPtr;
  {$endif}

  TSafeTensorType = (
    stUNKNOWN = -1,
    stF32 = 0,
    stF16 = 1,
    stBF16 = 2,
    stI32 = 3,
    stI64 = 4,
    stBOOL = 5
  );

const
  dtype_names : array[low(TSafeTensorType)..high(TSafeTensorType)] of ansistring = ('UNKNOWN', 'F32', 'F16', 'BF16', 'I32', 'I64', 'BOOL');

type

  { TSafeTensorTypeHelper }

  TSafeTensorTypeHelper = type helper for TSafeTensorType
    class function fromString(const str:string):TSafeTensorType; static;
  end;

  PSafeTensor = ^TSafeTensor;

  { TSafeTensor }

  TSafeTensor = record
  private
    data : pointer;
  public
    name : ansistring;
    dtype : TSafeTensorType ;
    ndim : longint ;
    shape : array {[0..7]} of int64;
    data_offset : int64;
    data_size : int64;
    function is_bf16():boolean;
    function asSingle:TArray<single>;
    function asBF16:TArray<BF16>;
    function asBF16Ptr:PBF16;
    function asFP16:TArray<FP16>;
    function count(): int64;
    procedure print();
  end;

  { TSafeTensorFile }

  TSafeTensorFile = record
    path : ansistring;
    data : pointer;              (* mmap'd file data *)
    file_size, header_size, data_size, offset : size_t;
    header_json : ansistring;
    tensors : array of TSafeTensor;
    sf_file : file;
    json : TJSON;
    function tensorCount : longint ;
    constructor open(const apath:string);
    procedure close();
    function find(const name:ansistring):PSafeTensor;
    function getData(const safeTensor:PSafeTensor):pointer;
    procedure print();
  end;

implementation

{$if defined(MSWINDOWS)}
function mmapFile(var f:file; const size:uint64=0; const offset:uint64=0):pointer;
var mapped_handle:THandle;
    ff : TFileRec absolute f;
    sz : uint64;
    err: longword;
    msg:array[0..255] of char;
begin
  //try
  //  file_size := FileSize(f);  // just checking if fileopened
  //finally
  //end;
  assert(ff.mode<>fmClosed, 'ERROR : File is not opened!');
  if size>0 then sz := size else sz := FileSize(f);

  mapped_handle := CreateFileMapping(ff.Handle, nil, PAGE_READONLY , sz shr 32, sz and $FFFFFFFF, nil);
  err := GetLastError;
  FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, nil, err, 0, msg, length(msg), nil);
  assert(mapped_handle<>0, 'ERROR : unable to create mapping file to memory!'+sLineBreak+msg) ;

  result := MapViewOfFile(mapped_handle, FILE_MAP_READ, offset shr 32, offset and $FFFFFFFF, sz);
  err := GetLastError;
  FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, nil, err, 0, msg, length(msg), nil);
  assert(CloseHandle(mapped_handle), 'ERROR : Unable to close mapped file!');
  assert(assigned(result), 'ERROR : unable to map file to memory!'+sLineBreak+msg )
end;

function mumapFile(var addr:pointer; const size:uint64):pointer;
begin
  assert(UnmapViewOfFile(addr),'ERROR : unable to unmap memory');
  addr := nil
end;

{$else} // assumably on POSIX systems
function mmapFile(var f:file; const size: uint64; const offset:uint64=0):pointer;
var
  ff : TFileRec absolute f;
  sz : uint64;
begin
  assert(ff.mode<>fmClosed, 'ERROR : File is not opened!');
  if size>0 then sz := size else sz := FileSize(f);
  result := mmap(nil, size, PROT_READ, MAP_PRIVATE, ff.handle, offset);
  assert(result<>MAP_FAILD, 'ERROR : Unable to map file to memory!');
end;

function mumapFile(var addr:pointer; const size:uint64):pointer;
var
  sz : uint64;
begin
  assert(mumap(addr, sz)=0, 'ERROR : unable to unmap memory');
  addr := nil
end;
{$endif}

{ TSafeTensorTypeHelper }

class function TSafeTensorTypeHelper.fromString(const str: string): TSafeTensorType;
var i:TSafeTensorType;
begin
  for i in TSafeTensorType do
    if CompareText(dtype_names[i], str)=0 then
      exit(i);
end;

{ TSafeTensor }

function TSafeTensor.is_bf16(): boolean;
begin
  result := dtype = stBF16;
end;

function TSafeTensor.asSingle: TArray<single>;
begin
  assert(dtype in [stBF16, stF16, stF32], 'ERROR : unsupported tensor type!');
  setlength(result, count);
  case dtype of
    stF32:
      begin
        move(data^, result[0], sizeof(single)*length(result))
      end;

    stF16:
      begin
        FP16ToSingle(length(result), data, pointer(result));
      end;

    stBF16:
      begin
        BF16ToSingle(length(result), data, pointer(result));
      end;
  end;
end;

function TSafeTensor.asBF16: TArray<BF16>;
begin
  assert(dtype = stBF16, 'ERROR : Tensor type is not BF16!');
  setLength(result, count);
  move(data^, result[0], sizeof(BF16)*length(result))
end;

function TSafeTensor.asBF16Ptr: PBF16;
begin
  assert(dtype = stBF16, 'ERROR : Tensor type is not BF16!');
  result := data
end;

function TSafeTensor.asFP16: TArray<FP16>;
begin
  assert(dtype = stF16, 'ERROR : Tensor type is not F16!');
  setLength(result, count);
  move(data^, result[0], sizeof(FP16)*length(result))
end;

function product(const ar:array of int64):int64;
var
  i: Integer;
begin
  //if not assigned(ar) then exit(0);
  result := 1;
  for i:=0 to high(ar) do
    result := result * ar[i]
end;

function TSafeTensor.count(): int64;
begin
  result := product(shape)
end;

procedure TSafeTensor.print();
var sShape:string;
  i: Integer;
begin
  sShape := '';
  for i:=0 to high(shape) do sShape:=sShape+','+intToStr(shape[i]); sShape[1]:='[';
  writeln(name,': dtype=', dtype,#9' shape=', sShape,']'#9' offset=', format('%.0n',[currency(data_offset)]),#9' size=', format('%.0n',[currency(data_size)]) );
end;

{ TSafeTensorFile }

function TSafeTensorFile.tensorCount: longint;
begin
  result := length(tensors)
end;

constructor TSafeTensorFile.open(const apath: string);
var
  i, o:int64;
  data_offsets : TArray<variant>;
  t : TJSON;
begin

  assert(FileExists(apath),'ERROR : File not found : '+apath);
  path := apath;
  assignfile(sf_file, path);
  reset(sf_file, 1);
  file_size := FileSize(sf_file);
  assert(file_size>7, 'ERROR : Invalide file size!');
  BlockRead(sf_file, header_size, sizeOf(header_size));
  assert(header_size<file_size, 'ERROR : Invalid SafeTensor header!');
  setLength(header_json, header_size);
  BlockRead(sf_file, header_json[1], header_size);
  data := mmapFile(sf_file);
  closefile(sf_file);
  offset := 8 + header_size;
  data_size := file_size - offset;
  json := TJSON.parse(header_json);
  setlength(tensors, json.count());
  o :=offset;
  for i:=0 to json.count()-1 do begin
    t := json[i];
    data_offsets := t['data_offsets'];
    tensors[i].name := json.childObjs[i].name;
    tensors[i].dtype:= TSafeTensorType.fromString(t['dtype']);
    tensors[i].data_offset := offset + data_offsets[0];
    tensors[i].data := data + tensors[i].data_offset;
    tensors[i].data_size := data_offsets[1] - data_offsets[0];
    tensors[i].shape := t['shape'];
    assert(assigned(tensors[i].shape), 'ERROR : tensor['+intToStr(i)+']['+tensors[i].name+'] has no shape!');
    tensors[i].ndim := length(tensors[i].shape);
    inc(o, tensors[i].data_size);
    if (o<offset) or (file_size < o) then begin
      close;
      assert(false, 'ERROR : Truncated Safetensor file, try to download again');
    end;
  end;

end;

procedure TSafeTensorFile.close();
begin
  //if TFileRec(sf_file).Mode<>fmClosed then begin
    mumapFile(data, file_size);
    //closeFile(sf_file);
  //end;
  self := default(TSafeTensorFile)
end;

function TSafeTensorFile.find(const name: ansistring): PSafeTensor;
var i:integer;
begin
  for i:=0 to high(tensors) do
    if name=tensors[i].name then exit(@tensors[i]);
  result := nil
end;

function TSafeTensorFile.getData(const safeTensor: PSafeTensor): pointer;
begin
  result := data + safeTensor.data_offset;
end;

procedure TSafeTensorFile.print;
var
  i: Integer;
begin
  writeln('Safetensors file: ', path);
  writeln('File size: ', format('%.0n',[currency(file_size)]),' bytes');
  writeln('Header size: ',  format('%.0n',[currency(header_size)]),' bytes');
  writeln('Number of tensors: ', tensorCount, sLineBreak,'---------------------------', sLineBreak, sLineBreak);

  for i:=0 to high(tensors) do
    tensors[i].print
end;

//var sf : TSafeTensorFile;
//  ss: TArray<single>;
initialization
  //sf:= TSafeTensorFile.open('C:\development\flux2.c\FLUX.2-klein-base-4B\flux-2-klein-base-4b.safetensors');
  //ss := sf.tensors[20].asSingle;
  //sf.print ;
  //readln

end.

