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
{$T+}

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

  //https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs

  TSafeTensorType = (
      /// Boolan type
      stBOOL,
      /// MXF4 (E2M1) <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      stF4,
      /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      stF6_E2M3,
      /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      stF6_E3M2,
      /// Unsigned byte
      stU8,
      /// Signed byte
      stI8,
      /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
      stF8_E5M2,
      /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
      stF8_E4M3,
      /// F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      stF8_E8M0,
      /// Signed integer (16-bit)
      stI16,
      /// Unsigned integer (16-bit)
      stU16,
      /// Half-precision floating point
      stF16,
      /// Brain floating point
      stBF16,
      /// Signed integer (32-bit)
      stI32,
      /// Unsigned integer (32-bit)
      stU32,
      /// Floating point (32-bit)
      stF32,
      /// Complex (32-bit parts)
      stC64,
      /// Floating point (64-bit)
      stF64,
      /// Signed integer (64-bit)
      stI64,
      /// Unsigned integer (64-bit)
      stU64,

      stUNKNOWN =-1
  );

const
  dtype_names : array[low(TSafeTensorType)..high(TSafeTensorType)] of ansistring = (
      /// Boolan type
      'UNKNOWN',
      'BOOL',
      'F4',
      'F6_E2M3',
      'F6_E3M2',
      'U8',
      'I8',
      'F8_E5M2',
      'F8_E4M3',
      'F8_E8M0',
      'I16',
      'U16',
      'F16',
      'BF16',
      'I32',
      'U32',
      'F32',
      'C64',
      'F64',
      'I64',
      'U64'
  );

  SAFETENSORTYPE_TO_DATATYPE : array[low(TSafeTensorType)..high(TSafeTensorType)] of TQNNDatatype = (
      /// Boolan type
      dtBoolean,
      dtF4_e2m1,
      dtUndef, //'F6_E2M3',
      dtUndef, // 'F6_E3M2',
      dtU8, // 'U8',
      dtS8, //'I8',
      dtF8_e5m2, //'F8_E5M2',
      dtF8_e4m3, //'F8_E4M3',
      dtUndef, //'F8_E8M0',
      dtS16,//'I16',
      dtU16,//'U16',
      dtF16,//'F16',
      dtBf16,//'BF16',
      dtS32 ,//'I32',
      dtUndef,// 'U32',
      dtF32,//'F32',
      dtUndef,//'C64',
      dtF64,//'F64',
      dtUndef,//'I64',
      dtUndef,//'U64',
      dtUndef //'UNKNOWN'
  );


type

  { TSafeTensorTypeHelper }

  TSafeTensorTypeHelper = record helper for TSafeTensorType
    function bits: byte;
    class function fromString(const str:string):TSafeTensorType; static;
  end;

  PSafeTensor = ^TSafeTensor;

  { TSafeTensor }

  TSafeTensor = record
  public
    name : ansistring;
    ndim : longint ;
    shape : array {[0..7]} of int64;
    data_offset : int64;
    data_size : int64;
    function is_bf16():boolean;
    function asSingle:TArray<single>;
    function asSinglePtr():PSingle;
    function asMappedPtr():pointer;
    function asBF16:TArray<BF16>;
    function asBF16Ptr:PBF16;
    function asFP16:TArray<FP16>;
    function count(): int64;
    procedure print();
    case dtype : TSafeTensorType of
      stf32 :
        (DataF32: PSIngle);
      stF16 :
        (DataF16: PFP16);
      stBF16 :
        (DataBF16: PBF16);
      stI8 :
        (DataI8: PShortInt);
      stI16 :
        (DataI16: PSmallInt);
      stI32 :
        (DataI32: PLongint);
      stI64 :
        (DataI64: PINT64);
      stBOOL :
        (DataBOOL: PBoolean);
      stUNKNOWN:
        (data : pointer);
  end;

  { TSafeTensorFile }

  TSafeTensorFile = record
    path : ansistring;
    data : pointer;              (* mmapped file data *)
    file_size, header_size, data_size, offset : size_t;
    header_json : ansistring;
    tensors : array of TSafeTensor;
    sf_file : file;
    json : TJSON;
    function tensorCount : longint ;
    constructor open(const apath:string);
    procedure close();
    function find(const name:ansistring):PSafeTensor;
    function getData(const safeTensor:PSafeTensor):pointer;  overload;
    function getData(const tensorName:string):pointer;  overload;
    procedure print();
  end;

  TSafeTensorFiles = TArray<TSafeTensorFile>;

  { TSafetensoFilesHelper }

  TSafetensoFilesHelper = record helper for TSafeTensorFiles
    // getData will allocate memory only if useMMap is false, result is always in F32
    function getTensorDataMemBlock(const name: string; const useMMap: boolean):TMemoryBlock;      overload;
    //function getTensorDataMemBlock(const name: string): TMemoryBlock; overload;
    function getTensorDataMemBlockBF16(const name:string; const useMMap: boolean): TMemoryBlock;      overload;

    function getTensor(const name:string):PSafeTensor;

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
begin
  assert(mumap(addr, size)=0, 'ERROR : unable to unmap memory');
  addr := nil
end;
{$endif}

{ TSafeTensorTypeHelper }

function TSafeTensorTypeHelper.bits: byte;
begin
  case self of
    stBOOL : result := 1;
    stF4 :   result := 4;
    stF6_E2M3, stF6_E3M2 : result := 6;
    stU8, stI8, stF8_E5M2, stF8_E4M3, stF8_E8M0 : result := 8;
    stI16, stU16, stF16, stBF16 : result := 16 ;
    stI32 ,stU32 ,stF32 : result :=32 ;
    stC64 ,stF64 ,stI64 ,stU64 : result := 64
  else
    result :=0;
  end;
end;

class function TSafeTensorTypeHelper.fromString(const str: string): TSafeTensorType;
var i:TSafeTensorType;
begin
  for i := low(TSafeTensorType) to high(TSafeTensorType) do
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
  result := nil;
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

function TSafeTensor.asSinglePtr(): PSingle;
begin
  result := data
end;

function TSafeTensor.asMappedPtr(): pointer;
begin
  result := data
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
  writeln(name,': dtype=', {$ifdef FPC}dtype{$else}ord(dtype){$endif},#9' shape=', sShape,']'#9' offset=', format('%.0n',[currency(data_offset)]),#9' size=', format('%.0n',[currency(data_size)]) );
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
    if not assigned(data_offsets) then continue;// its probably a metadata
    tensors[i].name := json.childObjs[i].name;
    tensors[i].dtype:= TSafeTensorType.fromString(t['dtype']);
    tensors[i].data_offset := offset + data_offsets[0];
    tensors[i].data := PByte(data) + tensors[i].data_offset;
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
  result := PByte(data) + safeTensor.data_offset;
end;

function TSafeTensorFile.getData(const tensorName: string): pointer;
var sf : PSafeTensor;
begin
  sf := find(tensorName);
  if assigned(sf) then exit(getData(sf));
  result := nil;
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


{ TSafetensoFilesHelper }

// result will be F32
function TSafetensoFilesHelper.getTensorDataMemBlock(const name: string; const useMMap:boolean): TMemoryBlock;
var i:integer;
  d:PSafeTensor;
begin
  for i:=0 to high(self) do
    begin
      d := self[i].find(name);
      if assigned(d) and assigned(d.data) then begin
        if useMMap then begin
          case d.dtype of
            stBF16:
              begin
                result := TMemoryBlock.Create(d.count, dtF32);
                BF16ToSingle(d.count(), d.data, result);
              end;
            stF16:
              begin
                result := TMemoryBlock.Create(d.count, dtF32);
                FP16ToSingle(d.count(), d.data, result);
              end;
            stF32: begin
              result.DataType := dtF32;
              result.DataPtr := d.data
            end
          else
            assert(false, 'ERROR [getTnsorData] : Unimplemented data type conversion : '+ name);
          end;
        end
        else begin
          // result must be freed later
          case d.dtype of
            stBF16:
              begin
                result := TMemoryBlock.Create(d.count, dtF32);
                BF16ToSingle(d.count(), d.data, result);
              end;
            stF16:
              begin
                result := TMemoryBlock.Create(d.count, dtF32);
                FP16ToSingle(d.count(), d.data, result);
              end;
            stF32:
              begin
                result := TMemoryBlock.Create(d.count, dtF32);
                move(d.data^, psingle(result)^, d.count*sizeOf(Single));
              end;
          else
            assert(false, 'ERROR [getTnsorData] : Unimplemented data type conversion : '+ name);
          end;
        end;
        exit
      end;
    end;
  result := nil;
  writeln(ErrOutput, 'ERROR : Tensor [', name, '] not found');
end;

// result will be BF16
function TSafetensoFilesHelper.getTensorDataMemBlockBF16(const name: string; const useMMap: boolean): TMemoryBlock;
var i, sz:integer;
  d:PSafeTensor;
begin
  for i:=0 to high(self) do
    begin
      d := TSafeTensorFile(self[i]).find(name);
      if assigned(d) and assigned(d.data) then begin
        assert(d.dtype in [stF32, stBF16], 'ERROR [getTnsorDataB16] : Unimplemented data type conversion : '+name);
        if useMMap then begin
          assert(d.dtype = stBF16, 'ERROR [getTnsorDataB16] : Cannot convert data type in mmap mode : '+name);
          result.DataType:=dtBf16;
          result.DataPtr := d.data
        end else begin
          sz := d.count() * sizeOf(BF16);
          result := TMemoryBlock.create(d.count(), dtBF16);
          assert(assigned(result.Data16), 'ERROR [getTnsorDataB16] : not enough memory to allocate for : '+name);
          if d.dtype=stBf16 then
            move(d.data^, PBF16(result)^, sz)
          else
            SingleToBF16(d.count(), d.data, result)
        end;
        exit
      end;
    end;
  result := nil;
  writeln(ErrOutput, 'ERROR : Tensor [', name, '] not found');
end;

//function TSafetensoFilesHelper.getTensorDataMemBlock(const name: string): TMemoryBlock;
//var i, sz:integer;
//  d:PSafeTensor;
//begin
//  for i:=0 to high(self) do
//    begin
//      d := TSafeTensorFile(self[i]).find(name);
//      if assigned(d) and assigned(d.data) then begin
//        result := TMemoryBlock.Create(d.count()*(d.dtype.bits div 8), SAFETENSORTYPE_TO_DATATYPE[d.dtype], d.data);
//        exit
//      end;
//    end;
//  result := nil;
//  writeln(StdErr, 'ERROR : Tensor [', name, '] not found');
//end;

function TSafetensoFilesHelper.getTensor(const name: string): PSafeTensor;
var i, j:integer;
begin
  for i:=0 to high(self) do
    begin
      result := TSafeTensorFile(self[i]).find(name);
      if assigned(result) then exit(result);
    end;
  result := nil;
  writeln(ErrOutput, 'ERROR : Tensor [', name, '] not found');
end;

procedure hello(const a:integer);begin writeln('hello', a); end;
//var sf : TSafeTensorFile;
//  ss: TArray<single>;

initialization

  //sf:= TSafeTensorFile.open('C:\development\flux2.c\FLUX.2-klein-base-4B\flux-2-klein-base-4b.safetensors');
  //ss := sf.tensors[20].asSingle;
  //sf.print ;
  //readln
end.

