unit nHttp;
{$ifdef FPC}
{$mode Delphi}
{$modeswitch advancedrecords}
{$endif}
{$assertions on}
interface

uses
  Classes, SysUtils
  {$ifdef FPC}
  {$ifdef MSWINDOWS}
  , WinHTTP
  {$else}
  , opensslsockets
  , fpHTTPClient
  {$endif}
  , zipper
  {$else}
  , Net.HttpClient
  , Zip
  {$endif}
  ;

   const
     clearLine = #$1B'[1K';
     zipExt : array of string =['.zip'];

     PROG_SYMB : array of rawbytestring = ['⡇', '⣆', '⣤', '⠶', '⠛', '⠶', '⣤', '⣤', '⠶', '⠛', '⠹', '⢸', '⣰', '⣤', '⣆', '⡇', '⠏', '⠛', '⠶', '⣤', '⠶', '⠛', '⠶', '⣤', '⣰', '⢸', '⠹', '⠛', '⠏'];
     DOWN_ARROW : array of rawbytestring = ['⠈', '⠙', '⠺', '⢼', '⣰', '⢠', '⢀', '⠁', '⠋', '⠗', '⡧', '⣆', '⡄', '⡀'];
type
  TNHttpMethod =(hmGET, hmPOST, hmPUT, hmOPTIONS, hmPATCH, hmDELETE);
const
  TNHttpMethodStrings : array[low(TNHttpMethod)..high(TNHttpMethod)] of string
     = ('GET', 'POST', 'PUT', 'OPTIONS', 'PATCH', 'DELETE') ;

type

{$if defined(fpc) and defined(MSWINDOWS)}

  { TWinHTTP }

  TWinHTTP = class
  private
    FSession, FConnect, FRequest : HINTERNET;
    FURI : URL_COMPONENTS;
    FResponseHeaders : TStringList;
    FTerminated : boolean;
  public
    OnDataReceived : procedure (Sender: TObject; const AContentLength, AReadCount: Int64) of object;
    OnHeaders : TNotifyEvent;
    constructor Create;
    function ResponseHeaders:TStrings;
    class function getHeader(Headers: TStrings; const aName:string):string;
    procedure HTTPMethod(Const AMethod, AURL : WideString; Stream : TStream; Const AllowedResponseCodes : TArray<Integer> = nil);
    procedure get(const AURL:string; stream:TStream);
    procedure Terminate();
    destructor Destroy; override;
    property Terminated :boolean read FTerminated;
  end;

{$endif}

  { TNHttp }

  TNHttp = class
  {$ifdef FPC}
    {$ifdef MSWINDOWS}
    FHTTP:TWinHttp;
    {$else}
    FHTTP  : TfpHttpClient;
    {$endif}
  {$else}
    FHTTP : THTTPClient;
  {$endif}
    FDownloadSize : int64;
  private
    FRunning : int64;
    FCurrentURL, FCurrentFile:string;
  public
    Method : TNHttpMethod;
    OnReceive : procedure (Sendert:TObject; const received, total:int64) of object;
    {$ifdef FPC}
    procedure FReceiveData(Sender: TObject; const AContentLength, AReadCount: Int64);
    {$else}
    procedure FReceiveData(const Sender: TObject; AContentLength, AReadCount: Int64; var AAbort: Boolean);
    {$endif}
    constructor Create;
    destructor Destroy; override;
    procedure addRequestHeader(const aHeader:rawbytestring);
    function getRequestHeader():rawbytestring;
    function getResponseHeader():rawbytestring;

    procedure Download(const aURL:string; toFile:string ='';const AUnZip:boolean = false);
    procedure FOnGetHeader(sender:TObject);

  end;

  {$ifdef MSWINDOWS}
  procedure printf(const fmt:pansichar);winapi;varargs;              external 'msvcrt.dll';
  //procedure sprintf(out a; const fmt:ansistring);winapi;varargs; external 'msvcrt.dll';
  {$else}
  procedure printf(const fmt:rawbytestring);winapi;varargs;external;
  {$endif}

  procedure unzip(const zipfile:rawbytestring);                           overload;
  procedure unzip(const zipfile, extractFileName, OutputFileName:rawbytestring);  overload;
//var
//  http: TNHttp;
implementation
uses
  termesc
{$ifdef MSWINDOWS}
   , windows
{$endif}
  ;

procedure unzip(const zipfile:rawbytestring);
begin
  {$ifdef FPC}
  zipper.TUnZipper.UnZip(zipfile);
  {$else}
  zip.TZipFile.ExtractZipFile(zipFile, '.')
  {$endif}

end;

procedure unzip(const zipfile, extractFileName, OutputFileName:rawbytestring);
{$ifndef fpc}
var
  LZip: TZipFile;
{$endif}
begin
  {$ifdef FPC}
    {$ifdef VER3_3}
    zipper.TUnZipper.UnZip(zipfile, extractFileName, outputFileName);
    {$else}
    zipper.TUnZipper.UnZip(zipfile, extractFileName);
    if LowerCase(extractFilePath(extractFileName)) <> LowerCase(ExtractFilePath(outputFileName)) then begin
      sysutils.RenameFile(extractFileName, outputFileName);
      sysutils.RemoveDir(extractFilePath(extractFileName));
    end;
    {$endif}
  {$else}
  LZip := TZipFile.Create;
  try
    LZip.Encoding := nil;
    LZip.Open(ZipFile, zmRead);
    LZip.Extract( extractFileName, outputFileName, false);
    LZip.Close;
  finally
    LZip.Free;
  end;
  {$endif}

end;

{$if defined(FPC) and defined(MSWINDOWS)}
{ TWinHTTP }
(*

.. fromm : https://stackoverflow.com/questions/822714/how-to-download-a-file-with-winhttp-in-c-c

hSession = WinHttpOpen( L"WinHTTP Example/1.0",
                        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                        WINHTTP_NO_PROXY_NAME,
                        WINHTTP_NO_PROXY_BYPASS, 0);

// Specify an HTTP server.
if (hSession)
    hConnect = WinHttpConnect( hSession, L"nytimes.com",
                               INTERNET_DEFAULT_HTTP_PORT, 0);

// Create an HTTP request handle.
if (hConnect)
    hRequest = WinHttpOpenRequest( hConnect, L"GET", L"/ref/multimedia/podcasts.html",
                                   NULL, WINHTTP_NO_REFERER,
                                   NULL,
                                   NULL);

// Send a request.
if (hRequest)
    bResults = WinHttpSendRequest( hRequest,
                                   WINHTTP_NO_ADDITIONAL_HEADERS,
                                   0, WINHTTP_NO_REQUEST_DATA, 0,
                                   0, 0);


// End the request.
if (bResults)
    bResults = WinHttpReceiveResponse( hRequest, NULL);

// Keep checking for data until there is nothing left.
if (bResults)
    do
    {

        // Check for available data.
        dwSize = 0;
        if (!WinHttpQueryDataAvailable( hRequest, &dwSize))
            printf( "Error %u in WinHttpQueryDataAvailable.\n",
                    GetLastError());

        // Allocate space for the buffer.
        pszOutBuffer = new char[dwSize+1];
        if (!pszOutBuffer)
        {
            printf("Out of memory\n");
            dwSize=0;
        }
        else
        {
            // Read the Data.
            ZeroMemory(pszOutBuffer, dwSize+1);

            if (!WinHttpReadData( hRequest, (LPVOID)pszOutBuffer,
                                  dwSize, &dwDownloaded))
            {
                printf( "Error %u in WinHttpReadData.\n",
                        GetLastError());
            }
            else
            {
                        printf("%s", pszOutBuffer);
                            // Data in vFileContent
                vFileContent.push_back(pszOutBuffer);
            }

            // Free the memory allocated to the buffer.
            delete [] pszOutBuffer;
        }

    } while (dwSize>0);


// Report any errors.
if (!bResults)
    printf("Error %d has occurred.\n",GetLastError());

// Close any open handles.
if (hRequest) WinHttpCloseHandle(hRequest);
if (hConnect) WinHttpCloseHandle(hConnect);
if (hSession) WinHttpCloseHandle(hSession);
*)

function WinHttpErrorToStr(const error:longword):string;
var
  amsg : array[0..255] of ansichar;
begin
  FillChar(amsg, length(amsg), #0);
  FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, nil, error,0, @amsg[0], length(amsg), nil);
  result := amsg;
  if result = '' then begin
    case error of
      ERROR_WINHTTPOF_HANDLES:
        result := 'ERROR_WINHTTPOF_HANDLES';
      ERROR_WINHTTP_TIME:
        result := 'ERROR_WINHTTP_TIME';
      ERROR_WINHTTP_INTERNAL_ERROR:
        result := 'ERROR_WINHTTP_INTERNAL_ERROR';
      ERROR_WINHTTP_INVALID_URL:
        result :=  'ERROR_WINHTTP_INVALID_URL';
      ERROR_WINHTTP_UNRECOGNIZED_SCHEME:
        result := 'ERROR_WINHTTP_UNRECOGNIZED_SCHEME';
      ERROR_WINHTTP_NAME_NOT_RESOLVED:
        result := 'ERROR_WINHTTP_NAME_NOT_RESOLVED';
      ERROR_WINHTTP_INVALID_OPTION:
        result := 'ERROR_WINHTTP_INVALID_OPTION';
      ERROR_WINHTTP_OPTION_NOT_SETTABLE:
        result := 'ERROR_WINHTTP_OPTION_NOT_SETTABLE';
      ERROR_WINHTTP_SHUTDOWN:
        result := 'ERROR_WINHTTP_SHUTDOWN';
      ERROR_WINHTTP_LOGIN_FAILURE:
        result := 'ERROR_WINHTTP_LOGIN_FAILURE';
      ERROR_WINHTTP_OPERATION_CANCELLED:
        result := 'ERROR_WINHTTP_OPERATION_CANCELLED';
      ERROR_WINHTTP_INCORRECT_HANDLE_TYPE:
        result := 'ERROR_WINHTTP_INCORRECT_HANDLE_TYPE';
      ERROR_WINHTTP_INCORRECT_HANDLE_STATE:
        result :=  'ERROR_WINHTTP_INCORRECT_HANDLE_STATE';
      ERROR_WINHTTP_CANNOT_CONNECT:
        result := 'ERROR_WINHTTP_CANNOT_CONNECT';
      ERROR_WINHTTP_CONNECTION_ERROR:
        result := 'ERROR_WINHTTP_CONNECTION_ERROR';
      ERROR_WINHTTP_RESEND_REQUEST:
        result := 'ERROR_WINHTTP_RESEND_REQUEST';
      ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED:
        result := 'ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED';
      ERROR_WINHTTP_CANNOT_CALL_BEFORE_OPEN:
        result := 'ERROR_WINHTTP_CANNOT_CALL_BEFORE_OPEN';
      ERROR_WINHTTP_CANNOT_CALL_BEFORE_SEND:
        result := 'ERROR_WINHTTP_CANNOT_CALL_BEFORE_SEND';
      ERROR_WINHTTP_CANNOT_CALL_AFTER_SEND:
        result := 'ERROR_WINHTTP_CANNOT_CALL_AFTER_SEND';
      ERROR_WINHTTP_CANNOT_CALL_AFTER_OPEN:
        result := 'ERROR_WINHTTP_CANNOT_CALL_AFTER_OPEN';
      ERROR_WINHTTP_HEADER_NOT_FOUND:
        result := 'ERROR_WINHTTP_HEADER_NOT_FOUND';
      ERROR_WINHTTP_INVALID_SERVER_RESPONSE:
        result := 'ERROR_WINHTTP_INVALID_SERVER_RESPONSE';
      ERROR_WINHTTP_INVALID_HEADER:
        result := 'ERROR_WINHTTP_INVALID_HEADER';
      ERROR_WINHTTP_INVALID_QUERY_REQUEST:
        result := 'ERROR_WINHTTP_INVALID_QUERY_REQUEST';
      ERROR_WINHTTP_HEADER_ALREADY_EXISTS:
        result := 'ERROR_WINHTTP_HEADER_ALREADY_EXISTS';
      ERROR_WINHTTP_REDIRECT_FAILED:
        result := 'ERROR_WINHTTP_REDIRECT_FAILED';
      ERROR_WINHTTP_AUTO_PROXY_SERVICE_ERROR:
        result := 'ERROR_WINHTTP_AUTO_PROXY_SERVICE_ERROR';
      ERROR_WINHTTP_BAD_AUTO_PROXY_SCRIPT:
        result := 'ERROR_WINHTTP_BAD_AUTO_PROXY_SCRIPT';
      ERROR_WINHTTP_UNABLE_TO_DOWNLOAD_SCRIPT:
        result := 'ERROR_WINHTTP_UNABLE_TO_DOWNLOAD_SCRIPT';
      ERROR_WINHTTP_UNHANDLED_SCRIPT_TYPE:
        result := 'ERROR_WINHTTP_UNHANDLED_SCRIPT_TYPE';
      ERROR_WINHTTP_SCRIPT_EXECUTION_ERROR:
        result := 'ERROR_WINHTTP_SCRIPT_EXECUTION_ERROR';
      ERROR_WINHTTP_NOT_INITIALIZED:
        result := 'ERROR_WINHTTP_NOT_INITIALIZED';
      ERROR_WINHTTP_SECURE_FAILURE:
        result := 'ERROR_WINHTTP_SECURE_FAILURE';
      ERROR_WINHTTP_SECURE_CERT_DATE_INVALID:
        result := 'ERROR_WINHTTP_SECURE_CERT_DATE_INVALID';
      ERROR_WINHTTP_SECURE_CERT_CN_INVALID:
        result := 'ERROR_WINHTTP_SECURE_CERT_CN_INVALID';
      ERROR_WINHTTP_SECURE_INVALID_CA:
        result := 'ERROR_WINHTTP_SECURE_INVALID_CA';
      ERROR_WINHTTP_SECURE_CERT_REV_FAILED:
        result := 'ERROR_WINHTTP_SECURE_CERT_REV_FAILED';
      ERROR_WINHTTP_SECURE_CHANNEL_ERROR:
        result := 'ERROR_WINHTTP_SECURE_CHANNEL_ERROR';
      ERROR_WINHTTP_SECURE_INVALID_CERT:
        result := 'ERROR_WINHTTP_SECURE_INVALID_CERT';
      ERROR_WINHTTP_SECURE_CERT_REVOKED:
        result := 'ERROR_WINHTTP_SECURE_CERT_REVOKED';
      ERROR_WINHTTP_SECURE_CERT_WRONG_USAGE:
        result := 'ERROR_WINHTTP_SECURE_CERT_WRONG_USAGE';
      ERROR_WINHTTP_AUTODETECTION_FAILED:
        result := 'ERROR_WINHTTP_AUTODETECTION_FAILED';
      ERROR_WINHTTP_HEADER_COUNT_EXCEEDED:
        result := 'ERROR_WINHTTP_HEADER_COUNT_EXCEEDED';
      ERROR_WINHTTP_HEADER_SIZE_OVERFLOW:
        result := 'ERROR_WINHTTP_HEADER_SIZE_OVERFLOW';
      ERROR_WINHTTP_CHUNKED_ENCODING_HEADER_SIZE_OVERFLOW:
        result := 'ERROR_WINHTTP_CHUNKED_ENCODING_HEADER_SIZE_OVERFLOW';
      ERROR_WINHTTP_RESPONSE_DRAIN_OVERFLOW:
        result := 'ERROR_WINHTTP_RESPONSE_DRAIN_OVERFLOW';
      ERROR_WINHTTP_CLIENT_CERT_NO_PRIVATE_KEY:
        result := 'ERROR_WINHTTP_CLIEENT_CERT_NO_PRIVATE_KEY';
      ERROR_WINHTTP_CLIENT_CERT_NO_ACCESS_PRIVATE_KEY:
        result := 'ERROR_WINHTTP_CLIENT_CERT_NO_ACCESS_PRIVATE_KEY';
      ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED_PROXY:
        result := 'ERROR_WINHTTP_CLIENT_AUTH_CERT_NEEDED_PROXY';
      ERROR_WINHTTP_SECURE_FAILURE_PROXY:
        result := 'ERROR_WINHTTP_SECURE_FAILURE_PROXY';
    else
        result := IntToStr(Error);
    end;
  end;
end;

constructor TWinHTTP.Create;
var USER_AGENT : widestring;
begin
  //USER_AGENT := 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'; // some chrome version
  FResponseHeaders := TStringList.Create;
  FResponseHeaders.NameValueSeparator:=':';
  USER_AGENT  := ApplicationName;
  FSession := WinHttpOpen(
           LPCWSTR(USER_AGENT),
           WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
           WINHTTP_NO_PROXY_NAME,
           WINHTTP_NO_PROXY_BYPASS,
           0);
  Assert(assigned(FSession), 'Failed WinHTTPOpen '+WinHttpErrorToStr(GetLastError));

  //FWebRequest:=Co;
end;

function TWinHTTP.ResponseHeaders: TStrings;
begin
  result := FResponseHeaders;
end;

class function TWinHTTP.getHeader(Headers: TStrings; const aName: string): string;
begin
  result := Trim(Headers.Values[AName]);
end;

function wcslen(const str:LPCWSTR):longword;winapi;external 'msvcrt.dll';

procedure TWinHTTP.HTTPMethod(const AMethod, AURL: WideString; Stream: TStream;
  const AllowedResponseCodes: TArray<Integer>);
var
  res : HINTERNET;
  iSize, iReceived : longword;
  iContentLength, iProgress: int64;
  pURL: LPCWSTR;
  head : widestring;
  data:rawbytestring;
  atrHost, strObj, strExtra:wideString;
  _flag :longword;
begin
  FTerminated := false;
  FillChar(FURI, sizeof(FURI), #0);
  FURI.dwSchemeLength    := DWORD(-1);
  FURI.dwHostNameLength  := DWORD(-1);
  FURI.dwUrlPathLength   := DWORD(-1);
  FURI.dwExtraInfoLength := DWORD(-1);
  FURI.dwStructSize:=sizeOf(FURI);
  pURL := LPCWSTR(AURL);
  Assert(WinHttpCrackUrl(pURL, wcslen(pURL), 0, @FURI), 'Failed WinHttpCrackUrl ' + WinHttpErrorToStr(GetLastError));
  try


    //_flag := WINHTTP_FLAG_SECURE_PROTOCOL_ALL;
    //Assert(WinHttpSetOption(FSession, WINHTTP_OPTION_SECURE_PROTOCOLS, @_flag, sizeof(_flag)), 'Failed WinHttpSetOption WINHTTP_OPTION_SECURE_PROTOCOLS ' + WinHttpErrorToStr(GetLastError()));
    //_flag := WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    //Assert(WinHttpSetOption(FSession, WINHTTP_OPTION_REDIRECT_POLICY, @_flag, sizeof(_flag)), 'Failed WinHttpSetOption WINHTTP_OPTION_REDIRECT_POLICY ' + WinHttpErrorToStr(GetLastError()));

    FConnect := WinHttpConnect(FSession, LPCWSTR(copy(FURI.lpszHostName,0,FURI.dwHostNameLength)), FURI.nPort, 0);
    Assert(assigned(FConnect), 'Failed WinHttpConnect ' +WinHttpErrorToStr(GetLastError));
    FRequest :=  WinHttpOpenRequest(FConnect, LPCWSTR(AMethod), FURI.lpszUrlPath, nil, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    Assert(assigned(FRequest), 'Failed WinHttpOpenRequest '+WinHttpErrorToStr(GetLastError));
    Assert(WinHttpSendRequest(FRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0), 'Failed WinHttpSendRequest '+ WinHttpErrorToStr(GetLastError));
    Assert(WinHttpReceiveResponse(FRequest, nil), 'Failed WinHttpReceiveResponse '+ WinHttpErrorToStr(GetLastError));

    _flag := 0;
    WinHttpQueryHeaders(FRequest, WINHTTP_QUERY_RAW_HEADERS_CRLF, nil, nil , @_flag, WINHTTP_NO_HEADER_INDEX);
    head := '';
    setLength(head, _flag div 2);
    if _flag>0 then begin
      assert(WinHttpQueryHeaders(FRequest, WINHTTP_QUERY_RAW_HEADERS_CRLF, nil, @head[1], @_flag, WINHTTP_NO_HEADER_INDEX), 'WinHttpQueryHeaders ' + WinHttpErrorToStr(getLastError()));
      FResponseHeaders.text := head;
      if assigned(OnHeaders) then OnHeaders(Self);
    end;
    iContentLength :=0;
    iProgress := 0;
    if FResponseHeaders.indexof('content-length')>=0 then
      tryStrToInt64(FResponseHeaders.Values['content-length'], iContentLength);

    Assert(WinHttpQueryDataAvailable(FRequest, @iSize), 'Failed WinHttpQueryDataAvailable '+ WinHttpErrorToStr(GetLastError));
    while (iSize>0) do begin
      if FTerminated then break;
      SetLength(data, iSize);
      assert(WinHttpReadData(FRequest, @data[1], iSize, @iReceived), 'Failed WinHttpReadData '+ WinHttpErrorToStr(GetLastError));
      stream.Write(data[1], iReceived);
      inc(iProgress, iReceived);
      if assigned(OnDataReceived) then OnDataReceived(self, 0, iProgress);
      Assert(WinHttpQueryDataAvailable(FRequest, @iSize), 'Failed WinHttpQueryDataAvailable '+ WinHttpErrorToStr(GetLastError));
    end;
    if not FTerminated then
      if assigned(OnDataReceived) then OnDataReceived(self, iContentLength, iProgress);
  finally
    if assigned(FConnect) then WinHttpCloseHandle(FConnect);
    if Assigned(FRequest) then WinHttpCloseHandle(FRequest);
  end;

end;

procedure TWinHTTP.get(const AURL: string; stream: TStream);
begin
  HTTPMethod('GET', AURL, stream);
end;

procedure TWinHTTP.Terminate();
begin
  FTerminated := true
end;

destructor TWinHTTP.Destroy;
begin
  Assert(WinHttpCloseHandle(FSession), 'Failed WinHTTPCloseSession');
  freeandNil(FResponseHeaders);
  inherited Destroy;
end;


{$endif}
{ TNHttp }

{$ifdef FPC}
procedure TNHttp.FReceiveData(Sender: TObject; const AContentLength, AReadCount: Int64);
{$else}

procedure TNHttp.FReceiveData(const Sender: TObject; AContentLength, AReadCount: Int64; var AAbort: Boolean);
{$endif}
const CHECK {:rawbytestring} = '✓';
var
  c : Int64;
  d: rawbytestring;

begin
  //if ResponseStatusCode<> 200 then exit;
  if AContentLength>0 then
    c := AContentLength
  else
    c := FDownloadSize;
  if (AContentLength<=0) and (FDownloadSize=0) then exit;
  if assigned(OnReceive) then begin
    OnReceive(self, AReadCount, FDownloadSize);
    exit
  end;
  if not IsConsole then exit;

  if AReadCount>=c then begin
    c := AReadCount;
    d := setColor4(colorLime)+ CHECK +resetColor
  end
  else
    d := setColor4(termesc.colorRed) + DOWN_ARROW[(FRunning div 300) mod (length(DOWN_ARROW) div 2)]+ resetColor ;//+ DOWN_ARROW[(length(DOWN_ARROW) div 2) + (FRunning div 300) mod (length(DOWN_ARROW) div 2)];
  //write() does not follow utf8 charcode
  printf(#13'%s %.3f / %.3f MB Downloaded', pansichar(d), AReadCount/1000000, c/1000000);
  //write(format(#13'%s %.3f / %.3f MB Downloaded', [d, AReadCount/1000000, c/1000000]));

  inc(FRunning);
  if AReadCount>=c then
    WriteLn('');
end;

function ExtractURLName(const aURL:string):string;
var i:integer;
begin
  I := aURL.LastDelimiter('/');
  result := copy(aURL, I+2)
end;

destructor TNHttp.Destroy;
begin
  freeAndNil(FHTTP);
  inherited;
end;

procedure TNHttp.addRequestHeader(const aHeader: rawbytestring);
begin
  //todo TNHTTP Implement headers set/get
  assert(false,'Not implemented!');
end;

function TNHttp.getRequestHeader(): rawbytestring;
begin
  //todo TNHTTP Implement headers set/get
  assert(false,'Not implemented!');
end;

function TNHttp.getResponseHeader(): rawbytestring;
begin
  //todo TNHTTP Implement headers set/get
  assert(false,'Not implemented!');
end;

procedure TNHttp.Download(const aURL: string; toFile: string;
  const AUnZip: boolean);
var fs :TFileStream;
  ext : string;
  i:integer;
begin

  if toFile='' then
    toFile := ExtractURLName(aURL);
  FCurrentURL  := aURL;
  FCurrentFile := toFile;
  fs := TFileStream.Create(toFile, fmCreate or fmShareDenyWrite);
  FRunning := 0;
  try
    // todo TNHttp [Download] is a get method, add other methods
    FHTTP.get(aURL, fs);
  finally
    freeAndNil(fs);
    if FHTTP.Terminated then begin
      if FileExists(FCurrentFile) then
        assert(DeleteFile(PChar(FCurrentFile)));
    end;
  end;
  if not AUnZip then exit;
  ext :=ExtractFileExt(toFile);
  for i:=0 to high(zipExt) do
    if lowerCase(ext)=zipExt[i] then begin
      unzip(toFile);
      SysUtils.DeleteFile(toFile);
      break
    end;
end;

procedure TNHttp.FOnGetHeader(sender: TObject);
var s : string;
begin
{$ifdef FPC}
  if not TryStrToInt64(FHTTP.GetHeader(FHTTP.ResponseHeaders, 'content-length'), FDownloadSize) then
{$else}
  if not TryStrToInt64(FHTTP.CustomHeaders['content-length'], FDownloadSize) then
{$endif}
    FDownloadSize:=-1;
end;

constructor TNHttp.Create;
begin
  {$ifdef FPC}
  {$ifdef MSWINDOWS}
  FHTTP := TWinHTTP.Create();
  {$else}
  FHTTP := TFPHTTPClient.Create(nil);
  FHTTP.AllowRedirect := true;
  {$endif}
  FHTTP.OnDataReceived := FReceiveData;
  FHTTP.OnHeaders := FOnGetHeader;
  {$else}
  FHTTP := THTTPClient.Create();
  FHTTP.HandleRedirects := true;
  FHTTP.OnReceiveData := FReceiveData;
  {$endif}
  Method := hmGET;
end;

{$ifdef MSWINDOWS}
var
  hConsole : THandle;
  cMode : longword;
{$endif}

initialization
  if isConsole then
  begin
  setUTF8Console();
  {$ifdef MSWINDOWS}
    hConsole := GetStdHandle(STD_OUTPUT_HANDLE);
  {$ifdef FPC}
    GetConsoleMode(hConsole, @cMode);
  {$else}
    GetConsoleMode(hConsole, cMode);
  {$endif}
    SetConsoleMode(hConsole, (cmode or ENABLE_VIRTUAL_TERMINAL_PROCESSING or
      ENABLE_PROCESSED_OUTPUT){ and not ENABLE_WRAP_AT_EOL_OUTPUT});
    //SetMultiByteConversionCodePage(CP_NONE);
  {$endif}
  end;
  //write(#$1B'[?1049h'); // set Console Alternative Buffer

  finalization


end.

 
