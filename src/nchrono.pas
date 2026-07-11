unit nChrono;
{$ifdef fpc}
{$mode Delphi}{$H+}
{$endif}
interface

uses
  SysUtils
  ;

{$ifdef MSWINDOWS}
//const CLOCKS_PER_SEC: int64 =1000;
{$else}
{$endif}

const CLOCKS_PER_SEC=1000000000;
const CLOCKS_PER_MS=1000000;
const CLOCKS_PER_uS=1000;




//{$if not declared(clock_t)}
//  clock_t = int64;
//{$endif}
{$if not declared(clock_t)}
type
  clock_t = int64;
{$endif}

{$if defined(unix) or defined(posix)}       // should work on darwin too!
  {.$linklib c}

type
  clockid_t=longint;

  PTimeSpec=^TTimeSpec;
  TTimeSpec = record
    tv_sec: int64;
    tv_nsec: int64;
  end;
  const
  {$if defined(LINUX)}
   //posix timer
  CLOCK_REALTIME                  = 0;
  CLOCK_MONOTONIC                 = 1;
  CLOCK_PROCESS_CPUTIME_ID        = 2;
  CLOCK_THREAD_CPUTIME_ID         = 3;
  CLOCK_MONOTONIC_RAW             = 4;
  CLOCK_REALTIME_COARSE           = 5;
  CLOCK_MONOTONIC_COARSE          = 6;

  {$elseif defined(DARWIN) or defined(MACOS)}
  CLOCK_REALTIME                  = 0;
  CLOCK_MONOTONIC_RAW             = 4;
  CLOCK_MONOTONIC_RAW_APPROX      = 5;
  CLOCK_MONOTONIC                 = 6;
  CLOCK_UPTIME_RAW                = 8;
  CLOCK_UPTIME_RAW_APPROX         = 9;
  CLOCK_PROCESS_CPUTIME_ID        = 12;
  CLOCK_THREAD_CPUTIME_ID         = 16;

  {$else}  // libc
   CLOCK_REALTIME           = 0;
   CLOCK_PROCESS_CPUTIME_ID = 2;
   CLOCK_THREAD_CPUTIME_ID  = 3;
   CLOCK_MONOTONIC_RAW      = 4;

  {$endif}
  THE_CLOCK = CLOCK_MONOTONIC_RAW;
  strTimeError = 'cannot read OS time!, ErrorNo [%s]';

  function clock_gettime(clk_id : clockid_t; tp: ptimespec) : longint  ;cdecl; external {$if defined(MACOS) or defined(DARWIN)} 'libc.dylib' {$else}'libc.so'{$endif};
{$endif}

{$ifdef MSWINDOWS}
var
   CPUFreq: clock_t;
   CPUFreqs : double;
function QueryPerformanceCounter(var lpPerformanceCount: clock_t): longbool;stdcall; external 'kernel32.dll';
function QueryPerformanceFrequency(var lpFrequency: clock_t): longbool;stdcall external 'kernel32.dll';
{$endif}

// get system tick in nano seconds, returns int64
function clock():clock_t;


implementation

function clock():clock_t;
{$ifdef MSWINDOWS}
{$else}
var
   TimeSpec : TTimeSpec;
{$endif}
begin
  // in NANO Seconds
  {$ifdef MSWINDOWS}
  //result:=GetTickCount64
  QueryPerformanceCounter(result);
  result := trunc(result / CPUFreqs)

  {$else}
  if clock_gettime(THE_CLOCK, @TimeSpec) <>0 then
      raise Exception.Createfmt(strTimeError,[SysErrorMessage({$ifdef FPC}GetLastOSError{$else}GetLastError{$endif})]);
  result:=TimeSpec.tv_sec*1000000000 + TimeSpec.tv_nsec;
  {$endif}
end;

initialization

{$ifdef MSWINDOWS}
  QueryPerformanceFrequency(CPUFreq);
  CPUFreqs := CPUFreq / 1000000000;
{$endif}

end.

