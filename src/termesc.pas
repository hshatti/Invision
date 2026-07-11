(*

  ************************************
  ANSI Escape sequence Terminl Helper
  ************************************

  Copyright (C) 2024 <Haitham Shatti> <haitham.shatti at gmail dot com>

  This library is free software; you can redistribute it and/or modify it under the terms of the GNU Library General Public License as published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version with the following modification:

  As a special exception, the copyright holders of this library give you permission to link this library with independent modules to produce an executable, regardless of the license terms of these
  independent modules,and to copy and distribute the resulting executable under terms of your choice, provided that you also meet, for each linked independent module, the terms and conditions of the
  license of that module. An independent module is a module which is not derived from or based on this library. If you modify this library, you may extend this exception to your version of the
  library, but you are not obligated to do so. If you do not wish to do so, delete this exception statement from your version.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public License along with this library; if not, write to the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA
  02110-1335, USA.
*)

unit termesc;
{$ifdef FPC}
  {$mode Delphi}
{$endif}

interface

type
  TCursorDirection =(cmUp, cmDown, cmBackward, cmForward);

const
  BRAILLE: array[0..255] of rawbytestring = (
      ' ', '⠁', '⠂', '⠃', '⠄', '⠅', '⠆', '⠇', '⡀','⡁', '⡂', '⡃', '⡄', '⡅', '⡆', '⡇'
    , '⠈', '⠉', '⠊', '⠋', '⠌', '⠍', '⠎', '⠏', '⡈','⡉', '⡊', '⡋', '⡌', '⡍', '⡎', '⡏'
    , '⠐', '⠑', '⠒', '⠓', '⠔', '⠕', '⠖', '⠗', '⡐','⡑', '⡒', '⡓', '⡔', '⡕', '⡖', '⡗'
    , '⠘', '⠙', '⠚', '⠛', '⠜', '⠝', '⠞', '⠟', '⡘','⡙', '⡚', '⡛', '⡜', '⡝', '⡞', '⡟'
    , '⠠', '⠡', '⠢', '⠣', '⠤', '⠥', '⠦', '⠧', '⡠','⡡', '⡢', '⡣', '⡤', '⡥', '⡦', '⡧'
    , '⠨', '⠩', '⠪', '⠫', '⠬', '⠭', '⠮', '⠯', '⡨','⡩', '⡪', '⡫', '⡬', '⡭', '⡮', '⡯'
    , '⠰', '⠱', '⠲', '⠳', '⠴', '⠵', '⠶', '⠷', '⡰','⡱', '⡲', '⡳', '⡴', '⡵', '⡶', '⡷'
    , '⠸', '⠹', '⠺', '⠻', '⠼', '⠽', '⠾', '⠿', '⡸','⡹', '⡺', '⡻', '⡼', '⡽', '⡾', '⡿'
    , '⢀', '⢁', '⢂', '⢃', '⢄', '⢅', '⢆', '⢇', '⣀','⣁', '⣂', '⣃', '⣄', '⣅', '⣆', '⣇'
    , '⢈', '⢉', '⢊', '⢋', '⢌', '⢍', '⢎', '⢏', '⣈','⣉', '⣊', '⣋', '⣌', '⣍', '⣎', '⣏'
    , '⢐', '⢑', '⢒', '⢓', '⢔', '⢕', '⢖', '⢗', '⣐','⣑', '⣒', '⣓', '⣔', '⣕', '⣖', '⣗'
    , '⢘', '⢙', '⢚', '⢛', '⢜', '⢝', '⢞', '⢟', '⣘','⣙', '⣚', '⣛', '⣜', '⣝', '⣞', '⣟'
    , '⢠', '⢡', '⢢', '⢣', '⢤', '⢥', '⢦', '⢧', '⣠','⣡', '⣢', '⣣', '⣤', '⣥', '⣦', '⣧'
    , '⢨', '⢩', '⢪', '⢫', '⢬', '⢭', '⢮', '⢯', '⣨','⣩', '⣪', '⣫', '⣬', '⣭', '⣮', '⣯'
    , '⢰', '⢱', '⢲', '⢳', '⢴', '⢵', '⢶', '⢷', '⣰','⣱', '⣲', '⣳', '⣴', '⣵', '⣶', '⣷'
    , '⢸', '⢹', '⢺', '⢻', '⢼', '⢽', '⢾', '⢿', '⣸','⣹', '⣺', '⣻', '⣼', '⣽', '⣾', '⣿');

  setOverline    = #$1B'[53m';
  setBold        = #$1B'[1m';
  setFaint       = #$1B'[2m';
  setItalic      = #$1B'[3m';
  setUnderline   = #$1B'[4m';
  setSlowBlink   = #$1B'[5m';
  setFastBlink   = #$1B'[6m';
  setInverse     = #$1B'[7m';
  setStrike      = #$1B'[9m';

  clearOverline  = #$1B'[55m';
  clearBold      = #$1B'[21m';
  clearFaint     = #$1B'[22m';
  clearItalic    = #$1B'[23m';
  clearUnderline = #$1B'[24m';
  clearBlink     = #$1B'[25m';
  clearInverse   = #$1B'[27m';
  clearStrike    = #$1B'[29m';
  clearFont      = #$1B'[10m';

  resetAllModes  = #$1B'[0m';
  resetColor     = #$1B'[39m';
  resetBackColor = #$1B'[49m';

  setClearLineEnd = #$1B'[0K';
  setClearLineStart = #$1B'[1K';
  setClearLine = #$1B'[2K';

  colorBlack   = 0;
  colorMaroon  = 1;
  colorGreen   = 2;
  colorOlive   = 3;
  colorNavy    = 4;
  colorMagenta = 5;
  colorTeal    = 6;
  colorSilver  = 7;
  colorGray    = 8;
  colorRed     = 9;
  colorLime    = 10;
  colorYellow  = 11;
  colorBlue    = 12;
  colorPurple  = 13;
  colorAqua    = 14;
  colorWhite   = 15;


function cursorMove(const direction:TCursorDirection; const count:integer=1):rawbytestring; overload;
function cursorMove(const relativeCol, relativeRow:integer):rawbytestring; overload;
procedure cursorShow(const visible:boolean);
procedure saveScreen();
procedure restorScreen();

function setColor4(const color:integer):rawbytestring;
function setBackColor4(const color:integer):rawbytestring;
function setColor5(const r,g,b:byte):rawbytestring;
function setBackColor5(const r,g,b:byte):rawbytestring;
function setColor(const r,g,b:byte):rawbytestring;
function setBackColor(const r,g,b:byte):rawbytestring;
function setGray(const brightness:byte):rawbytestring;
function setBackGray(const brightness:byte):rawbytestring;
function setCursorPos(const x, y:integer):rawbytestring;
function setCursorScrollDown(const count: integer =1):rawbytestring;
function setCursorScrollUp(const count: integer =1):rawbytestring;

procedure resetModes();
function setFont(font : byte):rawbytestring;

procedure cursorUp(const count: integer=1);
procedure cursorDown(const count: integer=1);
procedure cursorForward(const count: integer=1);
procedure cursorBack(const count: integer=1);
procedure cursorAbsPos(const x:integer=1; const y:integer=1);
procedure cursorScrollUp(const count:integer=1);
procedure cursorScrollDown(const count:integer=1);
procedure cursorClearDown();
procedure cursorClearUp();
procedure cursorClearScreen();
procedure cursorHome();
procedure saveCursorPos();
procedure restorCursorPos();
procedure setUTF8Console();

implementation

function intToStr(const i:integer):rawbytestring;
begin
  str(i, result)
end;

function cursorMove(const direction: TCursorDirection; const count: integer): rawbytestring;
begin
  case direction of
    cmUp:
      result := #$1B'['+intToStr(count)+'A';
    cmDown:
      result := #$1B'['+intToStr(count)+'B';
    cmForward:
      result := #$1B'['+intToStr(count)+'C';
    cmBackward:
      result := #$1B'['+intToStr(count)+'D';
  end;
end;

function cursorMove(const relativeCol, relativeRow: integer): rawbytestring;
var cm:TCursorDirection;
begin
  result :='';
  if (relativeCol<>0) then begin
    if relativeCol<0 then cm := cmBackward else cm := cmForward;
    result := cursorMove(cm, abs(relativeCol));
  end;

  if (relativeRow<>0) then begin
    if relativeRow<0 then cm := cmUp else cm := cmDown;
    result := result + cursorMove(cm, abs(relativeRow));
  end;
end;

procedure cursorShow(const visible: boolean);
begin
  if visible then
    write(#$1B'[?25h')
  else
    write(#$1B'[?25l')
end;

procedure saveScreen();
begin
  write(#$1B'[?47h');
end;

procedure restorScreen();
begin
  write(#$1B'[?47l');
end;


function setColor4(const color: integer): rawbytestring;
begin
  if color div 8>0 then
    result := #$1B'[9'+intToStr(color mod 8)+'m'
  else
    result := #$1B'[3'+intToStr(color)+'m';
end;

function setBackColor4(const color: integer): rawbytestring;
begin
  if color div 8>0 then
    result := #$1B'[10'+intToStr(color mod 8)+'m'
  else
    result := #$1B'[4'+intToStr(color)+'m';
end;

function setColor5(const r, g, b: byte): rawbytestring;
var c: byte;
begin
  c := 16 + round(b) + 6 * round(g) + 36 * round(r);
  result := #$1B'[38;5;'+intToStr(c)+'m';
end;

function setBackColor5(const r, g, b: byte): rawbytestring;
var c: byte;
begin
  c := 16 + round(b) + 6 * round(g) + 36 * round(r);
  result := #$1B'[48;5;'+intToStr(c)+'m';
end;

function setColor(const r, g, b: byte): rawbytestring;
begin
  result := #$1B'[38;2;'+intToStr(r)+';'+intToStr(g)+';'+intToStr(b)+'m';
end;

function setGray(const brightness: byte): rawbytestring;
begin
  case brightness of
    0 : result := #$1B'[30m';
    1..24: result := #$1B'[38;5;'+intToStr(brightness+231)+'m';
    25   : result := #$1B'[97m';
  end;
end;

function setBackGray(const brightness: byte): rawbytestring;
begin
  case brightness of
    0 : result := #$1B'[40m';
    1..24: result := #$1B'[48;5;'+intToStr(brightness+231)+'m';
    25   : result := #$1B'[107m';
  end;
end;

function setCursorPos(const x, y: integer): rawbytestring;
begin
  result := #$1B'[' + intTostr(y) + ';' + intToStr(x) + 'H'
end;

function setCursorScrollDown(const count: integer):rawbytestring;
begin
  result := #$1B'['+intToStr(count)+'T'
end;

function setCursorScrollUp(const count: integer):rawbytestring;
begin
  result := #$1B'['+intToStr(count)+'S'
end;

function setBackColor(const r, g, b: byte): rawbytestring;
begin
  result := #$1B'[48;2;'+intToStr(r)+';'+intToStr(g)+';'+intToStr(b)+'m';
end;

procedure resetModes;
begin
  write(resetAllModes);
end;

function setFont(font: byte): rawbytestring;
begin
  result := #$1B'[1'+intToStr(font)+'m';
end;

procedure cursorUp(const count: integer);
begin
  write(#$1B'[',count, 'A')
end;

procedure cursorDown(const count: integer);
begin
  write(#$1B'[',count, 'B')
end;

procedure cursorForward(const count: integer);
begin
  write(#$1B'[',count, 'C')
end;

procedure cursorBack(const count: integer);
begin
  write(#$1B'[',count, 'D')
end;

procedure cursorAbsPos(const x: integer; const y: integer);
begin
  write(#$1B'[', y, ';', x,'H')
end;

procedure cursorScrollUp(const count: integer);
begin
  write(#$1B'[',count,'S')
end;

procedure cursorScrollDown(const count: integer);
begin
  write(#$1B'[',count,'T')
end;

procedure cursorClearDown();
begin
  write(#$1B'[0J')
end;

procedure cursorClearUp();
begin
  write(#$1B'[1J')
end;

procedure cursorClearScreen();
begin
  write(#$1B'[2J')
end;

procedure cursorHome();
begin
  write(#$1B'[1H')
end;

procedure saveCursorPos();
begin
  write(#$1B'7')
end;

procedure restorCursorPos();
begin
  write(#$1B'8')
end;

procedure setUTF8Console();
begin
  writeln(#$1B'%G'); // set console UTF8;
end;

end.

