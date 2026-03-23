(*
  <a Quick JSON parser/generator for FPC and Delphi>

  usage :

```
    var
      json:TJSON;
    begin
      // direct string parse
      json := TJSON.parse('{"foo":"bar", "pi":3.14, "days": 365, "other":{"hello":"world"}}');

      // or load from file
      //json := TJSON.loadFromFile('path_to_json.json');

      //this  will print : elements = 4
      writeln('elements = ', json.count);

      // the json object can be assigned (casted) to a double, int64, string or an array of variants;
      // this will print : bar is jtString
      writeln(variant(json['foo']), ' is ', json['foo'].jsonType);
      // or simply
      // writeln(json['foo'].value, ' is ', json['foo'].jsonType);


      // this will print : 3.141592 is now jtFloat and 365 is jtInteger
      json['pi'] := 3.141592; // overwrite an existing key
      json['years'] := 365; // no previous key? a new element will be created
      writeln(variant(json['foo']), ' is now a ', json['foo'].jsonType, ' and ', json['years'].value, ' is ', json['years'].jsonType);

      // this will remove 'bar'
      json.remove('bar');


      // this will print back the original JSON string
      writeln(json.stringify());

      // and so on you get the idea...
    end;
```

//****************************************************************************
  Copyright (C) <2025> <rHaitham Shatti> <haitham.shatti at gmail dot com>

  This library is free software; you can redistribute it and/or modify it
  under the terms of the GNU Library General Public License as published by
  the Free Software Foundation; either version 2 of the License, or (at your
  option) any later version with the following modification:

  As a special exception, the copyright holders of this library give you
  permission to link this library with independent modules to produce an
  executable, regardless of the license terms of these independent modules,and
  to copy and distribute the resulting executable under terms of your choice,
  provided that you also meet, for each linked independent module, the terms
  and conditions of the license of that module. An independent module is a
  module which is not derived from or based on this library. If you modify
  this library, you may extend this exception to your version of the library,
  but you are not obligated to do so. If you do not wish to do so, delete this
  exception statement from your version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Library General Public License
  for more details.

  You should have received a copy of the GNU Library General Public License
  along with this library; if not, write to the Free Software Foundation,
  Inc., 51 Franklin Street - Fifth Floor, Boston, MA 02110-1335, USA.
//****************************************************************************
*)


unit quickjson;

{$ifdef FPC}
  {$mode Delphi}
  {$modeswitch advancedrecords}
  {$modeswitch typehelpers}
  {$ModeSwitch nestedprocvars}
{$endif}
{$pointermath on}
{$H+}
{$C+} // assertions always ON
interface

uses
  SysUtils, Variants;


const
  NUL = #0;
  SPACE = ' ';
  CARRIAGE_RETURN = #13;
  LINE_FEED = #10;
  TAB = #9;
  COLON = ':';
  OBJ_OPEN = '{';
  OBJ_CLOSE = '}';
  ARR_OPEN = '[';
  ARR_CLOSE = ']';
  ESCAPE = '\';
  QUOTE = '''';
  DBLQUOTE = '"';
  COMMA = ',';
  OPEN_BRACKETS='[{';
  CLOSE_BRACKETS=']}';
  MAX_ERR_MSG = 32;
  WHITE_SPACES = [SPACE, CARRIAGE_RETURN, LINE_FEED, TAB]; // a set of [space, line feed, carriage return, tab ]
  MAX_NESTING = $100;
  ELPS = '...';
type

  { TJSON }

  TJSON=record
  //const MAX_STACK = 128;
  type
    TJSONArray = TArray<TJSON>;
    TJSONType = (jtNone, jtNull, jtString, jtInteger, jtFloat, jtBoolean, jtArray, jtObject);
    //TNestedStack = array[0..MAX_STACK] of TToken;
  private
    function GetElement(key: variant): TJSON;
    procedure SetElement(key: variant; AValue: TJSON);
  public
    jsonType : TJSONType;
    name : string;
    Value : variant;
    //innerText : PCHAR;
    //innerTextLen : int64;
    //innerStr : string;
    childObjs: TJSONArray;
    function count():int64;
    function isArray():boolean;
    function keyExist(const key: variant): boolean;
    function get(const key:variant; const aDefault:TJSON):TJSON;
    function stringify():string;
    procedure saveToFile(const filename: string);

    // remove will return true on success, false if no key found
    function remove(const key:string):boolean; overload;
    function remove(const index:int64):boolean; overload;
    property element[const key:variant]:TJSON read GetElement write Setelement ;default;
    class function parse(jsonText:string):TJSON; overload; static;
    class function LoadFromFile(const filename:string):TJSON; static;  overload;

    class operator implicit(const val: TArray<variant>):TJSON;

    class operator implicit(const val: TArray<TJSON>):TJSON;
    class operator implicit(const val: TJSON):TArray<variant>;
    class operator implicit(const val: variant):TJSON;
    class operator implicit(const val: TJSON):variant;

    class operator implicit(const val: TJSON):TArray<string>;
    class operator implicit(const val: TJSON):TArray<int64>;
    class operator implicit(const val: TJSON):TArray<double>;
    class operator implicit(const val: TJSON):TArray<single>;

    class operator implicit(const val: TJSON):int64;
    class operator implicit(const val: TJSON):longint;
    class operator implicit(const val: TJSON):double;
    class operator implicit(const val: TJSON):string;
    class operator implicit(const val: TJSON):boolean;

    class operator implicit(const val: int64):TJSON;
    class operator implicit(const val: longint):TJSON;
    class operator implicit(const val: double):TJSON;
    class operator implicit(const val: string):TJSON;
    class operator implicit(const val: boolean):TJSON;


  end;

implementation

// instead of using the internal <pos> function
function indexOf(const needle:char; const heystack:string):int64;
var i:int64;
begin
  result :=0;
  for i:=1 to length(heystack) do
    if heystack[i] = needle then
      exit(i)
end;

// removes escape "\" character when used
function c_sanitize(const str:string; const aEsc:char=ESCAPE):string;
var i, j:int64;
begin
  setLength(result, length(str));
  j:=1;
  i:=1;
  while i<=length(str) do begin
    if (i<length(str)) and (str[i]=aEsc) then begin
      case LowerCase(str[i+1]) of
        ESCAPE:
          begin
            result[j]:=ESCAPE;
            inc(i);
          end;

        't':
          begin
            result[j]:=TAB;
            inc(i);
          end;

        'n':
          begin
            result[j]:=LINE_FEED;
            inc(i);
          end;
        'r':
          begin
            result[j]:=CARRIAGE_RETURN;
            inc(i);
          end;
        else
          begin
            result[j]:=str[i+1];
            inc(i);
          end
      end
    end else
      result[j]:=str[i];
    inc(i);
    inc(j);
  end;
  setLength(result, j-1)
end;

function c_desanitize(const str:string):string;
var i, j:int64;
begin
  result:='';
  j:=1;
  i:=1;
  while i<=length(str) do begin
    case str[i] of
      CARRIAGE_RETURN:
        result := result + ESCAPE+'r';
      LINE_FEED:
        result := result + ESCAPE+'n';

      DBLQUOTE,TAB, ESCAPE:
        result := result + ESCAPE+str[i]

      else
        result := result + str[i]
    end;
    inc(i);
  end
end;

// checka if the string is quoted and unquotes it if so
function c_unquote(const str:string; const assertQuote:boolean=false; const aQuote:char=DBLQUOTE):string;
begin
  result := c_sanitize(str);
  if result ='' then exit;
  if assertQuote then
    assert(str[1] = aQuote, 'ERROR : Invalide string format:'+sLineBreak+copy(str, 1, 4*MAX_ERR_MSG)+ELPS);
  if (length(str)>1) and (str[1] = aQuote) then begin
    assert(str[length(str)] = aQuote, 'ERROR : [c_unquote] unclosed quote:'+sLineBreak+copy(str, 1, 4*MAX_ERR_MSG)+ELPS);
    result := copy(result, 2, length(result)-2)
  end;
end;

function c_quote(const str:string; const aQuote:char=DBLQUOTE):string;
begin
  result := aQuote+c_desanitize(str)+aQuote ;
end;

function splitBy(str:string; out keys:TArray<string>;const seperator:char = COMMA; const openBrackets: string = OPEN_BRACKETS; const closeBrackets:string = CLOSE_BRACKETS; const aQuote:char = DBLQUOTE; const aEsc:char = ESCAPE):TArray<string>;
var
  openBracketIdx, closeBracketIdx : array [0..MAX_NESTING-1] of int64;  // make space for a quickstack
  start, i, j, k, escPos:int64;
  nestLevel : int64; // stack level
  inQuote, quotePresented, tokenPresented, secondWhiteSpaces, colonPresented : boolean;
  key : string;
begin
  keys := nil;
  inQuote := false;
  quotePresented := false;
  tokenPresented := false;
  secondWhiteSpaces := false;
  colonPresented := false;
  result := nil;
  escPos := -1;
  nestLevel := -1;
  start:=1;
  i:=1;
  assert(length(openBrackets)=length(closeBrackets), 'ERROR: Number of open/close brackets must be equal.');
  while i <= length(str) do begin

    // skip whitespaces
    if str[i] in WHITE_SPACES then begin
      if tokenPresented and not inQuote then
        secondWhiteSpaces := true;
      while (i<=length(str)) and (str[i] in WHITE_SPACES) do
        inc(i);
      continue
    end;


    // skip escapes
    if inQuote and (str[i]=aEsc) and (i<length(str)) {and ((str[i+1]=aEsc) or (str[i+1]=aQuote) or (str[i+1] in (openBrackets+closeBrackets)))} then begin
      inc(i,2);
      continue
    end;

    if str[i] = aQuote then begin
      assert(inQuote or not quotePresented, 'ERROR : unexpected open quote at :'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);

      quotePresented := true;
      tokenPresented := true;
      inQuote := not inQuote;
      inc(i);
      //if inQuote then begin
      //  while true do begin
      //    j := pos(aQuote, str, i);
      //    if (j>escPos) and (escPos<>0) then
      //      escPos := pos(aESC, str, i);
      //    assert(j>0, 'ERROR : string ended with unclosed quote!'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
      //    if (j<escPos) or (escPos=0) then begin
      //      i := j+1;
      //      inQuote := false;
      //      break
      //    end
      //    else if (escPos>0) and (escPos<length(str)) then
      //      i := escPos+2
      //  end;
      //end;
      continue
    end;

    //skip string literals
    if inQuote then begin
      inc(i);
      continue
    end;

    // check for open braket
    j := indexOf(str[i], openBrackets);
    if j>0 then begin
       assert(nestLevel<MAX_NESTING, 'ERROR : Maximum brackets nesting is reached:'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
       assert(not (quotePresented or secondWhiteSpaces), 'ERROR : unexpected open bracket at :'+sLineBreak+copy(str, i, 32)+ELPS) ;

       quotePresented:=false;
       secondWhiteSpaces:=false;
       inc(nestLevel);
       openBracketIdx[nestLevel] := j;
       inc(i);
       continue
    end;

    // check for close bracket
    j := indexOf(str[i], closeBrackets);
    if j>0 then begin
       assert((nestLevel>=0) and (j=openBracketIdx[nestLevel]), 'ERROR : Unexpected close bracket "'+str[i]+'" at :'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
       dec(nestLevel);
       secondWhiteSpaces:=true;
       inc(i);
       continue
    end;

    if (str[i]=seperator) then begin

      quotePresented:=false;
      tokenPresented:=false;
      secondWhiteSpaces:=false;
      if (nestLevel=-1) then begin
        colonPresented:=false;
        insert(trim(copy(str, start, i-start)), result, length(result));
        start := i+1;
      end;
      inc(i);
      continue
    end;

    if (str[i]=COLON) then begin
      quotePresented:=false;
      tokenPresented:=false;
      secondWhiteSpaces:=false;
      if nestLevel=-1 then begin
        assert(not colonPresented, 'ERROR : Unexpected colon ":" at :'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
        key := trim(copy(str, start, i-start));
        assert((length(key)>1) and (key[1]=aQuote) and (key[length(key)]=aQuote), 'ERROR : Invalid key at :'+sLineBreak+copy(str, start, MAX_ERR_MSG)+ELPS);
        insert(key, keys, length(keys));
        start := i+1;
        colonPresented := true;
      end;
      inc(i);
      continue
    end;
    assert(not(tokenPresented and secondWhiteSpaces), 'ERROR : Unexpected character at :'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
    tokenPresented:= true;
    inc(i)
  end;

  //check last round
  if (not inQuote) and (nestLevel=-1) then begin
    str := trim(copy(str, start, i-start));
    if str<>'' then
      insert(str, result, length(result));
    quotePresented:=false;
  end else
    if inQuote then
      assert(false, 'ERROR : string ended with unclosed quote!'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS)
    else
      assert(false, 'ERROR : string ended with unclosed bracket!'+sLineBreak+copy(str, i, MAX_ERR_MSG)+ELPS);
end;

{ TJSON }

function TJSON.GetElement(key: variant): TJSON;
var i:int64;
  obj: TArray<TJSON>;
begin
  result := default(TJSON);
  assert(assigned(childObjs), 'ERROR : empty object!');
  case tvardata(key).vtype of
    varString:
      begin
        assert(jsonType in [jtNone, jtObject],'ERROR : JSON is not of object type');
        if jsonType=jtNone then begin
          obj := childObjs[0].childObjs
        end
        else begin
          obj := childObjs;
        end;
        for i:=0 to high(Obj) do // todo implement a hashmap instead of scan search
          if sameStr(Obj[i].name, key) then begin
            result := Obj[i];
            break
          end;
      end;
    varInt64, varInteger, varSmallInt, varShortInt:
      begin
        assert(jsonType in [jtNone, jtObject, jtArray],'ERROR : JSON is not of objects type');
        //assert(int64(key)<length(chldObjs),'ERROR : out of index!');
        if jsonType=jtNone then begin
          obj := childObjs[0].childObjs
        end
        else begin
          obj := childObjs;
        end;
        result := Obj[int64(key)];
      end
    else
      assert(false, 'ERROR: Invalid name!');
  end;
  // traverse deeper if no object type associated
  while (result.jsonType=jtNone) and assigned(result.childObjs) do begin
    result:= result.childObjs[0];
  end
end;

procedure TJSON.SetElement(key: variant; AValue: TJSON);
var i:int64;
begin
  case tvardata(key).vtype of
    varString:
      begin
        assert(jsonType in [jtNone, jtObject],'ERROR : target JSON is not of object type!');
        jsonType:=jtObject;
        for i:=0 to high(childObjs) do // todo implement a hashmap instead of scan search
          if sameStr(childObjs[i].name, key) then begin
            AValue.name := key;
            childObjs[i] := AValue;
            exit
          end;
        AValue.name := key;
        insert(AValue, childObjs, length(childObjs));

      end;
    varInt64:
      begin
        assert(jsonType in [jtNone, jtArray],'ERROR : target JSON is not of array type!');
        jsonType:=jtArray;
        if int64(key) < length(childObjs) then
          childObjs[int64(key)] := AValue
        else
          insert(AValue, childObjs, int64(key))
      end
    else
      assert(false, 'ERROR: Invalid name!');
  end;
end;

function TJSON.count(): int64;
begin
  result := length(childObjs)
end;

function TJSON.isArray(): boolean;
begin
  result := jsonType=jtArray;
end;

function TJSON.keyExist(const key: variant): boolean;
var i:int64;
begin
  result := false;
  case tvardata(key).vtype of
    varString:
      begin
        assert(jsonType in [jtNone, jtObject],'ERROR : JSON is not of object type');
        for i:=0 to high(childObjs) do // todo implement a hashmap instead of scan search
          if SameStr(childObjs[i].name, key) then begin
            exit(true)
          end;
      end;
    varInt64:
      begin
        assert(jsonType in [jtNone, jtArray],'ERROR : JSON is not of array type');
        result := int64(key) < length(childObjs)
      end
    else
      assert(false, 'ERROR: Invalid key!');
  end;
end;

function TJSON.get(const key:variant; const aDefault:TJSON):TJSON;
var i:int64;
  obj: TArray<TJSON>;
  found:boolean;
begin
  result := default(TJSON);
  found := false;
  assert(assigned(childObjs), 'ERROR : empty object!');
  case tvardata(key).vtype of
    varString:
      begin
        assert(jsonType in [jtNone, jtObject],'ERROR : JSON is not of object type');
        if jsonType=jtNone then begin
          obj := childObjs[0].childObjs
        end
        else begin
          obj := childObjs;
        end;
        for i:=0 to high(Obj) do // todo implement a hashmap instead of scan search
          if sameStr(Obj[i].name, key) then begin
            result := Obj[i];
            found := true;
            break
          end;

      end;
    varInt64, varInteger, varSmallInt, varShortInt:
      begin
        assert(jsonType in [jtNone, jtObject, jtArray],'ERROR : JSON is not of objects type');
        //assert(int64(key)<length(chldObjs),'ERROR : out of index!');
        if jsonType=jtNone then begin
          obj := childObjs[0].childObjs
        end
        else begin
          obj := childObjs;
        end;
        result := Obj[int64(key)];
      end
    else
      assert(false, 'ERROR: Invalid name!');
  end;
  // traverse deeper if no object type associated
  if found then
    while (result.jsonType=jtNone) and assigned(result.childObjs) do begin
      result:= result.childObjs[0];
    end
  else result := aDefault
end;

function TJSON.stringify(): string;
var
  i:int64;
  fr:double;
begin
  result :='';
  case jsonType of
    jtNone:  (* it's an object/array container *)
    begin
      if name<>'' then
        result :=c_quote(name)+':';
      for i:=0 to high(childObjs) do
        result := result + childObjs[i].stringify() + COMMA;
      delete(result, length(result),1)
    end;

    jtObject:
      begin
        if name=''then
          result :=OBJ_OPEN
        else
          result :=OBJ_OPEN+c_quote(name)+':';
        if assigned(childObjs) then begin
          for i:=0 to high(childObjs) do
            result := result + childObjs[i].stringify() + COMMA;
          result[length(result)]:=OBJ_CLOSE
        end else
          result := result + OBJ_CLOSE;
      end;
    jtArray:
      begin
        if name=''then
          result :=ARR_OPEN
        else
          result :=c_quote(name)+':[';
          for i:=0 to high(childObjs) do
            result := result + childObjs[i].stringify() + COMMA;
        result[length(result)]:=ARR_CLOSE
      end;
    jtInteger, jtBoolean:
      begin
        if name='' then result := VarToStr(value)
        else result := c_quote(name)+':'+varToStr(value);
      end;
    jtFloat:
      begin
        //fr := frac(value);
        if name='' then begin
          result := varToStr(value);
          if fr=0 then result := result+'.0'
        end
        else begin
          result := c_quote(name)+':'+varToStr(value);
          if fr=0 then result := result+'.0'
        end;
      end;
    jtString:
      begin
        if name='' then result := c_quote(VarToStr(value))
        else result := c_quote(name)+':'+c_quote(varToStr(value));
      end;
    else
      //assert(false, 'ERROR : Object is empty')
  end;
end;

procedure TJSON.saveToFile(const filename: string);
var f:textFile;
begin
  try
    assignFile(f, filename);
    Rewrite(f);
    writeln(f, stringify());
  finally
    closeFile(f)
  end;
end;

function TJSON.remove(const key: string): boolean;
var
  i:int64;
begin
  assert(jsonType=jtObject, 'ERROR : JSON is not of object type!');
  result := false;
  for i:=0 to high(childObjs) do // todo implement as hashmap instead of scan
    if sameStr(childObjs[i].name, key) then begin
      delete(childObjs, i, 1);
      exit(true)
    end;
end;

function TJSON.remove(const index: int64): boolean;
var
  i:int64;
begin
  assert(jsonType=jtArray, 'ERROR : JSON is not of array type!');
  result := false;
  if (index>=0) and (index<length(childObjs)) then
    delete(childObjs, index, 1)
end;

class function TJSON.parse(jsonText: string): TJSON;
const ERR = 'ERROR : Invalid JSON format!';
var
  vals : array of string;
  keys: array of string;
  sVal:string;
  iVal, i, t:int64;
  fVal:double;
  bVal:boolean;
  elementPtr : ^TJSON;
begin
  jsonText:= trim(jsonText);
  result := default(TJSON);
  assert(length(jsonText)>1, ERR);
  case jsonText[1] of
    OBJ_OPEN :
      begin
        assert(jsonText[length(jsonText)]=OBJ_CLOSE,ERR+ ' incorrect character at the end of the string, must be a curly bracket.');
        result.jsonType:=jtObject;
        //t:=GetTickCount64;
        vals := splitBy(copy(jsonText,2, length(jsonText)-2), keys);
        //t:= GetTickCount64-t;
        if assigned(vals) then
          assert(assigned(keys), 'ERROR : no key specified:'+sLineBreak+copy(jsonText, 1, MAX_ERR_MSG)+ELPS);
        assert(length(keys)=length(vals), 'ERROR : Unable to parse :'+sLineBreak+copy(jsonText, 1, MAX_ERR_MSG)+ELPS);
        for i:=0 to high(vals) do begin
          setLength(result.childObjs, length(result.ChildObjs)+1);
          elementPtr := @result.childObjs[high(result.childObjs)];
          elementPtr.name:= c_unquote(keys[i]);
          if tryStrToInt64(vals[i], iVal) then begin
            elementPtr.value := iVal;
            elementPtr.jsonType:=jtInteger;
          end else
          if TryStrToBool(vals[i], bVal) then begin
            elementPtr.value := bVal;
            elementPtr.jsonType:=jtBoolean;
          end else
          if SameText(vals[i],'null') then begin
            elementPtr.value := Null;
            elementPtr.jsonType:=jtNull;
          end else
          if tryStrToFloat(vals[i], fVal) then begin
            elementPtr.value := fVal;
            elementPtr.jsonType:=jtFloat;
          end else
          if vals[i][1] in [OBJ_OPEN,ARR_OPEN] then
            insert(TJSON.parse(vals[i]), elementPtr.childObjs, length(elementPtr.childObjs))
          else begin
            elementPtr.value:=c_unquote(vals[i], true);
            elementPtr.jsonType:=jtString;
          end;
        end;
      end ;
    ARR_OPEN :
      begin
        assert(jsonText[length(jsonText)]=ARR_CLOSE, ERR+ ' incorrect character at the end of the string, must be a square bracket.');
        result.jsonType:=jtArray;
        //t:=GetTickCount64;
        vals := splitBy(copy(jsonText,2, length(jsonText)-2), keys);
        //t := GetTickCount64-t;
        assert(not assigned(keys), 'ERROR : incorrect JSON format, key''s are not allowed in arrayes! '+sLineBreak+copy(jsonText, 1, MAX_ERR_MSG)+ELPS);
        for i:=0 to high(vals) do begin
          setLength(result.childObjs, length(result.ChildObjs)+1);
          elementPtr := @result.childObjs[high(result.childObjs)];
          if tryStrToInt64(vals[i], iVal) then begin
            elementPtr.value := iVal;
            elementPtr.jsonType:=jtInteger;
          end else
          if TryStrToBool(vals[i], bVal) then begin
            elementPtr.value := bVal;
            elementPtr.jsonType:=jtBoolean;
          end else
          if SameText(vals[i],'null') then begin
            elementPtr.value := Null;
            elementPtr.jsonType:=jtNull;
          end else
          if tryStrToFloat(vals[i], fVal) then begin
            elementPtr.value := fVal;
            elementPtr.jsonType:=jtFloat;
          end else
          if vals[i][1] in [OBJ_OPEN,ARR_OPEN] then
            insert(TJSON.parse(vals[i]), elementPtr.childObjs, length(elementPtr.childObjs))
          else begin
            elementPtr.value:=c_unquote(vals[i], true);
            elementPtr.jsonType:=jtString;
          end;
        end;
      end;
    else
      assert(false, ERR);
    end;
end;

class function TJSON.LoadFromFile(const filename: string): TJSON;
var f:file;
  s,l:string;
begin
  s := '';
  assert(FileExists(filename),'ERROR : File not found:'+sLineBreak+filename);
  try
    assignfile(f, filename);
    reset(f, 1);
    setLength(s, fileSize(f));
    blockread(f, s[1], length(s));
  finally
    closeFile(f)
  end;
  result := TJSON.parse(S);
end;

class operator TJSON.implicit(const val: TArray<variant>): TJSON;
var
  i: Int64;
begin
  result := default(TJSON);
  setLength(result.childObjs, length(val));
  result.jsonType:=jtArray;
  for i:=0 to high(val) do
    result.childObjs[i] := val[i];
end;

class operator TJSON.implicit(const val: TJSON):TArray<string>;
var i: int64;
begin
  assert(val.jsonType=jtArray,'ERROR : JSON is not of array type!');
  setlength(result, val.count);
  for i:=0 to high(result) do
    result[i]:=val.childObjs[i].value
end;

class operator TJSON.implicit(const val: TJSON):TArray<int64>;
var i: int64;
begin
  assert(val.jsonType=jtArray,'ERROR : JSON is not of array type!');
  setlength(result, val.count);
  for i:=0 to high(result) do
    result[i]:=val.childObjs[i].value
end;

class operator TJSON.implicit(const val: TJSON):TArray<double>;
var i: int64;
begin
  assert(val.jsonType=jtArray,'ERROR : JSON is not of array type!');
  setlength(result, val.count);
  for i:=0 to high(result) do
    result[i]:=val.childObjs[i].value
end;

class operator TJSON.implicit(const val: TJSON):TArray<single>;
var i: int64;
begin
  assert(val.jsonType=jtArray,'ERROR : JSON is not of array type!');
  setlength(result, val.count);
  for i:=0 to high(result) do
    result[i]:=val.childObjs[i].value
end;

class operator TJSON.implicit(const val: TJSON): int64;
begin
  assert(val.jsonType=jtInteger, 'ERROR : element is not of an integer!');
  result := val.value
end;

class operator TJSON.implicit(const val: TJSON): LongInt;
begin
  assert(val.jsonType=jtInteger, 'ERROR : element is not of an integer!');
  result := val.value
end;

class operator TJSON.implicit(const val: TJSON): TArray<variant>;
var i:int64;
begin
  setlength(result, length(val.childObjs));
  for i:=0 to high(val.childObjs) do
    result[i] := val.childObjs[i].Value;
end;

class operator TJSON.implicit(const val: variant): TJSON;
begin
  result := default(TJSON);
  case TVarData(val).vtype of
    varShortInt, varSmallInt, varInteger, varInt64 :
      begin
        result.jsonType:=jtInteger;
        result.Value := int64(Val);
      end;
    varSingle, varDouble:
      begin
        result.jsonType:=jtFloat;
        result.Value:=double(val);
      end;
    varString:
      begin
        result.jsonType:=jtString;
        result.Value:=string(val);
      end;
    else
      assert(false, 'ERROR : value is not assignable to JSON!');
  end;
end;

class operator TJSON.implicit(const val: TJSON): variant;
begin
  result := val.Value;
end;

class operator TJSON.implicit(const val: TJSON): double;
begin
  assert(val.jsonType=jtFloat, 'ERROR : element is not of a float!');
  result := val.value
end;

class operator TJSON.implicit(const val: TJSON): string;
begin
  assert(val.jsonType=jtString, 'ERROR : element is not of a string!');
  result := val.value
end;

class operator TJSON.implicit(const val: TJSON): boolean;
begin
  assert(val.jsonType=jtBoolean, 'ERROR : element is not of a boolean!');
  result := val.value
end;

class operator TJSON.implicit(const val: int64): TJSON;
begin
  result := default(TJSON);
  result.jsonType := jtInteger;
  result.value := val
end;

class operator TJSON.implicit(const val: longint): TJSON;
begin
  result := default(TJSON);
  result.jsonType := jtInteger;
  result.value := val
end;

class operator TJSON.implicit(const val: double): TJSON;
begin
  result := default(TJSON);
  result.jsonType := jtFloat;
  result.value := val
end;

class operator TJSON.implicit(const val: string): TJSON;
begin
  result := default(TJSON);
  result.jsonType := jtString;
  result.value := val
end;

class operator TJSON.implicit(const val: boolean): TJSON;
begin
  result := default(TJSON);
  result.jsonType := jtBoolean;
  result.value := val
end;

class operator TJSON.implicit(const val: TArray<TJSON>): TJSON;
begin
  result := default(TJSON);
  result.jsonType:=jtArray;
  result.childObjs := val;
end;

// ********* test *********
//var
//  json:TJSON;
//  s : string;
//  ii: string;
//
//initialization
//
  //json := TJSON.parse('{           }');
  //json := TJSON.parse('{"foo":"bar", "pi":3.14, "bool":true, "days": 365, "other":["hello","world"]}');
//  json['pi'] := 3.141592; // overwrite existing key
//  json['years'] := 365; // no previous key? a new element will be added
//  ii := json.get('foo', 'koo'); // return <aDefault> if not found
//  writeln(variant(json['pi']), ' is now a ', json['pi'].jsonType, ' and ', json['years'].value, ' is ', json['years'].jsonType);
//  s:=json.stringify();
end.

