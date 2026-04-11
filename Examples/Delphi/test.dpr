program test;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils, quicknn_transformers;

begin
  try
    { TODO -oUser -cConsole Main : Insert code here }
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.
