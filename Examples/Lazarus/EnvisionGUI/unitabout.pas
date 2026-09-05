unit unitAbout;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls
  {$ifdef MSWINDOWS}
  , ShellApi
  {$else}
  , LCLIntf
  {$endif}
  ;

type

  { TfrmAbout }

  TfrmAbout = class(TForm)
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure FormPaint(Sender: TObject);
    procedure Label1Click(Sender: TObject);
    procedure Label4Click(Sender: TObject);
    procedure Panel1Click(Sender: TObject);
  private

  public

  end;

var
  frmAbout: TfrmAbout;

implementation

{$R *.lfm}

{ TfrmAbout }

procedure TfrmAbout.Panel1Click(Sender: TObject);
begin
  Close
end;

procedure TfrmAbout.FormPaint(Sender: TObject);
begin
  Canvas.GradientFill(ClientRect, $000044, $440000, TGradientDirection.gdVertical);
  Canvas.Pen.Color:=$CCCCCC;
  Canvas.Pen.Width:=1;
  Canvas.Brush.Style:=bsClear;
  Canvas.Rectangle(ClientRect);
end;

procedure TfrmAbout.FormCreate(Sender: TObject);
begin
  Close
end;

procedure TfrmAbout.Label1Click(Sender: TObject);
begin
  Close
end;

procedure TfrmAbout.Label4Click(Sender: TObject);
begin
  {$ifdef MSWINDOWS}
  ShellExecute(0, 'open', PCHAR(TLabel(Sender).Caption), '', '', 0);
  {$else}
  OpenURL(TLabel(Sender).Caption);
  {$endif}
end;

end.

