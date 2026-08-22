param(
    [string]$SourceWorld,
    [string]$OutputWorld
)
$ErrorActionPreference = 'Stop'
$text = Get-Content -LiteralPath $SourceWorld -Raw
$marker = 'DEF HUMAN_VIEWPOINT Viewpoint {'
$index = $text.IndexOf($marker)
if($index -lt 0){ throw 'HUMAN_VIEWPOINT marker not found.' }
$obstacle = @'
DEF CONTROLLED_SAFETY_OBSTACLE Solid {
  translation -8.95 0.50 -5.40
  children [ Shape { geometry Box { size 0.40 0.80 1.20 } } ]
  boundingObject Box { size 0.40 0.80 1.20 }
}
'@
$text = $text.Insert($index, $obstacle)
[IO.File]::WriteAllText($OutputWorld, $text, [Text.UTF8Encoding]::new($false))
