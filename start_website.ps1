$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Start-Process "http://127.0.0.1:8123/site/"
python -m http.server 8123 --directory $projectRoot
