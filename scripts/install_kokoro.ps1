$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelDir = Join-Path $repoRoot "model"

$downloads = @(
    @{
        Name = "kokoro-v1.0.onnx"
        Url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        Path = Join-Path $modelDir "kokoro-v1.0.onnx"
    },
    @{
        Name = "voices-v1.0.bin"
        Url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        Path = Join-Path $modelDir "voices-v1.0.bin"
    }
)

Write-Host "Marvin Kokoro TTS installer"
Write-Host "Model directory: $modelDir"

if (-not (Test-Path -LiteralPath $modelDir)) {
    Write-Host "Creating model directory..."
    New-Item -ItemType Directory -Path $modelDir | Out-Null
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    Write-Host "Could not force TLS 1.2; continuing with system defaults."
}

foreach ($download in $downloads) {
    $targetPath = $download.Path
    $tempPath = "$targetPath.download"

    if (Test-Path -LiteralPath $targetPath) {
        $sizeMb = [Math]::Round((Get-Item -LiteralPath $targetPath).Length / 1MB, 2)
        Write-Host "Skipping $($download.Name); already exists ($sizeMb MB)."
        continue
    }

    if (Test-Path -LiteralPath $tempPath) {
        Remove-Item -LiteralPath $tempPath -Force
    }

    Write-Host "Downloading $($download.Name)..."
    Write-Host "  $($download.Url)"

    try {
        Invoke-WebRequest -Uri $download.Url -OutFile $tempPath -UseBasicParsing
        Move-Item -LiteralPath $tempPath -Destination $targetPath
        $sizeMb = [Math]::Round((Get-Item -LiteralPath $targetPath).Length / 1MB, 2)
        Write-Host "Saved $($download.Name) ($sizeMb MB)."
    } catch {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force
        }

        Write-Error (
            "Failed to download $($download.Name). " +
            "Check your internet connection and try again. Details: $($_.Exception.Message)"
        )
    }
}

Write-Host "Kokoro TTS files are ready."
