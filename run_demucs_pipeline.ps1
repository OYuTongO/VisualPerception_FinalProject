Set-Location "E:\code\visual_perception\finalProject"

# Step 1: split
Write-Host "=== Step 1: Splitting audio ===" -ForegroundColor Cyan
python split_chunks.py
if ($LASTEXITCODE -ne 0) { Write-Error "Split failed"; exit 1 }

# Step 2: demucs each chunk (each in a fresh process, no parent holding memory)
$chunks = Get-ChildItem "diana_voice_raw\chunks\chunk_*.wav" | Sort-Object Name
$total  = $chunks.Count
Write-Host "=== Step 2: Running demucs on $total chunks ===" -ForegroundColor Cyan

for ($i = 0; $i -lt $total; $i++) {
    $chunk = $chunks[$i].FullName
    # skip if already done
    $chunkName = $chunks[$i].BaseName
    $doneFlag  = "diana_voice_raw\chunks\separated\htdemucs\$chunkName\vocals.wav"
    if (Test-Path $doneFlag) {
        Write-Host "[$($i+1)/$total] $chunkName already done, skipping." -ForegroundColor Gray
        continue
    }
    Write-Host "[$($i+1)/$total] Processing $chunkName ..." -ForegroundColor Yellow
    python -m demucs --two-stems vocals -o "diana_voice_raw\chunks\separated" $chunk
    if ($LASTEXITCODE -ne 0) { Write-Error "demucs failed on $chunk"; exit 1 }
}

# Step 3: concatenate
Write-Host "=== Step 3: Concatenating vocals ===" -ForegroundColor Cyan
python concat_vocals.py
if ($LASTEXITCODE -ne 0) { Write-Error "Concat failed"; exit 1 }

Write-Host "=== All done! ===" -ForegroundColor Green
