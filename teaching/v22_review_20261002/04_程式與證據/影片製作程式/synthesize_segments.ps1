[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InputJson,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$VoiceName = 'Microsoft Hanhan Desktop',
    [ValidateRange(-10,10)][int]$Rate = 0,
    [ValidateRange(1,100)][int]$Volume = 100,
    [ValidateSet(16000,22050,24000,44100,48000)][int]$SampleRate = 22050
)
# Local Windows speech only; source segments require {slide, text}.
# Sentence punctuation/newlines split segments into WAV files. No playback.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$inputPath = (Resolve-Path -LiteralPath $InputJson).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputPath) { throw "OutputDirectory exists; supply a new path: $outputPath" }
$source = [System.IO.File]::ReadAllText($inputPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
if (-not ($source.PSObject.Properties.Name -contains 'segments') -or @($source.segments).Count -eq 0) {
    throw 'Input must contain a nonempty segments array.'
}
$sentences = [System.Collections.Generic.List[object]]::new()
$segmentIndex = 0
foreach ($segment in @($source.segments)) {
    $segmentIndex++
    if (-not ($segment.PSObject.Properties.Name -contains 'slide') -or
        -not ($segment.PSObject.Properties.Name -contains 'text')) { throw "Segment $segmentIndex requires slide and text." }
    $slideNumber = 0
    if (-not [int]::TryParse([string]$segment.slide, [ref]$slideNumber) -or $slideNumber -lt 1) {
        throw "Segment $segmentIndex has an invalid positive slide number."
    }
    if ($segment.text -isnot [string] -or [string]::IsNullOrWhiteSpace($segment.text)) {
        throw "Segment $segmentIndex has empty or non-string text."
    }
    $parts = [regex]::Matches($segment.text, '[^\u3002\uff01\uff1f!?\uff1b;\r\n]+[\u3002\uff01\uff1f!?\uff1b;]*[\u300d\u300f\u201d\u2019"]*')
    foreach ($part in $parts) {
        $sentence = $part.Value.Trim()
        if (-not [string]::IsNullOrWhiteSpace($sentence)) {
            $sentences.Add([ordered]@{slide=$slideNumber;source_segment=$segmentIndex;text=$sentence})
        }
    }
}
if ($sentences.Count -eq 0) { throw 'No speakable sentences found.' }
function Get-WaveInfo([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $reader = [System.IO.BinaryReader]::new($stream)
    try {
        $tag = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        $null = $reader.ReadUInt32()
        $kind = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        if ($tag -ne 'RIFF' -or $kind -ne 'WAVE') { throw 'Not a RIFF WAV file.' }
        $format = $null; $dataBytes = [long]0
        while ($stream.Position + 8 -le $stream.Length) {
            $chunk = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $size = [long]$reader.ReadUInt32(); $payload = $stream.Position
            if ($payload + $size -gt $stream.Length) { throw 'Truncated WAV chunk.' }
            if ($chunk -eq 'fmt ') {
                if ($size -lt 16) { throw 'Invalid WAV format.' }
                $format = [ordered]@{
                    format_tag=[int]$reader.ReadUInt16(); channels=[int]$reader.ReadUInt16()
                    sample_rate=[int]$reader.ReadUInt32(); byte_rate=[int]$reader.ReadUInt32()
                    block_align=[int]$reader.ReadUInt16(); bits_per_sample=[int]$reader.ReadUInt16()
                }
            } elseif ($chunk -eq 'data') { $dataBytes += $size }
            $stream.Position = $payload + $size + ($size % 2)
        }
        if ($null -eq $format -or $format.format_tag -ne 1 -or $format.byte_rate -le 0 -or $dataBytes -le 0) {
            throw 'Expected nonempty uncompressed PCM audio.'
        }
        return [ordered]@{
            duration_seconds=$dataBytes/[double]$format.byte_rate; data_bytes=$dataBytes
            format_tag=$format.format_tag; channels=$format.channels; sample_rate=$format.sample_rate
            bits_per_sample=$format.bits_per_sample
        }
    } finally { $reader.Dispose(); $stream.Dispose() }
}
Add-Type -AssemblyName System.Speech
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
$receipt = $null
try {
    $voice = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Name -eq $VoiceName })
    if ($voice.Count -ne 1) { throw "Requested enabled voice not found: $VoiceName" }
    $synth.SelectVoice($VoiceName); $synth.Rate = $Rate; $synth.Volume = $Volume
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
        $SampleRate, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $null = [System.IO.Directory]::CreateDirectory($outputPath)
    $receipt = [ordered]@{
        schema_version='1.0'; status='running'; created_at_utc=[datetime]::UtcNow.ToString('o')
        input_path=$inputPath; input_sha256=(Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        powershell_version=$PSVersionTable.PSVersion.ToString(); engine='System.Speech.Synthesis.SpeechSynthesizer'
        voice_name=$synth.Voice.Name; voice_id=$synth.Voice.Id; voice_culture=$synth.Voice.Culture.Name
        rate=$Rate; volume=$Volume; sample_rate=$SampleRate; channels=1; bits_per_sample=16
        network_service_used=$false; sentence_count=$sentences.Count; completed_sentence_count=0
        total_audio_seconds=0.0; files=[System.Collections.Generic.List[object]]::new()
    }
    if ($source.PSObject.Properties.Name -contains 'purpose') { $receipt['purpose'] = $source.purpose }
    $metadataPath = Join-Path $outputPath 'metadata.json'
    [System.IO.File]::WriteAllText($metadataPath, ($receipt | ConvertTo-Json -Depth 12), $utf8)
    $timer = [System.Diagnostics.Stopwatch]::StartNew(); $index = 0
    foreach ($sentence in $sentences) {
        $index++
        $filename = 'slide_{0:D2}_sentence_{1:D3}.wav' -f $sentence.slide, $index
        $wavPath = Join-Path $outputPath $filename
        if (Test-Path -LiteralPath $wavPath) { throw "Refusing to overwrite $wavPath" }
        $synth.SetOutputToWaveFile($wavPath, $format)
        try { $synth.Speak($sentence.text) } finally { $synth.SetOutputToNull() }
        $wave = Get-WaveInfo $wavPath
        if ($wave.sample_rate -ne $SampleRate -or $wave.channels -ne 1 -or $wave.bits_per_sample -ne 16) {
            throw 'Audio does not match the requested PCM format.'
        }
        $receipt.files.Add([ordered]@{
            sequence=$index; slide=$sentence.slide; source_segment=$sentence.source_segment
            text=$sentence.text; filename=$filename; bytes=(Get-Item -LiteralPath $wavPath).Length
            sha256=(Get-FileHash -LiteralPath $wavPath -Algorithm SHA256).Hash.ToLowerInvariant(); audio=$wave
        })
        $receipt.completed_sentence_count = $index; $receipt.total_audio_seconds += $wave.duration_seconds
        [System.IO.File]::WriteAllText($metadataPath, ($receipt | ConvertTo-Json -Depth 12), $utf8)
    }
    $timer.Stop(); $receipt.status = 'completed'
    $receipt['elapsed_seconds'] = $timer.Elapsed.TotalSeconds
    $receipt['completed_at_utc'] = [datetime]::UtcNow.ToString('o')
    [System.IO.File]::WriteAllText($metadataPath, ($receipt | ConvertTo-Json -Depth 12), $utf8)
    [pscustomobject]@{status=$receipt.status;metadata=$metadataPath;sentences=$index;audio_seconds=$receipt.total_audio_seconds}
} catch {
    if ($null -ne $receipt -and (Test-Path -LiteralPath $outputPath)) {
        $receipt.status = 'failed'; $receipt['error'] = $_.Exception.Message
        [System.IO.File]::WriteAllText((Join-Path $outputPath 'metadata.json'), ($receipt | ConvertTo-Json -Depth 12), $utf8)
    }
    throw
} finally { $synth.Dispose() }
