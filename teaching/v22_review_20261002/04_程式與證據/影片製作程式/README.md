# 講解影片程式與重建

影片已完成。最新成品是 [15分鐘MP4](../../05_講解影片/v22_15分鐘講解_合成旁白.mp4)，含20頁主講與186段字幕。10月2日修正73段斷行，保護Public、Private及中文術語；全文、時間與原AAC旁白音軌均保持。旁白是Windows本機合成語音，並非使用者錄音或真人排練計時。

## 最短重建方式

包內現在附有完整20頁PNG、字幕及章節時間，放在 [video_assets/manifest.json](video_assets/manifest.json) 列示的檔案中。`recompose_video.py` 使用這些素材，以及成品MP4裡已有的AAC音軌，在新目錄重建畫面與字幕；不需重新合成語音或安裝簡報渲染套件。

需要Python 3.10以上、支援libx264與subtitles/libass的FFmpeg，以及Microsoft JhengHei字型。原受測環境是Windows、Python 3.12、FFmpeg 7.1。FFmpeg未隨包附上，程式不會自動安裝。下列命令請在解壓後的審查資料夾根目錄執行：

```powershell
$pack = (Get-Location).Path
$videoSource = Join-Path $pack '04_程式與證據/影片製作程式'
$python = Read-Host 'Python執行檔完整路徑'
$ffmpeg = Read-Host 'FFmpeg執行檔完整路徑'
$result = Join-Path (Split-Path $pack -Parent) 'v22-video-rebuilt'
& $python -B -X utf8 (Join-Path $videoSource 'recompose_video.py') --assets-dir (Join-Path $videoSource 'video_assets') --source-video (Join-Path $pack '05_講解影片/v22_15分鐘講解_合成旁白.mp4') --ffmpeg $ffmpeg --output-dir $result
if ($LASTEXITCODE -ne 0) { throw '重建失敗，請查看新目錄的failure.json或終端訊息。' }
```

`$result`必須是尚不存在、且與輸入分離的新目錄。失敗資料保留，重試時換新目錄，不要刪去身分檢查。成功後可看`main.mp4`、SRT/VTT及`receipt.json`；程式核對186段文字與時間、20章節、15分鐘規格、全片解碼，以及輸入／輸出壓縮音軌SHA相同。這是使用保存素材重組影片，不是從文字重新合成聲音，也不是房價模型重訓。

## 若要換旁白或重畫投影片

[speech_input.json](speech_input.json)有186段固定配音文字與字幕，`text`含發音替換，`caption`保留稿面原文。[synthesize_segments.ps1](synthesize_segments.ps1)可用Windows的System.Speech與Microsoft Hanhan Desktop產生WAV和metadata；原186個WAV沒有打包。`build_video.py`可將新WAV及20張PNG編成影片，目前採用同層`wrap_captions.py`的斷行修正。

完整參數與重畫PPTX的方法見[分階段渲染指南](重新配音與投影片渲染指南.md)。若沿用本包PNG給`build_video.py`，先在新工作目錄複製成`slide-01.png`至`slide-20.png`（原素材名是`slide_01.png`）；`recompose_video.py`則直接讀原素材名。重新合成受Windows聲音與版本影響，不能保證與原音訊相同。字幕時間保留原句子邊界估計，沒有做逐字強制對齊，也未人工逐句聽音評分。

## 保留的歷史

`build_video_20261001.py`保存字幕改版前、已修復章節驗證器的程式。`build_video_before_probe_fix.py`與`recover_video_receipt.py`是首次編碼成功後驗證器誤判的歷史證據，勿當成一般重建流程。`technical_feasibility.json`與`technical_test_input.json`是最早單句語音測試，不表示正式影片尚未完成。

`../本次製作程式`中的prepare/assemble/finish/package是舊工作區整合紀錄，仍依賴當時檔案配置；新環境請用上方有明確參數的影片入口。完整新電腦安裝、從零配音／渲染／重訓尚未驗證。本GitHub目錄已納入這份審查影片與來源；原本機包和影片保持。


## 本次實測

已只用本包程式、20張PNG、字幕時間與本包MP4音軌，於新的輸出目錄完成重組，耗時52.693秒。新MP4整檔SHA與本包影片相同；詳見[核對紀錄](保存素材重組核對.json)與[執行收據](保存素材重組收據.json)。這項實測使用既有Python／FFmpeg，沒有重做文字配音、重新渲染PPTX或乾淨環境安裝。
