param([Parameter(Mandatory = $true)][string]$ImagePath, [string]$LanguageTag = '')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
    $null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
    $null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
    $null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
    $null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
    $null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
    $null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1
    } | Select-Object -First 1
    function Wait-WinRT($Operation, [Type]$ResultType) {
        $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
        $task.GetAwaiter().GetResult()
    }
    $file = Wait-WinRT ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Wait-WinRT ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Wait-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-WinRT ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try {
            if ($LanguageTag) {
                if ($LanguageTag -notmatch '^[a-zA-Z]{2,3}(-[a-zA-Z]{2,8})?$') { throw 'Invalid local OCR language' }
                $requested = [Windows.Globalization.Language]::new($LanguageTag)
                $language = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | Where-Object {
                    $_.LanguageTag -eq $LanguageTag
                } | Select-Object -First 1
                if ($null -eq $language) {
                    $language = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | Where-Object {
                        $_.LanguageTag.Split('-')[0] -eq $requested.LanguageTag.Split('-')[0]
                    } | Select-Object -First 1
                }
                if ($null -eq $language) { throw 'Requested local OCR language is unavailable' }
                $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
            } else {
                $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
            }
            if ($null -eq $engine) { throw 'No local OCR language pack' }
            $result = Wait-WinRT ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            [Console]::Write(($result.Lines | ForEach-Object { $_.Text }) -join [Environment]::NewLine)
        } finally { $bitmap.Dispose() }
    } finally { $stream.Dispose() }
} catch {
    # Caller records the backend state; image paths/content and OS diagnostics stay private.
    exit 1
}
