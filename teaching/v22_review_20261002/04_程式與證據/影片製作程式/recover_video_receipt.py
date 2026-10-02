"""Recover a receipt after the known duplicated-chapter validator false failure.

Reads existing outputs and successful FFmpeg logs. Never reruns FFmpeg, playback,
or encoding, and never modifies the existing MP4 or failure evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, time
from pathlib import Path
from datetime import datetime, timezone
from build_video import sha, save, wav, probe_info, TARGET, FPS

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for name in ('receipt.json','recovery.json'):
        if (out/name).exists():
            raise ValueError('Refusing to overwrite '+name)
    started = time.perf_counter()
    before = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    failure = load(out/'failure.json')
    if failure['status'] != 'failed' or failure['error'] != 'Expected twenty embedded MP4 chapters':
        raise ValueError('This recovery applies only to the known chapter-count false failure')
    manifest = load(out/'input_manifest.json')
    original_compiler = Path(__file__).with_name('build_video_before_probe_fix.py')
    if sha(original_compiler) != manifest['compiler']['sha256']:
        raise ValueError('Preserved original compiler does not match original run')
    for key in ('input_json','tts_metadata','ffmpeg'):
        if sha(Path(manifest[key]['path'])) != manifest[key]['sha256']:
            raise ValueError('Input/executable identity mismatch: '+key)
    stages = ['ffmpeg_version','tempo','encode','decode','probe']
    if [x['label'] for x in failure['executions']] != stages:
        raise ValueError('Unexpected saved FFmpeg stage sequence')
    commands = {}
    for expected in failure['executions']:
        label = expected['label']
        stage = load(out/(label+'.execution.json'))
        command = load(out/(label+'.command.json'))
        if expected != stage or stage['exit_code'] != 0 or Path(command['cwd']).resolve() != out or Path(command['argv'][0]).resolve() != Path(manifest['ffmpeg']['path']).resolve():
            raise ValueError('Saved stage or executable mismatch: '+label)
        commands[label] = command['argv']
    encode, decode, probe = [commands[k] for k in ('encode','decode','probe')]
    if encode[-1] != 'main.mp4' or encode[encode.index('-t')+1] != str(TARGET) or 'captions.ass' not in encode[encode.index('-vf')+1] or 'narration.wav' not in encode:
        raise ValueError('Saved encode command does not match this deliverable')
    if decode[-2:] != ['null','-'] or '-xerror' not in decode or '0:v:0' not in decode or '0:a:0' not in decode or 'main.mp4' not in decode or any(x in decode for x in ('-t','-to','-frames:v','-frames:a')):
        raise ValueError('Saved decode was not an unlimited audio and video decode')
    if 'main.mp4' not in probe:
        raise ValueError('Saved probe used another file')
    progress = (out/'decode.progress.txt').read_text(encoding='utf-8')
    if not progress.rstrip().endswith('progress=end') or 'frame=13500' not in progress or 'out_time_us=900000000' not in progress:
        raise ValueError('Full decode progress does not finish at 15 minutes and 13,500 frames')
    if (out/'decode.stderr.log').read_bytes().strip():
        raise ValueError('Full decode logged errors')
    video = out/'main.mp4'
    if video.stat().st_size <= 0 or video.stat().st_mtime_ns > (out/'decode.execution.json').stat().st_mtime_ns:
        raise ValueError('Video is empty or has been modified after the saved decode')
    metadata = load(Path(manifest['tts_metadata']['path']))
    source = load(Path(manifest['input_json']['path']))
    segments = source['segments']
    if metadata['status'] != 'completed' or metadata['input_sha256'] != manifest['input_json']['sha256'] or len(metadata['files']) != len(segments):
        raise ValueError('TTS metadata/input mismatch')
    audio_dir = Path(manifest['tts_metadata']['path']).parent
    pieces, frame_cursor, settings = [], 0, None
    cues = load(out/'captions.json')
    timing = load(out/'timing_preflight.json')
    tempo = timing['atempo']
    if not .8 <= tempo <= 1.2 or len(cues) != len(segments):
        raise ValueError('Unexpected tempo or caption count')
    for index,(segment,item,identity,cue) in enumerate(zip(segments,metadata['files'],manifest['audio'],cues),1):
        path = (audio_dir/item['filename']).resolve()
        if path.parent != audio_dir.resolve() or sha(path) != item['sha256'] or item['sha256'] != identity['sha256'] or item['filename'] != identity['filename']:
            raise ValueError('Source WAV identity mismatch')
        current, frames, data = wav(path)
        settings = current if settings is None else settings
        if current != settings or current[:2] != (1,2) or not any(data) or frames != identity['frames']:
            raise ValueError('Source WAV format or frame mismatch')
        if item['source_segment'] != index or item['sequence'] != index or item['text'] != segment['text'].strip() or item['slide'] != segment['slide']:
            raise ValueError('Speech segment mapping mismatch')
        if cue['caption'] != segment['caption'] or cue['spoken_text'] != segment['text'] or cue['slide'] != segment['slide'] or cue['raw_start_frame'] != frame_cursor or cue['raw_end_frame'] != frame_cursor+frames:
            raise ValueError('Caption identity or frame range mismatch')
        if not math.isclose(cue['start_seconds'],frame_cursor/settings[2]/tempo,abs_tol=1e-9) or not math.isclose(cue['end_seconds'],(frame_cursor+frames)/settings[2]/tempo,abs_tol=1e-9):
            raise ValueError('Caption transformed timing mismatch')
        pieces.append(data); frame_cursor += frames
        if index < len(segments):
            silence = round(settings[2]*(.4 if segment['slide'] != segments[index]['slide'] else .08))
            pieces.append(b'\0'*(silence*2)); frame_cursor += silence
    tail = round(settings[2]*.5)
    pieces.append(b'\0'*(tail*2)); frame_cursor += tail
    raw_settings, raw_frames, raw_pcm = wav(out/'raw_narration.wav')
    if raw_settings != settings or raw_frames != frame_cursor or raw_pcm != b''.join(pieces) or not math.isclose(tempo,(frame_cursor/settings[2])/TARGET,abs_tol=1e-12):
        raise ValueError('Concatenated original narration does not match source WAVs and exact pauses')
    del pieces, raw_pcm
    tempo_settings, tempo_frames, tempo_pcm = wav(out/'tempo.wav')
    final_settings, final_frames, final_pcm = wav(out/'narration.wav')
    if tempo_settings != settings or final_settings != settings or final_frames != round(TARGET*settings[2]):
        raise ValueError('Final narration format/duration mismatch')
    target_bytes = final_frames*2
    if tempo_frames > final_frames and any(tempo_pcm[target_bytes:]):
        raise ValueError('Nonzero audio would have been trimmed')
    expected_pcm = tempo_pcm[:target_bytes] if tempo_frames >= final_frames else tempo_pcm+b'\0'*(target_bytes-len(tempo_pcm))
    if final_pcm != expected_pcm or any(final_pcm[-round(.15*settings[2])*2:]):
        raise ValueError('Final narration differs beyond silent tail adjustment')
    chapters = load(out/'chapters.json')
    for chapter in chapters:
        slide = chapter['slide']
        start = next(c['start_seconds'] for c in cues if c['slide'] == slide)
        end = next(c['start_seconds'] for c in cues if c['slide'] == slide+1) if slide < 20 else TARGET
        if chapter['start_ms'] != round(start*1000) or chapter['end_ms'] != round(end*1000):
            raise ValueError('Planned chapter and caption timing mismatch')
    parsed = probe_info((out/'probe.stderr.log').read_text(encoding='utf-8',errors='strict'),chapters)
    for image in load(out/'render_manifest.json'):
        if sha(Path(image['source_path'])) != image['sha256'] or sha(out/f"slide_{image['slide']:02}.png") != image['sha256']:
            raise ValueError('Slide render identity mismatch')
    for name,original_hash in before.items():
        if sha(out/name) != original_hash:
            raise ValueError('Existing output changed during recovery: '+name)
    elapsed = time.perf_counter()-started
    recovery = {'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
                'kind':'validator_only_recovery','elapsed_seconds':elapsed,
                'cause':'Original regex counted input chapters and FFmpeg output chapters together, producing forty instead of twenty.',
                'fix':'Validate only the input section and compare all twenty chapter times/titles against chapters.json.',
                'original_failure_preserved':True,'ffmpeg_rerun':False,'mp4_modified':False,
                'existing_files_unchanged_sha256':before,'parsed_input_chapters':parsed['chapters'],
                'original_compiler_sha256':sha(original_compiler),'corrected_compiler_sha256':sha(Path(__file__).with_name('build_video.py')),
                'recovery_script_sha256':sha(Path(__file__)),
                'full_decode_evidence':'Original unlimited video/audio decode exited zero, had empty error log and finished 13,500 frames at 900 seconds; video mtime predates its successful receipt.',
                'historical_identity_limit':'No MP4 hash was saved before the original validator failure; current MP4 hash is bound across this read-only recovery, not retroactively inserted into original decode logs.'}
    save(out/'recovery.json',recovery)
    save(out/'receipt.json',{'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
                            'recovered_validator_only':True,'original_failure_retained':'failure.json','recovery_evidence':'recovery.json',
                            'original_attempt_elapsed_seconds':failure['elapsed_seconds'],'recovery_elapsed_seconds':elapsed,
                            'executions':failure['executions'],
                            'main_mp4':{'bytes':video.stat().st_size,'sha256':before['main.mp4'],'duration_seconds':parsed['duration_seconds'],
                                        'width':1920,'height':1080,'fps':FPS,'video_codec':'h264','audio_codec':'aac'},
                            'main_slides':20,'sentence_count':len(cues),'embedded_chapters':20,'atempo':tempo,
                            'tts_voice':metadata.get('voice_name'),'tts_rate':metadata.get('rate'),'full_decode_passed':True,
                            'burned_subtitles':True,'external_subtitles':['captions.srt','captions.vtt'],
                            'real_human_delivery_timed':False,'human_audio_listening_performed':False,
                            'audio_playback_performed':False,'network_speech_service_used':False,
                            'disclosure':'合成語音示範 · v22 · 非真人上台計時'})
    print(json.dumps({'status':'completed','validator_only':True,'video_sha256':before['main.mp4'],'chapters':20,'elapsed_seconds':elapsed}))

if __name__ == '__main__':
    main()
