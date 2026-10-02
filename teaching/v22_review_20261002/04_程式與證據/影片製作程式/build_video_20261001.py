"""Compile local sentence WAVs and twenty slide PNGs into a subtitled 15-minute MP4.

Uses the explicitly supplied FFmpeg executable, never installs or replaces runtime files.
No audio playback or claim of human listening is performed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, subprocess, time, wave
from datetime import datetime, timezone
from pathlib import Path

TARGET = 900.0
FPS = 15

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)

def wav(path):
    with wave.open(str(path), 'rb') as f:
        settings = (f.getnchannels(), f.getsampwidth(), f.getframerate())
        frames = f.getnframes()
        data = f.readframes(frames)
        if f.getcomptype() != 'NONE' or frames <= 0 or len(data) != frames*settings[0]*settings[1]:
            raise ValueError('Expected a complete nonempty PCM WAV: '+str(path))
    return settings, frames, data

def write_wav(path, settings, data):
    with Path(path).open('xb') as raw:
        with wave.open(raw, 'wb') as f:
            f.setnchannels(settings[0]); f.setsampwidth(settings[1]); f.setframerate(settings[2])
            f.writeframes(data)

def timecode(seconds, kind='srt'):
    unit = 100 if kind == 'ass' else 1000
    ticks = round(seconds*unit)
    hours, ticks = divmod(ticks, 3600*unit)
    minutes, ticks = divmod(ticks, 60*unit)
    whole, fraction = divmod(ticks, unit)
    if kind == 'ass':
        return f'{hours}:{minutes:02}:{whole:02}.{fraction:02}'
    separator = ',' if kind == 'srt' else '.'
    return f'{hours:02}:{minutes:02}:{whole:02}{separator}{fraction:03}'

def caption_lines(caption):
    if not isinstance(caption, str) or not caption.strip() or '\n' in caption or '\r' in caption:
        raise ValueError('Caption must be one nonempty line')
    if len(caption) > 52:
        raise ValueError('Caption exceeds two 26-character lines; split the input sentence first')
    return [caption] if len(caption) <= 26 else [caption[:26], caption[26:]]

def ass_text(lines):
    return r'\N'.join(x.replace('\\',r'\\').replace('{',r'\{').replace('}',r'\}') for x in lines)

def ffmeta_escape(value):
    return str(value).replace('\\','\\\\').replace('=','\\=').replace(';','\\;').replace('#','\\#').replace('\n',' ')

def probe_info(probe, chapters):
    """Validate only demuxed MP4 input, not FFmpeg's repeated output chapters."""
    if not probe.startswith('Input #0,'):
        raise ValueError('Expected the main MP4 as FFmpeg input zero')
    input_section = re.split(r'(?m)^(?:Stream mapping:|Output #)', probe, maxsplit=1)[0]
    match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',input_section)
    if not match:
        raise ValueError('Cannot read final MP4 input duration from FFmpeg')
    duration = int(match[1])*3600+int(match[2])*60+float(match[3])
    if abs(duration-TARGET) > .1 or not re.search(r'Video: h264[^\n]*1920x1080[^\n]*15 fps',input_section) or 'Audio: aac' not in input_section:
        raise ValueError('Unexpected final MP4 input duration, codecs, size, or fps')
    actual = re.findall(r'Chapter #0:(\d+): start ([\d.]+), end ([\d.]+)\s+Metadata:\s+title\s*:\s*([^\r\n]+)',input_section)
    if len(actual) != 20 or len(chapters) != 20:
        raise ValueError('Expected twenty embedded MP4 input chapters')
    result = []
    for index, ((n,start,end,title),expected) in enumerate(zip(actual,chapters)):
        if int(n) != index or expected['slide'] != index+1 or round(float(start)*1000) != expected['start_ms'] or round(float(end)*1000) != expected['end_ms'] or title.strip() != expected['title']:
            raise ValueError('MP4 input chapter differs from chapters.json on slide '+str(index+1))
        result.append({'slide':index+1,'start_ms':round(float(start)*1000),'end_ms':round(float(end)*1000),'title':title.strip()})
    return {'duration_seconds':duration,'chapters':result,'input_only':True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('audio-dir','input-json','renders','ffmpeg','output-dir'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    audio, source, renders, ffmpeg, out = [p.resolve() for p in (args.audio_dir,args.input_json,args.renders,args.ffmpeg,args.output_dir)]
    if os.path.lexists(out):
        raise ValueError('Output directory already exists; use a new directory')
    for protected in (audio, renders, source.parent, ffmpeg.parent):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Output overlaps an input or executable directory')
    if not ffmpeg.is_file():
        raise ValueError('Explicit FFmpeg executable is missing')
    out.mkdir(parents=True, exist_ok=False)
    started, executions = time.perf_counter(), []

    def run(label, command):
        beginning = time.perf_counter()
        save(out/(label+'.command.json'), {'cwd':str(out),'argv':command})
        with (out/(label+'.stdout.log')).open('xb') as stdout, (out/(label+'.stderr.log')).open('xb') as stderr:
            result = subprocess.run(command,cwd=out,stdout=stdout,stderr=stderr)
        row = {'label':label,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-beginning}
        executions.append(row); save(out/(label+'.execution.json'),row)
        print(json.dumps(row),flush=True)
        if result.returncode:
            raise RuntimeError(label+' failed; see its saved stderr log')

    try:
        payload = json.loads(source.read_text(encoding='utf-8-sig'))
        metadata_path = audio/'metadata.json'
        meta = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        segments, files = payload['segments'], meta['files']
        if meta['status'] != 'completed' or meta['input_sha256'] != sha(source):
            raise ValueError('Audio metadata does not identify this completed speech input')
        if not segments or len(files) != len(segments) or meta['completed_sentence_count'] != len(files) or meta['sentence_count'] != len(files):
            raise ValueError('Each source segment must produce exactly one completed WAV')
        slide_order = [x['slide'] for x in segments]
        if any(type(n) is not int or not 1 <= n <= 20 for n in slide_order) or slide_order != sorted(slide_order) or set(slide_order) != set(range(1,21)):
            raise ValueError('Expected all twenty main slides in order; no appendices')
        evidence = {'input_json':{'path':str(source),'sha256':sha(source)},
                    'tts_metadata':{'path':str(metadata_path),'sha256':sha(metadata_path)},
                    'compiler':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
                    'ffmpeg':{'path':str(ffmpeg),'sha256':sha(ffmpeg)},'audio':[],'renders':[]}
        settings, pieces, cues, frame_cursor = None, [], [], 0
        for index,(segment,item) in enumerate(zip(segments,files),1):
            if item['sequence'] != index or item['source_segment'] != index or item['slide'] != segment['slide'] or item['text'] != segment['text'].strip():
                raise ValueError('Sentence mapping or pronunciation text differs at segment '+str(index))
            caption = segment['caption']
            lines = caption_lines(caption)
            path = (audio/item['filename']).resolve()
            if path.parent != audio or sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
                raise ValueError('Audio file identity mismatch at '+str(index))
            current, frames, data = wav(path)
            if current[:2] != (1,2) or not any(data):
                raise ValueError('Expected audible nonzero mono 16-bit PCM')
            if settings is None:
                settings = current
            if current != settings:
                raise ValueError('All WAVs must use identical PCM parameters')
            expected = item['audio']
            if (expected['channels'],expected['bits_per_sample']//8,expected['sample_rate']) != current or expected['data_bytes'] != len(data) or abs(expected['duration_seconds']-frames/current[2]) > 1/current[2]:
                raise ValueError('Measured WAV does not match SAPI metadata')
            cues.append({'sequence':index,'slide':segment['slide'],'caption':caption,'display_lines':lines,
                         'spoken_text':segment['text'],'raw_start_frame':frame_cursor,'raw_end_frame':frame_cursor+frames,
                         'source_filename':item['filename']})
            pieces.append(data); frame_cursor += frames
            if index < len(segments):
                gap = .4 if segment['slide'] != segments[index]['slide'] else .08
                gap_frames = round(current[2]*gap)
                pieces.append(b'\0'*(gap_frames*2)); frame_cursor += gap_frames
            evidence['audio'].append({'filename':item['filename'],'sha256':item['sha256'],'frames':frames,'sample_rate':current[2]})
        tail_frames = round(settings[2]*.5)
        pieces.append(b'\0'*(tail_frames*2)); frame_cursor += tail_frames
        total_seconds = frame_cursor/settings[2]
        tempo = total_seconds/TARGET
        save(out/'input_manifest.json',evidence)
        save(out/'timing_preflight.json',{'raw_frames':frame_cursor,'sample_rate':settings[2],'raw_seconds':total_seconds,
                                        'target_seconds':TARGET,'atempo':tempo,'allowed_atempo':[.8,1.2],
                                        'sentence_gap_seconds':.08,'slide_gap_seconds':.4,'final_silence_seconds':.5})
        if not .8 <= tempo <= 1.2:
            raise ValueError(f'atempo {tempo:.6f} is outside 0.8..1.2; regenerate SAPI with another rate in a new audio directory')
        for cue in cues:
            cue['start_seconds'] = cue['raw_start_frame']/settings[2]/tempo
            cue['end_seconds'] = cue['raw_end_frame']/settings[2]/tempo
        chapters = []
        for number in range(1,21):
            chapter_cues = [c for c in cues if c['slide'] == number]
            start = chapter_cues[0]['start_seconds']
            end = next(c['start_seconds'] for c in cues if c['slide'] == number+1) if number < 20 else TARGET
            chapters.append({'slide':number,'title':f'第{number:02}頁','start_seconds':start,'end_seconds':end,
                             'start_ms':round(start*1000),'end_ms':round(end*1000),
                             'raw_start_frame':chapter_cues[0]['raw_start_frame']})
            image = renders/f'slide-{number:02}.png'
            if not image.is_file():
                raise ValueError('Missing slide render '+str(image))
            shutil.copy2(image,out/f'slide_{number:02}.png')
            evidence['renders'].append({'slide':number,'source_path':str(image),'sha256':sha(image)})
        save(out/'render_manifest.json',evidence['renders'])
        save(out/'captions.json',cues); save(out/'chapters.json',chapters)
        srt, vtt = [], ['WEBVTT\n']
        for cue in cues:
            words = '\n'.join(cue['display_lines'])
            srt.append(f"{cue['sequence']}\n{timecode(cue['start_seconds'])} --> {timecode(cue['end_seconds'])}\n{words}\n")
            vtt.append(f"{timecode(cue['start_seconds'],'vtt')} --> {timecode(cue['end_seconds'],'vtt')}\n{words}\n")
        write(out/'captions.srt','\n'.join(srt)); write(out/'captions.vtt','\n'.join(vtt))
        header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Microsoft JhengHei,42,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,8,48,48,944,1
Style: Disclosure,Microsoft JhengHei,18,&H00BBBBBB,&H00BBBBBB,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,20,20,12,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
        events = [f"Dialogue: 0,{timecode(c['start_seconds'],'ass')},{timecode(c['end_seconds'],'ass')},Caption,,0,0,0,,{{\\an8\\pos(960,944)}}{ass_text(c['display_lines'])}" for c in cues]
        events.append('Dialogue: 1,0:00:00.00,0:15:00.00,Disclosure,,0,0,0,,{\\an2\\pos(960,1068)}合成語音示範 · v22 · 非真人上台計時')
        write(out/'captions.ass',header+'\n'.join(events)+'\n')
        concat = ['ffconcat version 1.0']
        for chapter in chapters:
            concat += [f"file 'slide_{chapter['slide']:02}.png'",f"duration {chapter['end_seconds']-chapter['start_seconds']:.12f}"]
        concat.append("file 'slide_20.png'")
        write(out/'slides.ffconcat','\n'.join(concat)+'\n')
        meta_lines = [';FFMETADATA1','title=臺灣房價預測 v22 講解示範','comment=合成語音示範，非真人上台計時']
        for chapter in chapters:
            meta_lines += ['[CHAPTER]','TIMEBASE=1/1000',f"START={chapter['start_ms']}",f"END={chapter['end_ms']}",'title='+ffmeta_escape(chapter['title'])]
        write(out/'chapters.ffmeta','\n'.join(meta_lines)+'\n')
        write_wav(out/'raw_narration.wav',settings,b''.join(pieces)); del pieces
        run('ffmpeg_version',[str(ffmpeg),'-version'])
        run('tempo',[str(ffmpeg),'-hide_banner','-nostdin','-n','-threads','4','-i','raw_narration.wav','-af',f'atempo={tempo:.12f}','-c:a','pcm_s16le','tempo.wav'])
        processed_settings, processed_frames, processed = wav(out/'tempo.wav')
        if processed_settings != settings:
            raise ValueError('Tempo transform changed PCM parameters')
        target_frames = round(TARGET*settings[2]); target_bytes = target_frames*2
        if abs(processed_frames-target_frames)/settings[2] > .25:
            raise ValueError('Tempo duration differs by more than 250 ms; inspect before continuing')
        # Only exact silence can be removed. Speech is never shortened to fit the target.
        if processed_frames > target_frames and any(processed[target_bytes:]):
            raise ValueError('Target boundary would remove nonzero audio; refusing to trim speech')
        final_pcm = processed[:target_bytes] if processed_frames >= target_frames else processed+b'\0'*(target_bytes-len(processed))
        if any(final_pcm[-round(.15*settings[2])*2:]):
            raise ValueError('Expected at least 150 ms silence after the final spoken sentence')
        write_wav(out/'narration.wav',settings,final_pcm)
        save(out/'audio_timing.json',{'atempo_output_frames':processed_frames,'atempo_output_seconds':processed_frames/settings[2],
                                    'target_frames':target_frames,'final_seconds':target_frames/settings[2],
                                    'silence_adjustment_frames':target_frames-processed_frames,
                                    'trimmed_only_exact_zero_pcm':True,'final_150ms_are_silent':True,
                                    'alignment_method':'Original PCM sample positions divided by atempo; atempo is not a forced aligner.',
                                    'subtitle_precision_note':'SRT/VTT round to milliseconds; ASS to 10 ms; images at 15 fps. Sentence timings are estimated from transformed segment boundaries.'})
        filters = f'scale=1664:936:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:0:color=black,setsar=1,fps={FPS},subtitles=captions.ass'
        command = [str(ffmpeg),'-hide_banner','-nostdin','-n','-threads','4','-filter_threads','4',
                   '-f','concat','-safe','0','-i','slides.ffconcat','-i','narration.wav','-f','ffmetadata','-i','chapters.ffmeta',
                   '-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2','-vf',filters,
                   '-c:v','libx264','-preset','veryfast','-crf','22','-tune','stillimage','-pix_fmt','yuv420p','-threads','4',
                   '-c:a','aac','-b:a','128k','-ar','48000','-t',str(TARGET),'-movflags','+faststart','main.mp4']
        run('encode',command)
        run('decode',[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-xerror','-err_detect','explode','-threads','4',
                      '-i','main.mp4','-map','0:v:0','-map','0:a:0','-progress','decode.progress.txt','-nostats','-f','null','-'])
        run('probe',[str(ffmpeg),'-hide_banner','-nostdin','-i','main.mp4','-map','0:v:0','-frames:v','1','-f','null','-'])
        probe = (out/'probe.stderr.log').read_text(encoding='utf-8',errors='replace')
        parsed = probe_info(probe,chapters)
        duration = parsed['duration_seconds']
        if sha(source) != evidence['input_json']['sha256'] or sha(metadata_path) != evidence['tts_metadata']['sha256']:
            raise ValueError('Speech inputs changed during compilation')
        save(out/'receipt.json',{'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
                                 'elapsed_seconds':time.perf_counter()-started,'executions':executions,
                                 'main_mp4':{'bytes':(out/'main.mp4').stat().st_size,'sha256':sha(out/'main.mp4'),'duration_seconds':duration,
                                              'width':1920,'height':1080,'fps':FPS,'video_codec':'h264','audio_codec':'aac'},
                                 'main_slides':20,'sentence_count':len(cues),'embedded_chapters':20,'atempo':tempo,
                                 'tts_voice':meta.get('voice_name'),'tts_rate':meta.get('rate'),'full_decode_passed':True,
                                 'burned_subtitles':True,'external_subtitles':['captions.srt','captions.vtt'],
                                 'real_human_delivery_timed':False,'human_audio_listening_performed':False,
                                 'audio_playback_performed':False,'network_speech_service_used':False,
                                 'disclosure':'合成語音示範 · v22 · 非真人上台計時'})
    except Exception as exc:
        save(out/'failure.json',{'status':'failed','error':str(exc),'elapsed_seconds':time.perf_counter()-started,'executions':executions})
        raise

if __name__ == '__main__':
    main()
