"""Rebuild subtitles/slide video with an existing AAC track; no TTS or network.

Inputs are preserved. Output must be a new directory outside the inputs/source.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, subprocess, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from wrap_captions import caption_lines
from build_video import timecode, ass_text, ffmeta_escape, probe_info

TARGET = 900.0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def write(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f: f.write(text)

def save(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets-dir', 'source-video', 'ffmpeg', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assets, source, ffmpeg, out = [p.resolve() for p in (args.assets_dir,args.source_video,args.ffmpeg,args.output_dir)]
    if os.path.lexists(out): raise ValueError('Output already exists; use a new directory')
    for protected in (assets,source.parent,ffmpeg.parent,Path(__file__).resolve().parent):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Output overlaps an input or executable/source directory')
    if not source.is_file() or not ffmpeg.is_file(): raise ValueError('Explicit video and FFmpeg files required')
    original = json.loads((assets/'captions.json').read_text('utf-8'))
    chapters = json.loads((assets/'chapters.json').read_text('utf-8'))
    if len(original)!=186 or len(chapters)!=20: raise ValueError('Expected 186 cues and twenty slides')
    previous_end=0.0
    for i,c in enumerate(original,1):
        if c['sequence']!=i or type(c['slide']) is not int or not 1<=c['slide']<=20:
            raise ValueError('Caption ordering/slide mismatch')
        start,end=c['start_seconds'],c['end_seconds']
        if not all(math.isfinite(v) for v in (start,end)) or not previous_end<=start<end<=TARGET:
            raise ValueError('Invalid or overlapping caption timings')
        if ''.join(c['display_lines'])!=c['caption']: raise ValueError('Displayed text differs from full caption')
        previous_end=end
    order=[c['slide'] for c in original]
    if order!=sorted(order) or set(order)!=set(range(1,21)): raise ValueError('Captions must cover twenty slides in order')
    for n,ch in enumerate(chapters,1):
        group=[c for c in original if c['slide']==n]
        end=next(c['start_seconds'] for c in original if c['slide']==n+1) if n<20 else TARGET
        if ch['slide']!=n or ch['start_seconds']!=group[0]['start_seconds'] or ch['end_seconds']!=end:
            raise ValueError('Chapter timing differs from cue boundaries')
        if ch['start_ms']!=round(ch['start_seconds']*1000) or ch['end_ms']!=round(end*1000):
            raise ValueError('Chapter milliseconds mismatch')
        if not (assets/f'slide_{n:02}.png').is_file(): raise ValueError('Missing slide PNG')
    cues=deepcopy(original)
    for c in cues: c['display_lines']=caption_lines(c['caption'])
    changed=[{'sequence':a['sequence'],'slide':a['slide'],'before':a['display_lines'],'after':b['display_lines']}
             for a,b in zip(original,cues) if a['display_lines']!=b['display_lines']]
    inputs=[assets/'captions.json',assets/'chapters.json',source,ffmpeg,Path(__file__).resolve(),
            Path(__file__).with_name('wrap_captions.py').resolve(),Path(__file__).with_name('build_video.py').resolve()]
    inputs += [assets/f'slide_{i:02}.png' for i in range(1,21)]
    manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs]
    out.mkdir(parents=True,exist_ok=False)
    started,executions=time.perf_counter(),[]
    def run(label,command):
        start=time.perf_counter()
        save(out/(label+'.command.json'),{'cwd':str(out),'argv':command})
        with (out/(label+'.stdout.log')).open('xb') as stdout,(out/(label+'.stderr.log')).open('xb') as stderr:
            result=subprocess.run(command,cwd=out,stdout=stdout,stderr=stderr)
        row={'label':label,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start}
        executions.append(row);save(out/(label+'.execution.json'),row);print(json.dumps(row),flush=True)
        if result.returncode: raise RuntimeError(label+' failed; see stderr log')
    try:
        save(out/'input_manifest.json',manifest);save(out/'caption_changes.json',changed)
        save(out/'captions.json',cues);save(out/'chapters.json',chapters)
        srt,vtt=[],['WEBVTT\n']
        for c in cues:
            words='\n'.join(c['display_lines'])
            srt.append(f"{c['sequence']}\n{timecode(c['start_seconds'])} --> {timecode(c['end_seconds'])}\n{words}\n")
            vtt.append(f"{timecode(c['start_seconds'],'vtt')} --> {timecode(c['end_seconds'],'vtt')}\n{words}\n")
        write(out/'captions.srt','\n'.join(srt));write(out/'captions.vtt','\n'.join(vtt))
        header='''[Script Info]
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
        events=[f"Dialogue: 0,{timecode(c['start_seconds'],'ass')},{timecode(c['end_seconds'],'ass')},Caption,,0,0,0,,{{\\an8\\pos(960,944)}}{ass_text(c['display_lines'])}" for c in cues]
        events.append('Dialogue: 1,0:00:00.00,0:15:00.00,Disclosure,,0,0,0,,{\\an2\\pos(960,1068)}合成語音示範 · v22 · 非真人上台計時')
        write(out/'captions.ass',header+'\n'.join(events)+'\n')
        concat=['ffconcat version 1.0']
        meta=[';FFMETADATA1','title=臺灣房價預測 v22 講解示範','comment=合成語音示範，非真人上台計時']
        for ch in chapters:
            name=f"slide_{ch['slide']:02}.png";shutil.copy2(assets/name,out/name)
            concat += [f"file '{name}'",f"duration {ch['end_seconds']-ch['start_seconds']:.12f}"]
            meta += ['[CHAPTER]','TIMEBASE=1/1000',f"START={ch['start_ms']}",f"END={ch['end_ms']}",'title='+ffmeta_escape(ch['title'])]
        concat.append("file 'slide_20.png'")
        write(out/'slides.ffconcat','\n'.join(concat)+'\n');write(out/'chapters.ffmeta','\n'.join(meta)+'\n')
        run('ffmpeg_version',[str(ffmpeg),'-version'])
        filters='scale=1664:936:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:0:color=black,setsar=1,fps=15,subtitles=captions.ass'
        run('encode',[str(ffmpeg),'-hide_banner','-nostdin','-n','-threads','4','-filter_threads','4',
            '-f','concat','-safe','0','-i','slides.ffconcat','-i',str(source),'-f','ffmetadata','-i','chapters.ffmeta',
            '-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2','-vf',filters,
            '-c:v','libx264','-preset','veryfast','-crf','22','-tune','stillimage','-pix_fmt','yuv420p','-threads','4',
            '-c:a','copy','-t',str(TARGET),'-movflags','+faststart','main.mp4'])
        for label,path in [('original_audio',source),('rebuilt_audio',out/'main.mp4')]:
            run(label,[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-c:a','copy','-f','hash','-hash','sha256','-'])
        old_hash=(out/'original_audio.stdout.log').read_text().strip()
        new_hash=(out/'rebuilt_audio.stdout.log').read_text().strip()
        if not re.fullmatch(r'SHA256=[0-9a-f]{64}',old_hash) or old_hash!=new_hash:
            raise ValueError('Compressed audio payload differs; inspect before continuing')
        run('decode',[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-xerror','-err_detect','explode','-threads','4',
            '-i','main.mp4','-map','0:v:0','-map','0:a:0','-progress','decode.progress.txt','-nostats','-f','null','-'])
        if 'progress=end' not in (out/'decode.progress.txt').read_text(): raise ValueError('Incomplete decode')
        run('probe',[str(ffmpeg),'-hide_banner','-nostdin','-i','main.mp4','-map','0:v:0','-frames:v','1','-f','null','-'])
        parsed=probe_info((out/'probe.stderr.log').read_text('utf-8',errors='replace'),chapters)
        if any(sha(row['path'])!=row['sha256'] for row in manifest): raise ValueError('Input changed during compilation')
        save(out/'receipt.json',{'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.perf_counter()-started,'executions':executions,
            'main_mp4':{'bytes':(out/'main.mp4').stat().st_size,'sha256':sha(out/'main.mp4'),
                'duration_seconds':parsed['duration_seconds'],'width':1920,'height':1080,'fps':15,'video_codec':'h264','audio_codec':'aac'},
            'captions':len(cues),'changed_wraps':len(changed),'embedded_chapters':len(parsed['chapters']),
            'caption_text_and_timing_unchanged':True,'slide_images_unchanged':True,
            'compressed_audio_payload_unchanged':True,'compressed_audio_sha256':old_hash.split('=')[1],
            'audio_reencoded':False,'new_tts_generated':False,'full_decode_passed':True,
            'human_audio_listening_performed':False,'real_human_delivery_timed':False,
            'alignment_method':'Preserved original estimated sentence timing, not forced alignment.'})
    except Exception as exc:
        save(out/'failure.json',{'status':'failed','error':str(exc),'executions':executions,'elapsed_seconds':time.perf_counter()-started})
        raise

if __name__=='__main__': main()
