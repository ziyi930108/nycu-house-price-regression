"""Build three notes-only review decks. Run the skill marker first, separately."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from preserve_package import note_order, save, sha, text, verify

NAMES = {'A':'白底學術', 'B':'深色科技', 'C':'暖色雜誌'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-dir','work-dir','runtime-node','runtime-python','runtime-node-modules','skill-dir'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--reference-receipt', type=Path, help='Optional original delivery_receipt.json for render identity checking')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without authoring or output writes')
    args = parser.parse_args()
    source, work, node, python, modules, skill = [p.resolve() for p in (args.source_dir,args.work_dir,args.runtime_node,args.runtime_python,args.runtime_node_modules,args.skill_dir)]
    if source.is_relative_to(work) or work.is_relative_to(source):
        raise ValueError('Source and work directories must be separate')
    for protected in (modules, skill, node.parent, python.parent):
        if work.is_relative_to(protected) or protected.is_relative_to(work):
            raise ValueError('Work directory overlaps runtime')
    if os.path.lexists(work / 'build'):
        raise ValueError('Build directory already exists; use a new work directory')
    for dependency in (node, python, modules/'@oai/artifact-tool/package.json', skill/'container_tools/artifact_tool_utils.mjs', skill/'container_tools/inspect_presentation_package_integrity.py', skill/'container_tools/inspect_presentation_layout_geometry.py'):
        if not dependency.is_file():
            raise ValueError('Missing runtime dependency: ' + str(dependency))
    plan_path, narrative_path = work/'content/notes_plan.json', work/'content/narrative.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    narrative = json.loads(narrative_path.read_text(encoding='utf-8'))
    if plan.get('narrative_sha256') != sha(narrative_path):
        raise ValueError('Notes plan narrative identity mismatch')
    if [x['style'] for x in plan['styles']] != list(NAMES):
        raise ValueError('Expected three styles A, B, C in order')
    slides = narrative['slides']
    if [x['number'] for x in slides] != list(range(1,24)) or [x['is_appendix'] for x in slides] != [False]*20+[True]*3:
        raise ValueError('Expected twenty main slides and three appendices')
    identities = {str(plan_path):sha(plan_path), str(narrative_path):sha(narrative_path)}
    entries = {}
    for entry in plan['styles']:
        key = entry['style']
        expected_name = f'臺灣房價預測_v22_{NAMES[key]}_15分鐘版.pptx'
        if entry['source_file'] != expected_name:
            raise ValueError('Unexpected source file for ' + key)
        original = source / expected_name
        if not original.resolve().is_relative_to(source) or sha(original) != entry['source_sha256']:
            raise ValueError('Source identity mismatch for ' + key)
        notes, changed = entry['notes'], entry['changed_slide_numbers']
        if len(notes) != 23 or not all(isinstance(x,str) and x for x in notes):
            raise ValueError('Expected 23 nonempty notes')
        if not changed or changed != sorted(set(changed)) or any(type(i) is not int or not 1 <= i <= 20 for i in changed):
            raise ValueError('Expected sorted unique main-slide changes')
        for slide, note in zip(slides, notes):
            if not slide['script'] or note.count(slide['script']) != 1:
                raise ValueError('Narrative script missing or duplicated in notes')
        with ZipFile(original) as z:
            order = note_order(z)
            actual = [i+1 for i, part in enumerate(order) if text(z.read(part)) != notes[i]]
            if actual != changed:
                raise ValueError('Declared changes disagree with source notes: ' + key)
        final = work/'deliverables'/expected_name.replace('_15分鐘版.pptx','_審查版.pptx')
        if os.path.lexists(final):
            raise ValueError('Final deck already exists: ' + str(final))
        identities[str(original)] = sha(original)
        entries[key] = entry
    reference = None
    if args.reference_receipt:
        reference = {x['style']:x for x in json.loads(args.reference_receipt.read_text(encoding='utf-8'))['styles']}
        if set(reference) != set(NAMES):
            raise ValueError('Reference receipt style mismatch')
        for key, entry in entries.items():
            if reference[key]['sha256'] != entry['source_sha256']:
                raise ValueError('Render reference uses another source deck')
        identities[str(args.reference_receipt.resolve())] = sha(args.reference_receipt)
    builder = Path(__file__).resolve().parent
    for name in ('run_build.py','preserve_package.py','edit_notes.mjs','finalize_notes.mjs'):
        identities[str(builder/name)] = sha(builder/name)
    preflight = {'status':'preflight_passed', 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                 'inputs_sha256':identities, 'styles':list(entries),
                 'changed_slide_numbers':{k:e['changed_slide_numbers'] for k,e in entries.items()},
                 'runtime_paths':{'node':str(node),'python':str(python),'node_modules':str(modules),'skill':str(skill)},
                 'marker_must_have_been_run_separately':True}
    if args.check_only:
        print(json.dumps(preflight,ensure_ascii=False,indent=2))
        return
    build = work/'build'
    build.mkdir(parents=False,exist_ok=False)
    (work/'deliverables').mkdir(exist_ok=True)
    save(build/'preflight.json',preflight)
    started, steps, outputs = time.perf_counter(), [], []
    try:
        for name in ('edit_notes.mjs','finalize_notes.mjs'):
            shutil.copy2(builder/name,build/name)
        link = build/'node_modules'
        if os.name == 'nt':
            def quote(value):
                return "'" + str(value).replace("'", "''") + "'"
            subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command','New-Item -ItemType Junction -Path '+quote(link)+' -Target '+quote(modules)+' -ErrorAction Stop | Out-Null'],check=True)
        else:
            link.symlink_to(modules,target_is_directory=True)
        env = os.environ.copy()
        env.update({'REVIEW_WORK_DIR':str(work),'REVIEW_SOURCE_DIR':str(source),'REVIEW_SKILL_DIR':str(skill),'RUNTIME_PYTHON':str(python),'RUNTIME_NODE_MODULES':str(modules)})
        for key, entry in entries.items():
            style_build = build/key
            style_build.mkdir()
            commands = [('edit',[str(node),str(build/'edit_notes.mjs'),key]),
                        ('preserve',[str(python),'-B','-X','utf8',str(builder/'preserve_package.py'),'--source-dir',str(source),'--work-dir',str(work),'--style',key]),
                        ('finalize',[str(node),str(build/'finalize_notes.mjs'),key])]
            for label, command in commands:
                step_start = time.perf_counter()
                with (style_build/(label+'.stdout.log')).open('xb') as stdout, (style_build/(label+'.stderr.log')).open('xb') as stderr:
                    process = subprocess.run(command,cwd=work,env=env,stdout=stdout,stderr=stderr)
                step = {'style':key,'stage':label,'exit_code':process.returncode,'elapsed_seconds':time.perf_counter()-step_start}
                steps.append(step)
                save(style_build/(label+'.execution.json'),step)
                print(json.dumps(step),flush=True)
                if process.returncode:
                    raise RuntimeError('Step failed; inspect saved logs. Rebuild in a new work directory.')
            final = work/'deliverables'/entry['source_file'].replace('_15分鐘版.pptx','_審查版.pptx')
            package = verify(source/entry['source_file'],final,entry)
            validation = json.loads((style_build/'final.validation.json').read_text(encoding='utf-8'))
            if validation['finalSha256'] != sha(final) or validation['packageIntegrity']['status'] != 'pass':
                raise ValueError('Finalizer receipt mismatch')
            render_hashes = {f'slide-{i:02}':sha(style_build/'renders'/f'slide-{i:02}.png') for i in range(1,24)}
            if reference and render_hashes != reference[key]['render_sha256']:
                raise ValueError('Rendered PNG bytes differ from the original edition')
            outputs.append({'style':key,'path':final.relative_to(work).as_posix(),'sha256':sha(final),
                            'slides':23,'render_sha256':render_hashes,
                            'render_bytes_identical_to_reference':True if reference else None,**package})
        if any(sha(Path(p)) != h for p,h in identities.items()):
            raise ValueError('An input changed during the build')
        save(build/'execution.json',{'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
                                    'elapsed_seconds':time.perf_counter()-started,'steps':steps,'outputs':outputs,
                                    'new_ML_fits':0,'PowerPoint_desktop_tested':False})
    except Exception as exc:
        save(build/'failure.json',{'status':'failed','error':str(exc),'steps':steps,'elapsed_seconds':time.perf_counter()-started})
        raise

if __name__ == '__main__':
    main()
