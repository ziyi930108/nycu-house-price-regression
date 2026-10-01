"""Rebuild the frozen v22 presentation styles into a new directory.

Requires the Codex bundled Node/Python runtime, presentation skill and original
fonts. Runtime paths are explicit; research workspace files are never inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

STYLE_NAMES = {'A': '白底學術', 'B': '深色科技', 'C': '暖色雜誌'}
TEACHING = Path('teaching/v22_styles_20261001')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Frozen source changed; expected one anchor: {old}')
    return source.replace(old, new, 1)


def replace_assignment(source, pattern, value):
    result, count = re.subn(pattern, lambda _: value, source)
    if count != 1:
        raise ValueError(f'Frozen source changed; expected one assignment: {pattern}')
    return result


def adapt(source, builder):
    source = replace_assignment(source, r"const SKILL='[^'\r\n]+';",
                                'const SKILL=process.env.TEACHING_PRESENTATION_SKILL;')
    if builder:
        source = replace_once(source, "const ROOT=path.resolve(DIR,'../..');",
                              'const ROOT=process.env.TEACHING_REPOSITORY;')
        for relative in ['round79/results.json',
                         'kaggle_feedback/public_submissions_v19_v24_20260927.json']:
            source = replace_once(source, f"ROOT+'/outputs/{relative}'",
                                  f"ROOT+'/evidence/outputs/{relative}'")
    else:
        source = replace_assignment(source, r"const PY='[^'\r\n]+';",
                                    'const PY=process.env.RUNTIME_PYTHON;')
        source = replace_assignment(source, r"process\.env\.RUNTIME_NODE_MODULES='[^'\r\n]+';",
                                    "if(!process.env.RUNTIME_NODE_MODULES)throw Error('Missing bundled Node modules');")
    return source


def validate_output(output, protected):
    if os.path.lexists(output):
        raise ValueError(f'Output must not exist: {output}')
    for path in protected:
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f'Output overlaps a source or runtime directory: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--node-modules', type=Path, required=True)
    parser.add_argument('--presentation-skill', type=Path, required=True)
    parser.add_argument('--styles', nargs='+', choices=STYLE_NAMES, default=list(STYLE_NAMES))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if len(args.styles) != len(set(args.styles)):
        parser.error('Each style may appear only once')
    # Detect an existing dangling link before resolve() follows it.
    if os.path.lexists(args.output):
        parser.error(f'Output must not exist: {args.output}')
    repo, output = args.repository.resolve(), args.output.resolve()
    node, python = args.node.resolve(), args.python.resolve()
    modules, skill = args.node_modules.resolve(), args.presentation_skill.resolve()
    validate_output(output, [repo, modules, skill, node.parent, python.parent])
    source = repo / TEACHING
    required = [node, python, modules / '@oai/artifact-tool/package.json']
    required += [skill / 'container_tools' / n for n in
                 ['artifact_tool_utils.mjs', 'inspect_presentation_package_integrity.py',
                  'inspect_presentation_layout_geometry.py']]
    for path in required:
        if not path.is_file():
            raise ValueError(f'Missing runtime dependency: {path}')

    manifest_path = repo / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {e['path']: e['sha256'] for e in manifest['files']}
    inputs = [source / 'content/narrative.json', source / 'content/完整講稿.md',
              source / 'content/排練與問答.md', source / 'delivery_receipt.json',
              repo / 'evidence/outputs/round79/results.json',
              repo / 'evidence/outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json']
    for style in args.styles:
        inputs += [source / 'build' / style / n for n in ['build_deck.mjs', 'finalize_deck.mjs']]
    if 'C' in args.styles:
        inputs.append(source / 'build/C/assets/taipei_markus_winkler.jpg')
    hashes = {}
    for path in inputs:
        resolved = path.resolve()
        if not resolved.is_relative_to(repo):
            raise ValueError(f'Source escaped the repository: {path}')
        relative = path.relative_to(repo).as_posix()
        value = sha(path)
        if value != expected.get(relative):
            raise ValueError(f'Source identity mismatch: {relative}')
        hashes[relative] = value
    adapted = {}
    for style in args.styles:
        for name in ['build_deck.mjs', 'finalize_deck.mjs']:
            path = source / 'build' / style / name
            adapted[(style, name)] = adapt(path.read_text(encoding='utf-8-sig'), name == 'build_deck.mjs')
    env = os.environ.copy()
    env.update({'RUNTIME_NODE_MODULES': str(modules), 'RUNTIME_PYTHON': str(python),
                'TEACHING_PRESENTATION_SKILL': str(skill), 'TEACHING_REPOSITORY': repo.as_posix()})
    versions = {'node': subprocess.check_output([str(node), '--version'], text=True).strip(),
                'python': subprocess.check_output([str(python), '--version'], text=True).strip()}
    preflight = {'repository': str(repo), 'output': str(output), 'styles': args.styles,
                 'runtime_versions': versions, 'runtime_paths': {k: env[k] for k in
                 ['RUNTIME_NODE_MODULES', 'RUNTIME_PYTHON', 'TEACHING_PRESENTATION_SKILL']},
                 'required_fonts': ['Microsoft JhengHei', 'Consolas'] + (['PMingLiU'] if 'C' in args.styles else []),
                 'input_sha256': hashes, 'source_manifest_sha256': sha(manifest_path),
                 'runner_sha256': sha(Path(__file__)), 'new_ML_fits': 0}
    if args.check_only:
        print(json.dumps({'status': 'preflight_passed', **preflight}, ensure_ascii=False))
        return

    output.mkdir(parents=True, exist_ok=False)
    save(output / 'preflight.json', preflight)
    started = time.perf_counter()
    steps = []
    try:
        (output / 'deliverables').mkdir()
        (output / 'content').mkdir()
        for name in ['narrative.json', '完整講稿.md', '排練與問答.md']:
            shutil.copy2(source / 'content' / name, output / 'content' / name)
        for (style, name), value in adapted.items():
            path = output / 'build' / style / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as f:
                f.write(value)
        if 'C' in args.styles:
            asset = output / 'build/C/assets/taipei_markus_winkler.jpg'
            asset.parent.mkdir()
            shutil.copy2(source / 'build/C/assets/taipei_markus_winkler.jpg', asset)
        link = output / 'build/node_modules'
        if os.name == 'nt':
            def ps_quote(value): return "'" + str(value).replace("'", "''") + "'"
            command = ('New-Item -ItemType Junction -Path ' + ps_quote(link) +
                       ' -Target ' + ps_quote(modules) + ' -ErrorAction Stop | Out-Null')
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], check=True)
        else:
            link.symlink_to(modules, target_is_directory=True)
        for style in args.styles:
            for name in ['build_deck.mjs', 'finalize_deck.mjs']:
                command = [str(node), str(output / 'build' / style / name)]
                start = time.perf_counter()
                with (output / 'build' / style / (name + '.stdout.log')).open('xb') as out, \
                     (output / 'build' / style / (name + '.stderr.log')).open('xb') as err:
                    result = subprocess.run(command, cwd=output, env=env, stdout=out, stderr=err)
                steps.append({'style': style, 'step': name, 'command': command,
                              'exit_code': result.returncode, 'elapsed_seconds': time.perf_counter() - start})
                print(json.dumps(steps[-1], ensure_ascii=False), flush=True)
                if result.returncode:
                    raise RuntimeError(f'{style}/{name} failed; see saved stderr, do not reuse this output directory')
        for relative, value in hashes.items():
            if sha(repo / relative) != value:
                raise ValueError(f'Frozen source changed during build: {relative}')
        original = json.loads((source / 'delivery_receipt.json').read_text(encoding='utf-8'))
        outputs = []
        for style in args.styles:
            deck = output / 'deliverables' / f'臺灣房價預測_v22_{STYLE_NAMES[style]}_r2.pptx'
            original_deck = next(s for s in original['styles'] if s['style'] == style)
            validation = json.loads((output / 'build' / style / 'final_r2.validation.json').read_text(encoding='utf-8'))
            value = sha(deck)
            if validation['finalSha256'] != value or validation['packageIntegrity']['status'] != 'pass':
                raise ValueError(f'Finalizer validation mismatch: {style}')
            render_hashes = {f'slide-{n:02}': sha(output / 'build' / style / 'renders_r2' / f'slide-{n:02}.png')
                             for n in range(1, 24)}
            outputs.append({'style': style, 'path': deck.relative_to(output).as_posix(),
                            'sha256': value, 'bytes': deck.stat().st_size,
                            'original_pptx_sha256': original_deck['sha256'],
                            'pptx_byte_identical': value == original_deck['sha256'],
                            'all_23_render_bytes_identical': render_hashes == original_deck['render_sha256'],
                            'render_sha256': render_hashes})
        save(output / 'rebuild_receipt.json', {'status': 'completed',
             'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - started,
             'steps': steps, 'outputs': outputs, 'frozen_sources_unchanged': True,
             'all_project_inputs_from_repository': True, 'new_ML_fits': 0,
             'limitations': ['Uses the provided existing Codex runtime and installed fonts; no fresh installation tested.',
                             'Rendered image comparison does not prove desktop PowerPoint behavior.']})
        print(json.dumps({'status': 'completed', 'output': str(output)}, ensure_ascii=False))
    except Exception as exc:
        save(output / 'failure.json', {'status': 'failed', 'error': str(exc), 'steps': steps,
                                      'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
