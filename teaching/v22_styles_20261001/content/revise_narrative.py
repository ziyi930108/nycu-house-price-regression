"""Create a new teaching edition; never overwrite the frozen source content."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/v22_full_style_20261001/content/narrative.json'
SOURCE_SCRIPT = ROOT / 'outputs/v22_full_style_20261001/content/完整講稿.md'
SOURCE_QA = ROOT / 'outputs/v22_beginner_20261001/deliverables/臺灣房價預測_v22_排練與問答.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def han(value: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', value))


def mmss(seconds: int) -> str:
    return f'{seconds // 60}:{seconds % 60:02d}'


def once(value: str, old: str, new: str) -> str:
    if value.count(old) != 1:
        raise ValueError(f'Expected exactly one source passage: {old[:45]}')
    return value.replace(old, new, 1)


def write_new(path: Path, value: str) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(value)


def main() -> None:
    targets = [OUT / name for name in (
        'narrative.json', '完整講稿.md', '排練與問答.md', 'content_revision_receipt.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Refusing to overwrite an existing content edition')
    sources = [SOURCE, SOURCE_SCRIPT, SOURCE_QA]
    before_sha = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    new = copy.deepcopy(original)
    by_number = {slide['number']: slide for slide in new['slides']}

    slide = by_number[13]
    anchor = '三條分支各自再加入四分之一的LightGBM預測，主要部分占四分之三。'
    addition = ('用一筆教學假設來算：同一間房，單價CatBoost換回總價後估一千萬元，'
                'LightGBM估一千兩百萬元。前者取七成五，後者取兩成五，相加就是一千零五十萬元。'
                '這是一條分支的初估，還要進入下一頁的修正。')
    slide['script'] = once(slide['script'], anchor, anchor + addition)
    slide['takeaway'] = '六個模型先換回總價，再按固定比例形成三條分支的初估。'
    slide['key_points'] = [
        '三個 CatBoost、一個 LightGBM、兩個神經網路',
        '三條分支各取主要估價 75%，共同 LightGBM 25%',
        '教學假設：1,000萬 × 75% + 1,200萬 × 25% = 1,050萬元初估',
    ]
    slide['why'] = ('分開解釋六模型和三分支，以同一間房的總價示例算一次75／25，'
                    '再銜接後續修正。例子只說明運算，沒有證明這組比例最佳。')

    slide = by_number[14]
    anchor = '第二步在參考來源較弱時，再利用粗略道路位置附近的歷史資訊補充，並考慮城市、建物型態與用途是否相容。'
    addition = '空間修正會同時參考這條分支的原初估，以及第一步的成交修正結果。'
    slide['script'] = once(slide['script'], anchor, anchor + addition)
    slide['script'] += '三條分支各自完成這個流程，手上仍有三個總價，下一頁才取中位數。'
    slide['takeaway'] = '每條分支各自修正，完成後仍有三個總價。'
    slide['key_points'] = [
        '每支先參考同建案、地址或道路的可比成交',
        '空間修正同時看原初估與成交修正結果，支持不足時回退',
        '三支各自完成修正，最後才一起取中位數',
    ]
    slide['why'] = ('補足初估、成交參考和空間修正的資料流，明說每支各自處理、'
                    '修正後仍是三個價格。保留當期訓練來源、支持條件及道路精度限制。')

    for slide in new['slides']:
        slide['spoken_han_characters'] = han(slide['script'])
        slide['seconds'] = 0 if slide['is_appendix'] else (
            int(math.floor(slide['spoken_han_characters'] / 285 * 60 / 5 + 0.5)) * 5)
    main_slides = [slide for slide in new['slides'] if not slide['is_appendix']]
    before_count = sum(han(slide['script']) for slide in original['slides'] if not slide['is_appendix'])
    count = sum(slide['spoken_han_characters'] for slide in main_slides)
    seconds = sum(slide['seconds'] for slide in main_slides)
    meta = new['meta']
    meta.update(
        edition='v22_teaching_refinement_20261001',
        main_spoken_han_characters=count,
        planned_main_seconds=seconds,
        main_exceeds_12_minutes_at_300=count / 300 >= 12,
        content_status='revised_content_ready_for_deck_build',
    )
    for rate in [250, 275, 285, 300]:
        meta[f'estimated_minutes_at_{rate}'] = count / rate

    changed_scripts = [a['number'] for a, b in zip(original['slides'], new['slides'])
                       if a['script'] != b['script']]
    if changed_scripts != [13, 14] or len(new['slides']) != 23 or len(main_slides) != 20:
        raise AssertionError('Unexpected scope or slide count')
    for a, b in zip(original['slides'], new['slides']):
        if a['number'] not in [13, 14] and a != b:
            raise AssertionError(f'Unrelated slide changed: {a["number"]}')
    if count / 300 < 12:
        raise AssertionError('Main script does not meet estimated twelve-minute minimum')
    if 1000 * .75 + 1200 * .25 != 1050:
        raise AssertionError('Illustrative branch arithmetic failed')

    estimates = '、'.join(f'{count / rate:.2f}' for rate in [250, 275, 285, 300])
    lines = [
        '# 臺灣房屋總價預測：完整講稿', '',
        '版本：v22 零基礎說明版（分支計算與修正銜接補充）。日期：2026-10-01。', '',
        '主講為第1至20頁，附錄第21至23頁供提問查閱。主講完整口述稿與簡報備忘稿使用同一份 narrative.json。', '',
        f'主講正文共 **{count:,} 個純漢字**。每分鐘250、275、285、300字的估計時間依序為 **{estimates} 分鐘**。'
        '這是依講稿字數估計，尚未本人計時排練。英文、阿拉伯數字、標點、標題、重點、來源、附錄與問答均不計入這個字數。'
        '至少12分鐘的主講不依靠問答或刻意停頓補時。', '',
        '講述原則：先解釋意思，再說技術名稱。第3、10、11、13、15頁的數字例子都是教學示意，不能說成實際競賽交易。', '',
        '## 主講配速', '',
        '| 頁次 | 題目 | 正文純漢字 | 建議秒數 | 累計 |',
        '| --- | --- | ---: | ---: | --- |',
    ]
    cumulative = 0
    for slide in main_slides:
        cumulative += slide['seconds']
        lines.append(f'| {slide["number"]:02d} | {slide["title"]} | {slide["spoken_han_characters"]} | '
                     f'{slide["seconds"]} | {mmss(cumulative)} |')
    lines += ['', '秒數只是以每分鐘285字換算的逐頁配速，並不取代親自排練。不要為了追秒數加速解釋例子。', '']
    for slide in new['slides']:
        role = '附錄，不計主講' if slide['is_appendix'] else '主講'
        lines += [f'## 第{slide["number"]:02d}頁　{slide["title"]}', '',
                  f'**{role}。這頁要讓聽眾理解：** {slide["takeaway"]}', '', slide['script'], '',
                  f'**理解提示：** {slide["why"]}', '', '**證據位置（相對工作區根目錄）：**', '']
        lines += [f'- `{source}`' for source in slide['sources']]
        lines.append('')
    script_md = '\n'.join(lines) + '\n'
    if not all(slide['script'] in script_md for slide in new['slides']):
        raise AssertionError('Missing spoken script in Markdown')

    qa = SOURCE_QA.read_text(encoding='utf-8')
    old_timing = ('主講完整稿目前有 **4,658 個純漢字**，以每分鐘300字估計 **15.53 分鐘**，'
                  '以285字估計 **16.34 分鐘**，以250字估計 **18.63 分鐘**。逐頁配速合計 **16分20秒**。')
    new_timing = (f'主講完整稿目前有 **{count:,} 個純漢字**，以每分鐘300字估計 **{count / 300:.2f} 分鐘**，'
                  f'以285字估計 **{count / 285:.2f} 分鐘**，以250字估計 **{count / 250:.2f} 分鐘**。'
                  f'逐頁配速合計 **{seconds // 60}分{seconds % 60:02d}秒**。')
    qa = once(qa, old_timing, new_timing)
    qa = once(qa, '第3、10、11、15頁的例子要把計算說完', '第3、10、11、13、15頁的例子要把計算說完')
    qa = once(qa,
        '| 13 | 六模型先組三分支，共用同一個LightGBM預測 | 說成六模型直接平均，或把LightGBM算成三個模型 |',
        '| 13 | 六模型先換回總價再組三分支；教學假設1,000萬×75%＋1,200萬×25%＝1,050萬元初估，還需修正 | 說成六模型直接平均，或把75／25當成已證明最佳比例 |')
    qa = once(qa,
        '| 14 | 價格參考只用當期訓練成交，支持不足回退 | 說找到最近一間就直接複製價格 |',
        '| 14 | 每支各自修正；空間步驟同時參考原初估與成交修正結果，完成後仍有三個總價 | 三支先合成一價才修正，或說找到最近一間就直接複製價格 |')
    # Existing sixteen answers and all other rehearsal guidance remain unchanged.
    if qa.split('## 常見問答', 1)[1] != SOURCE_QA.read_text(encoding='utf-8').split('## 常見問答', 1)[1]:
        raise AssertionError('Unexpected Q&A answer change')

    write_new(targets[0], json.dumps(new, ensure_ascii=False, indent=2) + '\n')
    write_new(targets[1], script_md)
    write_new(targets[2], qa)
    after_sha = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    if before_sha != after_sha:
        raise AssertionError('Frozen source files changed')
    receipt = {
        'status': 'content_revised_and_scope_verified',
        'created_date': '2026-10-01',
        'scope': 'Only slides 13 and 14 spoken text and teaching cues changed; no model or predictions changed',
        'source_sha256': before_sha,
        'frozen_sources_unchanged': before_sha == after_sha,
        'script_changed_slides': changed_scripts,
        'other_21_slide_objects_identical': True,
        'slide_count': len(new['slides']),
        'main_slide_count': len(main_slides),
        'before_main_spoken_han_characters': before_count,
        'after_main_spoken_han_characters': count,
        'before_planned_main_seconds': original['meta']['planned_main_seconds'],
        'after_planned_main_seconds': seconds,
        'estimated_minutes_at_250': count / 250,
        'estimated_minutes_at_285': count / 285,
        'estimated_minutes_at_300': count / 300,
        'duration_measured_by_presenter': False,
        'duration_excludes_appendix_and_questions': True,
        'main_exceeds_12_minutes_at_300': count / 300 >= 12,
        'per_changed_slide': [{
            'number': a['number'], 'before_han': han(a['script']), 'after_han': han(b['script']),
            'before_seconds': a['seconds'], 'after_seconds': b['seconds'],
        } for a, b in zip(original['slides'], new['slides']) if a['number'] in changed_scripts],
        'example': {'kind': 'hypothetical_after_total_price_restoration', 'unit': 'TWD_10000',
                    'C_unit': 1000, 'L_unit': 1200, 'C_weight': .75, 'L_weight': .25, 'B_unit': 1050,
                    'reference_and_spatial_correction_already_applied': False,
                    'claims_75_25_is_optimal': False},
        'checks': {'all_scripts_in_markdown': True, 'all_16_qa_answers_unchanged': True,
                   'branch_example_arithmetic': True},
        'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in targets[:3]},
        'source_script_sha256': sha(Path(__file__)),
        'not_performed': ['No PPTX built by this script', 'No new ML fit or prediction', 'No measured human rehearsal'],
    }
    write_new(targets[3], json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
