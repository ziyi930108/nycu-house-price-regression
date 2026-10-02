"""Wrap one caption without altering text, breaking terms, or orphaning punctuation."""
from __future__ import annotations
import re
import unicodedata

MAX_COLUMNS = 26
MIN_SECOND_LINE_COLUMNS = 5
PROTECTED_TERMS = (
    '總價', '單價', '先驗', '中位數', '神經網路', '平方公尺', '有效面積',
    '參考單價', '模型', '機器學習', '建案', '成交參考', '空間修正', '推論',
    '標籤', '驗證', '測試', '過擬合',
)
WORD = re.compile(r'[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]+')
CLAUSE_ENDINGS = '，、：；。！？,.!?;:'
OPENING_PUNCTUATION = '（([｛{「『【《〈“‘'


def caption_lines(caption: str) -> list[str]:
    """Return one or two <=26-character lines, or refuse an unsafe split.

    The width rule intentionally preserves the existing video compositor's
    character-based maximum. English tokens and explicit teaching terms stay
    together. Among eligible boundaries, prefer clause breaks, then balance.
    """
    if not isinstance(caption, str) or not caption.strip() or '\n' in caption or '\r' in caption:
        raise ValueError('Caption must be one nonempty line')
    length = len(caption)
    if length <= MAX_COLUMNS:
        return [caption]
    if length > 2 * MAX_COLUMNS:
        raise ValueError('Caption exceeds two 26-character lines; split the input sentence first')
    protected = [m.span() for m in WORD.finditer(caption)]
    for term in PROTECTED_TERMS:
        protected.extend(m.span() for m in re.finditer(re.escape(term), caption))
    candidates = []
    for split in range(max(MIN_SECOND_LINE_COLUMNS, length - MAX_COLUMNS), MAX_COLUMNS + 1):
        left, right = caption[:split], caption[split:]
        if len(right) < MIN_SECOND_LINE_COLUMNS:
            continue
        if any(start < split < end for start, end in protected):
            continue
        if unicodedata.category(right[0]).startswith('P') or right[0].isspace() or left[-1].isspace():
            continue
        if left[-1] in OPENING_PUNCTUATION:
            continue
        if not any(not unicodedata.category(c).startswith('P') for c in right):
            continue
        score = (0 if left[-1] in CLAUSE_ENDINGS else 1, abs(len(left)-len(right)), -len(left))
        candidates.append((score, [left, right]))
    if not candidates:
        raise ValueError('No safe two-line boundary; revise the caption segmentation explicitly')
    return min(candidates, key=lambda x: x[0])[1]
