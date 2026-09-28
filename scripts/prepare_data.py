"""Build reproducible chronic-disease SFT data from pinned Huatuo JSONL files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

SYSTEM = '你是中文慢病健康科普助手。提供清晰、审慎的健康教育信息，说明不确定性；不根据简短描述确诊，不开处方或建议自行调整药物。涉及个体治疗应建议咨询医生，遇到危险信号应提示及时就医。'
TOPICS = {
    '高血压': r'高血压|血压',
    '糖尿病': r'糖尿病|血糖|糖化血红蛋白',
    '血脂与肥胖': r'高血脂|高脂血|血脂|胆固醇|甘油三酯|肥胖|脂肪肝|代谢综合征',
    '心血管': r'冠心病|冠状动脉|动脉粥样硬化|心血管|心力衰竭|心衰|脑卒中|脑梗|中风',
    '其他慢病': r'痛风|高尿酸|尿酸|慢性肾|慢阻肺|慢性阻塞性肺|骨质疏松|哮喘|慢性肝炎|肝硬化|甲状腺功能减退|甲减|类风湿|银屑病',
}
PII = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[\dXx](?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|微信|QQ号|联系电话')
DOSE = re.compile(r'\d+(?:\.\d+)?\s*(?:mg|μg|ug|毫克|微克|片/|粒/|支/|单位/)|(?:每次|一次|一日|每日|每天).{0,10}\d+\s*(?:片|粒|支|毫升)|用法用量', re.I)
RISK = re.compile(r'包治|根治|保证治愈|彻底治愈|祖传|秘方|偏方|特效药|无需就医|不用去医院|不要去医院|停用所有|自行停药|抗癌|防癌|排毒|壮阳|补肾|针灸|中药|中医|食疗方|药方|清血脂|净化血液|大血球|血尿不可怕|杜绝|排除毒素|活多久|寿命|能活几年')


def flatten(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten(item)


def normalize(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', unicodedata.normalize('NFKC', text).lower())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hash_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_records(records):
    accepted, audit = [], Counter()
    seen_questions, seen_answers = set(), set()
    for index, row in enumerate(records):
        questions = list(dict.fromkeys(q.strip() for q in flatten(row.get('questions')) if q.strip()))
        answers = [a.strip() for a in flatten(row.get('answers')) if a.strip()]
        if not questions or not answers:
            audit['empty'] += 1
            continue
        # Preserve complete source answers; never truncate text to meet token limits.
        answer = '\n'.join(dict.fromkeys(answers))
        question = questions[0]
        if not 5 <= len(question) <= 240 or not 40 <= len(answer) <= 1400:
            audit['length'] += 1
            continue
        topics = [k for k, pattern in TOPICS.items() if re.search(pattern, ' '.join(questions))]
        if not topics:
            audit['off_topic'] += 1
            continue
        text = '\n'.join(questions + [answer])
        if PII.search(text):
            audit['contact_or_identifier'] += 1
            continue
        if DOSE.search(text) or RISK.search(text):
            audit['risk_text'] += 1
            continue
        qkeys = {digest(normalize(q)) for q in questions}
        akey = digest(normalize(answer))
        if qkeys & seen_questions or akey in seen_answers:
            # Also register discarded variants so duplicate chains remain excluded.
            seen_questions.update(qkeys)
            seen_answers.add(akey)
            audit['duplicate'] += 1
            continue
        seen_questions.update(qkeys)
        seen_answers.add(akey)
        accepted.append({
            'instruction': question, 'input': '', 'output': answer, 'system': SYSTEM,
            '_meta': {'source': row.get('_source', str(index)), 'question_hashes': sorted(qkeys),
                      'answer_hash': akey, 'topic': topics[0]},
        })
        audit['accepted'] += 1
    return accepted, dict(audit)


def read_records(raw):
    for split in ('train', 'validation', 'test'):
        path = raw / f'{split}_datasets.jsonl'
        if not path.exists():
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                row['_source'] = f'{path.name}:{line_number}'
                yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.root
    clean, audit = clean_records(read_records(root / 'data/raw'))
    random.Random(42).shuffle(clean)
    # All question variants/answers were grouped and deduplicated BEFORE splitting.
    # This creates project-specific splits, not the upstream benchmark split.
    evaluation_size = min(1000, len(clean) // 5)
    splits = {'test': clean[:evaluation_size], 'validation': clean[evaluation_size:2*evaluation_size],
              'train': clean[2*evaluation_size:2*evaluation_size+8000]}
    if len(splits['train']) < 100:
        raise ValueError('Insufficient eligible data; refusing to train')
    root.joinpath('reports').mkdir(exist_ok=True)
    info, allq, alla = {}, set(), set()
    for name, rows in splits.items():
        q = {h for r in rows for h in r['_meta']['question_hashes']}
        a = {r['_meta']['answer_hash'] for r in rows}
        assert not q & allq and not a & alla, 'Cross-split duplication'
        allq.update(q); alla.update(a)
        (root / f'data/{name}.json').write_text(json.dumps([{k:v for k,v in r.items() if k != '_meta'} for r in rows], ensure_ascii=False, indent=2))
        (root / f'reports/{name}_provenance.json').write_text(json.dumps([r['_meta'] for r in rows], ensure_ascii=False, indent=2))
        info[f'medical_{name}'] = {'file_name': f'{name}.json', 'columns': {'prompt':'instruction', 'query':'input', 'response':'output', 'system':'system'}}
    (root / 'data/dataset_info.json').write_text(json.dumps(info, indent=2))
    report = {'source': 'FreedomIntelligence/huatuo_encyclopedia_qa', 'revision': '2989899dd5bed83cf9bd17cf9fff9889705436e2',
              'seed': 42, 'filter_counts': audit, 'split_counts': {k:len(v) for k,v in splits.items()},
              'topics': {k:dict(Counter(r['_meta']['topic'] for r in v)) for k,v in splits.items()},
              'source_files_sha256': {p.name:hash_file(p) for p in (root/'data/raw').glob('*_datasets.jsonl')},
              'split_policy': 'Pool source records, deduplicate question variants and exact normalized answers, shuffle with seed 42, reserve test/validation before training.',
              'limitations': 'Rule-based filtering, not medical review. Exact normalization does not guarantee removal of all semantic near-duplicates. Source answers retained without invented corrections.'}
    (root / 'reports/data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (root / 'reports/data_samples.json').write_text(json.dumps(splits['train'][:20], ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
