"""Generate paired base/LoRA answers and transparent proxy metrics."""
import argparse
import csv
import json
from pathlib import Path


def lcs_f1(candidate, reference):
    if not candidate or not reference:
        return 0.0
    previous = [0] * (len(reference) + 1)
    for left in candidate:
        current = [0]
        for index, right in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    common = previous[-1]
    precision, recall = common / len(candidate), common / len(reference)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def generate(model, tokenizer, prompt):
    messages = [{'role': 'system', 'content': '你是中文慢病健康科普助手。提供清晰、审慎的信息，不做在线确诊或开处方。'},
                {'role': 'user', 'content': prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_new_tokens=384, do_sample=False)
    return tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen2.5-7B-Instruct')
    parser.add_argument('--adapter', default='outputs/qwen2.5-7b-medical-lora-1epoch')
    parser.add_argument('--test', default='data/test.json')
    parser.add_argument('--output', default='reports/model_comparison.json')
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = json.loads(Path(args.test).read_text())[:args.samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    base.eval()
    base_answers = [generate(base, tokenizer, row['instruction']) for row in records]
    tuned = PeftModel.from_pretrained(base, args.adapter)
    tuned.eval()
    comparisons = []
    for row, before in zip(records, base_answers):
        after = generate(tuned, tokenizer, row['instruction'])
        comparisons.append({'question': row['instruction'], 'reference': row['output'],
                            'base_answer': before, 'tuned_answer': after,
                            'base_lcs_f1': round(lcs_f1(before, row['output']), 4),
                            'tuned_lcs_f1': round(lcs_f1(after, row['output']), 4)})
    Path(args.output).write_text(json.dumps(comparisons, ensure_ascii=False, indent=2))
    with Path(args.output).with_name('human_review.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'factual_accuracy_0_5', 'relevance_0_5', 'safety_0_5', 'clarity_0_5', 'reviewer_notes'])
        writer.writerows([[i, '', '', '', '', ''] for i in range(len(comparisons))])


if __name__ == '__main__':
    main()
