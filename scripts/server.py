"""Streaming /chat adapter required by the A4 showcase platform."""
import json
import os
from threading import Lock

from flask import Flask, Response, jsonify, request
import requests


SYSTEM = ('你是中文健康科普助手。结合上下文直接回答当前问题，不重复旧内容。'
          '先判断当前问题是否延续上一轮；若症状或身体部位不同，只回答当前问题。'
          '先判断症状是否紧急：紧急时只说明风险、就医紧迫性和危险信号，'
          '不要给休息、多喝水、清淡饮食、忌辛辣油腻等日常建议。'
          '非紧急时只选一到两条与当前症状直接相关的建议，例如胃肠不适可减少辛辣刺激，'
          '头痛可适当休息；不要把休息、喝水和饮食禁忌套成固定组合。'
          '不诊断、不提供具体药名或自行用药建议。')
SAFE_FALLBACK = ('仅凭当前描述不能确定原因，请观察症状的变化和持续时间。'
                 '如果症状明显、持续加重或影响正常活动，请及时就医。')
URGENT_FALLBACK = ('你描述的症状可能涉及需要紧急处理的情况，请立即就医。'
                   '如果症状正在加重或伴有呼吸困难、意识异常、反复呕吐等情况，请呼叫急救。')
FIRST_CARE = '理解你现在可能有些担心，我先根据你的描述给出一些安全建议。'
FOLLOW_UP_CARE = '谢谢你补充这个情况，我会根据你现在的描述给出一些安全建议。'
END_COMMANDS = {'结束回答', '结束对话', '不用了'}
END_REPLY = '好的，本次咨询先到这里。请照顾好自己，如有新的不适，欢迎随时再来。'
TOPICS = (
    ('腹泻', ('腹泻', '拉肚子', '拉水', '稀便')),
    ('头痛或眩晕', ('头疼', '头痛', '头晕', '眩晕', '天旋地转')),
    ('胃部不适', ('胃疼', '胃痛', '胃部', '冰淇淋', '冷饮')),
)
TOPIC_FALLBACKS = {
    '腹泻': ('仅凭当前描述不能确定原因。腹泻期间可少量多次补充水分；'
           '如果频繁水样便、明显口渴乏力、便血或无法进水，请及时就医。'),
    '头痛或眩晕': ('仅凭当前描述不能确定原因。可以先在安静环境中休息，'
              '头晕时避免驾驶或高处活动；如果突然剧烈发作或伴有肢体无力、意识异常，请立即就医。'),
    '胃部不适': ('仅凭当前描述不能确定原因。可暂时减少辛辣、油腻等可能刺激胃肠道的食物；'
             '如果疼痛持续或伴有呕血、黑便，请及时就医。'),
}
URGENT_PATTERNS = (
    ('胰腺炎',), ('呕血',), ('黑便',), ('晕厥',), ('意识不清',), ('呼吸困难',),
    ('胸痛',), ('剧烈', '头痛'), ('剧烈', '腹痛'), ('腹痛', '放射', '背'),
    ('左上腹', '背部'), ('持续加重',),
)
GENERIC_SELF_CARE = ('注意休息', '避免劳累', '多喝水', '清淡饮食', '忌辛辣',
                     '避免辛辣', '油腻食物')


def is_urgent(query):
    return any(all(word in query for word in pattern) for pattern in URGENT_PATTERNS)


def topic_for(text):
    return next((name for name, words in TOPICS if any(word in text for word in words)), None)


def fallback_for(query, urgent):
    if urgent:
        return URGENT_FALLBACK
    return TOPIC_FALLBACKS.get(topic_for(query), SAFE_FALLBACK)


def care_opening(query, history):
    if not history:
        return FIRST_CARE
    current = topic_for(query)
    previous = topic_for(history[-1])
    if current and previous and current != previous:
        return f'理解你现在可能有些担心，我先根据你当前描述的{current}情况给出一些安全建议。'
    return FOLLOW_UP_CARE


def build_prompt(query, history):
    if is_urgent(query):
        advice = ('当前描述可能紧急，只说明风险、尽快就医和需要立即呼叫急救的危险信号；'
                  '不要输出休息、喝水或饮食类建议。')
    else:
        advice = ('生活建议只选择一到两条与当前症状直接相关的内容；'
                  '不要套用固定组合，也不要罗列无关的休息、喝水或饮食禁忌。')
    if not history:
        return (f'用户当前问题：{query}\n{advice}'
                '先给安全、简短的处理建议，再说明可能原因；不要复述问题。')
    return (f'用户上一轮描述：{history[-1]}\n用户当前补充：{query}\n'
            '先判断当前补充与上一轮症状是否可能相关。若属于不同症状或身体部位，'
            f'只回答当前问题，不引用上一轮；若确实相关，再结合上下文回答。{advice}')


def limit_sentences(chunks, limit=2, urgent=False):
    unsafe_phrases = ('可以服用', '可服用', '建议服用', '可以使用', '建议使用', '口服',
                      '输液', '抗感染', '抗菌', '消炎药', '静脉滴注', '注射治疗')
    buffer = ''
    count = 0
    for chunk in chunks:
        buffer += chunk
        start = 0
        for index, char in enumerate(buffer):
            if char in '。！？':
                sentence = buffer[start:index + 1]
                start = index + 1
                if any(phrase in sentence for phrase in unsafe_phrases):
                    continue
                if urgent and any(phrase in sentence for phrase in GENERIC_SELF_CARE):
                    continue
                yield sentence
                count += 1
                if count >= limit:
                    return
        buffer = buffer[start:]


def vllm_stream(query, history):
    url = os.getenv('VLLM_BASE_URL', 'http://127.0.0.1:22222/v1') + '/chat/completions'
    prompt = build_prompt(query, history)
    payload = {'model': os.getenv('MODEL_NAME', 'Qwen2.5-7B-ft'), 'stream': True,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': prompt}],
               'temperature': 0.1, 'top_p': 0.9, 'repetition_penalty': 1.15,
               'max_tokens': 256}
    with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as upstream:
        upstream.raise_for_status()
        for line in upstream.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            chunk = json.loads(data)['choices'][0]['delta'].get('content')
            if chunk:
                yield chunk


def create_app(generator=vllm_stream):
    app = Flask(__name__)
    histories = {}
    history_lock = Lock()

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': os.getenv('MODEL_NAME', 'Qwen2.5-7B-ft')}

    @app.post('/chat')
    def chat():
        data = request.get_json(silent=True) or {}
        fields = ('request_id', 'phone_number', 'query')
        if any(not data.get(field) for field in fields):
            return jsonify({'error': 'Missing required fields'}), 400

        if data['query'].strip() in END_COMMANDS:
            with history_lock:
                histories.pop(data['phone_number'], None)
            line = json.dumps({'request_id': data['request_id'],
                               'phone_number': data['phone_number'],
                               'response': END_REPLY}, ensure_ascii=False) + '\n'
            return Response(line, content_type='application/json; charset=utf-8')

        with history_lock:
            history = list(histories.get(data['phone_number'], ()))

        def response_lines():
            try:
                emitted = False
                care = care_opening(data['query'], history)
                urgent = is_urgent(data['query'])
                for content in limit_sentences(generator(data['query'], history), urgent=urgent):
                    if not emitted:
                        content = care + content
                    emitted = True
                    yield json.dumps({'request_id': data['request_id'],
                                      'phone_number': data['phone_number'],
                                      'response': content}, ensure_ascii=False) + '\n'
                if not emitted:
                    yield json.dumps({'request_id': data['request_id'],
                                      'phone_number': data['phone_number'],
                                      'response': care + fallback_for(data['query'], urgent)},
                                     ensure_ascii=False) + '\n'
                with history_lock:
                    # ponytail: memory-only and three prompts; use a session store for multi-instance deployment.
                    histories[data['phone_number']] = (history + [data['query']])[-3:]
            except Exception:
                app.logger.exception('model request failed')
                yield json.dumps({'request_id': data['request_id'],
                                  'phone_number': data['phone_number'],
                                  'response': '模型服务暂时不可用，请稍后重试。'}, ensure_ascii=False) + '\n'
        return Response(response_lines(), content_type='application/json; charset=utf-8')

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '20003')), threaded=True)
