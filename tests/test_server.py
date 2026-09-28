import json
import unittest

from scripts.server import create_app


class ServerTests(unittest.TestCase):
    def test_urgent_symptom_omits_generic_self_care_and_prioritizes_hospital(self):
        app = create_app(lambda _, __: iter([
            '平时注意休息避免劳累，清淡饮食多喝水，忌辛辣油腻食物。',
            '这种情况需要尽快去医院就诊。',
        ]))
        response = app.test_client().post('/chat', json={
            'request_id': 'r-urgent', 'phone_number': 'masked',
            'query': '我左上腹疼痛并放射到背部，我是得了胰腺炎吗？'
        })

        text = ''.join(json.loads(line)['response'] for line in response.text.splitlines())
        self.assertIn('尽快去医院就诊', text)
        self.assertNotIn('注意休息', text)
        self.assertNotIn('多喝水', text)
        self.assertNotIn('清淡饮食', text)
        self.assertNotIn('辛辣油腻', text)

    def test_urgent_symptom_uses_urgent_fallback_when_model_only_gives_self_care(self):
        app = create_app(lambda _, __: iter([
            '平时注意休息，清淡饮食多喝水，避免辛辣油腻食物。'
        ]))
        response = app.test_client().post('/chat', json={
            'request_id': 'r-urgent', 'phone_number': 'masked',
            'query': '我胸痛并且呼吸困难'
        })

        text = json.loads(response.text.splitlines()[0])['response']
        self.assertIn('立即就医', text)
        self.assertNotIn('注意休息', text)
        self.assertNotIn('多喝水', text)

    def test_prompt_requires_advice_to_match_the_current_symptom(self):
        from scripts.server import build_prompt

        prompt = build_prompt('我最近经常头疼', [])
        self.assertIn('只选择一到两条与当前症状直接相关', prompt)
        self.assertIn('不要套用固定组合', prompt)

    def test_safe_fallback_matches_the_symptom_topic(self):
        client = create_app(lambda _, __: iter(['建议服用某某药。'])).test_client()

        stomach = client.post('/chat', json={
            'request_id': 'r-stomach', 'phone_number': 'stomach', 'query': '我胃疼'
        }).text
        headache = client.post('/chat', json={
            'request_id': 'r-head', 'phone_number': 'head', 'query': '我有点头疼'
        }).text

        self.assertIn('辛辣', stomach)
        self.assertNotIn('注意休息', stomach)
        self.assertIn('休息', headache)
        self.assertNotIn('辛辣', headache)

    def test_new_symptom_uses_current_topic_care_opening(self):
        app = create_app(lambda _, __: iter(['正文。']))
        client = app.test_client()
        client.post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked',
            'query': '我早起后头疼剧烈，感觉天旋地转'
        }).get_data()
        response = client.post('/chat', json={
            'request_id': 'r-2', 'phone_number': 'masked',
            'query': '我父亲现在有点拉肚子，一直在拉水'
        })

        text = json.loads(response.text.splitlines()[0])['response']
        self.assertTrue(text.startswith(
            '理解你现在可能有些担心，我先根据你当前描述的腹泻情况给出一些安全建议。'))

    def test_end_command_returns_once_without_calling_model_and_clears_history(self):
        calls = []

        def generator(query, history):
            calls.append((query, history))
            return iter(['正文。'])

        client = create_app(generator).test_client()
        client.post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked', 'query': '我胃痛'
        }).get_data()
        ended = client.post('/chat', json={
            'request_id': 'r-2', 'phone_number': 'masked', 'query': '结束回答'
        })
        client.post('/chat', json={
            'request_id': 'r-3', 'phone_number': 'masked', 'query': '重新咨询'
        }).get_data()

        lines = [json.loads(line) for line in ended.text.splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertIn('本次咨询先到这里', lines[0]['response'])
        self.assertEqual([call[0] for call in calls], ['我胃痛', '重新咨询'])
        self.assertEqual(calls[-1][1], [])

    def test_adds_professional_care_opening_to_first_and_follow_up_turns(self):
        app = create_app(lambda _, __: iter(['正文。']))
        client = app.test_client()

        first = client.post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked', 'query': '我胃痛'
        })
        first.get_data()
        follow_up = client.post('/chat', json={
            'request_id': 'r-2', 'phone_number': 'masked', 'query': '刚吃了冰淇淋'
        })

        first_text = json.loads(first.text.splitlines()[0])['response']
        follow_up_text = json.loads(follow_up.text.splitlines()[0])['response']
        self.assertTrue(first_text.startswith('理解你现在可能有些担心，我先根据你的描述给出一些安全建议。'))
        self.assertTrue(follow_up_text.startswith('谢谢你补充这个情况，我会根据你现在的描述给出一些安全建议。'))

    def test_follow_up_prompt_requires_link_to_previous_symptom(self):
        from scripts.server import build_prompt

        prompt = build_prompt('刚吃了冰淇淋，是什么原因？', ['我胃痛'])
        self.assertIn('我胃痛', prompt)
        self.assertIn('刚吃了冰淇淋', prompt)
        self.assertIn('是否可能相关', prompt)

    def test_follow_up_prompt_ignores_history_when_the_symptom_is_different(self):
        from scripts.server import build_prompt

        prompt = build_prompt('我早起感觉天旋地转，这是怎么了？', ['我胃痛'])
        self.assertIn('若属于不同症状或身体部位', prompt)
        self.assertIn('只回答当前问题', prompt)

    def test_limits_stream_to_two_complete_sentences(self):
        from scripts.server import limit_sentences

        chunks = ['第一句。第二', '句。第三句。']
        self.assertEqual(''.join(limit_sentences(chunks)), '第一句。第二句。')

    def test_skips_self_medication_sentence_and_incomplete_tail(self):
        from scripts.server import limit_sentences

        chunks = ['先少量喝温水。可以服', '用某某药。也可以输液治疗。最后半句']
        self.assertEqual(''.join(limit_sentences(chunks)), '先少量喝温水。')

    def test_rejects_missing_required_fields(self):
        response = create_app(lambda _, __: iter(())).test_client().post('/chat', json={'query': '你好'})
        self.assertEqual(response.status_code, 400)

    def test_streams_platform_json_lines(self):
        app = create_app(lambda _, __: iter(['第一段。', '第二段。']))
        response = app.test_client().post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked', 'query': '高血压如何管理？'
        })
        lines = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(lines, [
            {'request_id': 'r-1', 'phone_number': 'masked',
             'response': '理解你现在可能有些担心，我先根据你的描述给出一些安全建议。第一段。'},
            {'request_id': 'r-1', 'phone_number': 'masked', 'response': '第二段。'},
        ])

    def test_returns_safe_fallback_when_every_sentence_is_filtered(self):
        app = create_app(lambda _, __: iter(['建议服用某某药。']))
        response = app.test_client().post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked', 'query': '胃痛怎么办？'
        })
        lines = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertIn('仅凭当前描述不能确定原因', lines[0]['response'])
        self.assertNotIn('上一轮', lines[0]['response'])
        self.assertNotIn('胃部', lines[0]['response'])

    def test_same_user_receives_previous_turn_as_context(self):
        calls = []

        def generator(query, history):
            calls.append((query, history))
            return iter(['第一次回答' if len(calls) == 1 else '第二次回答'])

        client = create_app(generator).test_client()
        client.post('/chat', json={
            'request_id': 'r-1', 'phone_number': 'masked', 'query': '我胃痛'
        }).get_data()
        client.post('/chat', json={
            'request_id': 'r-2', 'phone_number': 'masked', 'query': '刚吃了冰淇淋，是什么原因？'
        }).get_data()

        self.assertEqual(calls[1][1], ['我胃痛'])


if __name__ == '__main__':
    unittest.main()
