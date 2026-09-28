import importlib.util
import unittest
from pathlib import Path
import tempfile

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_data.py'


class PreparationTests(unittest.TestCase):
    def test_hash_file_works_without_python_311_file_digest(self):
        self.assertTrue(MODULE.exists(), 'data preparation implementation is missing')
        spec = importlib.util.spec_from_file_location('prepare', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b'abc')
            handle.flush()
            self.assertEqual(module.hash_file(Path(handle.name)),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_nested_variants_and_duplicate_answers_cannot_leak(self):
        self.assertTrue(MODULE.exists(), 'data preparation implementation is missing')
        spec = importlib.util.spec_from_file_location('prepare', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        answer = '这是用于测试的回答文本。' * 5
        rows = [
            {'questions': [['高血压日常如何管理？', '高血压生活需要注意什么']], 'answers': [answer]},
            {'questions': [['高血压生活需要注意什么？']], 'answers': ['另一条回答，重复问题必须识别。' * 5]},
            {'questions': [['糖尿病生活如何管理']], 'answers': [answer]},
        ]
        accepted, audit = module.clean_records(rows)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['instruction'], '高血压日常如何管理？')
        self.assertEqual(audit['duplicate'], 2)

    def test_filters_do_not_turn_empty_or_dosage_content_into_training_answers(self):
        self.assertTrue(MODULE.exists(), 'data preparation implementation is missing')
        spec = importlib.util.spec_from_file_location('prepare', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [
            {'questions': [['糖尿病如何管理']], 'answers': ['']},
            {'questions': [['高血压吃什么药']], 'answers': ['建议服用药物，每次50mg，每日三次。' * 5]},
            {'questions': [['糖尿病日常管理']], 'answers': ['可以拨打13812345678联系我们。' * 5]},
            {'questions': [['普通感冒怎么办']], 'answers': ['这段内容不属于慢病问答范围。' * 5]},
        ]
        accepted, audit = module.clean_records(rows)
        self.assertEqual(accepted, [])
        self.assertEqual(sum(audit.values()), 4)

    def test_common_chronic_respiratory_question_is_in_scope(self):
        self.assertTrue(MODULE.exists(), 'data preparation implementation is missing')
        spec = importlib.util.spec_from_file_location('prepare', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        answer = '哮喘需要长期规范管理，并根据医生制定的方案复诊和监测。' * 4
        accepted, _ = module.clean_records([{'questions': [['哮喘平时怎样管理？']], 'answers': [answer]}])
        self.assertEqual(len(accepted), 1)


if __name__ == '__main__':
    unittest.main()
