import os
import unittest
from collections.abc import Mapping

class MyTestCase(unittest.TestCase):
    def test_something(self):
        # type(...) 不是 str，需用 str() 或 f-string 转成可打印文本
        print(f"os.environ 的类型是 {type(os.environ)}")
        is_mapping = isinstance(os.environ, Mapping)
        print(f"isinstance(os.environ, Mapping) -> {is_mapping}")
        self.assertTrue(is_mapping)
        for k, v in os.environ.items():
            print(f"{k}={v}")


if __name__ == '__main__':
    # 默认 buffer=True 时，测试通过会吞掉 print；想看到输出请关掉缓冲
    unittest.main(buffer=False)
