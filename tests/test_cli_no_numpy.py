import subprocess
import sys
import textwrap
import unittest


class TestCliNoNumpy(unittest.TestCase):
    def test_cli_imports_for_server_mode_without_numpy(self):
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockNumpy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'numpy' or fullname.startswith('numpy.'):
                        raise ImportError('blocked numpy for server-only smoke test')
                    return None

            sys.meta_path.insert(0, BlockNumpy())
            import src.cli
            parser = src.cli.build_parser()
            args = parser.parse_args(['serve-api', '--help'])
            """
        )
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
