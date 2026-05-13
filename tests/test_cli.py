import subprocess
import sys


def test_cli_help():
    "Verifies that the 'schicorr' command is installed and returns the help message."
    try:
        result = subprocess.run(
            ["schiccorr", "--help"], capture_output=True, text=True, check=True
        )
        assert "usage: schiccorr" in result.stdout

        print("\nCLI Help Test Passed!")

    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command failed with error: {e.stderr}")
    except FileNotFoundError:
        pytest.fail("CLI command 'schicorr' not found. Is the package installed?")
