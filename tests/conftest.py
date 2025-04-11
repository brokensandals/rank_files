def pytest_addoption(parser):
    parser.addoption("--ollama", action="store_true", help="run tests that invoke Ollama API")
    parser.addoption("--anthropic", action="store_true", help="run tests that invoke Anthropic API")
