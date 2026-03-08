## Cursor Cloud specific instructions

This repository is currently an empty scaffold. It contains only a GitHub Actions CI workflow template (`.github/workflows/blank.yml`) that echoes "Hello, world!". There is no application code, no dependencies, no tests, and no lint configuration.

- **No services to run.** There are no backend/frontend services, databases, or other infrastructure.
- **No package manager.** No `package.json`, `requirements.txt`, `go.mod`, or similar exists yet.
- **No build/test/lint commands.** The only executable content is the GitHub Actions workflow, which can be validated locally by checking YAML syntax (`python3 -c "import yaml; yaml.safe_load(open('.github/workflows/blank.yml'))"`)

When application code is added to this repository, this file should be updated with relevant service startup, testing, and build instructions.
