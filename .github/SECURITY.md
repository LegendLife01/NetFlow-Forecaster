# Security Policy

## Supported Versions

This is a portfolio and experimentation project. Security fixes are handled on
the `main` branch.

## Reporting a Vulnerability

Please do not open public issues for vulnerabilities involving credentials,
private telemetry, command injection, dependency compromise, or unsafe file
handling.

Report privately through GitHub Security Advisories when available, or contact
the repository owner through GitHub.

Include:

- A short description of the issue.
- Steps to reproduce.
- Affected files or commands.
- Potential impact.
- Suggested fix, if you have one.

## Scope

Relevant reports include:

- Unsafe handling of local files or generated run artifacts.
- Command execution risks in runners or scripts.
- Exposure of secrets, API tokens, or Kaggle credentials.
- Dependency vulnerabilities that affect normal project usage.

Out of scope:

- Claims based only on model accuracy or benchmark quality.
- Generated run data that was intentionally created locally and ignored by git.
- Vulnerabilities in external services outside this repository.
