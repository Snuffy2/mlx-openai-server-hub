# Coding Guidelines for AI Agents

This document outlines the coding standards and best practices that AI agents must follow when contributing to the `mlx-openai-server-hub` project. These guidelines ensure consistency, maintainability, and high-quality code.

## Development Environment

- **Virtual Environment**: All Python commands must be executed within a virtual environment. Create a `.venv` directory if it does not exist, and install dependencies and development dependencies from `pyproject.toml` as needed.
- **Dependency Management**: Never install packages globally using `pip`. Update `pyproject.toml` when adding new dependencies, and document the rationale in commit messages or pull request descriptions.
- **Pre-commit Hooks**: Utilize pre-commit with Ruff for code formatting and linting.
- **Type Checking**: Employ MyPy for static type checking.

## Code Quality

- **Docstrings**: Provide docstrings for all files, classes, and methods. Use NumPy-style docstrings for methods.
- **Type Annotations**: Include type annotations on all function signatures, method signatures, and class attributes. Always specify return types, including `None` where applicable.
- **Future Annotations**: Import `from __future__ import annotations` to enable deferred evaluation of type annotations, facilitating forward references without string literals.
- **Generic Types**: For Python 3.11+, use built-in generic types (e.g., `dict[str, Any]`, `list[str]`) instead of their `typing` module counterparts (e.g., `Dict[str, Any]`, `List[str]`).
- **Import Statements**: Place all import statements at the top of the file.
- **Exception Handling**: Avoid broad `except Exception` clauses; opt for specific exceptions. Use tuple syntax for catching multiple exceptions. If an exception cannot be handled meaningfully, re-raise it or raise a more descriptive custom exception.
- **Comments**: Preserve existing comments and maintain clear, up-to-date comments for complex algorithms, configuration settings, and architectural decisions.

## Testing and Validation

- **Testing Framework**: Pytest is not currently in use, and tests do not need to be created at this time.
- **Input Validation**: Validate all user inputs, particularly file paths and model parameters.
- **Security**: Do not hardcode API keys, tokens, or credentials. Use environment variables or secure configuration files instead.

## Asynchronous Programming

- **Async/Await**: Use `async/await` for I/O-bound operations to ensure responsiveness.

## Documentation and Maintenance

- **README Updates**: Revise the README when introducing significant features, altering CLI options, or changing installation procedures.
- **API Documentation**: For new API endpoints, document request/response schemas and provide usage examples.
- **Dependency Review**: Regularly check dependencies for known vulnerabilities.

## Version Control

- **Branching and Pull Requests**: Only create new branches or open pull requests when explicitly requested by the user via the `#github-pull-request-agent` hashtag. By default, work on the current branch and commit changes locally without initiating pull requests.

## Additional Recommendations

- **Code Style**: Adhere to PEP 8 guidelines for Python code style.
- **Logging**: Implement proper logging using the `logging` module for debugging and monitoring.
- **Error Messages**: Provide clear and informative error messages to aid in troubleshooting.
- **Code Reviews**: Encourage code reviews for significant changes to maintain quality.
- **Performance**: Optimize code for performance, especially in resource-intensive operations.
- **Modularity**: Write modular, reusable code with clear separation of concerns.
- **Version Compatibility**: Ensure code compatibility with the target Python version (3.11+).
- **Continuous Integration**: Integrate CI/CD pipelines for automated testing and deployment where applicable.