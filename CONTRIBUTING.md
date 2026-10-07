# Contributing to Phyne

Hey there! First off, thank you for considering contributing to the `phyne` crate! This is a young crate in active development, so any contributions are welcome.

To keep the code clean and good quality, please check out the [Rust API guidelines]( https://rust-lang.github.io/api-guidelines/about.html ) and try to follow them when contributing to the codebase.

And just as a gentle reminder, all contributors are expected to follow [Rust's Code of Conduct]( https://www.rust-lang.org/policies/code-of-conduct ).

We welcome all types of contributions:

- **🐛 Bug fixes**: Help us squash bugs!
- **✨ New features**: Enhance the crate's functionality;
- **📚 Documentation**: Improve docs, examples, or guides;
- **🔧 Performance**: Optimise existing code;
- **🧪 Tests**: Add test coverage;
- **🎨 Code quality**: Refactoring and cleanup.

[Bug Reports](#bug-reports-and-feature-requests) • [Getting Started](#getting-started) • [Code Quality](#ensuring-code-quality) • [Code Review](#code-review-process) • [Support](#community--support)

## Bug Reports and Feature Requests

For questions, bug reports, or feature requests, please go to the [phyne discussion page]( https://github.com/acg-team/phyne/discussions ) and/or open an issue on [GitHub]( https://github.com/acg-team/phyne/issues ).

In case you would like to add a feature to the crate, please open an issue first and then tackle it with a pull request.

## Getting Started

### Prerequisites

- **Rust**: install the latest stable Rust toolchain via [rustup](https://rustup.rs/);
- **Git**: for version control;
- **A GitHub account**: for submitting pull requests.

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/phyne.git
   cd phyne/phyne
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/acg-team/phyne.git
   ```
4. **Install development dependencies**:
   ```bash
   rustup component add rustfmt clippy
   ```
5. **Run the test suite** to make sure everything works:
   ```bash
   cargo test
   ```

### Contribution Workflow

1. **Create a new branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the guidelines below
3. **Test thoroughly**:
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```
4. **Commit your changes** with clear, descriptive commit messages
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** on GitHub with a clear description of your changes

## Ensuring Code Quality

If you decide to tackle an issue from our [issue list]( https://github.com/acg-team/phyne/issues ), please follow these guidelines to make the review process go smoothly. We recommend reading [Rust crate guidelines]( https://rust-lang.github.io/api-guidelines/about.html ) to make sure your contribution is up to scratch!

When you submit a pull request, it will be automatically tested and code coverage will run with GitHub Actions. We are aiming to maintain our current code coverage percentage, so if the report claims that some new features are missing tests, please add some! Bear in mind that the code coverage report will take around 1.5 hours to generate, so please be patient. In addition to running the tests, GitHub Actions runs Clippy and `rustfmt` on each PR.

### Running Tests

Before submitting a pull request, please run the test suite locally:

```bash
cargo test
```

For faster test runs during development, you can also use:

```bash
cargo test --features precomputed-test-results
```

### Formatting Code with `rustfmt`

Before you make your pull request to the project, please run it through the `rustfmt` utility. This will ensure we have good quality source code that is better for us all to maintain.

1. Install it (`rustfmt` is usually installed by default via [rustup](https://rustup.rs/)):
    ```
    rustup component add rustfmt
    ```
2. You can now run `rustfmt` on a single file simply by...
    ```
    rustfmt src/path/to/your/file.rs
    ```
   ... or you can format the entire project with
   ```
   cargo fmt
   ```
   When run through `cargo` it will format all bin and lib files in the current package.

**Visual Studio Code users**: the [rust-analyzer]( https://rust-analyzer.github.io/ ) extension will automatically run `rustfmt` for you when you save the files.

### Finding Issues with Clippy

[Clippy]( https://doc.rust-lang.org/clippy/ ) is a code analyser/linter detecting mistakes, and therefore helps to improve your code. Like formatting your code with `rustfmt`, running clippy regularly and before your pull request will help us maintain awesome code.

1. To install
    ```
    rustup component add clippy
    ```
2. Running clippy
    ```
    cargo clippy
    ```

**Visual Studio Code users**: you can set Clippy as your default linter if you install the [rust-analyzer]( https://rust-analyzer.github.io/ ) extension and set its `rust-analyzer.check.command` to `clippy`. This will highlight all of Clippy lints in your workspace.

### Documentation

Good documentation is crucial for users and maintainers. For your PR, please ensure:

- **Public APIs** have comprehensive doc comments with doctests;
- **Complex algorithms** include explanatory comments;
- **New features** are documented in relevant modules;
- **Breaking changes** are noted in commit messages.

<!-- To check documentation:
```bash
cargo doc --open
```

### Benchmarks

If your changes affect performance, please run benchmarks:
```bash
cargo bench
```

Add new benchmarks for significant new functionality. -->

## Code Review Process

### What to Expect

- **Automated checks**: Your PR will be tested with GitHub Actions;
- **Review timeline**: We aim to provide initial feedback within a few days;
- **Iterative process**: Expect suggestions and requests for changes;
- **Learning opportunity**: Reviews are collaborative - ask questions!

### Review Criteria

We look for:
- **Correctness**: Does the code work as intended?
- **Performance**: Are there obvious performance issues?
- **API design**: Is the interface intuitive and consistent?
- **Tests**: Are edge cases covered?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow Rust conventions?

## Community & Support

- **Questions about contributing**: Open a [discussion](https://github.com/acg-team/phyne/discussions);
- **Technical questions**: Check existing [issues](https://github.com/acg-team/phyne/issues) or open a new one;
- **Be patient**: Reviews take time;
- **Be collaborative**: We're all here to make the crate better together!

---

Thank you for contributing to `phyne`! 🦀
