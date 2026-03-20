# Contributing 🙏

Thanks for taking the time to contribute! 🎉👍

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the api please create an issue first. This way we can ensure that your precious
work is not in vain.


## Setup with make ⚙️

- Clone the repo `git clone https://github.com/deel-ai/xplique.git`.
- Go to your freshly downloaded repo `cd xplique`
- Be sure to have `make` and `uv` installed on your machine.
- Create a virtual environment and install the necessary dependencies for development with `make prepare-dev` and `. .venv/bin/activate`.
- Run the test suite with `make test`.

Welcome to the team 🔥🚀 !

## Setup without make ⚙️

- Clone the repo `git clone https://github.com/deel-ai/xplique.git`.
- Go to your freshly downloaded repo `cd xplique`
### Install virtualenv with `uv`:
```
uv venv
```
```
. .venv/bin/activate
```

Or with Powershell:
```
PS ~/xplique> path\to\.venv\bin\Activate.ps1
```
```
uv sync
```

You are now ready to code and to be part of the team 🔥🚀 !

## Tests ✅

A pretty fair question would be to know what is `make test` doing ?
It is actually just a command which activate your virtual environment and launch the `tox` command.
So basically, if you do not succeed to use `make` just activate your virtual env and do `tox` !

`tox` on the otherhand will do the following:
- run pytest on the tests folder with python 3.10, python 3.11, python 3.12, and python 3.13.
> Note: If you do not have those 3 interpreters the tests would be only performs with your current interpreter
- run `ruff check .` and `ruff format .` on the xplique main files, also with python 3.10, python 3.11, python 3.12, and python 3.13.
> Note: It is possible that ruff throw false-positive errors. If the linting test failed please check first ruff output to point out the reasons.

Please, make sure you run all the tests at least once before opening a pull request.

A word toward [Ruff](https://pypi.org/project/ruff/) for those that don't know it:
> Ruff is a Python static code analysis tool which looks for programming errors, helps enforcing a coding standard, sniffs for code smells and offers simple refactoring suggestions.

Basically, it will check that your code follow a certain number of convention. Any Pull Request will go through a Github workflow ensuring that your code respect the Ruff conventions (most of them at least).

## Submitting Changes 🔃

After getting some feedback, push to your fork and submit a pull request. We
may suggest some changes or improvements or alternatives, but for small changes
your pull request should be accepted quickly.

Something that will increase the chance that your pull request is accepted:

- Write tests and ensure that the existing ones pass.
- If `make test` is succesful, you have fair chances to pass the CI workflows (linting and test)
- Follow the existing coding style.
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please create one. This would help discussion with the community, and polishing ideas in case of a new feature.

## Documentation 📚
Xplique is a small library but documentation is often a huge time sink for
users. That's why we greatly appreciate any time spent fixing typos or
clarifying sections in the documentation. To setup a local live-server to update
the documentation: `make serve-doc` or activate your virtual env and:
```
CUDA_VISIBLE_DEVICES=-1 mkdocs serve
```
