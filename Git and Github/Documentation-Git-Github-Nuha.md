# Introduction to GitHub

## Git vs GitHub

**Git** is a distributed version control system that lets developers track changes to files and collaborate on code. It works locally on the machine.

**GitHub** is a cloud-based platform built around Git. It hosts Git repositories and provides collaboration tools like pull requests, issues, and actions. We use Git locally and push our work to GitHub to share it.

## Why use GitHub?

GitHub makes collaboration easy. It offers:
- A centralized place for Git repositories
- Tools to manage code reviews and pull requests
- Integration with CI/CD tools
- Issue tracking and project management
- Community engagement via stars, forks, and discussions

---

# Basic Commands on Git

## `git init` and `git clone`

- `git init`: Initializes a new Git repository in the current directory.
  ```bash
  git init
  ```
  This creates a `.git` folder, enabling version control.

- `git clone`: Clones an existing remote repository to the local machine.
  ```bash
  git clone https://github.com/username/repo.git
  ```

## `git add`, `git commit`, `git status`, `git log`

- `git add`: Adds changes to the staging area.
  ```bash
  git add file.txt
  ```

- `git commit`: Records changes in the repository.
  ```bash
  git commit -m "Add new feature"
  ```

- `git status`: Shows the current state of the working directory.
  ```bash
  git status
  ```

- `git log`: Displays the commit history.
  ```bash
  git log
  ```

## `.gitignore`

`.gitignore` tells Git which files or folders to ignore.
Example `.gitignore`:
```
*.log
node_modules/
.env
```
This avoids committing unnecessary files to the repo.

---

# GitHub Repo & Branch

- A **GitHub repo** is a hosted version of the local Git repository.
- A **branch** is a parallel version of the project. The default is `main`, but other branches can be created as well:
  ```bash
  git branch feature-x
  ```

Branches allow multiple developers to work on features independently.

---

# Working with Branch & Remote

- To create and switch to a new branch:
  ```bash
  git checkout -b new-feature
  ```

- Push the branch to GitHub:
  ```bash
  git push origin new-feature
  ```

- To link a local repo to a remote GitHub repo:
  ```bash
  git remote add origin https://github.com/username/repo.git
  ```

---


## `git fetch`

Fetches changes from the remote repository but doesn’t merge them into the current branch.
```bash
git fetch origin
```
This is used when we want to review changes before merging.

## `git merge`

Merges changes from one branch into the current branch.
```bash
git merge feature-branch
```
This brings the commits from `feature-branch` into the current branch (e.g., `main`).

## `git pull`

Pulls changes from the remote repo and merges them into the current branch.
```bash
git pull origin main
```
Equivalent to `git fetch` followed by `git merge`.

## `git stash`

Temporarily stores uncommitted changes so one can switch branches safely.
```bash
git stash
```

## `git stash pop`

Applies the last stashed changes and removes them from the stash list.
```bash
git stash pop
```

## `git rebase`

Re-applies the commits on top of another base tip. Used to make a linear history.
```bash
git rebase main
```
If user is on `feature`, this replays `feature` on top of `main`.

**Example:**
If we are on a feature branch and want to sync it with updated `main`:
```bash
git checkout feature
git fetch origin
git rebase origin/main
```

This keeps the history clean and avoids unnecessary merge commits.

---

# GitHub Actions

**GitHub Actions** is a CI/CD tool built into GitHub. It allows us to automate workflows like testing, building, and deploying code every time we push.


This workflow:
1. Runs on every `push` or `pull_request`
2. Checks out the code
3. Sets up Python
4. Installs dependencies
5. Runs our tests

GitHub Actions help maintain code quality and automate deployment, reducing manual tasks.
