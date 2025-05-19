# Git & GitHub Documentation

## Table of Contents
- [Introduction](#introduction)
- [Git Basics](#git-basics)
  - [What is Git?](#what-is-git)
  - [Core Concepts](#core-concepts)
  - [Installation & Setup](#installation--setup)
- [Essential Git Commands](#essential-git-commands)
  - [Initializing Repositories](#initializing-repositories)
  - [Basic Workflow](#basic-workflow)
  - [Viewing History](#viewing-history)
  - [Undoing Changes](#undoing-changes)
- [Branching & Merging](#branching--merging)
  - [Creating & Switching Branches](#creating--switching-branches)
  - [Merging Branches](#merging-branches)
  - [Resolving Conflicts](#resolving-conflicts)
- [GitHub](#github)
  - [What is GitHub?](#what-is-github)
  - [Setting Up a GitHub Account](#setting-up-a-github-account)
  - [Creating Repositories](#creating-repositories)
  - [Remote Operations](#remote-operations)
- [Collaboration with GitHub](#collaboration-with-github)
  - [Forking](#forking)
  - [Pull Requests](#pull-requests)
  - [Issues](#issues)
  - [Code Reviews](#code-reviews)
- [Advanced Git Features](#advanced-git-features)
  - [Rebasing](#rebasing)
  - [Interactive Rebasing](#interactive-rebasing)
  - [Cherry-Picking](#cherry-picking)
  - [Stashing](#stashing)
  - [Git Hooks](#git-hooks)
- [GitHub Additional Features](#github-additional-features)
  - [GitHub Actions](#github-actions)
  - [GitHub Pages](#github-pages)
  - [GitHub Projects](#github-projects)
- [Best Practices](#best-practices)
  - [Commit Messages](#commit-messages)
  - [Branching Strategies](#branching-strategies)
  - [Git Workflow Models](#git-workflow-models)
- [Troubleshooting](#troubleshooting)

## Introduction

Version control is essential for managing changes to code and collaborating with others. Git is the most widely used version control system, and GitHub is a popular platform built around Git that adds collaboration features.

This documentation aims to provide a comprehensive guide to Git and GitHub, covering everything from basic commands to advanced features and best practices.

## Git Basics

### What is Git?

Git is a distributed version control system created by Linus Torvalds in 2005. It allows developers to track changes in their code, revert to previous versions, and collaborate with others without overwriting each other's work.

Key characteristics of Git:
- **Distributed**: Every developer has a complete copy of the repository
- **Speed**: Operations are performed locally, making them fast
- **Data integrity**: Git uses checksums to ensure data integrity
- **Non-linear development**: Supports branching and merging efficiently

### Core Concepts

Understanding these core concepts is crucial for working with Git effectively:

- **Repository (Repo)**: A collection of files and their complete history
- **Commit**: A snapshot of changes at a specific point in time
- **Branch**: An independent line of development
- **Working Directory**: The files you're currently working on
- **Staging Area (Index)**: A place to prepare changes before committing
- **Remote**: A version of your repository hosted on another server (like GitHub)
- **Clone**: A copy of a repository
- **Push/Pull**: Commands to send/receive changes to/from a remote repository

### Installation & Setup

#### Installing Git

**Windows**:
1. Download the installer from [git-scm.com](https://git-scm.com/)
2. Run the installer and follow the instructions

**macOS**:
1. Install via Homebrew: `brew install git`
2. Or download the installer from [git-scm.com](https://git-scm.com/)

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install git
```

**Linux (Fedora)**:
```bash
sudo dnf install git
```

#### Initial Configuration

After installation, set up your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Optional but recommended configurations:

```bash
# Set default editor
git config --global core.editor "vim"  # or "nano", "code --wait", etc.

# Set default branch name
git config --global init.defaultBranch main

# Configure line endings
git config --global core.autocrlf true  # Windows
git config --global core.autocrlf input  # macOS/Linux
```

Verify your configuration:
```bash
git config --list
```

## Essential Git Commands

### Initializing Repositories

**Create a new repository**:
```bash
mkdir my-project
cd my-project
git init
```

**Clone an existing repository**:
```bash
git clone https://github.com/username/repository.git
git clone https://github.com/username/repository.git my-folder  # Clone to specific folder
```

### Basic Workflow

The typical Git workflow involves these steps:

1. Make changes to files
2. Stage the changes
3. Commit the changes with a message
4. Push changes to a remote repository (optional)

**Check status of working directory**:
```bash
git status
```

**Stage changes**:
```bash
git add filename.txt                # Stage a specific file
git add directory/                  # Stage a directory
git add .                           # Stage all changes
git add -p                          # Interactively stage parts of files
```

**Commit staged changes**:
```bash
git commit -m "Add feature X"       # Commit with a message
git commit                          # Opens editor for a detailed commit message
git commit -a -m "Fix bug Y"        # Stage all tracked files and commit
```

**View differences**:
```bash
git diff                            # Changes in working directory not yet staged
git diff --staged                   # Changes between staging area and last commit
git diff HEAD                       # All changes since last commit
git diff commit1 commit2            # Changes between two commits
```

### Viewing History

**View commit history**:
```bash
git log                             # Full commit history
git log --oneline                   # Compact view
git log --graph --oneline --all     # Graphical view of all branches
git log -p                          # Show patches (changes) for each commit
git log --stat                      # Show stats for each commit
git log --author="username"         # Filter by author
git log --since="2 weeks ago"       # Filter by date
```

**Examine a specific commit**:
```bash
git show commit_hash                # Show details of a specific commit
```

### Undoing Changes

**Discard changes in working directory**:
```bash
git checkout -- filename.txt        # Discard changes in a file
git restore filename.txt            # Modern alternative (Git 2.23+)
```

**Unstage changes**:
```bash
git reset HEAD filename.txt         # Unstage a file
git restore --staged filename.txt   # Modern alternative (Git 2.23+)
```

**Amend the last commit**:
```bash
git commit --amend                  # Amend with staged changes
git commit --amend -m "New message" # Amend and change commit message
```

**Reset to a previous state**:
```bash
git reset --soft HEAD~1             # Undo last commit, keep changes staged
git reset HEAD~1                    # Undo last commit, keep changes unstaged
git reset --hard HEAD~1             # Undo last commit, discard changes (DANGEROUS)
```

**Revert a commit**:
```bash
git revert commit_hash              # Create new commit that undoes specified commit
```

## Branching & Merging

Branches allow you to develop features, fix bugs, or experiment without affecting the main codebase.

### Creating & Switching Branches

**List branches**:
```bash
git branch                          # List local branches
git branch -a                       # List all branches (local and remote)
```

**Create a new branch**:
```bash
git branch feature-x                # Create branch
git checkout -b feature-x           # Create and switch to branch
git switch -c feature-x             # Create and switch (Git 2.23+)
```

**Switch branches**:
```bash
git checkout branch-name            # Switch to branch
git switch branch-name              # Modern alternative (Git 2.23+)
```

**Rename a branch**:
```bash
git branch -m old-name new-name     # Rename branch
```

**Delete a branch**:
```bash
git branch -d branch-name           # Delete branch (safe)
git branch -D branch-name           # Force delete branch
```

### Merging Branches

Once you've completed work on a branch, you'll want to incorporate those changes back into another branch (usually main).

**Merge a branch**:
```bash
git checkout main                   # Switch to target branch
git merge feature-x                 # Merge feature-x into current branch
```

**Types of merges**:
- **Fast-forward**: When there are no new changes in the target branch
- **Recursive/Three-way**: When both branches have new commits

**Abort a merge**:
```bash
git merge --abort                   # Abort an in-progress merge
```

### Resolving Conflicts

Conflicts occur when Git can't automatically merge changes. When a conflict happens:

1. Git marks the conflicted files
2. You need to manually edit the files to resolve conflicts
3. After resolving, you stage and commit the changes

**Example conflict markers**:
```
<<<<<<< HEAD
Code from current branch
=======
Code from branch being merged
>>>>>>> feature-branch
```

**Resolve conflicts**:
```bash
# After editing files to resolve conflicts
git add resolved-file.txt
git commit                          # Completes the merge
```

**Merge tools**:
```bash
git mergetool                       # Launch configured merge tool
```

## GitHub

### What is GitHub?

GitHub is a web-based platform built around Git that provides hosting for software development and version control using Git. It adds collaboration features such as:

- Repository hosting
- Pull requests
- Issues and bug tracking
- Project management tools
- Wikis and documentation
- Actions for CI/CD
- Security features

### Setting Up a GitHub Account

1. Go to [github.com](https://github.com/)
2. Click "Sign Up"
3. Follow the instructions to create an account
4. Optional: Set up two-factor authentication for security

### Creating Repositories

**Create a new repository on GitHub**:
1. Click the "+" icon in the top-right corner
2. Select "New repository"
3. Enter a name, description, and select options
4. Click "Create repository"

**Push an existing local repository to GitHub**:
```bash
# Add the remote repository
git remote add origin https://github.com/username/repository.git

# Push your code
git push -u origin main
```

**Create a new repository and push from command line**:
```bash
echo "# My New Repository" >> README.md
git init
git add README.md
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/username/new-repository.git
git push -u origin main
```

### Remote Operations

**Add a remote**:
```bash
git remote add origin https://github.com/username/repository.git
git remote add upstream https://github.com/original-owner/original-repository.git
```

**View remotes**:
```bash
git remote -v                       # List all remotes
```

**Fetch changes from remote**:
```bash
git fetch origin                    # Fetch changes without merging
git fetch --all                     # Fetch from all remotes
```

**Pull changes from remote**:
```bash
git pull origin main                # Fetch and merge changes
git pull --rebase origin main       # Fetch and rebase changes
```

**Push changes to remote**:
```bash
git push origin main                # Push local commits to remote
git push -u origin feature-branch   # Push and set upstream
git push --force origin branch      # Force push (use with caution!)
git push --tags                     # Push tags
```

## Collaboration with GitHub

### Forking

Forking creates a personal copy of someone else's repository, allowing you to freely experiment without affecting the original project.

**To fork a repository**:
1. Navigate to the repository on GitHub
2. Click the "Fork" button in the top-right corner
3. Select your account as the destination

**Clone your fork**:
```bash
git clone https://github.com/your-username/repository.git
```

**Keep your fork in sync**:
```bash
# Add the original repository as "upstream"
git remote add upstream https://github.com/original-owner/repository.git

# Fetch changes from upstream
git fetch upstream

# Merge changes from upstream into your local main branch
git checkout main
git merge upstream/main

# Push the updated main to your fork
git push origin main
```

### Pull Requests

Pull requests (PRs) are proposals to merge code from one branch or fork into another.

**Creating a pull request**:
1. Push your branch to GitHub:
   ```bash
   git push origin feature-branch
   ```
2. Navigate to your repository on GitHub
3. Click "Compare & pull request"
4. Fill in the title and description
5. Click "Create pull request"

**PR best practices**:
- Use a clear, descriptive title
- Include thorough description with context
- Reference related issues using #issue-number
- Keep PRs focused on a single task
- Add appropriate reviewers

### Issues

GitHub Issues is a tracking system for bugs, feature requests, and tasks.

**Creating an issue**:
1. Navigate to the repository
2. Click on "Issues" tab
3. Click "New issue"
4. Fill in the title and description
5. Assign labels, milestones, and people as needed
6. Click "Submit new issue"

**Linking issues and PRs**:
- Include "Fixes #123" or "Closes #123" in a PR description or commit message to automatically close the issue when merged
- Reference an issue without closing it using #123

### Code Reviews

Code reviews are a critical part of the collaborative development process.

**Reviewing a PR**:
1. Navigate to the PR
2. Click on "Files changed"
3. Review the code:
   - Add comments on specific lines
   - Suggest changes with the "suggestion" feature
   - Approve, request changes, or comment on the entire PR

**Responding to reviews**:
- Address comments by making changes
- Respond to comments with explanations or questions
- Request re-review after making changes

## Advanced Git Features

### Rebasing

Rebasing is an alternative to merging that rewrites commit history by applying changes on top of another branch.

```bash
git checkout feature-branch
git rebase main                     # Rebase feature-branch onto main
```

**Interactive rebase**:
```bash
git rebase -i HEAD~3                # Interactively rebase the last 3 commits
```

**When to use rebase**:
- To maintain a cleaner, linear history
- To update a feature branch with latest changes from main
- To clean up commits before sharing

**When NOT to use rebase**:
- On commits that have been pushed and shared with others
- When you want to preserve the branch history

### Interactive Rebasing

Interactive rebasing allows you to modify commits in various ways:

```bash
git rebase -i HEAD~5                # Start interactive rebase for last 5 commits
```

In the interactive rebase editor, you can:
- `pick` - Keep the commit as is
- `reword` - Change the commit message
- `edit` - Pause at this commit to make changes
- `squash` - Combine with previous commit (keeps both messages)
- `fixup` - Combine with previous commit (discards this message)
- `drop` - Remove the commit

**Example interactive rebase file**:
```
pick abc123 Add feature X
reword def456 Fix bug in feature X
squash ghi789 Minor adjustments to X
fixup jkl012 Fix typo
drop mno345 Temporary commit
```

### Cherry-Picking

Cherry-picking allows you to apply specific commits from one branch to another.

```bash
git cherry-pick commit_hash         # Apply a specific commit to current branch
git cherry-pick commit1..commit2    # Apply a range of commits
git cherry-pick -x commit_hash      # Include source in commit message
git cherry-pick --no-commit commit  # Apply changes without committing
```

### Stashing

Stashing lets you temporarily save changes without committing them.

```bash
git stash                           # Stash changes in working directory
git stash save "Work in progress"   # Stash with description
git stash list                      # List all stashes
git stash show stash@{0}            # Show details of a stash
git stash apply                     # Apply most recent stash without removing it
git stash pop                       # Apply most recent stash and remove it
git stash drop stash@{1}            # Delete a specific stash
git stash clear                     # Delete all stashes
```

### Git Hooks

Hooks are scripts that run automatically when certain Git events occur.

Common hooks:
- `pre-commit`: Runs before a commit is created
- `prepare-commit-msg`: Prepares the default commit message
- `commit-msg`: Validates commit messages
- `post-commit`: Runs after a commit is created
- `pre-push`: Runs before pushing commits

**Setting up a hook**:
1. Navigate to `.git/hooks/` in your repository
2. Create a file named after the hook (e.g., `pre-commit`)
3. Make it executable: `chmod +x .git/hooks/pre-commit`
4. Write your script

**Example pre-commit hook** (run tests before committing):
```bash
#!/bin/sh
npm test
if [ $? -ne 0 ]; then
  echo "Tests failed, commit aborted"
  exit 1
fi
```

## GitHub Additional Features

### GitHub Actions

GitHub Actions is an automation and CI/CD platform that allows you to define workflows in YAML files.

**Creating a workflow**:
1. Create a `.github/workflows` directory in your repository
2. Add a YAML file (e.g., `ci.yml`)

**Example workflow** (run tests on push and pull requests):
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '16'
    - name: Install dependencies
      run: npm ci
    - name: Run tests
      run: npm test
```

### GitHub Pages

GitHub Pages is a static site hosting service that takes files directly from your repository.

**Setting up GitHub Pages**:
1. Go to repository settings
2. Scroll to "GitHub Pages" section
3. Select the source branch and folder
4. Save

**Custom domain**:
1. Add a CNAME file to your repository with your domain
2. Update DNS settings with your provider
3. Configure in GitHub Pages settings

### GitHub Projects

GitHub Projects is a flexible project management tool integrated with GitHub.

**Creating a project board**:
1. On repository or organization page, click "Projects"
2. Click "New project"
3. Select a template or start from scratch
4. Name your project and add a description

**Project views**:
- Table: Spreadsheet-like view
- Board: Kanban-style columns
- Roadmap: Timeline view

**Automating projects**:
Set up workflows to automatically move issues and PRs through your board:
- When issues are created
- When PRs are opened
- When PRs are merged

## Best Practices

### Commit Messages

Good commit messages make your repository history more useful.

**Structure**:
```
Short summary line (50 chars or less)

More detailed explanatory text, if necessary. Wrap it to about 72
characters. The blank line separating the summary from the body is
critical.

- Bullet points are okay
- Typically a hyphen or asterisk is used, followed by a space

Fixes: #123
```

**Guidelines**:
- Use the imperative mood ("Add feature" not "Added feature")
- Capitalize the first letter of the summary line
- Don't end the summary line with a period
- Reference relevant issues or PRs
- Explain what and why, not how

### Branching Strategies

Different projects may use different branching strategies:

**GitHub Flow**:
- Simple, lightweight flow
- Main branch should always be deployable
- Create feature branches from main
- Open PR early for discussion
- Merge to main after review and CI passes

**Git Flow**:
More structured with specific branch types:
- `main`: Production-ready code
- `develop`: Latest delivered development changes
- `feature/*`: New features
- `release/*`: Preparing for a release
- `hotfix/*`: Urgent fixes for production

**Trunk-Based Development**:
- Short-lived feature branches
- Frequent merges to main (trunk)
- Feature flags for incomplete features
- Emphasizes CI/CD

### Git Workflow Models

**Centralized Workflow**:
- Single main branch
- All developers commit to main
- Simple but can lead to conflicts

**Feature Branch Workflow**:
- Feature development occurs in dedicated branches
- PRs for code review
- Main branch remains stable

**Forking Workflow**:
- Each developer works on their own fork
- Changes proposed via PRs from forks
- Common in open-source projects

## Troubleshooting

**Common Issues and Solutions**:

1. **Merge conflicts**:
   - Use `git status` to identify conflicted files
   - Edit files to resolve conflicts
   - Use `git add` to mark as resolved
   - Complete with `git commit`

2. **Detached HEAD state**:
   ```bash
   # Save your work in a branch
   git branch temp-branch
   git checkout temp-branch
   # Or in one command
   git checkout -b temp-branch
   ```

3. **Accidentally committed to wrong branch**:
   ```bash
   # Create a new branch with your changes
   git branch correct-branch
   # Reset current branch
   git reset --hard HEAD~1
   # Switch to the new branch
   git checkout correct-branch
   ```

4. **Need to undo a push**:
   ```bash
   # Revert the commit locally
   git revert commit_hash
   # Push the revert
   git push origin branch-name
   ```

5. **Fix authentication issues**:
   ```bash
   # Use a personal access token instead of password
   git remote set-url origin https://username:token@github.com/username/repository.git
   
   # Or use SSH keys
   git remote set-url origin git@github.com:username/repository.git
   ```

6. **Large files rejected by GitHub**:
   - Remove large files and use Git LFS instead
   - Use `.gitignore` to prevent them from being added

7. **Reset to remote state**:
   ```bash
   git fetch origin
   git reset --hard origin/main
   ```