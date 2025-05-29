# `Git and Github : Essential Commands for Industry Use`
## `Author : Rakib Abdullah`
---
## Introduction to Git and GitHub

### What is Git?

Git is a distributed version control system created by Linus Torvalds in 2005. It's designed to track changes in source code during software development and enables multiple developers to work together on non-linear development.

### What is Github

GitHub is a web-based hosting service for Git repositories. Launched in 2008 and acquired by Microsoft in 2018, GitHub provides the distributed version control of Git plus access control, bug tracking, feature requests, task management, continuous integration, and other collaborative features.

### Why Use Git and GitHub?

Together, Git and GitHub form the backbone of modern software development by enabling:

- **Collaboration**: Multiple developers can work on the same project simultaneously
- **History tracking**: Every change is recorded with author information
- **Rollback capability**: Revert to previous states if needed
- **Branching strategies**: Isolate work in progress from production code
- **Code reviews**: Improve code quality through peer review
- **Documentation**: Track why changes were made, not just what changed
- **Open-source contributions**: Facilitate participation in and maintenance of open-source projects
- **Portfolio building**: Showcase your work and contributions to potential employers

Nearly every software development team in industry uses Git for version control, with GitHub being the most popular hosting platform. Understanding these tools is an essential skill for developers, DevOps engineers, and many IT professionals.

---

## Initial Configuration

#### Set username and email (required for commits)
`git config --global user.name "My Name"`
`git config --global user.email "my.email@example.com"`

#### Set default editor
`git config --global core.editor "code --wait"  # For VS Code`

#### Set default branch name
`git config --global init.defaultBranch main`

## Starting a New Repository

#### Initialize a new Git repository
`git init`

#### Clone an existing repository
`git clone https://github.com/username/repository.git`

#### Clone a specific branch
`git clone -b branch-name https://github.com/username/repository.git`

---
## Basic Git Commands

#### Check status of working directory
`git status`

#### Add files to staging area
`git add filename`                # Add specific file
`git add directory/`              # Add all files in directory
`git add .`                         # Add all files in current directory
`git add -p`                       # Interactively add changes in hunks

#### Remove files
`git rm filename`                   # Remove file from Git and working directory
`git rm --cached filename`          # Stop tracking file but keep in working directory

### Committing Changes

#### Commit staged changes
`git commit -m "Commit message"``

#### Add changes and commit in one command
`git commit -am "Commit message"`   # Only works for modified files, not new files

#### Amend the most recent commit
`git commit --amend -m "Updated commit message"`

---

## Branching and Merging

### Branch Management

#### Create a new branch
`git branch branch-name`

#### Switch to a branch
`git checkout branch-name`
`git switch branch-name`            # Git 2.23+ alternative

#### Create and switch to a new branch
`git checkout -b new-branch`
`git switch -c new-branch`          # Git 2.23+ alternative

#### Delete a remote branch
`git push origin --delete branch-name`
`git push origin :branch-name`      # Alternative syntax

#### Rename a branch
`git branch -m old-name new-name`

---
### Merging and Handling Conflicts

#### Merge a branch into current branch
`git merge branch-name`

#### Abort a merge during conflicts
`git merge --abort`

#### Continue merge after resolving conflicts
`git add <resolved-files>`
`git commit`

### Rebasing

#### Rebase current branch onto another branch
`git rebase base-branch`

#### Interactive rebase for editing commits
`git rebase -i HEAD~3`              # Rebase last 3 commits

#### Continue rebase after resolving conflicts
`git rebase --continue`

#### Abort rebase
`git rebase --abort`

#### Remote Repositories

#### List remote repositories
`git remote -v`

#### Add a remote repository
`git remote add origin https://github.com/username/repository.git`

#### Change the URL of an existing remote
`git remote set-url origin https://github.com/username/new-repository.git`

#### Remove a remote
`git remote remove remote-name`

## Syncing with Remote

#### Fetch changes from remote without merging
`git fetch origin`
`git fetch --all`                   # Fetch from all remotes

#### Pull changes (fetch and merge)
`git pull origin branch-name`
`git pull`                          # Pull from tracking branch

#### Push changes to remote
`git push origin branch-name`
`git push`                         # Push to tracking branch
`git push -u origin branch-name`    # Push and set tracking branch

#### Push all branches
`git push --all origin`

# Force push (use with caution!)
`git push -f origin branch-name`    # Overwrites remote branch
`git push --force-with-lease`       # Safer force push - fails if remote has changes you haven't pulled

---
## Collaborative Workflows

### Working with Forks

#### Add original repository as remote
`git remote add upstream https://github.com/original-owner/original-repository.git`

#### Sync fork with original repo
`git fetch upstream`
`git merge upstream/main`

---
## Viewing History
#### View commit history
`git log`
`git log --oneline`                 # Compact view
`git log --graph --oneline --all`   # Visual representation of branches
`git log -p`                        # Show patches (changes)
`git log --stat`                    # Show stats for each commit
`git log --author="username"`       # Filter by author
`git log --since="2 weeks ago"`     # Filter by time

#### Show details of a specific commit
`git show commit-hash`

---
## Undoing Changes

### Reverting Working Directory

#### Discard changes in working directory for specific file
`git checkout -- filename`
`git restore filename`               # Git 2.23+ alternative

#### Discard all changes in working directory
`git checkout .`
`git restore .`                      # Git 2.23+ alternative

### Unstaging Files

#### Remove file from staging area
`git reset filename`
`git restore --staged filename`      # Git 2.23+ alternative

#### Reverting Commits

#### Create new commit that undoes changes from a previous commit
`git revert commit-hash`

#### Reset to specific commit (removes commits after it)
`git reset --soft commit-hash`       # Keeps changes staged
`git reset commit-hash`              # Keeps changes unstaged
`git reset --hard commit-hash`       # Discards all changes (DANGEROUS)

---

### Recovering Lost Work

#### Show all actions including those that might be lost
`git reflog`

#### Recover branch deleted accidentally
`git checkout -b recovered-branch commit-hash`

### Stashing Changes

#### Save changes temporarily
`git stash`

#### Save with description
`git stash save "Work in progress on feature"`

#### List stashed changes
`git stash list`

#### Apply most recent stash
`git stash apply`

#### Apply specific stash
`git stash apply stash@{2}`

#### Apply and remove stash
`git stash pop`

#### Remove a stash
`git stash drop stash@{0}`

#### Clear all stashes
`git stash clear`



